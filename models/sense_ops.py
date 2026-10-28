"""
Shared SENSE operations for the multicoil TRUST-MRI pipeline (torch, GPU-friendly).

Conventions (identical to eval_sense_block.py / eval_sens_maps.py / MediTok_official):
  * complex tensors: coils / maps (B, C, H, W) complex64, single images (B, H, W) complex64;
    the GPT side uses real views (B, H, W, 2) float32 -> `c2r` / `r2c` convert between them
  * fft2c / ifft2c: centered, orthonormal
  * sensitivity maps S_c: SENSE combine x0 = sum_c conj(S_c) x_c / sum_c |S_c|^2 on the support
    (support = sum|S|^2 > 1e-6; ESPIRiT maps are 0 outside their eigenvalue support, the
    VarNet sens-net maps are RSS-normalised everywhere), expand x_c = S_c x0
  * per-coil hard data consistency: k_c = fft2c(S_c x);  k_c <- y_c on the mask;  x <- combine(ifft2c(k_c))
  * 1D Cartesian masks are column masks; helpers accept (B,H,W,1|2), (B,1,W,1), (B,1,1,W,1), (H,W,1), (W,)
"""
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FFT / dtype helpers
# ---------------------------------------------------------------------------

def fft2c(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def ifft2c(k: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def c2r(z: torch.Tensor) -> torch.Tensor:
    """complex (...,) -> real (..., 2) float32"""
    return torch.view_as_real(z.contiguous()).float()


def r2c(t: torch.Tensor) -> torch.Tensor:
    """real (..., 2) -> complex64"""
    return torch.view_as_complex(t.float().contiguous())


def center_crop_or_pad(x: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """Center crop (or zero-pad, then crop) the last two dims to `shape`."""
    h, w = shape
    H, W = x.shape[-2:]
    if H < h or W < w:
        ph, pw = max(h - H, 0), max(w - W, 0)
        x = torch.nn.functional.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        H, W = x.shape[-2:]
    top, left = (H - h) // 2, (W - w) // 2
    return x[..., top:top + h, left:left + w]


def crop_native_to_grid(x: torch.Tensor, map_hw: Sequence[int], crop: int) -> torch.Tensor:
    """Native-grid image (..., H, W) -> ESPIRiT-map grid (e.g. 320x320) -> (crop x crop).
    The single shared crop used by the dataset, the trainers and the evals."""
    return center_crop_or_pad(center_crop_or_pad(x, tuple(map_hw)), (crop, crop))


# ---------------------------------------------------------------------------
# ACS / mask layout helpers
# ---------------------------------------------------------------------------

def acs_num_low_freqs(W: int, center_fraction: float) -> int:
    """Number of fully sampled centre columns of the 1D masks (dataset/data/subsample.py)."""
    return int(round(W * center_fraction))


def acs_col_range(W: int, center_fraction: float) -> Tuple[int, int]:
    n = acs_num_low_freqs(W, center_fraction)
    pad = (W - n + 1) // 2
    return pad, pad + n


def mask_cols(mask: torch.Tensor, batch: Optional[int] = None) -> torch.Tensor:
    """1D-Cartesian mask in any layout -> (B, W) bool (row 0 of 2D layouts)."""
    if mask.dim() == 5:          # (B,1,1,W,1)
        m = mask[:, 0, 0, :, 0]
    elif mask.dim() == 4:        # (B,H,W,1|2) or (B,1,W,1)
        m = mask[:, 0, :, 0]
    elif mask.dim() == 3:        # (H,W,1)
        m = mask[0, :, 0][None]
    elif mask.dim() == 2:        # (H,W) if square else (B,W)
        m = mask[0][None] if mask.shape[0] == mask.shape[1] else mask
    elif mask.dim() == 1:        # (W,)
        m = mask[None]
    else:
        raise ValueError(f"unsupported mask shape {tuple(mask.shape)}")
    m = m.bool()
    if batch is not None and m.shape[0] != batch:
        assert m.shape[0] == 1, f"cannot broadcast mask batch {m.shape[0]} to {batch}"
        m = m.expand(batch, -1)
    return m


def mask_for_sens_net(mask: torch.Tensor, batch: Optional[int] = None) -> torch.Tensor:
    """-> (B,1,1,W,1) float, the layout SensitivityModel expects (only the ACS columns matter)."""
    m = mask_cols(mask, batch)
    return m[:, None, None, :, None].float()


def mask_bhw(mask: torch.Tensor, H: int, batch: Optional[int] = None) -> torch.Tensor:
    """Any mask layout -> (B,H,W) bool. Full 2D layouts ((B,H,W,1|2), (H,W,1), (H,W)) are kept as they are
    (the training-time radial far mask is 2D); column layouts are expanded over H."""
    if mask.dim() == 4 and mask.shape[1] == H and mask.shape[2] != 1:
        m = mask[..., 0].bool()
    elif mask.dim() == 3 and mask.shape[0] == H:
        m = mask[..., 0].bool()[None]
    elif mask.dim() == 2 and mask.shape[0] == H and mask.shape[0] == mask.shape[1]:
        m = mask.bool()[None]
    else:
        m = mask_cols(mask)[:, None, :].expand(-1, H, -1)
    if batch is not None and m.shape[0] != batch:
        assert m.shape[0] == 1, f"cannot broadcast mask batch {m.shape[0]} to {batch}"
        m = m.expand(batch, -1, -1)
    return m


# ---------------------------------------------------------------------------
# SENSE combine / expand / DC
# ---------------------------------------------------------------------------

def support_from_maps(maps: torch.Tensor, thresh: float = 1e-6) -> torch.Tensor:
    """(B,C,H,W) complex -> (B,H,W) bool"""
    return (maps.abs() ** 2).sum(1) > thresh


def sense_combine(coils: torch.Tensor, maps: torch.Tensor, thresh: float = 1e-6) -> torch.Tensor:
    """x0 = sum_c conj(S_c) x_c / sum_c |S_c|^2 on the support, 0 elsewhere.  (B,C,H,W) -> (B,H,W)"""
    ss = (maps.abs() ** 2).sum(1)
    num = (maps.conj() * coils).sum(1)
    return torch.where(ss > thresh, num / ss.clamp_min(thresh), torch.zeros_like(num))


def sense_expand(x: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
    """x_c = S_c x.  (B,H,W), (B,C,H,W) -> (B,C,H,W)"""
    return maps * x[:, None]


def sense_dc(xhat: torch.Tensor, maps: torch.Tensor, k_meas: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-coil hard data consistency.
    xhat (B,H,W) complex, maps / k_meas (B,C,H,W) complex, mask any 1D layout (or (B,1,H,W) bool)."""
    B, C, H, W = k_meas.shape
    m = mask if (mask.dtype == torch.bool and mask.dim() == 4 and mask.shape[1] == 1) else mask_bhw(mask, H, B)[:, None]
    k = fft2c(sense_expand(xhat, maps))
    k = torch.where(m, k_meas, k)
    return sense_combine(ifft2c(k), maps)


def phase_ref_coil0(maps: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Multiply every coil by conj(S_0/|S_0|) so coil 0 is real-positive (the sigpy ESPIRiT gauge).
    Where |S_0| <= eps the phase factor is 1 (no NaN forward, bounded gradient)."""
    s0 = maps[:, 0]
    a = s0.abs()
    ph = s0 / a.clamp_min(eps)
    ph = torch.where(a > eps, ph, torch.ones_like(ph))
    return maps * ph.conj()[:, None]


# ---------------------------------------------------------------------------
# Sensitivity net wrapper and DC closure
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_maps(sens_net, k_us: torch.Tensor, mask: torch.Tensor, num_coils: Optional[torch.Tensor] = None,
                  num_low_freqs: Optional[int] = None) -> torch.Tensor:
    """Run the (frozen) sensitivity net on undersampled k-space.
    k_us (B,C,H,W) complex (zero-padded coils allowed when num_coils is given), mask any 1D layout.
    Returns (B,C,H,W) complex64 maps, RSS-normalised, coil-0 phase referenced; padded coils are 0."""
    B = k_us.shape[0]
    with torch.autocast("cuda", enabled=False):
        out = sens_net(c2r(k_us), mask_for_sens_net(mask, B).to(k_us.device),
                       num_low_frequencies=num_low_freqs, num_coils=num_coils)
    return phase_ref_coil0(r2c(out.float()))


def make_sense_dc_fn(k_meas: torch.Tensor, maps: torch.Tensor, mask: torch.Tensor):
    """Closure for `reconstruct_far(..., dc_fn=...)`: dc_fn(i_hat (B',H,W,2), sample_mask) -> (B',H,W,2).
    B' may be n*B for posterior sampling (sample-major `.repeat(n,1,1,1)` tiling); k_meas / maps are
    never tiled, the n draws are processed one after the other."""
    B, C, H, W = k_meas.shape
    m_own = mask_bhw(mask, H, B)[:, None]                                   # (B,1,H,W) bool

    def dc_fn(i_hat: torch.Tensor, sample_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        n, rem = divmod(i_hat.shape[0], B)
        assert rem == 0 and n >= 1, f"batch {i_hat.shape[0]} is not a multiple of {B}"
        x = r2c(i_hat).reshape(n, B, H, W)
        if sample_mask is not None:
            m = mask_bhw(sample_mask, H, n * B).reshape(n, B, 1, H, W)
        else:
            m = m_own[None].expand(n, -1, -1, -1, -1)
        out = torch.stack([sense_dc(x[j], maps, k_meas, m[j]) for j in range(n)])
        return c2r(out.reshape(n * B, H, W))

    return dc_fn


def prepare_sense_batch(batch: dict, device, sens_net=None, mode: str = "net", center_fractions: float = 0.04,
                        mask: Optional[torch.Tensor] = None) -> dict:
    """GPU-side SENSE preparation of a SenseSliceDataset batch.
    mode "net": maps from the ACS lines of `mask` with `sens_net`; "oracle": batch["maps_oracle"].
    Returns kc (full k-space, B,C,H,W), k_meas (= kc * mask), maps, masked_c_image (ZF SENSE, B,H,W,2),
    dc_fn (per-coil DC closure for reconstruct_far), coils (complex), num_coils, mask."""
    coils = r2c(batch["coils"].to(device, non_blocking=True))                     # (B,C,H,W)
    num_coils = batch["num_coils"].to(device)
    if mask is None:
        mask = batch["mask"].to(device, non_blocking=True)
    B, C, H, W = coils.shape
    kc = fft2c(coils)
    k_meas = kc * mask_bhw(mask, H, B)[:, None]
    if mode == "oracle":
        assert "maps_oracle" in batch, "oracle maps requested but the dataset was built without return_oracle_maps"
        maps = r2c(batch["maps_oracle"].to(device, non_blocking=True))
    elif mode == "net":
        assert sens_net is not None, "mode='net' needs a sensitivity net"
        maps = estimate_maps(sens_net, k_meas, mask, num_coils, acs_num_low_freqs(W, center_fractions))
    else:
        raise ValueError(f"unknown sens-maps mode {mode}")
    x_zf = sense_combine(ifft2c(k_meas), maps)
    return dict(kc=kc, k_meas=k_meas, maps=maps, masked_c_image=c2r(x_zf),
                dc_fn=make_sense_dc_fn(k_meas, maps, mask), coils=coils, num_coils=num_coils, mask=mask)


# ---------------------------------------------------------------------------
# Token-grid resampling (tokenizer / GPT at a higher resolution than the acquisition grid)
# ---------------------------------------------------------------------------
# Convention of the MediTok_official `res512` SENSE tokenizer (utilities/fastmri_eval.py): the Re and Im images
# of the 256 slice are bicubic-upsampled (align_corners=False, clamped to [-1, 1]) before the tokenizer, and the
# decoded output is bilinear-downsampled with antialiasing back to the native grid for the loss / metrics.

def tok_upsample(x: torch.Tensor, size: int, mode: str = "bicubic") -> torch.Tensor:
    """(B,H,W,2) real view -> (B,size,size,2); Re and Im resampled independently, clamped to [-1, 1]."""
    if x.shape[-3] == size and x.shape[-2] == size:
        return x
    y = F.interpolate(x.permute(0, 3, 1, 2).float(), size=(size, size), mode=mode, align_corners=False)
    return y.clamp(-1, 1).permute(0, 2, 3, 1).contiguous()


def tok_downsample(x: torch.Tensor, size: int) -> torch.Tensor:
    """(B,H,W,2) -> (B,size,size,2): bilinear, antialias, align_corners=False; no clamp."""
    if x.shape[-3] == size and x.shape[-2] == size:
        return x
    y = F.interpolate(x.permute(0, 3, 1, 2).float(), size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return y.permute(0, 2, 3, 1).contiguous()


def wrap_dc_fn_resized(dc_fn, crop_size: int, token_size: int, mode: str = "bicubic"):
    """Make a crop-grid dc_fn usable on token-grid images: downsample -> per-coil DC at crop_size -> upsample."""
    if crop_size == token_size:
        return dc_fn

    def f(i_hat: torch.Tensor, sample_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return tok_upsample(dc_fn(tok_downsample(i_hat, crop_size), sample_mask), token_size, mode)

    return f


# ---------------------------------------------------------------------------
# Expected-embedding ("token mixture") decoding
# ---------------------------------------------------------------------------

def expected_latent(quantizer, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """logits (B, K, L, V) over the K product codebooks -> (B, L, K*d) softmax-weighted codebook vectors
    (the same rows `quantizer.idx_to_f` looks up). With one-hot probabilities this equals idx_to_f(argmax).
    Differentiable w.r.t. the logits, so it can carry an image-domain loss through the frozen decoder."""
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    feats = [probs[:, k] @ cb.codebook.get_norm_weight().float() for k, cb in enumerate(quantizer.codebooks)]   # (B, L, d) each; NormalizedEmbedding rows
    return torch.cat(feats, dim=-1)


def decode_latent(tokenizer, f: torch.Tensor) -> torch.Tensor:
    """(B, L, K*d) latent -> (B, H, W) grayscale image in [-1, 1] (RGB-averaged decoder output)."""
    dec = tokenizer.decoder(tokenizer.post_quant_proj(f)).clamp(-1, 1)
    return dec.float().mean(1)


def decode_expected(tokenizer, logits_re_im: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """logits (B, 2, K, L, V) for the Re / Im streams -> (B, H, W, 2) real view of the decoded complex image."""
    ims = [decode_latent(tokenizer, expected_latent(tokenizer.quantizer, logits_re_im[:, ch], temperature)) for ch in range(2)]
    return torch.stack(ims, dim=-1)


# ---------------------------------------------------------------------------
# Continuous-latent (regression) path: targets from the tokenizer encoder, decode without quantisation
# ---------------------------------------------------------------------------

def encode_features(tokenizer, img_bhw: torch.Tensor) -> torch.Tensor:
    """(B,H,W) grayscale in [-1,1] -> (B, L, D) pre-quantisation encoder features (fp32): one encoder pass that serves both the
    tokens (features_to_tokens) and the latent-regression target (features_to_latent)."""
    img = img_bhw.unsqueeze(-1).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)
    return tokenizer.quant_proj(tokenizer.encoder(img)).float()


def features_to_tokens(tokenizer, f: torch.Tensor) -> torch.Tensor:
    """(B, L, D) features -> (B, K, L) codebook indices (what `encode` returns)."""
    return tokenizer.quantizer.f_to_idx(f)


def features_to_latent(f: torch.Tensor, nchunk: int = 8) -> torch.Tensor:
    """(B, L, D) features -> (B, L, D) with each codebook chunk L2-normalised (what `latent_target` returns)."""
    B, L, D = f.shape
    return F.normalize(f.reshape(B, L, nchunk, -1), dim=-1).reshape(B, L, D)


def latent_target(tokenizer, img_bhw: torch.Tensor, nchunk: int = 8) -> torch.Tensor:
    """(B,H,W) grayscale in [-1,1] -> (B, L, D) pre-quantisation encoder latent with each codebook chunk L2-normalised
    (the quantiser matches normalised chunks against normalised codebook rows; the decoder consumes unit-norm chunks)."""
    img = img_bhw.unsqueeze(-1).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)
    z = tokenizer.quant_proj(tokenizer.encoder(img)).float()                 # (B, L, D)
    B, L, D = z.shape
    return F.normalize(z.reshape(B, L, nchunk, -1), dim=-1).reshape(B, L, D)


def latent_to_image(tokenizer, z: torch.Tensor) -> torch.Tensor:
    """(B, L, D) latent (unit-norm chunks) -> (B, H, W) grayscale image, decoded without quantisation."""
    dec = tokenizer.decoder(tokenizer.post_quant_proj(z)).clamp(-1, 1)
    return dec.float().mean(1)


def decode_latent_pair(tokenizer, z2: torch.Tensor) -> torch.Tensor:
    """(B, 2, L, D) Re/Im latents -> (B, H, W, 2) real view of the decoded complex image."""
    return torch.stack([latent_to_image(tokenizer, z2[:, c]) for c in range(2)], dim=-1)
