"""
Multicoil MRI reconstruction model built on frozen MedITok encoder/decoder.

Architecture (simple mode forward pass):
    x_us : (B, C, H, W, 2)          per-coil undersampled complex images
        ↓
    _encode_to_tokens → (B*C_eff, K, L)  discrete MedITok token indices
        ↓
    CoilEmbedFuser × C_eff           TokenEmbedder(re) + TokenEmbedder(im) fused per coil
        ↓  (B, C_eff, L, D)
    CrossCoilFusion                  attention pool over coil axis
        ↓  h : (B, L, D)
    SharedTransformer (KAR_GPT_L: dim=1024, 24 layers, 16 heads)
        ↓  z : (B, L, D)
    AutoRegressiveHeadSingleStream   teacher-forced AR prediction over K codebooks
        ↓  logits : (B, K, L, vpc)
    CE loss vs GT RSS tokens (B, K, L)

All MedITok operations run under torch.no_grad() — only the fusion / transformer
/ AR-head modules are trainable.
"""

import sys
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from dataset.data.fftc import fft2c_new as fft2c
from .gpt_kar import (
    ModelArgs,
    TokenEmbedder,
    AutoRegressiveHeadSingleStream,
    TransformerBlock,
    precompute_freqs_cis_2d,
)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def rss_from_coils(x: Tensor) -> Tensor:
    """Root-sum-of-squares across coil dimension.

    Args:
        x: (B, C, H, W, 2) complex coil images

    Returns:
        (B, H, W) RSS magnitude
    """
    mag = torch.sqrt((x ** 2).sum(dim=-1))   # (B, C, H, W)
    return torch.sqrt((mag ** 2).sum(dim=1)) # (B, H, W)


# ---------------------------------------------------------------------------
# CoilEmbedFuser  (replaces ComplexFusion)
# ---------------------------------------------------------------------------

class CoilEmbedFuser(nn.Module):
    """Fuse real and imaginary discrete token embeddings for one coil.

    Follows the same pattern as gpt_kar's Transformer input stage:
      - TokenEmbedder sums K codebook embeddings → (B, L, D) per channel
      - Linear(2D → D) + LayerNorm fuses re + im
      - Learnable spatial and coil embeddings added

    Args:
        vocab_size:    total VQ vocabulary (32768)
        num_codebooks: number of VQ codebooks K (8)
        D:             output hidden dimension
        L:             number of spatial tokens (256 for 256×256 @ patch 16)
        max_coils:     size of coil embedding table
    """

    def __init__(self, vocab_size: int, num_codebooks: int, D: int,
                 L: int, max_coils: int):
        super().__init__()
        self.re_embedder = TokenEmbedder(vocab_size, D, num_codebooks)
        self.im_embedder = TokenEmbedder(vocab_size, D, num_codebooks)
        self.fuse = nn.Linear(2 * D, D)
        self.ln   = nn.LayerNorm(D)
        self.spatial_emb = nn.Parameter(torch.zeros(1, L, D))
        self.coil_emb    = nn.Embedding(max_coils, D)

    def forward(self, idx_re: Tensor, idx_im: Tensor, c: int) -> Tensor:
        """
        Args:
            idx_re: (B, K, L)  discrete token indices for real channel
            idx_im: (B, K, L)  discrete token indices for imag channel
            c:      int         coil index in [0, max_coils)

        Returns:
            h_c: (B, L, D)
        """
        h_re = self.re_embedder(idx_re)                               # (B, L, D)
        h_im = self.im_embedder(idx_im)                               # (B, L, D)
        h_c  = self.ln(self.fuse(torch.cat([h_re, h_im], dim=-1)))   # (B, L, D)
        c_emb = self.coil_emb(torch.tensor([c], device=idx_re.device)).view(1, 1, -1)
        return h_c + self.spatial_emb + c_emb                         # (B, L, D)


# ---------------------------------------------------------------------------
# SharedTransformer
# ---------------------------------------------------------------------------

class SharedTransformer(nn.Module):
    """Bidirectional Transformer operating on fused coil token embeddings.

    Uses TransformerBlock from gpt_kar with full_attn=True (no causal mask)
    and 2-D rotary position embeddings matching the spatial grid layout.

    Args:
        dim:            hidden dimension
        n_layer:        number of TransformerBlock layers
        n_head:         number of attention heads
        grid_size:      spatial grid side (L = grid_size²; default 16 for 256px)
        drop_path_rate: stochastic depth rate
    """

    def __init__(self, dim: int, n_layer: int, n_head: int,
                 grid_size: int = 16, drop_path_rate: float = 0.0):
        super().__init__()

        cfg = ModelArgs(
            dim             = dim,
            n_layer         = n_layer,
            n_head          = n_head,
            n_kv_head       = n_head,
            vocab_size       = 32768,
            cls_token_num   = 0,
            cond_token_num  = 0,
            block_size      = grid_size * grid_size,
            max_batch_size  = 1,
            max_seq_len     = grid_size * grid_size,
            attn_dropout_p  = 0.0,
            resid_dropout_p = 0.0,
            ffn_dropout_p   = 0.0,
            token_dropout_p = 0.0,
            drop_path_rate  = drop_path_rate,
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layer)]
        self.layers = nn.ModuleList([TransformerBlock(cfg, dpr[i]) for i in range(n_layer)])

        head_dim = dim // n_head
        freqs = precompute_freqs_cis_2d(grid_size, head_dim, zero_prefix_len=0)
        self.register_buffer("freqs_cis", freqs, persistent=False)

    def forward(self, h: Tensor) -> Tensor:
        """Args: h (B, L, D)  →  Returns: z (B, L, D)"""
        for layer in self.layers:
            h = layer(h, self.freqs_cis, start_pos=0, full_attn=True)
        return h


# ---------------------------------------------------------------------------
# CoilDecoder  (kept for full mode; not used in simple mode)
# ---------------------------------------------------------------------------

def _make_mlp(D: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(D), nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D),
    )


class CoilDecoder(nn.Module):
    """Coil-conditioned latent decoder (full mode only)."""

    def __init__(self, D: int, Dh: int, max_coils: int, De: int = 64):
        super().__init__()
        self.coil_emb  = nn.Embedding(max_coils, De)
        self.in_proj   = nn.Linear(2 * D + De, D)
        self.mlp1      = _make_mlp(D)
        self.mlp2      = _make_mlp(D)
        self.u_re_proj = nn.Linear(D, Dh)
        self.u_im_proj = nn.Linear(D, Dh)
        self.A         = nn.Linear(Dh, Dh, bias=False)
        self.B         = nn.Linear(Dh, Dh, bias=False)

    def forward(self, z: Tensor, h_c: Tensor, c: int) -> Tuple[Tensor, Tensor]:
        B, L, _ = z.shape
        e_c  = self.coil_emb(torch.tensor([c], device=z.device)).view(1, 1, -1).expand(B, L, -1)
        r    = self.in_proj(torch.cat([z, h_c, e_c], dim=-1))
        r    = r + self.mlp1(r)
        r    = r + self.mlp2(r)
        u_re = self.u_re_proj(r)
        u_im = self.u_im_proj(r)
        return self.A(u_re) - self.B(u_im), self.B(u_re) + self.A(u_im)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(
    out: dict,
    lambda_coil: float = 0.1,
    lambda_dc: float   = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute total training loss.

    Returns:
        (loss_total, loss_ce, loss_lat, loss_dc)
    """
    # logits: (B, K, L, vpc)  targets: (B, K, L)
    B, K, L, vpc = out["logits"].shape
    loss_ce = F.cross_entropy(
        out["logits"].reshape(B * K * L, vpc),
        out["targets"]["rss_tokens"].reshape(B * K * L),
    )

    if "h_hat_re" in out:
        loss_lat = (
            (out["h_hat_re"] - out["targets"]["h_gt_re"]).abs().mean()
            + (out["h_hat_im"] - out["targets"]["h_gt_im"]).abs().mean()
        )
    else:
        loss_lat = loss_ce.new_zeros(())

    if "kspace_dc_err" in out:
        loss_dc = out["kspace_dc_err"].abs().mean()
    else:
        loss_dc = loss_ce.new_zeros(())

    total = loss_ce + lambda_coil * loss_lat + lambda_dc * loss_dc
    return total, loss_ce, loss_lat, loss_dc


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def decode_coil_latents(meditok, h_hat_re: Tensor, h_hat_im: Tensor) -> Tuple[Tensor, Tensor]:
    with torch.no_grad():
        idx_re = meditok.quantizer.f_to_idx(h_hat_re)
        idx_im = meditok.quantizer.f_to_idx(h_hat_im)
        img_re = meditok.idx_to_img(idx_re)
        img_im = meditok.idx_to_img(idx_im)
    return img_re[:, 0], img_im[:, 0]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MulticoilRecon(nn.Module):
    """Multicoil MRI reconstruction model.

    Encodes undersampled coil images to discrete MedITok token indices,
    fuses them via CoilEmbedFuser + CrossCoilFusion, runs a bidirectional
    SharedTransformer, and predicts GT RSS tokens with an autoregressive head.
    """

    def __init__(
        self,
        meditok,
        coil_fuser:         CoilEmbedFuser,
        cross_coil_fusion,
        shared_transformer: SharedTransformer,
        coil_decoder:       CoilDecoder,
        token_head:         AutoRegressiveHeadSingleStream,
        enc_chunk_size: int = 8,
    ):
        super().__init__()

        self.meditok = meditok
        for p in meditok.parameters():
            p.requires_grad_(False)
        meditok.eval()

        self.enc_chunk_size     = enc_chunk_size
        self.coil_fuser         = coil_fuser
        self.cross_coil_fusion  = cross_coil_fusion
        self.shared_transformer = shared_transformer
        self.coil_decoder       = coil_decoder
        self.token_head         = token_head

    def _encode_to_tokens(self, imgs: Tensor) -> Tensor:
        """Encode (N, H, W) grayscale images to discrete MedITok token indices.

        Matches the encoding path used in train.py:
            encoder → quant_proj → f_to_idx

        Args:
            imgs: (N, H, W)  float in [-1, 1]

        Returns:
            (N, K, L)  long  — discrete codebook indices, no gradients
        """
        chunks  = imgs.split(self.enc_chunk_size, dim=0)
        results = []
        with torch.no_grad():
            for chunk in chunks:
                img4 = chunk.unsqueeze(-1).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)  # (n,3,H,W)
                z    = self.meditok.quant_proj(self.meditok.encoder(img4))           # (n, L, Dh)
                results.append(self.meditok.quantizer.f_to_idx(z))                  # (n, K, L)
        return torch.cat(results, dim=0)                                             # (N, K, L)

    def forward(
        self,
        x_us:       Optional[Tensor] = None,  # (B, C, H, W, 2)  — not needed when tokens pre-computed
        x_gt:       Optional[Tensor] = None,  # (B, C, H, W, 2)  — not needed in simple mode
        kspace_us:  Optional[Tensor] = None,  # unused in simple mode
        mask:       Optional[Tensor] = None,  # unused in simple mode
        num_coils:  Optional[Tensor] = None,  # (B,) actual coil counts
        simple:     bool = False,
        # Pre-computed tokens from TokenCachedDataset (skip MedITok encoding)
        tok_us_re:  Optional[Tensor] = None,  # (B, C, K, L)
        tok_us_im:  Optional[Tensor] = None,  # (B, C, K, L)
        tok_gt_rss: Optional[Tensor] = None,  # (B, K, L)
        _timings:   Optional[dict] = None,    # if provided, filled with section timings (s)
    ) -> dict:
        import time

        def _t():
            torch.cuda.synchronize()
            return time.perf_counter()

        precomputed = tok_us_re is not None  # fast path: skip all MedITok encoding

        # C_eff = min real coils across the batch. Padded coil slots (indices >= num_coils[b])
        # are naturally excluded — we never reach them in the encoding loop.
        if precomputed:
            B, C_eff_max, K, L = tok_us_re.shape
            C_eff = max(int(num_coils.min().item()) if num_coils is not None else C_eff_max, 1)
        else:
            B, C, H, W, _ = x_us.shape
            C_eff = max(int(num_coils.min().item()) if num_coils is not None else C, 1)

        N = B * C_eff  # flat batch×coil size used in full mode
        if not precomputed:
            pass  # H, W already set above
        elif x_gt is not None:
            _, _, H, W, _ = x_gt.shape  # needed by full mode DC loss

        # ------------------------------------------------------------------
        # 1. Encode undersampled coils to discrete token indices.
        #    Only indices 0..C_eff-1 are encoded; padded slots are skipped.
        # ------------------------------------------------------------------
        t0 = _t()
        if precomputed:
            tok_us_re_eff = tok_us_re[:, :C_eff]   # (B, C_eff, K, L)
            tok_us_im_eff = tok_us_im[:, :C_eff]
            K = tok_us_re_eff.shape[2]
            L = tok_us_re_eff.shape[3]
            t0b = t0
        else:
            N = B * C_eff
            us_re = x_us[:, :C_eff, :, :, 0].reshape(N, H, W)
            us_im = x_us[:, :C_eff, :, :, 1].reshape(N, H, W)
            _tok_re = self._encode_to_tokens(us_re)   # (N, K, L)
            t0b = _t()
            _tok_im = self._encode_to_tokens(us_im)
            K, L = _tok_re.shape[1], _tok_re.shape[2]
            tok_us_re_eff = _tok_re.view(B, C_eff, K, L)
            tok_us_im_eff = _tok_im.view(B, C_eff, K, L)
        t1 = _t()

        # ------------------------------------------------------------------
        # 2. Per-coil fuse re + im token embeddings  →  (B, L, D)
        # ------------------------------------------------------------------
        per_coil_h = []
        for c in range(C_eff):
            h_c = self.coil_fuser(tok_us_re_eff[:, c], tok_us_im_eff[:, c], c)  # (B, L, D)
            per_coil_h.append(h_c)
        t2 = _t()

        # ------------------------------------------------------------------
        # 3. Cross-coil fusion + SharedTransformer
        # ------------------------------------------------------------------
        H_stk = torch.stack(per_coil_h, dim=1)   # (B, C_eff, L, D)
        h     = self.cross_coil_fusion(H_stk)    # (B, L, D)
        t3 = _t()

        # Per-layer transformer timing (only when _timings is requested)
        if _timings is not None:
            layer_times = []
            for layer in self.shared_transformer.layers:
                _lt0 = _t()
                h = layer(h, self.shared_transformer.freqs_cis, start_pos=0, full_attn=True)
                layer_times.append(_t() - _lt0)
            z = h
        else:
            z = self.shared_transformer(h)
        t4 = _t()

        # ------------------------------------------------------------------
        # 4. GT RSS target tokens
        #    Skipped when tokens are passed in directly.
        # ------------------------------------------------------------------
        if precomputed:
            t_rss = tok_gt_rss   # (B, K, L) — from data loader
        else:
            with torch.no_grad():
                rss_gt  = rss_from_coils(x_gt[:, :C_eff]) * 2.0 - 1.0
                rss_4ch = rss_gt.unsqueeze(-1).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)
                t_rss   = self.meditok.img_to_idx(rss_4ch)
        t5 = _t()

        # ------------------------------------------------------------------
        # 5. AR token prediction (teacher-forced)
        # ------------------------------------------------------------------
        logits = self.token_head.forward_train(z, t_rss)  # (B, K, L, vpc)
        t6 = _t()

        if _timings is not None:
            _timings["encode_us_coils"] = t1 - t0
            _timings["encode_us_re"]    = t0b - t0   # real channel only
            _timings["encode_us_im"]    = t1 - t0b   # imag channel only
            _timings["n_coils_encoded"] = C_eff
            _timings["coil_fuse"]       = t2 - t1
            _timings["cross_coil"]      = t3 - t2
            _timings["transformer"]     = t4 - t3
            _timings["transformer_layers"] = layer_times  # list[float] per layer
            _timings["encode_gt_rss"]   = t5 - t4
            _timings["ar_head"]         = t6 - t5

        if simple:
            return {
                "logits":  logits,
                "z":       z,
                "targets": {"rss_tokens": t_rss},
            }

        # ------------------------------------------------------------------
        # 6. Coil decoder (full mode)
        # ------------------------------------------------------------------
        # For full mode we also need encoded GT coil features (continuous)
        with torch.no_grad():
            gt_re = x_gt[:, :C_eff, :, :, 0].reshape(N, H, W)
            gt_im = x_gt[:, :C_eff, :, :, 1].reshape(N, H, W)
            gt_imgs = torch.cat([gt_re, gt_im], dim=0)
            # encode_with_ste returns z_e (continuous pre-quant features)
            chunks_h = []
            for c in gt_imgs.split(self.enc_chunk_size):
                img4 = c.unsqueeze(-1).repeat(1,1,1,3).permute(0,3,1,2)
                chunks_h.append(self.meditok.encode_with_ste(img4)[2])
            gt_h = torch.cat(chunks_h, dim=0)  # (2N, L, Dh)

        Dh = gt_h.shape[-1]
        h_gt_re = gt_h[:N].view(B, C_eff, L, Dh)
        h_gt_im = gt_h[N:].view(B, C_eff, L, Dh)

        h_hat_re_list, h_hat_im_list = [], []
        for c in range(C_eff):
            h_hat_re_c, h_hat_im_c = self.coil_decoder(z, per_coil_h[c], c)
            h_hat_re_list.append(h_hat_re_c)
            h_hat_im_list.append(h_hat_im_c)

        h_hat_re = torch.stack(h_hat_re_list, dim=1)
        h_hat_im = torch.stack(h_hat_im_list, dim=1)

        # ------------------------------------------------------------------
        # 7. k-space DC loss
        # ------------------------------------------------------------------
        mask_dc      = mask.squeeze(1)
        kspace_gt_f  = fft2c(x_gt[:, :C_eff].reshape(N, H, W, 2))
        kspace_us_f  = kspace_us[:, :C_eff].reshape(N, H, W, 2)
        mask_dc_flat = mask_dc.unsqueeze(1).expand(B, C_eff, 1, W, 1).reshape(N, 1, W, 1)
        kspace_dc_err = (kspace_gt_f - kspace_us_f) * mask_dc_flat

        return {
            "logits":        logits,
            "z":             z,
            "h_hat_re":      h_hat_re,
            "h_hat_im":      h_hat_im,
            "kspace_dc_err": kspace_dc_err,
            "targets": {
                "rss_tokens": t_rss,
                "h_gt_re":    h_gt_re,
                "h_gt_im":    h_gt_im,
                "x_gt":       x_gt[:, :C_eff],
            },
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_multicoil_recon(
    meditok,
    D: int             = 1024,
    n_layer: int       = 24,
    n_head: int        = 16,
    De: int            = 64,
    max_coils: int     = 20,
    Dh: int            = 64,
    num_codebooks: int = 8,
    vocab_size: int    = 32768,
    crop_size: int     = 256,
    patch_size: int    = 16,
    drop_path_rate: float = 0.0,
    enc_chunk_size: int   = 8,
    n_ar_layer: int       = 2,
) -> MulticoilRecon:
    """Build a MulticoilRecon model with KAR_GPT_L-scale backbone.

    Defaults match KAR_GPT_L: dim=1024, 24 layers, 16 heads.
    """
    assert crop_size % patch_size == 0
    grid_size = crop_size // patch_size   # 16
    L         = grid_size * grid_size     # 256

    from models.multicoil_fusion import CrossCoilFusion

    coil_fuser  = CoilEmbedFuser(
        vocab_size=vocab_size, num_codebooks=num_codebooks,
        D=D, L=L, max_coils=max_coils,
    )
    cross_coil_fusion  = CrossCoilFusion(D=D, n_heads=n_head)
    shared_transformer = SharedTransformer(
        dim=D, n_layer=n_layer, n_head=n_head,
        grid_size=grid_size, drop_path_rate=drop_path_rate,
    )
    coil_decoder = CoilDecoder(D=D, Dh=Dh, max_coils=max_coils, De=De)

    ar_cfg    = ModelArgs(
        dim           = D,
        n_layer       = n_ar_layer,
        n_head        = n_head,
        n_kv_head     = n_head,
        num_codebooks = num_codebooks,
        vocab_size    = vocab_size,
        n_output_layer= n_ar_layer,
        norm_eps      = 1e-5,
        block_size    = L,
    )
    token_head = AutoRegressiveHeadSingleStream(ar_cfg)

    return MulticoilRecon(
        meditok            = meditok,
        coil_fuser         = coil_fuser,
        cross_coil_fusion  = cross_coil_fusion,
        shared_transformer = shared_transformer,
        coil_decoder       = coil_decoder,
        token_head         = token_head,
        enc_chunk_size     = enc_chunk_size,
    )
