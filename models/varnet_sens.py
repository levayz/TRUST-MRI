"""
Sensitivity-map estimators for the SENSE-combined TRUST pipeline.

1. `SensitivityModel` — the E2E-VarNet sensitivity U-Net, vendored from the
   fastMRI repository (Facebook AI Research, MIT license):
   https://github.com/facebookresearch/fastMRI  (fastmri/models/varnet.py, unet.py).
   Copied here (with only the fft / rss helpers swapped for the repo's own)
   so the pretrained leaderboard weights can be used from any environment
   without installing the fastmri package.

   Pretrained weights: the fastMRI brain leaderboard E2E-VarNet state dict
   (https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/brain_leaderboard_state_dict.pt)
   contains the sensitivity net under the `sens_net.` prefix, built with
   sens_chans=8, sens_pools=4. Trained jointly with the 12 cascades on the
   fastMRI brain multicoil train set, equispaced x4/x8 masks, cf 0.08/0.04.

2. `espirit_maps` — ESPIRiT (Uecker et al. 2014) in torch, following the
   structure of sigpy.mri.app.EspiritCalib but with a rectangular calibration
   region so all acquired ACS columns x the full readout extent are used, and
   an exact batched eigendecomposition instead of power iteration.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FFT / RSS helpers (centered, orthonormal — same convention as fastmri and
# dataset/data/fftc.py)
# ---------------------------------------------------------------------------

def ifft2c_real(x: torch.Tensor) -> torch.Tensor:
    """(..., H, W, 2) real view -> (..., H, W, 2) real view."""
    xc = torch.view_as_complex(x.contiguous())
    xc = torch.fft.ifftshift(xc, dim=(-2, -1))
    xc = torch.fft.ifft2(xc, norm="ortho")
    xc = torch.fft.fftshift(xc, dim=(-2, -1))
    return torch.view_as_real(xc)


def rss_complex(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.sqrt((data ** 2).sum(dim=-1).sum(dim))


def batched_mask_center(x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(x)
    if mask_from.shape[0] == 1:
        mask[:, :, :, int(mask_from):int(mask_to)] = x[:, :, :, int(mask_from):int(mask_to)]
    else:
        for i, (start, end) in enumerate(zip(mask_from, mask_to)):
            mask[i, :, :, start:end] = x[i, :, :, start:end]
    return mask


# ---------------------------------------------------------------------------
# U-Net building blocks (fastmri/models/unet.py)
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


class Unet(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, chans: int = 32,
                 num_pool_layers: int = 4, drop_prob: float = 0.0):
        super().__init__()
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2
        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(nn.Sequential(
            ConvBlock(ch * 2, ch, drop_prob),
            nn.Conv2d(ch, out_chans, kernel_size=1, stride=1),
        ))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        stack = []
        output = image
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
        output = self.conv(output)
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1
            if sum(padding) != 0:
                output = F.pad(output, padding, "reflect")
            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)
        return output


# ---------------------------------------------------------------------------
# NormUnet + SensitivityModel (fastmri/models/varnet.py)
# ---------------------------------------------------------------------------

class NormUnet(nn.Module):
    def __init__(self, chans: int, num_pools: int, in_chans: int = 2, out_chans: int = 2, drop_prob: float = 0.0):
        super().__init__()
        self.unet = Unet(in_chans=in_chans, out_chans=out_chans, chans=chans,
                         num_pool_layers=num_pools, drop_prob=drop_prob)

    @staticmethod
    def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    @staticmethod
    def chan_complex_to_last_dim(x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    @staticmethod
    def norm(x: torch.Tensor):
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)
        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)
        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    @staticmethod
    def unnorm(x, mean, std):
        return x * std + mean

    @staticmethod
    def pad(x: torch.Tensor):
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    @staticmethod
    def unpad(x, h_pad: List[int], w_pad: List[int], h_mult: int, w_mult: int):
        return x[..., h_pad[0]:h_mult - h_pad[1], w_pad[0]:w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)
        x = self.unet(x)
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        return self.chan_complex_to_last_dim(x)


class SensitivityModel(nn.Module):
    """IFFT of the ACS-only k-space -> per-coil U-Net -> divide by RSS.

    forward(masked_kspace (B,C,H,W,2), mask (B,1,1,W,1), num_low_frequencies=None) -> (B,C,H,W,2)
    Maps are RSS-normalised at every pixel (no support mask, unlike ESPIRiT).
    """

    def __init__(self, chans: int, num_pools: int, in_chans: int = 2, out_chans: int = 2,
                 drop_prob: float = 0.0, mask_center: bool = True):
        super().__init__()
        self.mask_center = mask_center
        self.norm_unet = NormUnet(chans, num_pools, in_chans=in_chans, out_chans=out_chans, drop_prob=drop_prob)

    @staticmethod
    def chans_to_batch_dim(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape
        return x.view(b * c, 1, h, w, comp), b

    @staticmethod
    def batch_chans_to_chan_dim(x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        return x.view(batch_size, bc // batch_size, h, w, comp)

    @staticmethod
    def divide_root_sum_of_squares(x: torch.Tensor) -> torch.Tensor:
        return x / rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    @staticmethod
    def get_pad_and_num_low_freqs(mask: torch.Tensor, num_low_frequencies: Optional[int] = None):
        if num_low_frequencies is None or num_low_frequencies == 0:
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(2 * torch.min(left, right), torch.ones_like(left))
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(mask.shape[0], dtype=mask.dtype, device=mask.device)
        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2
        return pad.type(torch.long), num_low_frequencies_tensor.type(torch.long)

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor,
                num_low_frequencies: Optional[int] = None,
                num_coils: Optional[torch.Tensor] = None) -> torch.Tensor:
        """masked_kspace (B,C,H,W,2), mask (B,1,1,W,1), num_low_frequencies: ACS width (pass it
        explicitly for random masks, otherwise it is inferred from the mask and a random column next to
        the ACS block widens the calibration region), num_coils (B,) long: number of valid coils when the
        coil dim is zero-padded. Padded coils must not go through NormUnet (its per-coil std is 0 ->
        NaN that spreads to every coil through the RSS); they are skipped and returned as exact zeros."""
        if self.mask_center:
            pad, num_low_freqs = self.get_pad_and_num_low_freqs(mask, num_low_frequencies)
            masked_kspace = batched_mask_center(masked_kspace, pad, pad + num_low_freqs)
        images, batches = self.chans_to_batch_dim(ifft2c_real(masked_kspace))
        if num_coils is None:
            out = self.norm_unet(images)
        else:
            B, C = masked_kspace.shape[:2]
            valid = (torch.arange(C, device=masked_kspace.device)[None, :] < num_coils.view(-1, 1)).reshape(-1)
            out = torch.zeros_like(images)
            out[valid] = self.norm_unet(images[valid]).to(out.dtype)
        maps = self.batch_chans_to_chan_dim(out, batches)
        return maps / rss_complex(maps, dim=1).clamp_min(1e-8).unsqueeze(-1).unsqueeze(1)


def load_sens_net(path: str, device: Optional[torch.device] = None, trainable: bool = False,
                  chans: int = 8, num_pools: int = 4) -> SensitivityModel:
    """Build a SensitivityModel from either
      * a full fastMRI E2E-VarNet state dict (`sens_net.*` keys; leaderboard checkpoints), or
      * a bare sens-net state dict, or
      * a TRUST-MRI stage-1 checkpoint from train_sens_net.py: {"model": state_dict, "chans", "num_pools", ...}.
    `trainable=False` returns the net in eval mode with requires_grad off."""
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and isinstance(sd.get("model"), dict):          # stage-1 checkpoint
        chans = int(sd.get("chans", chans)); num_pools = int(sd.get("num_pools", num_pools))
        sub = sd["model"]
    else:
        if isinstance(sd, dict) and isinstance(sd.get("state_dict"), dict):
            sd = sd["state_dict"]
        prefix = "sens_net."
        sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if not sub:  # maybe already a bare sens-net state dict
            sub = sd
    sub = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sub.items()}
    model = SensitivityModel(chans=chans, num_pools=num_pools)
    missing, unexpected = model.load_state_dict(sub, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"sens_net load mismatch — missing={missing} unexpected={unexpected}")
    model.train(trainable)
    for p in model.parameters():
        p.requires_grad_(trainable)
    return model.to(device) if device is not None else model


def load_pretrained_sens_net(state_dict_path: str, chans: int = 8, num_pools: int = 4,
                             device: Optional[torch.device] = None) -> SensitivityModel:
    """Frozen SensitivityModel from a fastMRI E2E-VarNet state dict (kept for the existing eval scripts)."""
    return load_sens_net(state_dict_path, device=device, trainable=False, chans=chans, num_pools=num_pools)


# ---------------------------------------------------------------------------
# ESPIRiT in torch (rectangular calibration region, exact eigendecomposition)
# ---------------------------------------------------------------------------

@torch.no_grad()
def espirit_maps(
    kspace: torch.Tensor,
    calib_cols: Tuple[int, int],
    calib_rows: Optional[Tuple[int, int]] = None,
    kernel_width: int = 6,
    thresh: float = 0.02,
    crop: float = 0.95,
    kernel_chunk: int = 48,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ESPIRiT sensitivity maps from a calibration region of a single slice.

    Args:
        kspace:     (C, H, W) complex k-space (only the calibration region is read)
        calib_cols: (c0, c1) column range of the calibration region (ACS lines)
        calib_rows: (r0, r1) row range; default = full readout extent
        kernel_width, thresh, crop: as in sigpy.mri.app.EspiritCalib

    Returns:
        maps:  (C, H, W) complex, unit norm inside the support, 0 outside
        eigv:  (H, W) leading eigenvalue map (≈1 inside the object)
    """
    C, H, W = kspace.shape
    r0, r1 = (0, H) if calib_rows is None else calib_rows
    c0, c1 = calib_cols
    k = kernel_width
    calib = kspace[:, r0:r1, c0:c1]                                          # (C, hc, wc)
    hc, wc = calib.shape[-2:]
    assert hc >= k and wc >= k, "calibration region smaller than kernel"

    # Hankel / calibration matrix: all k×k blocks, stride 1  -> (n_blocks, C*k*k)
    blocks = calib.unfold(1, k, 1).unfold(2, k, 1)                           # (C, nh, nw, k, k)
    mat = blocks.permute(1, 2, 0, 3, 4).reshape(-1, C * k * k)               # (n_blocks, C*k*k)
    _, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    keep = S > thresh * S.max()
    kernels = Vh[keep].reshape(-1, C, k, k)                                  # (n_k, C, k, k)

    # image-domain kernels: zero-pad each kernel to (H, W) at the k-space centre, centred IFFT.
    # Accumulate the per-pixel covariance  Σ_k g_k(p) g_k(p)ᴴ  (C×C) in chunks of kernels to bound memory
    # (n_k can reach C·k² ≈ 700 for 20 coils; all kernels at once would need several GB).
    top, left = (H - k) // 2, (W - k) // 2
    scale = math.sqrt(H * W / (k * k))                                       # sigpy scaling -> eigenvalues ≈ 1 in-object
    AHA = torch.zeros(H, W, C, C, dtype=kspace.dtype, device=kspace.device)
    for kchunk in kernels.split(kernel_chunk, dim=0):
        padded = torch.zeros(kchunk.shape[0], C, H, W, dtype=kspace.dtype, device=kspace.device)
        padded[:, :, top:top + k, left:left + k] = kchunk
        img_k = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(padded, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))
        a = img_k.permute(2, 3, 0, 1) * scale                                # (H, W, n_chunk, C)  rows g_k(p)
        AHA += a.transpose(-1, -2) @ a.conj()                                # (H, W, C, C)  (sigpy: aH @ a)
        del padded, img_k, a

    # leading eigenvector per pixel
    evals, evecs = torch.linalg.eigh(AHA)                                    # ascending
    lead = evals[..., -1]                                                    # (H, W)
    maps = evecs[..., -1]                                                    # (H, W, C)
    # fix the arbitrary per-pixel phase: make the first-coil component real-positive (sigpy convention-free, smooth)
    ref = maps[..., 0]
    phase = torch.where(ref.abs() > 0, ref / ref.abs().clamp_min(1e-12), torch.ones_like(ref))
    maps = maps * phase.conj().unsqueeze(-1)
    maps = maps * (lead > crop).unsqueeze(-1)
    return maps.permute(2, 0, 1).contiguous(), lead


# ---------------------------------------------------------------------------
# Full E2E-VarNet (fastmri/models/varnet.py), state-dict compatible with the fastMRI checkpoints
# (`sens_net.*`, `cascades.N.model.*`, `cascades.N.dc_weight`), with padded-coil support through
# SensitivityModel(num_coils=...). Used by train_varnet256.py to train the baseline on the 256-grid setup.
# ---------------------------------------------------------------------------

def fft2c_real(x: torch.Tensor) -> torch.Tensor:
    """(..., H, W, 2) real view -> (..., H, W, 2) real view (centered, orthonormal)."""
    xc = torch.view_as_complex(x.contiguous())
    xc = torch.fft.ifftshift(xc, dim=(-2, -1))
    xc = torch.fft.fft2(xc, norm="ortho")
    xc = torch.fft.fftshift(xc, dim=(-2, -1))
    return torch.view_as_real(xc)


class VarNetBlock(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    @staticmethod
    def sens_expand(x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """x (B,1,H,W,2) image -> per-coil k-space (B,C,H,W,2)"""
        xc = torch.view_as_complex(x.contiguous()) * torch.view_as_complex(sens_maps.contiguous())
        return fft2c_real(torch.view_as_real(xc))

    @staticmethod
    def sens_reduce(k: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """per-coil k-space (B,C,H,W,2) -> combined image (B,1,H,W,2)"""
        xc = torch.view_as_complex(ifft2c_real(k).contiguous()) * torch.view_as_complex(sens_maps.contiguous()).conj()
        return torch.view_as_real(xc.sum(dim=1, keepdim=True))

    def forward(self, current_kspace, ref_kspace, mask, sens_maps):
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
        model_term = self.sens_expand(self.model(self.sens_reduce(current_kspace, sens_maps)), sens_maps)
        return current_kspace - soft_dc - model_term


class VarNet(nn.Module):
    """forward(masked_kspace (B,C,H,W,2), mask (B,1,1,W,1) bool, num_low_frequencies, num_coils) -> predicted k-space (B,C,H,W,2)."""

    def __init__(self, num_cascades: int = 12, sens_chans: int = 8, sens_pools: int = 4, chans: int = 18, pools: int = 4):
        super().__init__()
        self.sens_net = SensitivityModel(chans=sens_chans, num_pools=sens_pools)
        self.cascades = nn.ModuleList([VarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)])

    def forward(self, masked_kspace, mask, num_low_frequencies=None, num_coils=None):
        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies, num_coils=num_coils)
        kspace_pred = masked_kspace.clone()
        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)
        return kspace_pred

    @staticmethod
    def rss(kspace_pred: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W,2) k-space -> (B,H,W) RSS magnitude"""
        return rss_complex(ifft2c_real(kspace_pred), dim=1)


def load_varnet(path: str, device=None, trainable: bool = False) -> VarNet:
    """fastMRI leaderboard bare state dict, a Lightning checkpoint (`varnet.` prefix) or a train_varnet256.py checkpoint."""
    ck = torch.load(path, map_location="cpu")
    if isinstance(ck, dict) and "hyper_parameters" in ck:
        sd = {k[len("varnet."):]: v for k, v in ck["state_dict"].items() if k.startswith("varnet.")}
    elif isinstance(ck, dict) and "state_dict" in ck:
        sd = ck["state_dict"]
    else:
        sd = ck
    n_casc = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("cascades."))
    chans = sd["cascades.0.model.unet.down_sample_layers.0.layers.0.weight"].shape[0]
    sens_chans = sd["sens_net.norm_unet.unet.down_sample_layers.0.layers.0.weight"].shape[0]
    model = VarNet(num_cascades=n_casc, sens_chans=sens_chans, sens_pools=4, chans=chans, pools=4)
    model.load_state_dict(sd, strict=True)
    model.train(trainable)
    for p in model.parameters():
        p.requires_grad_(trainable)
    return model.to(device) if device is not None else model
