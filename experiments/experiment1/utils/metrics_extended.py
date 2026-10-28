"""Extended metrics for Experiment 1.

Adds token-level metrics (accuracy, NLL) and SSFD on top of the standard
metrics already in autoregressive/utils/metrics.py.
"""

from __future__ import annotations

import sys
import pathlib
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# TRUST-MRI root
_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# deep-feature-mr-recon for SSFD
_DFD_ROOT = pathlib.Path("/fast/lev/deep-feature-mr-recon")
_DFD_METRICS = _DFD_ROOT / "dfd_mr_recon"
for _p in (str(_DFD_ROOT), str(_DFD_METRICS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# SSFD (lazy-loaded singleton so the HuggingFace download happens once)
# ---------------------------------------------------------------------------

_ssfd_instance: Optional[object] = None
_ssfd_device: Optional[torch.device] = None


def _get_ssfd(device: Optional[torch.device] = None):
    """Return an SSFD singleton with its encoder on `device`.

    SSFD inherits from a torchmetrics-style Metric class that stores non-tensor
    attributes in its state dict.  Calling .to(device) on the whole object fails
    because PyTorch tries to convert those strings.  We move only ssfd.net (a
    plain nn.Module) to the target device instead.
    """
    global _ssfd_instance, _ssfd_device
    if _ssfd_instance is None:
        from dfd_mr_recon.metrics.ssfd import SSFD
        _ssfd_instance = SSFD(
            mode="grayscale",
            layer_names=("block4_relu1",),
            distance_func="mse",
        ).eval()
    if device is not None and device != _ssfd_device:
        _ssfd_instance.net = _ssfd_instance.net.to(device)
        _ssfd_device = device
    return _ssfd_instance


def compute_ssfd(
    pred_rss: np.ndarray,
    gt_rss: np.ndarray,
    device: torch.device,  # kept for API compatibility, not used
) -> float:
    """Compute SSFD between two (H, W) magnitude images.

    Images are normalised to [0, 1] using the GT range before scoring.
    SSFD runs on CPU regardless of device argument.

    Returns:
        Scalar SSFD value (lower is better).
    """
    gt_max = float(gt_rss.max())
    gt_min = float(gt_rss.min())
    scale = gt_max - gt_min + 1e-8

    def _to_tensor(arr):
        t = torch.from_numpy(arr.astype(np.float32))
        t = (t - gt_min) / scale
        return t.clamp(0, 1).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

    with torch.no_grad():
        ssfd_model = _get_ssfd(device)
        # Call .func() directly to bypass the torchmetrics state-accumulation
        # wrapper, which would fail when trying to sync non-tensor attributes.
        score = ssfd_model.func(_to_tensor(pred_rss), _to_tensor(gt_rss))
    return float(score.mean().item())


# ---------------------------------------------------------------------------
# GT token encoding
# ---------------------------------------------------------------------------

def encode_gt_tokens(
    tokenizer,
    gt_c_image: Tensor,
) -> Tensor:
    """Encode GT complex image to discrete token indices.

    Mirrors the encoding in reconstruct_far:
        encode(channel) = tokenizer.img_to_idx(channel_as_3ch)

    Args:
        tokenizer:    MedITok model (frozen).
        gt_c_image:   (B, H, W, 2) complex GT image, float, each channel in [-1, 1].

    Returns:
        gt_tokens: (B, 2, K, L)  long  — real/imag × codebook × spatial token.
    """
    with torch.no_grad():
        def _encode_channel(ch: Tensor) -> Tensor:
            # ch: (B, H, W)  →  (B, 3, H, W)  →  img_to_idx → (B, K, L)
            img4 = ch.unsqueeze(-1).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)
            return tokenizer.img_to_idx(img4)  # (B, K, L)

        tok_re = _encode_channel(gt_c_image[..., 0])  # (B, K, L)
        tok_im = _encode_channel(gt_c_image[..., 1])  # (B, K, L)
    # Stack to (B, 2, K, L)
    return torch.stack([tok_re, tok_im], dim=1)


# ---------------------------------------------------------------------------
# Token accuracy and NLL
# ---------------------------------------------------------------------------

def compute_token_accuracy(
    logits: Tensor,
    gt_tokens: Tensor,
) -> float:
    """Fraction of tokens predicted correctly.

    Args:
        logits:    (B, H_p, W_p, 2, K, V_K) — raw logits.
        gt_tokens: (B, 2, K, L)             — GT token indices.

    Returns:
        Scalar accuracy in [0, 1].
    """
    B, H_p, W_p, two, K, V_K = logits.shape
    L = H_p * W_p
    # pred: (B, H_p, W_p, 2, K)  →  (B, 2, K, L)
    pred = logits.argmax(-1).reshape(B, L, two, K).permute(0, 2, 3, 1)
    # gt_tokens: (B, 2, K, L)
    gt = gt_tokens.to(pred.device)
    return float((pred == gt).float().mean().item())


def compute_token_nll(
    logits: Tensor,
    gt_tokens: Tensor,
) -> float:
    """Mean negative log-likelihood (cross-entropy) of GT tokens.

    Args:
        logits:    (B, H_p, W_p, 2, K, V_K)
        gt_tokens: (B, 2, K, L)

    Returns:
        Scalar NLL (nats).
    """
    B, H_p, W_p, two, K, V_K = logits.shape
    L = H_p * W_p
    # Reshape logits to (N, V_K)
    log_flat = logits.reshape(B, L, two, K, V_K).permute(0, 2, 3, 1, 4)  # (B, 2, K, L, V_K)
    log_flat = log_flat.reshape(-1, V_K)
    # Reshape gt to (N,)
    gt_flat = gt_tokens.to(logits.device).reshape(-1)
    return float(F.cross_entropy(log_flat.float(), gt_flat).item())


# ---------------------------------------------------------------------------
# Combined metrics dict
# ---------------------------------------------------------------------------

def compute_all_metrics(
    pred_rss: np.ndarray,
    gt_rss: np.ndarray,
    device: torch.device,
    compute_ssfd_flag: bool = True,
) -> dict:
    """Compute PSNR, SSIM, NMSE, SSFD for a single (H, W) image pair.

    Uses skimage for PSNR/SSIM, piq for LPIPS/DISTS (same as
    autoregressive/utils/metrics.py), plus SSFD.

    Args:
        pred_rss: (H, W) float numpy array, not necessarily normalised.
        gt_rss:   (H, W) float numpy array.
        device:   Torch device for SSFD.

    Returns:
        dict with keys: psnr, ssim, nmse, ssfd.
    """
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    data_range = float(gt_rss.max() - gt_rss.min()) + 1e-8
    psnr = float(peak_signal_noise_ratio(gt_rss, pred_rss, data_range=data_range))
    ssim = float(structural_similarity(gt_rss, pred_rss, data_range=data_range))
    nmse = float(np.linalg.norm(gt_rss - pred_rss) ** 2 / (np.linalg.norm(gt_rss) ** 2 + 1e-8))

    out = {"psnr": psnr, "ssim": ssim, "nmse": nmse}
    if compute_ssfd_flag:
        out["ssfd"] = compute_ssfd(pred_rss, gt_rss, device)
    return out
