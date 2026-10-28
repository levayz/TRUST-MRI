#!/usr/bin/env python
"""
Offline analysis: does high Transformer token entropy correspond to locally
complex anatomical regions under random 1D Cartesian undersampling?

For each validation slice and each random mask realisation:
  1. Apply a random 1D Cartesian mask → zero-filled input.
  2. Forward-pass through tokenizer + Transformer → patch-level logits.
  3. Compute token entropy H_token(l) from the predicted token distribution.
  4. Compute image complexity measures on the GT magnitude image.
  5. Flatten over slices/patches and compute Spearman & Pearson correlations.

Usage:
  python scripts/analyze_token_entropy_complexity.py \
      --dataset knee --acceleration 8 \
      --output_dir /fast/lev/TRUST-MRI/results/entropy_complexity/knee_R8 \
      --num_random_masks 3 --seed 42 --save_detail
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy import stats
from tqdm import tqdm

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data import transforms
from autoregressive.models.generate import reconstruct_far, data_consistency
from experiments.experiment1.run_normal_trajectory import load_slice
from experiments.experiment1.utils.model_loader import load_singlecoil_model, load_meditok
from experiments.experiment1.utils.slice_selector import select_slices_center_out


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEASURE_NAMES = [
    "pixel_entropy",
    "local_variance",
    "gradient_energy",
    "laplacian_energy",
    "high_frequency_energy",
]

MEASURE_LABELS = {
    "pixel_entropy": "Pixel Entropy",
    "local_variance": "Local Variance",
    "gradient_energy": "Gradient Energy",
    "laplacian_energy": "Laplacian Energy",
    "high_frequency_energy": "HF Energy",
}

FAR_KWARGS = dict(n_steps=1, far_mask_mode="cartesian", active_sampling_mask_type="cartesian")

# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def make_random_cartesian_mask(
    H: int,
    W: int,
    acceleration: int,
    center_fraction: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Random 1D Cartesian undersampling mask, shape (1, H, W, 2).

    Acquires W // acceleration phase-encoding lines total.
    Center lines: max(1, round(W * center_fraction)).
    Remaining lines are drawn uniformly at random from outer k-space.
    """
    total_lines = max(1, W // acceleration)
    num_center = max(1, round(W * center_fraction))
    num_center = min(num_center, total_lines)

    mask_1d = np.zeros(W, dtype=np.float32)
    c_start = W // 2 - num_center // 2
    c_end = c_start + num_center
    mask_1d[c_start:c_end] = 1.0

    outer_cols = np.where(mask_1d == 0)[0]
    num_outer = total_lines - num_center
    if num_outer > 0 and len(outer_cols) > 0:
        rng = np.random.RandomState(seed)
        chosen = rng.choice(outer_cols, size=min(num_outer, len(outer_cols)), replace=False)
        mask_1d[chosen] = 1.0

    mask_t = torch.from_numpy(mask_1d).to(device)           # (W,)
    mask_4d = mask_t.view(1, 1, W, 1).expand(1, H, W, 2).clone()
    return mask_4d


# ---------------------------------------------------------------------------
# Token entropy
# ---------------------------------------------------------------------------

def compute_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy summed over real/imag and codebooks.

    Args:
        logits: (B, H_p, W_p, 2, K, V_K)
    Returns:
        (B, H_p, W_p)
    """
    log_p = F.log_softmax(logits.float(), dim=-1)
    p = log_p.exp()
    H_per_cb = -(p * log_p).sum(-1)        # (B, H_p, W_p, 2, K)
    return H_per_cb.sum(dim=(-2, -1))       # (B, H_p, W_p)


# ---------------------------------------------------------------------------
# Image complexity measures
# ---------------------------------------------------------------------------

def _sobel_kernels(device: torch.device, dtype: torch.dtype):
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                      device=device, dtype=dtype).view(1, 1, 3, 3)
    return kx, ky


def _laplacian_kernel(device: torch.device, dtype: torch.dtype):
    return torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                        device=device, dtype=dtype).view(1, 1, 3, 3)


def compute_complexity_measures(
    gt_mag_norm: torch.Tensor,  # (B, H, W) float in [0, 1]
    patch_size: int,
    hist_bins: int,
) -> dict[str, torch.Tensor]:
    """Compute five per-patch image complexity maps.

    All returned tensors have shape (B, H_lat, W_lat).
    """
    B, H, W = gt_mag_norm.shape
    p = patch_size
    H_lat, W_lat = H // p, W // p
    L = H_lat * W_lat
    eps = 1e-10
    device = gt_mag_norm.device

    x = gt_mag_norm.unsqueeze(1)                                  # (B, 1, H, W)

    # --- shared patch extraction (intensity) ---
    patches = F.unfold(x, kernel_size=p, stride=p)               # (B, p*p, L)

    # A. Pixel entropy via histogram scatter
    patches_BL = patches.permute(0, 2, 1).reshape(B * L, p * p)  # (B*L, p*p)
    bin_idx = (patches_BL.clamp(0., 1. - 1e-7) * hist_bins).long()
    counts = torch.zeros(B * L, hist_bins, device=device)
    counts.scatter_add_(1, bin_idx, torch.ones_like(patches_BL))
    probs = counts / (p * p)
    pixel_entropy = -(probs * torch.log(probs + eps)).sum(1)      # (B*L,)
    pixel_entropy = pixel_entropy.view(B, H_lat, W_lat)

    # B. Local variance
    local_var = patches.var(dim=1, unbiased=False).view(B, H_lat, W_lat)

    # C. Gradient energy (Sobel magnitude, then mean per patch)
    kx, ky = _sobel_kernels(device, x.dtype)
    grad_mag = ((F.conv2d(x, kx, padding=1)**2 + F.conv2d(x, ky, padding=1)**2)
                .sqrt())                                           # (B, 1, H, W)
    gradient_energy = (F.unfold(grad_mag, kernel_size=p, stride=p)
                       .mean(dim=1).view(B, H_lat, W_lat))

    # D. Laplacian energy (abs response, mean per patch)
    lap = F.conv2d(x, _laplacian_kernel(device, x.dtype), padding=1).abs()
    laplacian_energy = (F.unfold(lap, kernel_size=p, stride=p)
                        .mean(dim=1).view(B, H_lat, W_lat))

    # E. High-frequency energy (per-patch 2D FFT, fraction outside centre disc)
    patches_spatial = patches.permute(0, 2, 1).view(B, L, p, p)  # (B, L, p, p)
    fft_out = torch.fft.fftshift(
        torch.fft.fft2(patches_spatial.float()), dim=(-2, -1)
    )
    power = fft_out.abs() ** 2                                    # (B, L, p, p)

    cy, cx = p // 2, p // 2
    ys = torch.arange(p, device=device)
    xs = torch.arange(p, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    low_mask = ((gy - cy) ** 2 + (gx - cx) ** 2) < (p // 4) ** 2  # (p, p)

    low_power = power[:, :, low_mask].sum(-1)                     # (B, L)
    total_power = power.sum(dim=(-2, -1))                         # (B, L)
    hf_energy = (total_power - low_power) / (total_power + eps)
    hf_energy = hf_energy.view(B, H_lat, W_lat)

    return {
        "pixel_entropy":        pixel_entropy,
        "local_variance":       local_var,
        "gradient_energy":      gradient_energy,
        "laplacian_energy":     laplacian_energy,
        "high_frequency_energy": hf_energy,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_slice(mag: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-slice min-max normalisation to [0, 1]. Input: (B, H, W) or (H, W)."""
    squeeze = mag.ndim == 2
    if squeeze:
        mag = mag.unsqueeze(0)
    mn = mag.amin(dim=(-2, -1), keepdim=True)
    mx = mag.amax(dim=(-2, -1), keepdim=True)
    out = (mag - mn) / (mx - mn + eps)
    return out.squeeze(0) if squeeze else out


def crop_to_patch_grid(t: torch.Tensor, p: int) -> torch.Tensor:
    """Crop spatial dims of (B, H, W) or (B, H, W, C) to be divisible by p."""
    if t.ndim == 3:
        B, H, W = t.shape
        return t[:, :H // p * p, :W // p * p]
    B, H, W, C = t.shape
    return t[:, :H // p * p, :W // p * p, :]


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def correlate(h_flat: np.ndarray, c_flat: np.ndarray) -> dict:
    mask = np.isfinite(h_flat) & np.isfinite(c_flat)
    h, c = h_flat[mask], c_flat[mask]
    sp_r, sp_p = stats.spearmanr(h, c)
    pe_r, pe_p = stats.pearsonr(h, c)
    return dict(spearman_rho=float(sp_r), spearman_p=float(sp_p),
                pearson_r=float(pe_r), pearson_p=float(pe_p),
                num_patches=int(mask.sum()))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_quintile_bars(df: pd.DataFrame, tag: str, out_dir: pathlib.Path,
                       n_boot: int = 500, rng_seed: int = 0) -> None:
    """Mean complexity per token-entropy quintile with bootstrap 95% CI."""
    rng = np.random.default_rng(rng_seed)
    df = df.copy()
    df["quintile"] = pd.qcut(df["token_entropy"], q=5,
                             labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    qs = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    fig, axes = plt.subplots(1, len(MEASURE_NAMES),
                             figsize=(4 * len(MEASURE_NAMES), 4), squeeze=False)
    axes = axes[0]
    for ax, m in zip(axes, MEASURE_NAMES):
        means, lo, hi = [], [], []
        for q in qs:
            vals = df[df["quintile"] == q][m].values
            mu = vals.mean()
            boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
            means.append(mu)
            lo.append(mu - np.percentile(boots, 2.5))
            hi.append(np.percentile(boots, 97.5) - mu)
        ax.bar(qs, means, yerr=[lo, hi], capsize=4, color="steelblue", alpha=0.85)
        ax.set_title(MEASURE_LABELS[m], fontsize=9)
        ax.set_xlabel("Token Entropy Quintile", fontsize=8)
        ax.set_ylabel("Mean Complexity", fontsize=8)
        ax.tick_params(labelsize=7)
    plt.suptitle(tag, y=1.01, fontsize=10)
    plt.tight_layout()
    path = out_dir / f"quintile_{tag}.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_scatter(df: pd.DataFrame, tag: str, out_dir: pathlib.Path,
                 corr_rows: list[dict]) -> None:
    """Hexbin scatter of token_entropy vs each complexity measure."""
    corr_map = {r["measure"]: r for r in corr_rows}
    fig, axes = plt.subplots(1, len(MEASURE_NAMES),
                             figsize=(4 * len(MEASURE_NAMES), 4), squeeze=False)
    axes = axes[0]
    for ax, m in zip(axes, MEASURE_NAMES):
        ax.hexbin(df["token_entropy"].values, df[m].values,
                  gridsize=40, mincnt=1, cmap="Blues")
        rho = corr_map.get(m, {}).get("spearman_rho", float("nan"))
        ax.set_title(f"{MEASURE_LABELS[m]}\nρ={rho:.3f}", fontsize=9)
        ax.set_xlabel("Token Entropy", fontsize=8)
        ax.set_ylabel(MEASURE_LABELS[m], fontsize=8)
        ax.tick_params(labelsize=7)
    plt.suptitle(tag, y=1.01, fontsize=10)
    plt.tight_layout()
    path = out_dir / f"scatter_{tag}.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved {path}")


def save_sanity_check(
    gt_mag: torch.Tensor,     # (H, W)
    zf_mag: torch.Tensor,     # (H, W)
    token_ent: torch.Tensor,  # (H_p, W_p)
    pixel_ent: torch.Tensor,  # (H_p, W_p)
    grad_ent: torch.Tensor,   # (H_p, W_p)
    hf_ent: torch.Tensor,     # (H_p, W_p)
    out_dir: pathlib.Path,
    tag: str,
    idx: int,
) -> None:
    H, W = gt_mag.shape

    def up(t: torch.Tensor) -> np.ndarray:
        return (F.interpolate(t.float().cpu().unsqueeze(0).unsqueeze(0),
                              size=(H, W), mode="bilinear", align_corners=False)
                .squeeze().numpy())

    panels = [
        ("GT Magnitude",    gt_mag.float().cpu().numpy(),   "gray"),
        ("Zero-filled",     zf_mag.float().cpu().numpy(),   "gray"),
        ("Token Entropy",   up(token_ent),                  "hot"),
        ("Pixel Entropy",   up(pixel_ent),                  "hot"),
        ("Gradient Energy", up(grad_ent),                   "hot"),
        ("HF Energy",       up(hf_ent),                     "hot"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3))
    for ax, (title, img, cmap) in zip(axes, panels):
        vmax = img.max() if img.max() > 0 else 1.0
        ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    plt.suptitle(tag, fontsize=9, y=1.01)
    plt.tight_layout()
    path = out_dir / f"sanity_{tag}_{idx:04d}.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Per-slice processing
# ---------------------------------------------------------------------------

@torch.no_grad()
def process_slice(
    model,
    tokenizer,
    slice_rec: dict,
    data_root: pathlib.Path,
    acceleration: int,
    center_fraction: float,
    patch_size: int,
    hist_bins: int,
    num_random_masks: int,
    base_seed: int,
    device: torch.device,
    resolution: int,
    sanity_dir: Optional[pathlib.Path],
    sanity_idx: int,
) -> Optional[pd.DataFrame]:
    """Process one slice; returns a DataFrame with one row per (mask, patch)."""
    h5_path = data_root / slice_rec["scan_id"]
    if not h5_path.exists():
        return None

    slice_data = load_slice(h5_path, slice_rec["slice_index"], resolution)
    gt_c = slice_data["gt_c_image"].to(device)  # (1, H, W, 2)
    gt_c = crop_to_patch_grid(gt_c, patch_size)
    _, H, W, _ = gt_c.shape
    H_lat, W_lat, L = H // patch_size, W // patch_size, (H // patch_size) * (W // patch_size)

    # GT magnitude, normalised once (same for all masks)
    gt_mag = transforms.complex_abs(gt_c[0]).unsqueeze(0)  # (1, H, W)
    gt_mag_norm = normalize_slice(gt_mag)                   # (1, H, W)

    complexity = compute_complexity_measures(gt_mag_norm, patch_size, hist_bins)
    # each value: (1, H_lat, W_lat)
    for name, cm in complexity.items():
        assert cm.shape == (1, H_lat, W_lat), \
            f"{name}: expected (1,{H_lat},{W_lat}), got {tuple(cm.shape)}"

    dfs: list[pd.DataFrame] = []
    global_idx = slice_rec.get("global_index", 0)

    for mask_id in range(num_random_masks):
        mask_seed = base_seed + mask_id * 100_000 + global_idx
        mask = make_random_cartesian_mask(H, W, acceleration, center_fraction,
                                          mask_seed, device)
        zf = data_consistency(gt_c, torch.zeros_like(gt_c), mask)

        far_kw = {**FAR_KWARGS, "center_fractions": center_fraction}
        _, aux = reconstruct_far(model, tokenizer, zf, mask, **far_kw)
        logits = aux["logits"]  # (1, H_p, W_p, 2, K, V_K)

        token_entropy = compute_token_entropy(logits)  # (1, H_p, W_p)
        _, H_p, W_p = token_entropy.shape
        assert H_p == H_lat and W_p == W_lat, \
            f"Token entropy {H_p}×{W_p} != patch grid {H_lat}×{W_lat}"

        te_flat = token_entropy[0].cpu().float().numpy().ravel()  # (L,)
        patch_y = np.repeat(np.arange(H_lat), W_lat)
        patch_x = np.tile(np.arange(W_lat), H_lat)

        row_data: dict[str, object] = {
            "volume_id": slice_rec["scan_id"],
            "slice_id":  slice_rec["slice_index"],
            "mask_id":   mask_id,
            "mask_seed": mask_seed,
            "patch_y":   patch_y,
            "patch_x":   patch_x,
            "token_entropy": te_flat,
        }
        for name, cm in complexity.items():
            row_data[name] = cm[0].cpu().float().numpy().ravel()

        dfs.append(pd.DataFrame(row_data))

        # Visual sanity check for first few slices (mask_id=0 only)
        if mask_id == 0 and sanity_dir is not None and sanity_idx >= 0:
            zf_mag = normalize_slice(transforms.complex_abs(zf[0]).unsqueeze(0))[0]
            save_sanity_check(
                gt_mag=gt_mag_norm[0],
                zf_mag=zf_mag,
                token_ent=token_entropy[0],
                pixel_ent=complexity["pixel_entropy"][0],
                grad_ent=complexity["gradient_energy"][0],
                hf_ent=complexity["high_frequency_energy"][0],
                out_dir=sanity_dir,
                tag=slice_rec["scan_id"].replace(".h5", ""),
                idx=sanity_idx,
            )

    return pd.concat(dfs, ignore_index=True) if dfs else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Token entropy vs image complexity analysis")
    p.add_argument("--dataset",          required=True, choices=["brain", "knee"])
    p.add_argument("--acceleration",     required=True, type=int, choices=[8, 16])
    p.add_argument("--checkpoint",       default=None)
    p.add_argument("--meditok_config",   default=None)
    p.add_argument("--meditok_ckpt",     default=None)
    p.add_argument("--data_path",        default=None)
    p.add_argument("--split_file",       default=None)
    p.add_argument("--config",           default="experiments/experiment1/config.yaml")
    p.add_argument("--output_dir",       required=True)
    p.add_argument("--patch_size",       type=int, default=16)
    p.add_argument("--hist_bins",        type=int, default=32)
    p.add_argument("--num_random_masks", type=int, default=1)
    p.add_argument("--max_slices",       type=int, default=None)
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--num_sanity_checks", type=int, default=4)
    p.add_argument("--save_detail",      action="store_true",
                   help="Save per-patch detail CSV")
    p.add_argument("--save_scatter",     action="store_true",
                   help="Save token-entropy vs complexity scatter plots")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = pathlib.Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = _ROOT / cfg_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    center_fraction = float(cfg.get("center_fraction", 0.04))

    # Load models
    ckpt_path = args.checkpoint or (_ROOT / cfg["checkpoints"][args.dataset][args.acceleration])
    model, ckpt_args = load_singlecoil_model(ckpt_path, device)
    model._ckpt_args = ckpt_args
    resolution = getattr(ckpt_args, "image_size", 256)

    mt_cfg  = args.meditok_config or (_ROOT / cfg["meditok_config"])
    mt_ckpt = args.meditok_ckpt   or (_ROOT / cfg["meditok_ckpt"])
    tokenizer = load_meditok(mt_cfg, mt_ckpt, device)

    data_root  = pathlib.Path(args.data_path   or (_ROOT / cfg["data_paths"][args.dataset]))
    split_file = pathlib.Path(args.split_file  or (_ROOT / cfg["split_files"][args.dataset]))

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sanity_dir = out_dir / "sanity_checks"
    sanity_dir.mkdir(exist_ok=True)

    # Slice selection
    slices = select_slices_center_out(
        data_root=data_root, split_file=split_file,
        dataset=args.dataset, acceleration=args.acceleration,
        num_slices=args.max_slices, seed=args.seed,
    )
    print(f"[{args.dataset} R={args.acceleration}]  {len(slices)} slices selected")

    # Persist run config
    run_cfg = {**vars(args), "center_fraction": center_fraction, "resolution": resolution}
    (out_dir / "config.json").write_text(json.dumps(run_cfg, indent=2, default=str))

    # Main loop
    all_dfs: list[pd.DataFrame] = []
    sanity_count = 0

    for i, rec in enumerate(tqdm(slices, desc=f"{args.dataset} R={args.acceleration}")):
        s_idx = sanity_count if sanity_count < args.num_sanity_checks else -1
        df_slice = process_slice(
            model, tokenizer, rec, data_root,
            acceleration=args.acceleration,
            center_fraction=center_fraction,
            patch_size=args.patch_size,
            hist_bins=args.hist_bins,
            num_random_masks=args.num_random_masks,
            base_seed=args.seed,
            device=device,
            resolution=resolution,
            sanity_dir=sanity_dir if s_idx >= 0 else None,
            sanity_idx=s_idx,
        )
        if df_slice is None:
            continue
        all_dfs.append(df_slice)
        if s_idx >= 0:
            sanity_count += 1

    if not all_dfs:
        print("No data collected — check data paths.")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    df["dataset"]      = args.dataset
    df["acceleration"] = args.acceleration

    # Drop non-finite values
    key_cols = ["token_entropy"] + MEASURE_NAMES
    n_before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=key_cols)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped} rows with NaN/Inf values.")

    # Assertions
    assert df["token_entropy"].notna().all()
    for m in MEASURE_NAMES:
        assert df[m].notna().all()

    # Correlation analysis
    tag = f"{args.dataset}_R{args.acceleration}"
    summary_rows: list[dict] = []
    for m in MEASURE_NAMES:
        corr = correlate(df["token_entropy"].values, df[m].values)
        corr["measure"]      = m
        corr["dataset"]      = args.dataset
        corr["acceleration"] = args.acceleration
        corr["num_slices"]   = int(df["slice_id"].nunique())
        corr["num_masks"]    = args.num_random_masks
        summary_rows.append(corr)

    col_order = ["dataset", "acceleration", "measure",
                 "spearman_rho", "spearman_p",
                 "pearson_r", "pearson_p",
                 "num_patches", "num_slices", "num_masks"]
    summary_df = pd.DataFrame(summary_rows)[col_order]
    print(f"\n=== {tag} ===")
    print(summary_df.to_string(index=False))

    summary_path = out_dir / f"summary_{tag}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary → {summary_path}")

    if args.save_detail:
        detail_path = out_dir / f"detail_{tag}.csv"
        df.to_csv(detail_path, index=False)
        print(f"Saved detail → {detail_path}")

    # Plots
    plot_quintile_bars(df, tag, out_dir)
    if args.save_scatter:
        plot_scatter(df, tag, out_dir, summary_rows)

    print("\nDone.")


if __name__ == "__main__":
    main()
