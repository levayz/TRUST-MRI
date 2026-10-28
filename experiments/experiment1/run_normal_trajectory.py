#!/usr/bin/env python
"""Run Normal Active-Sampling Trajectories for Experiment 1.

For each (dataset, acceleration, method, selected_slice), runs the active
sampling loop and saves per-step .npz files with reconstructions, token
stats, entropy, and line-selection scores.

Methods:
    LES — Logit-Based Spectral Uncertainty (entropy → FFT → select)
    GEO — Gradient-based Entropy Optimization

Usage (single setting):
    python run_normal_trajectory.py \\
        --config experiments/experiment1/config.yaml \\
        --run_dir /fast/lev/TRUST-MRI/results/exp1 \\
        --dataset brain --acceleration 8 --method LES

Usage (all settings, in parallel launch separate processes):
    for d in brain knee; do
      for r in 8 16; do
        for m in LES GEO; do
          python run_normal_trajectory.py --dataset $d --acceleration $r --method $m &
        done
      done
    done
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.experiment1.utils.model_loader import load_singlecoil_model, load_meditok
from experiments.experiment1.utils.metrics_extended import (
    encode_gt_tokens, compute_token_accuracy, compute_token_nll, compute_all_metrics,
)
from experiments.experiment1.utils.storage import save_step_npz
from experiments.experiment1.utils.slice_selector import make_sample_id

from data import transforms  # data/transforms.py — has ifft2/fft2 aliases used by generate.py
from autoregressive.models.generate import reconstruct_far, data_consistency
from autoregressive.sample.adaptive_sampling import kspace_uncertainty_from_patch_logits


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_slice(h5_path: pathlib.Path, slice_idx: int, resolution: int) -> dict:
    """Load a single slice from H5, applying the same preprocessing as SliceMRICodeDataset.

    Returns:
        dict with:
            gt_c_image:  (1, H, W, 2) complex ground-truth image in [-1, 1]
            max_val:     float, normalization factor
    """
    with h5py.File(h5_path, "r") as f:
        kspace_np = f["kspace"][slice_idx]  # (H, W) complex or (H, W, 2)
    kspace = transforms.to_tensor(kspace_np)  # (H, W, 2)
    image = transforms.ifft2(kspace)  # (H, W, 2)

    # Pad and crop to target resolution
    H, W, _ = image.shape
    if H < resolution or W < resolution:
        image = transforms.complex_pad(image, resolution, resolution)
    image = transforms.complex_center_crop(image, (resolution, resolution))

    # Normalise by max magnitude
    mag = transforms.complex_abs(image)  # (H, W)
    max_val = float(mag.max())
    if max_val > 0:
        image = image / max_val

    return {
        "gt_c_image": image.unsqueeze(0),  # (1, H, W, 2)
        "max_val": max_val,
    }


def make_center_mask(W: int, center_fraction: float, device: torch.device) -> torch.Tensor:
    """Build a 1D Cartesian center-fraction mask of shape (1, W, W, 2)."""
    cf_cols = max(1, int(W * center_fraction))
    # Make it even
    cf_cols = cf_cols if cf_cols % 2 == 0 else cf_cols + 1
    mask = torch.zeros(1, W, W, 2, device=device)
    center_start = W // 2 - cf_cols // 2
    center_end = center_start + cf_cols
    mask[:, :, center_start:center_end, :] = 1.0
    return mask


# GEO step lives in autoregressive/sample/geo.py — import here for backward compat
from autoregressive.sample.adaptive_sampling import geo_step as _geo_step  # noqa: E402


# ---------------------------------------------------------------------------
# Main sampling loop
# ---------------------------------------------------------------------------

def run_trajectory_for_slice(
    model,
    tokenizer,
    slice_rec: dict,
    h5_path: pathlib.Path,
    dataset: str,
    acceleration: int,
    method: str,
    run_dir: pathlib.Path,
    cfg: dict,
    device: torch.device,
    rng: random.Random,
) -> None:
    """Run the full active sampling trajectory for a single slice and save results."""
    sample_id = make_sample_id(dataset, slice_rec["scan_id"], slice_rec["slice_index"])
    step_dir = (
        run_dir / "normal_trajectory"
        / f"dataset={dataset}"
        / f"R={acceleration}"
        / f"method={method}"
        / f"sample_id={sample_id}"
    )
    step_dir.mkdir(parents=True, exist_ok=True)

    if (step_dir / "summary.json").exists():
        return  # already completed

    slice_idx = slice_rec["slice_index"]
    ckpt_args_res = getattr(model, "_ckpt_args", None)
    resolution = cfg.get("resolution", 256)
    if ckpt_args_res is not None and hasattr(ckpt_args_res, "image_size"):
        resolution = ckpt_args_res.image_size

    # Load and preprocess slice
    slice_data = load_slice(h5_path, slice_idx, resolution)
    gt_c_image = slice_data["gt_c_image"].to(device)  # (1, H, W, 2)
    B, H, W, _ = gt_c_image.shape

    # Compute budget and n_steps
    center_fraction = cfg.get("center_fraction", 0.04)
    rank = cfg.get("rank", 1)
    budget = max(1, W // acceleration - int(W * center_fraction))
    n_steps = budget // rank

    # Initial center mask and masked image
    sample_mask = make_center_mask(W, center_fraction, device)
    masked_image = data_consistency(gt_c_image, torch.zeros_like(gt_c_image), sample_mask)

    # Shared reconstruct_far kwargs
    far_kwargs = {
        "n_steps": 1,  # single-pass: masked image → model → full image
        "far_mask_mode": "cartesian",
        "center_fractions": center_fraction,
        "active_sampling_mask_type": "cartesian",
    }

    save_topk = cfg.get("save_token_probabilities", True)
    top_k_probs = cfg.get("top_k_probs", 64)
    save_dtype = np.float16

    for step in tqdm(range(n_steps), desc=f"{sample_id[:30]} {method}", leave=False):
        t_start = time.perf_counter()

        # ---- Forward pass to get reconstruction + logits ----
        with torch.no_grad():
            rec_image, aux = reconstruct_far(
                model, tokenizer, masked_image, sample_mask, **far_kwargs
            )
        logits = aux["logits"]  # (B, H_p, W_p, 2, K, V_K)
        B_, H_p, W_p, two, K, V_K = logits.shape

        # ---- Token statistics ----
        with torch.no_grad():
            log_probs = F.log_softmax(logits.float(), dim=-1)
            probs = torch.exp(log_probs)
            # entropy: (B, H_p, W_p, 2, K)
            token_entropy = -(probs * log_probs).sum(-1)
            total_entropy = float(token_entropy.sum().item())

            # pred_tokens: (B, H_p, W_p, 2, K)
            pred_tokens = logits.argmax(-1)

            # GT tokens: (B, 2, K, L) → reshape to (B, H_p, W_p, 2, K)
            gt_tokens_BKL = encode_gt_tokens(tokenizer, gt_c_image)  # (B, 2, K, L)
            L = H_p * W_p
            gt_tokens = (
                gt_tokens_BKL
                .permute(0, 2, 3, 1)          # (B, K, L, 2)
                .reshape(B_, K, H_p, W_p, two)
                .permute(0, 2, 3, 4, 1)       # (B, H_p, W_p, 2, K)
            )

            tok_acc = compute_token_accuracy(logits, gt_tokens_BKL)
            tok_nll = compute_token_nll(logits, gt_tokens_BKL)

        # ---- RSS reconstruction for image metrics ----
        rec_rss = transforms.complex_abs(rec_image[0]).cpu().numpy()  # (H, W)
        gt_rss  = transforms.complex_abs(gt_c_image[0]).cpu().numpy()  # (H, W)
        metrics = compute_all_metrics(rec_rss, gt_rss, device, compute_ssfd_flag=True)

        # ---- Method-specific line selection ----
        les_line_scores: Optional[np.ndarray] = None
        geo_line_scores: Optional[np.ndarray] = None
        selected_line: Optional[int] = None
        old_mask = sample_mask.clone()

        if method == "LES":
            mask_for_fn = sample_mask.permute(0, 3, 1, 2)  # (B, 2, H, W)
            new_mask_permuted, _, freq_map = kspace_uncertainty_from_patch_logits(
                logits,
                mask_for_fn,
                vertical_mask=True,
                rank=rank,
                les_zero_mean=False,
                les_radial_weight_alpha=0.0,
                mask_type="cartesian",
            )
            sample_mask = new_mask_permuted.permute(0, 2, 3, 1).float()
            # LES scores: sum freq_magnitude over rows for each column
            les_line_scores = freq_map[0].mean(dim=0).cpu().numpy()  # (W,)
            # Determine selected column
            diff = (sample_mask[0, 0, :, 0] - old_mask[0, 0, :, 0])
            sel_cols = diff.nonzero(as_tuple=True)[0]
            if len(sel_cols) > 0:
                selected_line = int(sel_cols[0].item())

        elif method == "GEO":
            new_mask, geo_scores = _geo_step(
                model, tokenizer, masked_image, sample_mask, rank, device
            )
            geo_line_scores = geo_scores.cpu().numpy()  # (W,)
            sample_mask = new_mask
            diff = (sample_mask[0, 0, :, 0] - old_mask[0, 0, :, 0])
            sel_cols = diff.nonzero(as_tuple=True)[0]
            if len(sel_cols) > 0:
                selected_line = int(sel_cols[0].item())

        # Update masked image
        masked_image = data_consistency(
            gt_c_image, torch.zeros_like(gt_c_image), sample_mask
        )

        # ---- Compressed token probs (top-k) ----
        topk_probs = topk_indices = None
        if save_topk:
            with torch.no_grad():
                tk_vals, tk_idx = torch.topk(
                    probs[0].float(), k=min(top_k_probs, V_K), dim=-1
                )  # (H_p, W_p, 2, K, top_k)
                topk_probs   = tk_vals.cpu().numpy().astype(np.float16)
                topk_indices = tk_idx.cpu().numpy().astype(np.int16)

        # ---- Save step ----
        step_data = {
            "sample_id":     sample_id,
            "dataset":       dataset,
            "acceleration":  acceleration,
            "method":        method,
            "step":          step,
            "mask":          sample_mask[0, 0, :, 0].cpu().numpy().astype(np.int8),
            "selected_line": selected_line if selected_line is not None else -1,
            "reconstruction": rec_image[0].detach().cpu().numpy(),   # (H, W, 2)
            "token_entropy": token_entropy[0].cpu().numpy(),          # (H_p, W_p, 2, K)
            "total_entropy": total_entropy,
            "pred_tokens":   pred_tokens[0].cpu().numpy().astype(np.int16),  # (H_p, W_p, 2, K)
            "gt_tokens":     gt_tokens[0].cpu().numpy().astype(np.int16),    # (H_p, W_p, 2, K)
            "token_accuracy": tok_acc,
            "token_nll":     tok_nll,
            "psnr":  metrics["psnr"],
            "ssim":  metrics["ssim"],
            "nmse":  metrics["nmse"],
            "ssfd":  metrics.get("ssfd", float("nan")),
            "runtime_seconds": time.perf_counter() - t_start,
        }
        if les_line_scores is not None:
            step_data["LES_line_scores"] = les_line_scores
        if geo_line_scores is not None:
            step_data["GEO_line_scores"] = geo_line_scores
        if topk_probs is not None:
            step_data["topk_probs"]   = topk_probs    # (H_p, W_p, 2, K, top_k)
            step_data["topk_indices"] = topk_indices  # (H_p, W_p, 2, K, top_k)

        save_step_npz(step_dir / f"step_{step:03d}.npz", step_data, dtype=save_dtype)

    # Save final reconstruction
    with torch.no_grad():
        final_rec, _ = reconstruct_far(
            model, tokenizer, masked_image, sample_mask, **far_kwargs
        )
    final_rss = transforms.complex_abs(final_rec[0]).cpu().numpy()
    gt_rss_final = transforms.complex_abs(gt_c_image[0]).cpu().numpy()
    final_metrics = compute_all_metrics(final_rss, gt_rss_final, device)
    final_meta = {
        "sample_id": sample_id,
        "n_steps": n_steps,
        "budget": budget,
        **{f"final_{k}": v for k, v in final_metrics.items()},
    }
    (step_dir / "summary.json").write_text(json.dumps(final_meta, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run normal active-sampling trajectories")
    p.add_argument("--config", default="experiments/experiment1/config.yaml")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--dataset", required=True, choices=["brain", "knee"])
    p.add_argument("--acceleration", required=True, type=int, choices=[8, 16])
    p.add_argument("--method", required=True, choices=["LES", "GEO"])
    p.add_argument("--checkpoint", default=None,
                   help="Override checkpoint path (default: from config)")
    p.add_argument("--meditok_config", default=None)
    p.add_argument("--meditok_ckpt", default=None)
    p.add_argument("--data_path", default=None,
                   help="Override data directory for this dataset")
    p.add_argument("--num_slices", default=None, type=int,
                   help="Limit number of slices (for quick smoke tests)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    root = _ROOT

    config_path = pathlib.Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    run_dir = pathlib.Path(args.run_dir)

    # ---- Model ----
    ckpt_path = args.checkpoint or root / cfg["checkpoints"][args.dataset][args.acceleration]
    model, ckpt_args = load_singlecoil_model(ckpt_path, device)
    model._ckpt_args = ckpt_args  # stash for resolution lookup

    # ---- Tokenizer ----
    mt_cfg = args.meditok_config or root / cfg["meditok_config"]
    mt_ckpt = args.meditok_ckpt  or root / cfg["meditok_ckpt"]
    tokenizer = load_meditok(mt_cfg, mt_ckpt, device)

    # ---- Selected slices ----
    meta_dir = run_dir / "metadata"
    slices_json = meta_dir / f"selected_slices_{args.dataset}_R{args.acceleration}.json"
    if not slices_json.exists():
        print(f"ERROR: slice metadata not found at {slices_json}. "
              "Run select_slices_center_out.py first.")
        sys.exit(1)
    with open(slices_json) as f:
        slice_records = json.load(f)
    if args.num_slices is not None:
        slice_records = slice_records[:args.num_slices]

    # ---- Data path ----
    data_root = pathlib.Path(args.data_path or root / cfg["data_paths"][args.dataset])

    rng = random.Random(cfg.get("seed", 42))

    print(f"Running {args.method} trajectories: {len(slice_records)} slices, "
          f"dataset={args.dataset}, R={args.acceleration}")

    for rec in tqdm(slice_records, desc="Slices"):
        h5_path = data_root / rec["scan_id"]
        if not h5_path.exists():
            print(f"  SKIP (not found): {h5_path}")
            continue
        try:
            run_trajectory_for_slice(
                model=model,
                tokenizer=tokenizer,
                slice_rec=rec,
                h5_path=h5_path,
                dataset=args.dataset,
                acceleration=args.acceleration,
                method=args.method,
                run_dir=run_dir,
                cfg=cfg,
                device=device,
                rng=rng,
            )
        except Exception as e:
            print(f"  ERROR on {rec['scan_id']} slice {rec['slice_index']}: {e}")
            raise

    print("Done.")


if __name__ == "__main__":
    main()
