#!/usr/bin/env python
"""Run Candidate-Line Counterfactual Analysis for Experiment 1.

For each slice at steps [early, middle, late], evaluates every currently
unsampled k-space line by temporarily adding it and re-reconstructing.
Saves per-candidate metrics to .parquet files.

Reads:
    <run_dir>/normal_trajectory/dataset=<d>/R=<r>/method=<m>/sample_id=<id>/step_*.npz
    <run_dir>/metadata/selected_slices_{dataset}_R{acc}.json

Writes:
    <run_dir>/candidate_counterfactuals/dataset=<d>/R=<r>/base_method=<m>/
        sample_id=<id>/step_<t>.parquet

Usage:
    python run_candidate_counterfactuals.py \\
        --config experiments/experiment1/config.yaml \\
        --run_dir /fast/lev/TRUST-MRI/results/exp1 \\
        --dataset brain --acceleration 8 --base_method LES
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

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
    encode_gt_tokens, compute_token_accuracy, compute_token_nll, compute_ssfd,
)
from experiments.experiment1.utils.storage import load_step_npz, save_parquet
from experiments.experiment1.utils.slice_selector import make_sample_id
from experiments.experiment1.run_normal_trajectory import load_slice, make_center_mask

from data import transforms
from autoregressive.models.generate import reconstruct_far, data_consistency
from autoregressive.sample.adaptive_sampling import kspace_uncertainty_from_patch_logits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candidate_step_indices(n_steps: int) -> dict[str, int]:
    """Map ['early', 'middle', 'late'] to step indices."""
    return {
        "early":  0,
        "middle": n_steps // 2,
        "late":   max(0, n_steps - 1),
    }


def _les_column_scores(logits: torch.Tensor, sample_mask: torch.Tensor) -> np.ndarray:
    """Compute LES per-column scores from logits without updating the mask.

    Returns: (W,) float32 array of frequency entropy per column.
    """
    mask_for_fn = sample_mask.permute(0, 3, 1, 2)  # (B, 2, H, W)
    _, entropy_complex, freq_map = kspace_uncertainty_from_patch_logits(
        logits, mask_for_fn,
        vertical_mask=True, rank=1,
        les_zero_mean=False, les_radial_weight_alpha=0.0,
        mask_type="cartesian",
    )
    return freq_map[0].mean(dim=0).cpu().numpy()  # (W,)


def _rss(img: torch.Tensor) -> np.ndarray:
    """(1, H, W, 2) → (H, W) magnitude."""
    return transforms.complex_abs(img[0]).cpu().numpy()


# ---------------------------------------------------------------------------
# Per-candidate evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_candidates(
    model,
    tokenizer,
    gt_c_image: torch.Tensor,          # (1, H, W, 2)
    sample_mask: torch.Tensor,          # (1, H, W, 2)  — current mask at this step
    base_rec: np.ndarray,               # (H, W) base reconstruction RSS
    base_logits: torch.Tensor,          # (1, H_p, W_p, 2, K, V_K)
    base_les_scores: np.ndarray,        # (W,) or None
    candidate_cols: list[int],
    far_kwargs: dict,
    device: torch.device,
    seed_base: int,
) -> list[dict]:
    """Evaluate each candidate k-space column, returning one row per candidate."""
    B, H, W, _ = gt_c_image.shape
    gt_rss = _rss(gt_c_image)

    # Base metrics
    base_mse  = float(np.mean((base_rec - gt_rss) ** 2))
    base_ssfd = compute_ssfd(base_rec, gt_rss, device)

    # Token stats from base logits
    gt_tokens_BKL = encode_gt_tokens(tokenizer, gt_c_image)
    base_tok_acc = compute_token_accuracy(base_logits, gt_tokens_BKL)
    base_tok_nll = compute_token_nll(base_logits, gt_tokens_BKL)

    # Base entropy (per column, summed over patch rows)
    base_logits_float = base_logits.float()
    log_p = F.log_softmax(base_logits_float, dim=-1)
    base_entropy_patch = -(torch.exp(log_p) * log_p).sum(-1)  # (1, H_p, W_p, 2, K)
    _, H_p, W_p, _, K = base_entropy_patch.shape
    # Map patch columns to image columns via nearest-column assignment
    patch_entropy_col = base_entropy_patch[0].sum(dim=(0, 2, 3))  # (W_p,) sum over H_p,2,K
    # Will be interpolated to W columns for per-column entropy score
    patch_entropy_col_W = F.interpolate(
        patch_entropy_col.unsqueeze(0).unsqueeze(0).float(),
        size=W, mode="linear", align_corners=False,
    ).squeeze().cpu().numpy()  # (W,)

    base_total_entropy = float(base_entropy_patch.sum().item())

    rows = []
    for col in candidate_cols:
        # 1. Build candidate mask
        cand_mask = sample_mask.clone()
        cand_mask[0, :, col, :] = 1.0

        # 2. Build candidate masked image
        cand_img = data_consistency(gt_c_image, torch.zeros_like(gt_c_image), cand_mask)

        # 3. Reconstruct
        cand_rec_img, cand_aux = reconstruct_far(
            model, tokenizer, cand_img, cand_mask, **far_kwargs
        )
        cand_logits = cand_aux["logits"]
        cand_rss = _rss(cand_rec_img)

        # 4. Metrics after acquiring col
        cand_mse  = float(np.mean((cand_rss - gt_rss) ** 2))
        cand_ssfd = compute_ssfd(cand_rss, gt_rss, device)
        cand_tok_acc = compute_token_accuracy(cand_logits, gt_tokens_BKL)
        cand_tok_nll = compute_token_nll(cand_logits, gt_tokens_BKL)

        # Candidate entropy
        cand_lp = F.log_softmax(cand_logits.float(), dim=-1)
        cand_entropy = -(torch.exp(cand_lp) * cand_lp).sum(-1)
        cand_total_entropy = float(cand_entropy.sum().item())

        # 5. Control scores
        rng_ctrl = random.Random(seed_base + col)
        random_score = rng_ctrl.random()

        # Shuffled entropy: shuffle columns of patch_entropy_col_W
        shuffled_ent = patch_entropy_col_W.copy()
        rng_ctrl2 = random.Random(seed_base + col + 10000)
        shuffled_ent = shuffled_ent[
            np.array(rng_ctrl2.sample(range(W), W), dtype=int)
        ]
        shuffled_entropy_score = float(shuffled_ent[col])

        uniform_entropy_score = 1.0
        center_distance_score = 1.0 / (1.0 + abs(col - W // 2))

        # k-space magnitude score: mean |kspace_gt[:, col]|
        with torch.no_grad():
            kspace_gt = transforms.fft2(gt_c_image)  # (1, H, W, 2)
            kspace_col_mag = float(
                (kspace_gt[0, :, col, 0] ** 2 + kspace_gt[0, :, col, 1] ** 2)
                .sqrt().mean().item()
            )

        # LES and GEO scores for this column
        les_score = float(base_les_scores[col]) if base_les_scores is not None else float("nan")

        row = {
            "candidate_line":     col,
            "base_mse":           float(base_mse),
            "candidate_mse":      float(cand_mse),
            "delta_mse":          float(cand_mse - base_mse),
            "base_ssfd":          float(base_ssfd),
            "candidate_ssfd":     float(cand_ssfd),
            "delta_ssfd":         float(cand_ssfd - base_ssfd),
            "base_token_accuracy":     float(base_tok_acc),
            "candidate_token_accuracy": float(cand_tok_acc),
            "delta_token_accuracy":     float(cand_tok_acc - base_tok_acc),
            "base_total_entropy":     float(base_total_entropy),
            "candidate_total_entropy": float(cand_total_entropy),
            "delta_entropy":          float(cand_total_entropy - base_total_entropy),
            "LES_score":           les_score,
            "GEO_score":           float("nan"),  # filled separately if GEO method
            "random_score":        float(random_score),
            "shuffled_entropy_score":  float(shuffled_entropy_score),
            "uniform_entropy_score":   float(uniform_entropy_score),
            "center_distance_score":   float(center_distance_score),
            "kspace_magnitude_score":  float(kspace_col_mag),
        }
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_counterfactuals_for_slice(
    model,
    tokenizer,
    slice_rec: dict,
    h5_path: pathlib.Path,
    dataset: str,
    acceleration: int,
    base_method: str,
    run_dir: pathlib.Path,
    cfg: dict,
    device: torch.device,
) -> None:
    sample_id = make_sample_id(dataset, slice_rec["scan_id"], slice_rec["slice_index"])
    traj_dir = (
        run_dir / "normal_trajectory"
        / f"dataset={dataset}" / f"R={acceleration}"
        / f"method={base_method}" / f"sample_id={sample_id}"
    )
    out_dir = (
        run_dir / "candidate_counterfactuals"
        / f"dataset={dataset}" / f"R={acceleration}"
        / f"base_method={base_method}" / f"sample_id={sample_id}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover saved steps
    step_files = sorted(traj_dir.glob("step_*.npz"))
    if not step_files:
        print(f"  SKIP (no trajectory found): {sample_id}")
        return
    n_steps = len(step_files)

    resolution = cfg.get("resolution", 256)
    center_fraction = cfg.get("center_fraction", 0.04)
    ckpt_args = getattr(model, "_ckpt_args", None)
    if ckpt_args is not None and hasattr(ckpt_args, "image_size"):
        resolution = ckpt_args.image_size

    # Load GT image
    slice_data = load_slice(h5_path, slice_rec["slice_index"], resolution)
    gt_c_image = slice_data["gt_c_image"].to(device)
    _, H, W, _ = gt_c_image.shape

    far_kwargs = {
        "n_steps": 1,  # single-pass reconstruction
        "far_mask_mode": "cartesian",
        "center_fractions": center_fraction,
        "active_sampling_mask_type": "cartesian",
    }

    candidate_step_map = _candidate_step_indices(n_steps)
    target_steps = {v for v in candidate_step_map.values()}

    for step_name, step_idx in candidate_step_map.items():
        out_path = out_dir / f"step_{step_idx:03d}.parquet"
        if out_path.exists():
            continue  # already done

        step_path = traj_dir / f"step_{step_idx:03d}.npz"
        if not step_path.exists():
            print(f"  SKIP step {step_idx}: file not found at {step_path}")
            continue

        step_data = load_step_npz(step_path)

        # Reconstruct current mask from saved data
        saved_mask_1d = step_data["mask"]  # (W,) int8
        sample_mask = torch.zeros(1, H, W, 2, device=device)
        acquired_cols = np.where(saved_mask_1d > 0)[0]
        for c in acquired_cols:
            sample_mask[0, :, int(c), :] = 1.0

        # Build masked image from current mask
        masked_image = data_consistency(
            gt_c_image, torch.zeros_like(gt_c_image), sample_mask
        )

        # Get base logits for this step
        with torch.no_grad():
            base_rec_img, base_aux = reconstruct_far(
                model, tokenizer, masked_image, sample_mask, **far_kwargs
            )
        base_logits = base_aux["logits"]
        base_rss    = _rss(base_rec_img)

        # LES scores for base state
        base_les_scores = _les_column_scores(base_logits, sample_mask)

        # Candidate lines: all unsampled columns, optionally subsampled
        all_cols = list(range(W))
        unsampled_cols = [c for c in all_cols if saved_mask_1d[c] == 0]

        if not unsampled_cols:
            print(f"  SKIP step {step_idx}: no unsampled lines")
            continue

        seed_base = cfg.get("seed", 42) + slice_rec["global_index"] * 1000 + step_idx

        max_candidates = cfg.get("candidate_lines_per_step", None)
        if max_candidates is not None and len(unsampled_cols) > max_candidates:
            rng_cand = np.random.default_rng(seed_base)
            unsampled_cols = sorted(rng_cand.choice(
                unsampled_cols, size=max_candidates, replace=False
            ).tolist())

        print(f"  Evaluating {len(unsampled_cols)} candidates at step {step_idx} "
              f"({step_name}) for {sample_id[:30]}")

        rows = evaluate_candidates(
            model=model,
            tokenizer=tokenizer,
            gt_c_image=gt_c_image,
            sample_mask=sample_mask,
            base_rec=base_rss,
            base_logits=base_logits,
            base_les_scores=base_les_scores,
            candidate_cols=unsampled_cols,
            far_kwargs=far_kwargs,
            device=device,
            seed_base=seed_base,
        )

        # Add metadata to each row
        for row in rows:
            row["sample_id"]    = sample_id
            row["dataset"]      = dataset
            row["acceleration"] = acceleration
            row["base_method"]  = base_method
            row["step"]         = step_idx
            row["step_name"]    = step_name

        save_parquet(out_path, rows)
        print(f"    → saved {len(rows)} rows to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Candidate-line counterfactual analysis")
    p.add_argument("--config", default="experiments/experiment1/config.yaml")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--dataset", required=True, choices=["brain", "knee"])
    p.add_argument("--acceleration", required=True, type=int, choices=[8, 16])
    p.add_argument("--base_method", required=True, choices=["LES", "GEO"])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--meditok_config", default=None)
    p.add_argument("--meditok_ckpt", default=None)
    p.add_argument("--data_path", default=None)
    p.add_argument("--num_slices", default=None, type=int)
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

    ckpt_path = args.checkpoint or root / cfg["checkpoints"][args.dataset][args.acceleration]
    model, ckpt_args = load_singlecoil_model(ckpt_path, device)
    model._ckpt_args = ckpt_args

    mt_cfg  = args.meditok_config or root / cfg["meditok_config"]
    mt_ckpt = args.meditok_ckpt  or root / cfg["meditok_ckpt"]
    tokenizer = load_meditok(mt_cfg, mt_ckpt, device)

    meta_dir = run_dir / "metadata"
    slices_json = meta_dir / f"selected_slices_{args.dataset}_R{args.acceleration}.json"
    with open(slices_json) as f:
        slice_records = json.load(f)
    if args.num_slices is not None:
        slice_records = slice_records[:args.num_slices]

    data_root = pathlib.Path(
        args.data_path or root / cfg["data_paths"][args.dataset]
    )

    print(f"Running candidate counterfactuals: {args.base_method}, "
          f"dataset={args.dataset}, R={args.acceleration}, "
          f"{len(slice_records)} slices")

    for rec in tqdm(slice_records, desc="Slices"):
        h5_path = data_root / rec["scan_id"]
        if not h5_path.exists():
            print(f"  SKIP: {h5_path}")
            continue
        try:
            run_counterfactuals_for_slice(
                model=model,
                tokenizer=tokenizer,
                slice_rec=rec,
                h5_path=h5_path,
                dataset=args.dataset,
                acceleration=args.acceleration,
                base_method=args.base_method,
                run_dir=run_dir,
                cfg=cfg,
                device=device,
            )
        except Exception as e:
            print(f"  ERROR on {rec['scan_id']} slice {rec['slice_index']}: {e}")
            raise

    print("Done.")


if __name__ == "__main__":
    main()
