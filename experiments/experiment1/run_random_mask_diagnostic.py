#!/usr/bin/env python
"""Experiment A: Patch entropy predicts patch reconstruction error (random-mask diagnostic).

For each (dataset, R, scan, slice, seed), generates B random k-space trajectories
and at each acquisition step measures:
  - per-patch token entropy (H_p × W_p)
  - per-patch reconstruction MSE against fully-sampled target (H_p × W_p)
  - Spearman ρ between entropy and MSE maps
  - mean MSE per entropy quintile (Q1–Q5, normalized by mean patch MSE)

Using random masks avoids the circularity of using entropy-guided masks to
evaluate whether entropy correlates with error.

Budgets (derived from W // R - floor(W × center_fraction)):
    Brain R=8:  B=22    Brain R=16: B=6
    Knee  R=8:  B=28    Knee  R=16: B=8

Usage:
    python run_random_mask_diagnostic.py \\
        --config experiments/experiment1/config.yaml \\
        --run_dir results/exp1_entropy \\
        --dataset brain --acceleration 16 \\
        --num_scans 50 --num_seeds 5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr
from tqdm import tqdm

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.experiment1.utils.model_loader import load_singlecoil_model, load_meditok
from experiments.experiment1.utils.metrics_extended import compute_all_metrics
from experiments.experiment1.utils.storage import save_parquet
from experiments.experiment1.utils.slice_selector import make_sample_id, select_slices_fixed_fractions
from experiments.experiment1.run_normal_trajectory import load_slice, make_center_mask

from data import transforms
from autoregressive.models.generate import reconstruct_far, data_consistency


FAR_KWARGS = dict(
    n_steps=1,
    far_mask_mode="cartesian",
    active_sampling_mask_type="cartesian",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entropy_map(logits: torch.Tensor) -> np.ndarray:
    """(1, H_p, W_p, 2, K, V_K) → (H_p, W_p) entropy summed over channels+codebooks."""
    log_p = F.log_softmax(logits.float(), dim=-1)
    p = log_p.exp()
    H = -(p * log_p).sum(-1)          # (1, H_p, W_p, 2, K)
    return H[0].sum(dim=(2, 3)).cpu().numpy()  # (H_p, W_p)


def _patch_mse(rec_rss: np.ndarray, gt_rss: np.ndarray, H_p: int, W_p: int) -> np.ndarray:
    """Compute per-patch MSE. Returns (H_p, W_p) array."""
    H, W = rec_rss.shape
    ph, pw = H // H_p, W // W_p
    rec_p = rec_rss.reshape(H_p, ph, W_p, pw)
    gt_p  = gt_rss.reshape(H_p, ph, W_p, pw)
    return ((rec_p - gt_p) ** 2).mean(axis=(1, 3))  # (H_p, W_p)


def _entropy_bin_errors(
    entropy_map: np.ndarray,
    patch_mse: np.ndarray,
    n_bins: int = 5,
) -> tuple[list[float], list[float]]:
    """Split patches into n_bins entropy quintiles; return (raw, normalized) mean MSE per bin."""
    flat_e = entropy_map.flatten()
    flat_m = patch_mse.flatten()
    mean_mse = float(flat_m.mean()) + 1e-12
    thresholds = np.quantile(flat_e, np.linspace(0, 1, n_bins + 1))
    raw_bins, norm_bins = [], []
    for i in range(n_bins):
        lo, hi = thresholds[i], thresholds[i + 1]
        if i < n_bins - 1:
            mask = (flat_e >= lo) & (flat_e < hi)
        else:
            mask = flat_e >= lo
        vals = flat_m[mask]
        raw = float(vals.mean()) if len(vals) > 0 else float("nan")
        raw_bins.append(raw)
        norm_bins.append(raw / mean_mse)
    return raw_bins, norm_bins


# ---------------------------------------------------------------------------
# Per-slice computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_slice(
    model,
    tokenizer,
    slice_rec: dict,
    h5_path: pathlib.Path,
    dataset: str,
    acceleration: int,
    center_fraction: float,
    num_seeds: int,
    run_dir: pathlib.Path,
    device: torch.device,
    global_slice_idx: int,
    batch_size: int = 16,
) -> list[dict]:
    sample_id = make_sample_id(dataset, slice_rec["scan_id"], slice_rec["slice_index"])
    slice_dir = (
        run_dir / "random_mask_diagnostic"
        / f"dataset={dataset}" / f"R={acceleration}"
        / f"sample_id={sample_id}"
    )
    slice_dir.mkdir(parents=True, exist_ok=True)

    ckpt_args = getattr(model, "_ckpt_args", None)
    resolution = 256
    if ckpt_args is not None and hasattr(ckpt_args, "image_size"):
        resolution = ckpt_args.image_size

    slice_data = load_slice(h5_path, slice_rec["slice_index"], resolution)
    gt_c_image = slice_data["gt_c_image"].to(device)   # (1, H, W, 2)
    _, H, W, _ = gt_c_image.shape
    gt_rss = transforms.complex_abs(gt_c_image[0]).cpu().numpy()  # (H, W)

    center_mask = make_center_mask(W, center_fraction, device)
    acquired = center_mask[0, 0, :, 0].bool()
    outer_cols = [c for c in range(W) if not acquired[c].item()]
    B_budget = max(1, W // acceleration - int(W * center_fraction))

    far_kwargs = {**FAR_KWARGS, "center_fractions": center_fraction}

    # --- Phase 1: collect pending (seed, t) items; short-circuit done seeds ---
    pending_items: list[tuple[int, int, list[int], torch.Tensor]] = []
    seeds_done: dict[int, list[dict]] = {}

    for seed in range(1, num_seeds + 1):
        seed_dir = slice_dir / f"seed={seed}"
        done_file = seed_dir / "done.json"
        if done_file.exists():
            with open(done_file) as f:
                seeds_done[seed] = json.load(f)
            continue
        seed_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed * 100000 + global_slice_idx)
        shuffled = rng.permutation(outer_cols).tolist()
        for t in range(1, B_budget + 1):
            mask = center_mask.clone()
            mask[0, :, shuffled[:t], :] = 1.0
            pending_items.append((seed, t, shuffled, mask))

    # --- Phase 2: batched forward passes over all pending (seed, t) pairs ---
    # results[(seed, t)] → dict with emap, rec_rss, pmap, mask_vec, metrics
    results: dict[tuple[int, int], dict] = {}

    for i in range(0, len(pending_items), batch_size):
        chunk = pending_items[i : i + batch_size]
        n = len(chunk)

        masks_batch = torch.cat([item[3] for item in chunk], dim=0)        # (n, H, W, 2)
        gt_batch = gt_c_image.expand(n, -1, -1, -1).contiguous()           # (n, H, W, 2)
        masked_batch = data_consistency(gt_batch, torch.zeros_like(gt_batch), masks_batch)

        rec_imgs, aux = reconstruct_far(
            model, tokenizer, masked_batch, masks_batch, **far_kwargs
        )
        logits_batch = aux["logits"]  # (n, H_p, W_p, 2, K, V_K)
        H_p, W_p = logits_batch.shape[1], logits_batch.shape[2]

        for j, (seed, t, _, mask) in enumerate(chunk):
            emap = _entropy_map(logits_batch[j : j + 1])
            rec_rss_j = transforms.complex_abs(rec_imgs[j]).cpu().numpy()
            pmap = _patch_mse(rec_rss_j, gt_rss, H_p, W_p)
            metrics = compute_all_metrics(rec_rss_j, gt_rss, device, compute_ssfd_flag=False)
            results[(seed, t)] = {
                "emap": emap, "rec_rss": rec_rss_j, "pmap": pmap,
                "mask_vec": mask[0, 0, :, 0].cpu().numpy().astype(np.int8),
                "metrics": metrics,
            }

    # --- Phase 3: assemble per-seed rows from results and save done files ---
    rows: list[dict] = []
    for seed in range(1, num_seeds + 1):
        if seed in seeds_done:
            rows.extend(seeds_done[seed])
            continue

        seed_dir = slice_dir / f"seed={seed}"
        seed_rows: list[dict] = []
        for t in range(1, B_budget + 1):
            r = results[(seed, t)]
            emap, pmap = r["emap"], r["pmap"]
            rec_rss_j = r["rec_rss"]
            mask_vec = r["mask_vec"]
            metrics = r["metrics"]

            mean_entropy = float(emap.mean())
            mean_patch_error = float(pmap.mean())
            rho = float(spearmanr(emap.flatten(), pmap.flatten()).statistic)
            raw_bins, norm_bins = _entropy_bin_errors(emap, pmap)
            n_sampled = int(mask_vec.sum())

            seed_rows.append({
                "dataset": dataset,
                "R": acceleration,
                "scan_id": slice_rec["scan_id"],
                "slice_id": slice_rec["slice_index"],
                "volume_fraction": slice_rec.get("volume_fraction", 0.5),
                "seed_id": seed,
                "step": t,
                "policy": "random_mask_diagnostic",
                "n_sampled_lines": n_sampled,
                "sampled_fraction": n_sampled / W,
                "mean_entropy": mean_entropy,
                "mean_patch_error": mean_patch_error,
                "MSE": float(((rec_rss_j - gt_rss) ** 2).mean()),
                "NMSE": metrics["nmse"],
                "PSNR": metrics["psnr"],
                "SSIM": metrics["ssim"],
                "spearman_rho": rho,
                "bin_error_Q1": raw_bins[0], "bin_error_Q2": raw_bins[1],
                "bin_error_Q3": raw_bins[2], "bin_error_Q4": raw_bins[3],
                "bin_error_Q5": raw_bins[4],
                "normalized_bin_error_Q1": norm_bins[0],
                "normalized_bin_error_Q2": norm_bins[1],
                "normalized_bin_error_Q3": norm_bins[2],
                "normalized_bin_error_Q4": norm_bins[3],
                "normalized_bin_error_Q5": norm_bins[4],
            })

            np.save(str(seed_dir / f"entropy_map_t{t:03d}.npy"), emap.astype(np.float32))
            np.save(str(seed_dir / f"patch_error_map_t{t:03d}.npy"), pmap.astype(np.float32))
            np.save(str(seed_dir / f"mask_t{t:03d}.npy"), mask_vec)

        (seed_dir / "done.json").write_text(json.dumps(seed_rows))
        rows.extend(seed_rows)

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Random-mask diagnostic (Experiment A)")
    p.add_argument("--config", default="experiments/experiment1/config.yaml")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--dataset", required=True, choices=["brain", "knee"])
    p.add_argument("--acceleration", required=True, type=int, choices=[8, 16])
    p.add_argument("--num_scans", default=50, type=int)
    p.add_argument("--num_seeds", default=5, type=int)
    p.add_argument("--batch_size", default=16, type=int,
                   help="Number of (seed, step) pairs to process in a single GPU forward pass")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--meditok_config", default=None)
    p.add_argument("--meditok_ckpt", default=None)
    p.add_argument("--data_path", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    root = _ROOT
    with open(root / args.config if not pathlib.Path(args.config).is_absolute() else args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    run_dir = pathlib.Path(args.run_dir)
    center_fraction = cfg.get("center_fraction", 0.04)

    ckpt_path = args.checkpoint or root / cfg["checkpoints"][args.dataset][args.acceleration]
    model, ckpt_args = load_singlecoil_model(ckpt_path, device)
    model._ckpt_args = ckpt_args

    mt_cfg  = args.meditok_config or root / cfg["meditok_config"]
    mt_ckpt = args.meditok_ckpt  or root / cfg["meditok_ckpt"]
    tokenizer = load_meditok(mt_cfg, mt_ckpt, device)

    data_root = pathlib.Path(args.data_path or root / cfg["data_paths"][args.dataset])
    split_file = root / cfg["split_files"][args.dataset]

    slice_records = select_slices_fixed_fractions(
        data_root, split_file,
        dataset=args.dataset,
        acceleration=args.acceleration,
        num_scans=args.num_scans,
        fractions=(0.25, 0.5, 0.75),
    )

    print(
        f"Random-mask diagnostic: dataset={args.dataset}, R={args.acceleration}, "
        f"{len(slice_records)} slices ({args.num_scans} scans × 3), "
        f"{args.num_seeds} seeds, batch_size={args.batch_size}"
    )

    all_rows = []
    for global_idx, rec in enumerate(tqdm(slice_records, desc="Slices")):
        h5_path = data_root / rec["scan_id"]
        if not h5_path.exists():
            print(f"  SKIP: {h5_path}")
            continue
        try:
            rows = run_slice(
                model=model, tokenizer=tokenizer,
                slice_rec=rec, h5_path=h5_path,
                dataset=args.dataset, acceleration=args.acceleration,
                center_fraction=center_fraction,
                num_seeds=args.num_seeds,
                run_dir=run_dir, device=device,
                global_slice_idx=global_idx,
                batch_size=args.batch_size,
            )
            all_rows.extend(rows)
        except Exception as e:
            print(f"  ERROR on {rec['scan_id']} slice {rec['slice_index']}: {e}")
            raise

    out_dir = run_dir / "random_mask_diagnostic" / f"dataset={args.dataset}" / f"R={args.acceleration}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_parquet(out_dir / "metadata.parquet", all_rows)
    print(f"Saved {len(all_rows)} rows → {out_dir / 'metadata.parquet'}")


if __name__ == "__main__":
    main()
