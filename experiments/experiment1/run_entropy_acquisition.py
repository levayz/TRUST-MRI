#!/usr/bin/env python
"""Experiment B: Entropy decreases during closed-loop entropy-guided acquisition.

For each (dataset, R, method, scan, slice), runs active acquisition from
step 0 (center mask only) through B steps, saving per-step entropy and
reconstruction metrics normalized by their step-0 values.

Methods:
    LES — Logit-based Entropy Scoring (fast, no backprop)
    GEO — Gradient Entropy Optimization (slower, uses backprop)

Budgets:
    Brain R=8: B=22    Brain R=16: B=6
    Knee  R=8: B=28    Knee  R=16: B=8

Usage:
    python run_entropy_acquisition.py \\
        --config experiments/experiment1/config.yaml \\
        --run_dir results/exp1_entropy \\
        --dataset brain --acceleration 16 --method LES \\
        --num_scans 50
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import collections
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
from experiments.experiment1.utils.metrics_extended import compute_all_metrics
from experiments.experiment1.utils.storage import save_parquet
from experiments.experiment1.utils.slice_selector import (
    make_sample_id, select_slices_fixed_fractions,
    select_all_slices_from_split, select_stratified_center_slices,
)
from experiments.experiment1.run_normal_trajectory import load_slice, make_center_mask
from autoregressive.sample.adaptive_sampling import (
    geo_step as _geo_step,
    gles_step as _gles_step,
    erse_step as _erse_step,
)

from data import transforms
from autoregressive.models.generate import reconstruct_far, data_consistency
from autoregressive.sample.adaptive_sampling import (
    kspace_uncertainty_from_patch_logits,
    posterior_sampling_scores,
    posterior_sampling_scores_kspace,
)
from autoregressive.sample.tds import tds_score_step


FAR_KWARGS = dict(
    n_steps=1,
    far_mask_mode="cartesian",
    active_sampling_mask_type="cartesian",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entropy_map(logits: torch.Tensor) -> np.ndarray:
    """(B, H_p, W_p, 2, K, V_K) → (H_p, W_p) entropy summed over channels+codebooks."""
    log_p = F.log_softmax(logits.float(), dim=-1)
    p = log_p.exp()
    H = -(p * log_p).sum(-1)           # (B, H_p, W_p, 2, K)
    return H[0].sum(dim=(2, 3)).cpu().numpy()  # (H_p, W_p)


def _patch_mse(rec_rss: np.ndarray, gt_rss: np.ndarray, H_p: int, W_p: int) -> np.ndarray:
    """Compute per-patch MSE. Returns (H_p, W_p) array."""
    H, W = rec_rss.shape
    ph, pw = H // H_p, W // W_p
    rec_p = rec_rss.reshape(H_p, ph, W_p, pw)
    gt_p  = gt_rss.reshape(H_p, ph, W_p, pw)
    return ((rec_p - gt_p) ** 2).mean(axis=(1, 3))  # (H_p, W_p)


# ---------------------------------------------------------------------------
# Per-slice acquisition loop
# ---------------------------------------------------------------------------

def run_slice(
    model,
    tokenizer,
    slice_rec: dict,
    h5_path: pathlib.Path,
    dataset: str,
    acceleration: int,
    method: str,
    center_fraction: float,
    run_dir: pathlib.Path,
    device: torch.device,
    compute_ssfd: bool = False,
    ps_n_samples: int = 8,
    ps_top_k: int = 100,
    ps_temperature: float = 1.0,
    ps_top_p: float = 1.0,
    save_ps_samples: bool = False,
    save_per_step_recon: bool = False,
    # TDS hyperparameters
    tds_distance: str = "js",
    tds_foreground_mode: str = "percentile",
    tds_foreground_threshold: float = 0.0,
    tds_foreground_percentile: float = 20.0,
    tds_top_m: int = 64,
    tds_alpha: float = 1.0,
    tds_beta: float = 0.0,
    tds_candidate_mode: str = "all",
    tds_candidate_k: int = 16,
    tds_entropy_normalization: bool = True,
    tds_line_aggregation: str = "sum",
    tds_candidate_batch_size: int = 16,
    # GLES hyperparameters
    gles_m: int = 8,
    gles_use_foreground: bool = False,
    gles_foreground_percentile: float = 20.0,
    # ERSE hyperparameters
    erse_residual_mode: str = "pred_minus_current",
    erse_weight_mode: str = "soft_entropy",
    erse_top_m: int = 32,
    erse_smoothing: bool = True,
    erse_smoothing_kernel: int = 5,
    erse_entropy_normalization: bool = True,
    erse_line_aggregation: str = "mean",
    erse_use_complex_residual: bool = True,
) -> list[dict]:
    sample_id = make_sample_id(dataset, slice_rec["scan_id"], slice_rec["slice_index"])
    slice_dir = (
        run_dir / "entropy_acquisition"
        / f"dataset={dataset}" / f"R={acceleration}"
        / f"method={method}"
        / f"sample_id={sample_id}"
    )

    done_file = slice_dir / "done.json"
    if done_file.exists():
        with open(done_file) as f:
            return json.load(f)

    slice_dir.mkdir(parents=True, exist_ok=True)

    ckpt_args = getattr(model, "_ckpt_args", None)
    resolution = 256
    if ckpt_args is not None and hasattr(ckpt_args, "image_size"):
        resolution = ckpt_args.image_size

    slice_data = load_slice(h5_path, slice_rec["slice_index"], resolution)
    gt_c_image = slice_data["gt_c_image"].to(device)   # (1, H, W, 2)
    max_val    = float(slice_data["max_val"])
    _, H, W, _ = gt_c_image.shape
    gt_rss = transforms.complex_abs(gt_c_image[0]).cpu().numpy()  # (H, W), normalised

    center_mask = make_center_mask(W, center_fraction, device)
    B_budget = max(1, W // acceleration - int(W * center_fraction))

    far_kwargs = {**FAR_KWARGS, "center_fractions": center_fraction}
    rank = 1

    sample_mask = center_mask.clone()
    selected_line = -1

    mean_entropy_0: float | None = None
    mse_0: float | None = None
    nmse_0: float | None = None

    ps_sampling_kwargs = {"temperature": ps_temperature, "top_k": ps_top_k, "top_p": ps_top_p}
    samples_dir = slice_dir / "samples" if save_ps_samples else None

    rows = []
    for t in range(B_budget + 1):  # 0..B inclusive
        masked_image = data_consistency(
            gt_c_image, torch.zeros_like(gt_c_image), sample_mask
        )

        with torch.no_grad():
            rec_img, aux = reconstruct_far(
                model, tokenizer, masked_image, sample_mask,
                sampling_kwargs={"temperature": 1.0, "top_k": 0, "top_p": 1.0},
                **far_kwargs,
            )
        logits = aux["logits"]  # (1, H_p, W_p, 2, K, V_K)
        H_p, W_p = logits.shape[1], logits.shape[2]

        rec_rss = transforms.complex_abs(rec_img[0]).cpu().numpy()
        emap = _entropy_map(logits)
        pmap = _patch_mse(rec_rss, gt_rss, H_p, W_p)

        mean_entropy = float(emap.mean())
        mse_val = float(((rec_rss - gt_rss) ** 2).mean())
        metrics = compute_all_metrics(rec_rss, gt_rss, device, compute_ssfd_flag=compute_ssfd)

        # --- PS: generate posterior samples at every step for avg-reconstruction metrics ---
        ps_avg_metrics: dict = {}
        ps_tensor_complex: torch.Tensor | None = None
        if method.startswith("PS"):
            masked_rep = masked_image.expand(ps_n_samples, -1, -1, -1).contiguous()
            mask_rep   = sample_mask.expand(ps_n_samples, -1, -1, -1).contiguous()
            with torch.no_grad():
                rec_batch, _ = reconstruct_far(
                    model, tokenizer, masked_rep, mask_rep,
                    sampling_kwargs=ps_sampling_kwargs, **far_kwargs
                )  # (S, H, W, 2)
            ps_tensor = transforms.complex_abs(rec_batch)          # (S, H, W)
            if method == "PS_kspace":
                ps_tensor_complex = rec_batch                      # (S, H, W, 2)
            ps_mean_rss = ps_tensor.mean(dim=0).cpu().numpy()      # (H, W)
            ps_avg_metrics = compute_all_metrics(
                ps_mean_rss, gt_rss, device, compute_ssfd_flag=compute_ssfd
            )

            if samples_dir is not None:
                samples_dir.mkdir(parents=True, exist_ok=True)
                for s_idx, s in enumerate(ps_tensor):
                    np.save(
                        str(samples_dir / f"step{t:03d}_sample{s_idx:03d}.npy"),
                        s.cpu().numpy().astype(np.float32),
                    )
                np.save(
                    str(samples_dir / f"step{t:03d}_mean.npy"),
                    ps_mean_rss.astype(np.float32),
                )
                np.save(
                    str(samples_dir / f"step{t:03d}_argmax.npy"),
                    rec_rss.astype(np.float32),
                )

        if t == 0:
            mean_entropy_0 = mean_entropy if mean_entropy > 0 else 1.0
            mse_0 = mse_val if mse_val > 0 else 1.0
            nmse_0 = metrics["nmse"] if metrics["nmse"] > 0 else 1.0

        mask_vec = sample_mask[0, 0, :, 0].cpu().numpy().astype(np.int8)
        n_sampled = int(mask_vec.sum())

        row = {
            "dataset": dataset,
            "R": acceleration,
            "method": method,
            "scan_id": slice_rec["scan_id"],
            "slice_id": slice_rec["slice_index"],
            "volume_fraction": slice_rec.get("volume_fraction", 0.5),
            "step": t,
            "normalized_step": t / B_budget if B_budget > 0 else 0.0,
            "policy": "entropy_as",
            "selected_line": selected_line,
            "n_sampled_lines": n_sampled,
            "sampled_fraction": n_sampled / W,
            "mean_entropy": mean_entropy,
            "normalized_mean_entropy": mean_entropy / mean_entropy_0,
            "mean_patch_error": float(pmap.mean()),
            "MSE": mse_val,
            "normalized_MSE": mse_val / mse_0,
            "NMSE": metrics["nmse"],
            "normalized_NMSE": metrics["nmse"] / nmse_0,
            "PSNR": metrics["psnr"],
            "SSIM": metrics["ssim"],
            "SSFD": metrics.get("ssfd", float("nan")),
            # PS avg-reconstruction metrics (NaN for non-PS methods)
            "PSNR_ps_avg": ps_avg_metrics.get("psnr", float("nan")),
            "SSIM_ps_avg": ps_avg_metrics.get("ssim", float("nan")),
            "NMSE_ps_avg": ps_avg_metrics.get("nmse", float("nan")),
            "SSFD_ps_avg": ps_avg_metrics.get("ssfd", float("nan")),
            # TDS-specific diagnostics (NaN for non-TDS methods)
            "tds_score":    float("nan"),
            "H_j_selected": float("nan"),
            "entropy_drop": float("nan"),
            "n_fg_patches": -1,
            "tds_loop_s":   float("nan"),
        }
        rows.append(row)

        np.save(str(slice_dir / f"entropy_map_t{t:03d}.npy"), emap.astype(np.float32))
        np.save(str(slice_dir / f"patch_error_map_t{t:03d}.npy"), pmap.astype(np.float32))
        np.save(str(slice_dir / f"mask_t{t:03d}.npy"), mask_vec)
        if t == B_budget:
            np.save(str(slice_dir / "recon_final.npy"), (rec_rss * max_val).astype(np.float32))
        if save_per_step_recon:
            np.save(str(slice_dir / f"recon_t{t:03d}.npy"), (rec_rss * max_val).astype(np.float32))

        if t < B_budget:
            old_mask = sample_mask.clone()

            if method == "LES":
                mask_2d = sample_mask.permute(0, 3, 1, 2)  # (1, 2, H, W)
                with torch.no_grad():
                    new_mask_2d, _, freq_map = kspace_uncertainty_from_patch_logits(
                        logits, mask_2d,
                        vertical_mask=True, rank=rank,
                        les_zero_mean=False, les_radial_weight_alpha=0.0,
                        mask_type="cartesian",
                    )
                sample_mask = new_mask_2d.permute(0, 2, 3, 1).float()
                scores = freq_map[0].mean(dim=0).cpu().numpy()

            elif method == "GEO":
                new_mask, geo_scores = _geo_step(
                    model, tokenizer, masked_image, sample_mask, rank, device
                )
                sample_mask = new_mask.detach()
                scores = geo_scores.cpu().numpy()

            elif method == "PS_kspace":
                sel_cols, col_scores = posterior_sampling_scores_kspace(
                    ps_tensor_complex, sample_mask, rank=rank
                )
                new_mask = sample_mask.clone()
                for c in sel_cols:
                    new_mask[0, :, c.item(), :] = 1.0
                sample_mask = new_mask
                scores = col_scores.cpu().numpy()

            elif method.startswith("PS"):
                # ps_tensor already computed above; reuse for line selection
                sel_cols, col_scores = posterior_sampling_scores(ps_tensor, sample_mask, rank=rank)
                new_mask = sample_mask.clone()
                for c in sel_cols:
                    new_mask[0, :, c.item(), :] = 1.0
                sample_mask = new_mask
                scores = col_scores.cpu().numpy()

            elif method == "GLES":
                new_mask, gles_scores = _gles_step(
                    model, tokenizer, masked_image, sample_mask, rank, device,
                    M=gles_m if gles_m > 0 else None,
                    use_foreground=gles_use_foreground,
                    foreground_percentile=gles_foreground_percentile,
                )
                sample_mask = new_mask.detach()
                scores = gles_scores.cpu().numpy()

            elif method == "ERSE":
                new_mask, erse_scores = _erse_step(
                    model, tokenizer, masked_image, sample_mask, rank, device,
                    residual_mode=erse_residual_mode,
                    weight_mode=erse_weight_mode,
                    top_m=erse_top_m if erse_top_m > 0 else None,
                    smoothing=erse_smoothing,
                    smoothing_kernel=erse_smoothing_kernel,
                    entropy_normalization=erse_entropy_normalization,
                    line_aggregation=erse_line_aggregation,
                    use_complex_residual=erse_use_complex_residual,
                )
                sample_mask = new_mask.detach()
                scores = erse_scores.cpu().numpy()

            elif method == "random":
                acquired = sample_mask[:, 0, :, 0].bool()
                unacquired = (~acquired[0]).nonzero(as_tuple=True)[0]
                idx = torch.randint(len(unacquired), (rank,), device=device)
                sel = unacquired[idx]
                new_mask = sample_mask.clone()
                for c in sel:
                    new_mask[:, :, c.item(), :] = 1.0
                sample_mask = new_mask
                scores = np.zeros(W, dtype=np.float32)

            elif method.startswith("TDS"):
                # Decode-once: x_hat is already in aux from the reconstruct_far call above.
                x_hat_no_dc = aux["decoded_no_dc"]   # (1, H, W, 2)
                y_omega_cur  = transforms.fft2(masked_image)
                tds_result = tds_score_step(
                    model, tokenizer,
                    y_omega_cur, sample_mask, x_hat_no_dc, logits,
                    device,
                    sampling_kwargs={"temperature": 1.0, "top_k": 0, "top_p": 1.0},
                    tds_distance=tds_distance,
                    tds_foreground_mode=tds_foreground_mode,
                    tds_foreground_threshold=tds_foreground_threshold,
                    tds_foreground_percentile=tds_foreground_percentile,
                    tds_top_m=tds_top_m,
                    tds_alpha=tds_alpha,
                    tds_beta=tds_beta,
                    tds_candidate_mode=tds_candidate_mode,
                    tds_candidate_k=tds_candidate_k,
                    tds_entropy_normalization=tds_entropy_normalization,
                    tds_line_aggregation=tds_line_aggregation,
                    tds_candidate_batch_size=tds_candidate_batch_size,
                )
                best_col = tds_result["selected_col"]
                new_mask = sample_mask.clone()
                new_mask[:, :, best_col, :] = 1.0
                sample_mask = new_mask
                scores = tds_result["scores_1d"][0].cpu().numpy()
                # Save TDS diagnostics
                tds_diag = tds_result["diagnostics"]
                np.save(
                    str(slice_dir / f"tds_diag_t{t:03d}.npy"),
                    np.array([tds_diag], dtype=object),
                )
                rows[-1]["tds_score"]      = tds_diag["selected_tds_score"]
                rows[-1]["H_j_selected"]   = tds_diag["H_j_selected"]
                rows[-1]["entropy_drop"]   = tds_diag["entropy_drop_selected"]
                rows[-1]["n_fg_patches"]   = tds_diag["n_fg_patches"]
                rows[-1]["tds_loop_s"]     = tds_diag["runtime_loop_s"]

            diff = sample_mask[0, 0, :, 0] - old_mask[0, 0, :, 0]
            sel = diff.nonzero(as_tuple=True)[0]
            selected_line = int(sel[0].item()) if len(sel) > 0 else -1
            np.save(str(slice_dir / f"kspace_scores_t{t:03d}.npy"), scores.astype(np.float32))

    done_file.write_text(json.dumps(rows))
    return rows


# ---------------------------------------------------------------------------
# Batched LES acquisition (multiple slices per GPU call)
# ---------------------------------------------------------------------------

def _run_batch_les(
    model,
    tokenizer,
    slice_recs: list[dict],
    h5_paths: list[pathlib.Path],
    dataset: str,
    acceleration: int,
    center_fraction: float,
    run_dir: pathlib.Path,
    device: torch.device,
    compute_ssfd: bool = False,
) -> list[dict]:
    """Process a batch of slices together through the LES acquisition loop.

    One reconstruct_far call per acquisition step instead of one per (slice, step).
    Returns all rows for the batch.
    """
    n = len(slice_recs)
    ckpt_args = getattr(model, "_ckpt_args", None)
    resolution = 256
    if ckpt_args is not None and hasattr(ckpt_args, "image_size"):
        resolution = ckpt_args.image_size

    gt_images, gt_rsses, gt_max_vals, sample_masks, slice_dirs = [], [], [], [], []
    for rec, h5_path in zip(slice_recs, h5_paths):
        sample_id = make_sample_id(dataset, rec["scan_id"], rec["slice_index"])
        sdir = (
            run_dir / "entropy_acquisition"
            / f"dataset={dataset}" / f"R={acceleration}"
            / f"method=LES" / f"sample_id={sample_id}"
        )
        sdir.mkdir(parents=True, exist_ok=True)
        slice_dirs.append(sdir)

        sd = load_slice(h5_path, rec["slice_index"], resolution)
        gt = sd["gt_c_image"].to(device)            # (1, H, W, 2)
        _, H, W, _ = gt.shape
        gt_images.append(gt)
        gt_rsses.append(transforms.complex_abs(gt[0]).cpu().numpy())
        gt_max_vals.append(float(sd["max_val"]))
        sample_masks.append(make_center_mask(W, center_fraction, device))

    _, H, W, _ = gt_images[0].shape
    B_budget = max(1, W // acceleration - int(W * center_fraction))
    far_kwargs = {**FAR_KWARGS, "center_fractions": center_fraction}
    rank = 1

    gt_batch = torch.cat(gt_images, dim=0)                    # (n, H, W, 2)
    mask_batch = torch.cat(sample_masks, dim=0)               # (n, H, W, 2)

    mean_entropy_0 = [None] * n
    mse_0          = [None] * n
    nmse_0         = [None] * n
    selected_lines = [-1] * n
    slice_rows: list[list[dict]] = [[] for _ in range(n)]

    for t in range(B_budget + 1):
        masked_batch = data_consistency(gt_batch, torch.zeros_like(gt_batch), mask_batch)

        with torch.no_grad():
            rec_imgs, aux = reconstruct_far(
                model, tokenizer, masked_batch, mask_batch, **far_kwargs
            )
        logits_batch = aux["logits"]  # (n, H_p, W_p, 2, K, V_K)
        H_p, W_p = logits_batch.shape[1], logits_batch.shape[2]

        for i, rec in enumerate(slice_recs):
            emap = _entropy_map(logits_batch[i : i + 1])
            rec_rss = transforms.complex_abs(rec_imgs[i]).cpu().numpy()
            pmap = _patch_mse(rec_rss, gt_rsses[i], H_p, W_p)

            mean_entropy = float(emap.mean())
            mse_val = float(((rec_rss - gt_rsses[i]) ** 2).mean())
            metrics = compute_all_metrics(rec_rss, gt_rsses[i], device, compute_ssfd_flag=compute_ssfd)

            if t == 0:
                mean_entropy_0[i] = mean_entropy if mean_entropy > 0 else 1.0
                mse_0[i]          = mse_val      if mse_val      > 0 else 1.0
                nmse_0[i]         = metrics["nmse"] if metrics["nmse"] > 0 else 1.0

            mask_vec = mask_batch[i, 0, :, 0].cpu().numpy().astype(np.int8)
            n_sampled = int(mask_vec.sum())

            slice_rows[i].append({
                "dataset": dataset, "R": acceleration, "method": "LES",
                "scan_id": rec["scan_id"], "slice_id": rec["slice_index"],
                "volume_fraction": rec.get("volume_fraction", 0.5),
                "step": t, "normalized_step": t / B_budget if B_budget > 0 else 0.0,
                "policy": "entropy_as", "selected_line": selected_lines[i],
                "n_sampled_lines": n_sampled, "sampled_fraction": n_sampled / W,
                "mean_entropy": mean_entropy,
                "normalized_mean_entropy": mean_entropy / mean_entropy_0[i],
                "mean_patch_error": float(pmap.mean()),
                "MSE": mse_val, "normalized_MSE": mse_val / mse_0[i],
                "NMSE": metrics["nmse"],
                "normalized_NMSE": metrics["nmse"] / nmse_0[i],
                "PSNR": metrics["psnr"], "SSIM": metrics["ssim"],
                "SSFD": metrics.get("ssfd", float("nan")),
            })

            np.save(str(slice_dirs[i] / f"entropy_map_t{t:03d}.npy"),    emap.astype(np.float32))
            np.save(str(slice_dirs[i] / f"patch_error_map_t{t:03d}.npy"), pmap.astype(np.float32))
            np.save(str(slice_dirs[i] / f"mask_t{t:03d}.npy"),           mask_vec)
            if t == B_budget:
                np.save(str(slice_dirs[i] / "recon_final.npy"), (rec_rss * gt_max_vals[i]).astype(np.float32))

        if t < B_budget:
            old_mask = mask_batch.clone()
            mask_2d = mask_batch.permute(0, 3, 1, 2)  # (n, 2, H, W)
            with torch.no_grad():
                new_mask_2d, _, freq_map = kspace_uncertainty_from_patch_logits(
                    logits_batch, mask_2d,
                    vertical_mask=True, rank=rank,
                    les_zero_mean=False, les_radial_weight_alpha=0.0,
                    mask_type="cartesian",
                )
            mask_batch = new_mask_2d.permute(0, 2, 3, 1).float()

            for i in range(n):
                diff = mask_batch[i, 0, :, 0] - old_mask[i, 0, :, 0]
                sel = diff.nonzero(as_tuple=True)[0]
                selected_lines[i] = int(sel[0].item()) if len(sel) > 0 else -1
                scores = freq_map[i].mean(dim=0).cpu().numpy()
                np.save(str(slice_dirs[i] / f"kspace_scores_t{t:03d}.npy"), scores.astype(np.float32))

    all_rows: list[dict] = []
    for i, sdir in enumerate(slice_dirs):
        (sdir / "done.json").write_text(json.dumps(slice_rows[i]))
        all_rows.extend(slice_rows[i])
    return all_rows


def _run_batch_geo(
    model,
    tokenizer,
    slice_recs: list[dict],
    h5_paths: list[pathlib.Path],
    dataset: str,
    acceleration: int,
    center_fraction: float,
    run_dir: pathlib.Path,
    device: torch.device,
    compute_ssfd: bool = False,
) -> list[dict]:
    """Process a batch of slices together through the GEO acquisition loop.

    geo_step computes a (B, W) gradient-magnitude score and selects a column
    independently per slice, so one forward+backward pass serves all n slices.
    """
    n = len(slice_recs)
    ckpt_args = getattr(model, "_ckpt_args", None)
    resolution = 256
    if ckpt_args is not None and hasattr(ckpt_args, "image_size"):
        resolution = ckpt_args.image_size

    gt_images, gt_rsses, gt_max_vals, sample_masks, slice_dirs = [], [], [], [], []
    for rec, h5_path in zip(slice_recs, h5_paths):
        sample_id = make_sample_id(dataset, rec["scan_id"], rec["slice_index"])
        sdir = (
            run_dir / "entropy_acquisition"
            / f"dataset={dataset}" / f"R={acceleration}"
            / f"method=GEO" / f"sample_id={sample_id}"
        )
        sdir.mkdir(parents=True, exist_ok=True)
        slice_dirs.append(sdir)

        sd = load_slice(h5_path, rec["slice_index"], resolution)
        gt = sd["gt_c_image"].to(device)
        _, H, W, _ = gt.shape
        gt_images.append(gt)
        gt_rsses.append(transforms.complex_abs(gt[0]).cpu().numpy())
        gt_max_vals.append(float(sd["max_val"]))
        sample_masks.append(make_center_mask(W, center_fraction, device))

    _, H, W, _ = gt_images[0].shape
    B_budget = max(1, W // acceleration - int(W * center_fraction))
    far_kwargs = {**FAR_KWARGS, "center_fractions": center_fraction}
    rank = 1

    gt_batch   = torch.cat(gt_images,    dim=0)    # (n, H, W, 2)
    mask_batch = torch.cat(sample_masks, dim=0)    # (n, H, W, 2)

    mean_entropy_0 = [None] * n
    mse_0          = [None] * n
    nmse_0         = [None] * n
    selected_lines = [-1] * n
    slice_rows: list[list[dict]] = [[] for _ in range(n)]

    for t in range(B_budget + 1):
        masked_batch = data_consistency(gt_batch, torch.zeros_like(gt_batch), mask_batch)

        with torch.no_grad():
            rec_imgs, aux = reconstruct_far(
                model, tokenizer, masked_batch, mask_batch, **far_kwargs
            )
        logits_batch = aux["logits"]
        H_p, W_p = logits_batch.shape[1], logits_batch.shape[2]

        for i, rec in enumerate(slice_recs):
            emap = _entropy_map(logits_batch[i : i + 1])
            rec_rss = transforms.complex_abs(rec_imgs[i]).cpu().numpy()
            pmap = _patch_mse(rec_rss, gt_rsses[i], H_p, W_p)

            mean_entropy = float(emap.mean())
            mse_val = float(((rec_rss - gt_rsses[i]) ** 2).mean())
            metrics = compute_all_metrics(rec_rss, gt_rsses[i], device, compute_ssfd_flag=compute_ssfd)

            if t == 0:
                mean_entropy_0[i] = mean_entropy if mean_entropy > 0 else 1.0
                mse_0[i]          = mse_val      if mse_val      > 0 else 1.0
                nmse_0[i]         = metrics["nmse"] if metrics["nmse"] > 0 else 1.0

            mask_vec  = mask_batch[i, 0, :, 0].cpu().numpy().astype(np.int8)
            n_sampled = int(mask_vec.sum())

            slice_rows[i].append({
                "dataset": dataset, "R": acceleration, "method": "GEO",
                "scan_id": rec["scan_id"], "slice_id": rec["slice_index"],
                "volume_fraction": rec.get("volume_fraction", 0.5),
                "step": t, "normalized_step": t / B_budget if B_budget > 0 else 0.0,
                "policy": "entropy_as", "selected_line": selected_lines[i],
                "n_sampled_lines": n_sampled, "sampled_fraction": n_sampled / W,
                "mean_entropy": mean_entropy,
                "normalized_mean_entropy": mean_entropy / mean_entropy_0[i],
                "mean_patch_error": float(pmap.mean()),
                "MSE": mse_val, "normalized_MSE": mse_val / mse_0[i],
                "NMSE": metrics["nmse"],
                "normalized_NMSE": metrics["nmse"] / nmse_0[i],
                "PSNR": metrics["psnr"], "SSIM": metrics["ssim"],
                "SSFD": metrics.get("ssfd", float("nan")),
                "PSNR_ps_avg": float("nan"), "SSIM_ps_avg": float("nan"),
                "NMSE_ps_avg": float("nan"), "SSFD_ps_avg": float("nan"),
                "tds_score": float("nan"), "H_j_selected": float("nan"),
                "entropy_drop": float("nan"), "n_fg_patches": -1,
                "tds_loop_s": float("nan"),
            })

            np.save(str(slice_dirs[i] / f"entropy_map_t{t:03d}.npy"),     emap.astype(np.float32))
            np.save(str(slice_dirs[i] / f"patch_error_map_t{t:03d}.npy"), pmap.astype(np.float32))
            np.save(str(slice_dirs[i] / f"mask_t{t:03d}.npy"),            mask_vec)
            if t == B_budget:
                np.save(str(slice_dirs[i] / "recon_final.npy"),
                        (rec_rss * gt_max_vals[i]).astype(np.float32))

        if t < B_budget:
            old_mask = mask_batch.clone()
            new_mask_batch, geo_scores = _geo_step(
                model, tokenizer, masked_batch, mask_batch, rank, device
            )
            mask_batch = new_mask_batch.detach()

            for i in range(n):
                diff = mask_batch[i, 0, :, 0] - old_mask[i, 0, :, 0]
                sel  = diff.nonzero(as_tuple=True)[0]
                selected_lines[i] = int(sel[0].item()) if len(sel) > 0 else -1

    all_rows: list[dict] = []
    for i, sdir in enumerate(slice_dirs):
        (sdir / "done.json").write_text(json.dumps(slice_rows[i]))
        all_rows.extend(slice_rows[i])
    return all_rows


def _run_batch_gles(
    model,
    tokenizer,
    slice_recs: list[dict],
    h5_paths: list[pathlib.Path],
    dataset: str,
    acceleration: int,
    center_fraction: float,
    run_dir: pathlib.Path,
    device: torch.device,
    compute_ssfd: bool = False,
    gles_m: int = 8,
    gles_use_foreground: bool = False,
    gles_foreground_percentile: float = 20.0,
) -> list[dict]:
    """Process a batch of slices together through the GLES acquisition loop.

    gles_step already supports B>1: it computes a (B, W) line_score and selects
    a column independently per slice. One reconstruct_far + one set of M backward
    passes serves all n slices in the batch simultaneously.
    """
    n = len(slice_recs)
    ckpt_args = getattr(model, "_ckpt_args", None)
    resolution = 256
    if ckpt_args is not None and hasattr(ckpt_args, "image_size"):
        resolution = ckpt_args.image_size

    gt_images, gt_rsses, gt_max_vals, sample_masks, slice_dirs = [], [], [], [], []
    for rec, h5_path in zip(slice_recs, h5_paths):
        sample_id = make_sample_id(dataset, rec["scan_id"], rec["slice_index"])
        sdir = (
            run_dir / "entropy_acquisition"
            / f"dataset={dataset}" / f"R={acceleration}"
            / f"method=GLES" / f"sample_id={sample_id}"
        )
        sdir.mkdir(parents=True, exist_ok=True)
        slice_dirs.append(sdir)

        sd = load_slice(h5_path, rec["slice_index"], resolution)
        gt = sd["gt_c_image"].to(device)            # (1, H, W, 2)
        _, H, W, _ = gt.shape
        gt_images.append(gt)
        gt_rsses.append(transforms.complex_abs(gt[0]).cpu().numpy())
        gt_max_vals.append(float(sd["max_val"]))
        sample_masks.append(make_center_mask(W, center_fraction, device))

    _, H, W, _ = gt_images[0].shape
    B_budget = max(1, W // acceleration - int(W * center_fraction))
    far_kwargs = {**FAR_KWARGS, "center_fractions": center_fraction}
    rank = 1

    gt_batch   = torch.cat(gt_images,   dim=0)     # (n, H, W, 2)
    mask_batch = torch.cat(sample_masks, dim=0)    # (n, H, W, 2)

    mean_entropy_0 = [None] * n
    mse_0          = [None] * n
    nmse_0         = [None] * n
    selected_lines = [-1] * n
    slice_rows: list[list[dict]] = [[] for _ in range(n)]

    for t in range(B_budget + 1):
        masked_batch = data_consistency(gt_batch, torch.zeros_like(gt_batch), mask_batch)

        with torch.no_grad():
            rec_imgs, aux = reconstruct_far(
                model, tokenizer, masked_batch, mask_batch, **far_kwargs
            )
        logits_batch = aux["logits"]   # (n, H_p, W_p, 2, K, V_K)
        H_p, W_p = logits_batch.shape[1], logits_batch.shape[2]

        for i, rec in enumerate(slice_recs):
            emap = _entropy_map(logits_batch[i : i + 1])
            rec_rss = transforms.complex_abs(rec_imgs[i]).cpu().numpy()
            pmap = _patch_mse(rec_rss, gt_rsses[i], H_p, W_p)

            mean_entropy = float(emap.mean())
            mse_val = float(((rec_rss - gt_rsses[i]) ** 2).mean())
            metrics = compute_all_metrics(rec_rss, gt_rsses[i], device, compute_ssfd_flag=compute_ssfd)

            if t == 0:
                mean_entropy_0[i] = mean_entropy if mean_entropy > 0 else 1.0
                mse_0[i]          = mse_val      if mse_val      > 0 else 1.0
                nmse_0[i]         = metrics["nmse"] if metrics["nmse"] > 0 else 1.0

            mask_vec  = mask_batch[i, 0, :, 0].cpu().numpy().astype(np.int8)
            n_sampled = int(mask_vec.sum())

            slice_rows[i].append({
                "dataset": dataset, "R": acceleration, "method": "GLES",
                "scan_id": rec["scan_id"], "slice_id": rec["slice_index"],
                "volume_fraction": rec.get("volume_fraction", 0.5),
                "step": t, "normalized_step": t / B_budget if B_budget > 0 else 0.0,
                "policy": "entropy_as", "selected_line": selected_lines[i],
                "n_sampled_lines": n_sampled, "sampled_fraction": n_sampled / W,
                "mean_entropy": mean_entropy,
                "normalized_mean_entropy": mean_entropy / mean_entropy_0[i],
                "mean_patch_error": float(pmap.mean()),
                "MSE": mse_val, "normalized_MSE": mse_val / mse_0[i],
                "NMSE": metrics["nmse"],
                "normalized_NMSE": metrics["nmse"] / nmse_0[i],
                "PSNR": metrics["psnr"], "SSIM": metrics["ssim"],
                "SSFD": metrics.get("ssfd", float("nan")),
                # TDS-specific fields — not applicable here
                "PSNR_ps_avg": float("nan"), "SSIM_ps_avg": float("nan"),
                "NMSE_ps_avg": float("nan"), "SSFD_ps_avg": float("nan"),
                "tds_score": float("nan"), "H_j_selected": float("nan"),
                "entropy_drop": float("nan"), "n_fg_patches": -1,
                "tds_loop_s": float("nan"),
            })

            np.save(str(slice_dirs[i] / f"entropy_map_t{t:03d}.npy"),     emap.astype(np.float32))
            np.save(str(slice_dirs[i] / f"patch_error_map_t{t:03d}.npy"), pmap.astype(np.float32))
            np.save(str(slice_dirs[i] / f"mask_t{t:03d}.npy"),            mask_vec)
            if t == B_budget:
                np.save(str(slice_dirs[i] / "recon_final.npy"),
                        (rec_rss * gt_max_vals[i]).astype(np.float32))

        if t < B_budget:
            old_mask = mask_batch.clone()
            new_mask_batch, _ = _gles_step(
                model, tokenizer, masked_batch, mask_batch, rank, device,
                M=gles_m if gles_m > 0 else None,
                use_foreground=gles_use_foreground,
                foreground_percentile=gles_foreground_percentile,
            )
            mask_batch = new_mask_batch.detach()

            for i in range(n):
                diff = mask_batch[i, 0, :, 0] - old_mask[i, 0, :, 0]
                sel  = diff.nonzero(as_tuple=True)[0]
                selected_lines[i] = int(sel[0].item()) if len(sel) > 0 else -1

    all_rows: list[dict] = []
    for i, sdir in enumerate(slice_dirs):
        (sdir / "done.json").write_text(json.dumps(slice_rows[i]))
        all_rows.extend(slice_rows[i])
    return all_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Entropy-guided acquisition dynamics (Experiment B)")
    p.add_argument("--config", default="experiments/experiment1/config.yaml")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--dataset", required=True, choices=["brain", "knee"])
    p.add_argument("--acceleration", required=True, type=int, choices=[8, 16])
    p.add_argument("--method", required=True, choices=["LES", "GEO", "PS", "PS_kspace", "TDS", "GLES", "ERSE", "random"])
    p.add_argument("--ps_n_samples", default=8, type=int,
                   help="Number of posterior samples per step (PS only)")
    p.add_argument("--ps_top_k", default=100, type=int,
                   help="top-k for stochastic posterior sampling (PS only)")
    p.add_argument("--ps_temperature", default=1.0, type=float,
                   help="Sampling temperature for PS (PS only)")
    p.add_argument("--ps_top_p", default=1.0, type=float,
                   help="Nucleus sampling top-p for PS (PS only)")
    p.add_argument("--ps_method_tag", default="", type=str,
                   help="Override method tag in output dirs/parquet (e.g. PS_T2_pure); default is 'PS'")
    # TDS hyperparameters
    p.add_argument("--tds_distance", default="js",
                   choices=["js", "symmetric_kl", "topk_l1"],
                   help="Token distribution distance metric (TDS only)")
    p.add_argument("--tds_foreground_mode", default="percentile",
                   choices=["none", "threshold", "percentile", "top_m_entropy_foreground"],
                   help="Foreground masking mode (TDS only)")
    p.add_argument("--tds_foreground_threshold", default=0.0, type=float,
                   help="Magnitude threshold for foreground=threshold mode")
    p.add_argument("--tds_foreground_percentile", default=20.0, type=float,
                   help="Percentile cutoff for foreground=percentile mode (default 20)")
    p.add_argument("--tds_top_m", default=64, type=int,
                   help="Top-M entropy patches to keep in top_m_entropy_foreground mode")
    p.add_argument("--tds_alpha", default=1.0, type=float,
                   help="Weight on TDS score (default 1.0)")
    p.add_argument("--tds_beta", default=0.0, type=float,
                   help="Weight on entropy-reduction term; 0 = plain TDS (default 0.0)")
    p.add_argument("--tds_candidate_mode", default="all",
                   choices=["all", "geo_topk"],
                   help="Candidate shortlist mode (TDS only)")
    p.add_argument("--tds_candidate_k", default=16, type=int,
                   help="Shortlist size for geo_topk mode")
    p.add_argument("--tds_entropy_normalization", default=True, type=lambda x: x.lower() != "false",
                   help="Normalize patch entropy by C*log(V_K) (default True)")
    p.add_argument("--tds_line_aggregation", default="sum",
                   choices=["sum", "mean"],
                   help="Aggregate patch scores by sum or mean (default sum)")
    p.add_argument("--tds_candidate_batch_size", default=16, type=int,
                   help="Candidates processed per model forward pass (TDS only, default 16)")
    # GLES hyperparameters
    p.add_argument("--gles_m", default=8, type=int,
                   help="Top-M high-entropy patches for GLES; 0 = all patches (default: 8)")
    p.add_argument("--gles_use_foreground", default=False,
                   type=lambda x: x.lower() != "false",
                   help="Restrict GLES patch selection to foreground patches (default: False)")
    p.add_argument("--gles_foreground_percentile", default=20.0, type=float,
                   help="Exclude bottom-p%% of patches by magnitude when --gles_use_foreground (default: 20)")
    # ERSE hyperparameters
    p.add_argument("--erse_residual_mode", default="pred_minus_current",
                   choices=["pred_minus_current", "pred", "current"],
                   help="Residual mode for ERSE (default: pred_minus_current)")
    p.add_argument("--erse_weight_mode", default="soft_entropy",
                   choices=["soft_entropy", "top_m_entropy", "foreground_soft_entropy"],
                   help="Entropy weight map mode for ERSE (default: soft_entropy)")
    p.add_argument("--erse_top_m", default=32, type=int,
                   help="Top-M patches for top_m_entropy mode; 0 = all (default: 32)")
    p.add_argument("--erse_smoothing", default=True,
                   type=lambda x: x.lower() != "false",
                   help="Apply box-filter smoothing to entropy weight map (default: True)")
    p.add_argument("--erse_smoothing_kernel", default=5, type=int,
                   help="Smoothing kernel size for ERSE (default: 5)")
    p.add_argument("--erse_entropy_normalization", default=True,
                   type=lambda x: x.lower() != "false",
                   help="Normalize patch entropy by log(V_K) (default: True)")
    p.add_argument("--erse_line_aggregation", default="mean",
                   choices=["mean", "sum"],
                   help="Column energy aggregation for ERSE (default: mean)")
    p.add_argument("--erse_use_complex_residual", default=True,
                   type=lambda x: x.lower() != "false",
                   help="Use complex residual in ERSE; False = magnitude only (default: True)")
    p.add_argument("--num_scans", default=50, type=int)
    p.add_argument("--batch_size", default=8, type=int,
                   help="Slices processed per GPU forward pass (LES only; GEO is always serial)")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--meditok_config", default=None)
    p.add_argument("--meditok_ckpt", default=None)
    p.add_argument("--data_path", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Split-file mode (replaces fixed-fractions default)
    p.add_argument("--split_file", default=None,
                   help="Text file of H5 filenames; if given, overrides config split file")
    p.add_argument("--stratified", action="store_true",
                   help="Select scans stratified by contrast with center slices (requires --split_file)")
    p.add_argument("--scans_per_contrast", default=8, type=int)
    p.add_argument("--center_slices", default=15, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--compute_ssfd", action="store_true",
                   help="Compute SSFD metric at each step (adds ~1-2s/step overhead)")
    p.add_argument("--save_ps_samples", action="store_true",
                   help="Save posterior sample images to <slice_dir>/samples/ for debugging (PS only)")
    p.add_argument("--save_per_step_recon", action="store_true",
                   help="Save recon_t{t:03d}.npy at every acquisition step (for per-step visualization)")
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

    if args.split_file is not None:
        sf = pathlib.Path(args.split_file)
        if not sf.is_absolute():
            sf = root / sf
        if args.stratified:
            slice_records = select_stratified_center_slices(
                data_root, sf,
                dataset=args.dataset,
                acceleration=args.acceleration,
                scans_per_contrast=args.scans_per_contrast,
                n_center_slices=args.center_slices,
                seed=args.seed,
            )
        else:
            slice_records = select_all_slices_from_split(
                data_root, sf,
                dataset=args.dataset,
                acceleration=args.acceleration,
                num_scans=args.num_scans,
            )
    else:
        split_file = root / cfg["split_files"][args.dataset]
        slice_records = select_slices_fixed_fractions(
            data_root, split_file,
            dataset=args.dataset,
            acceleration=args.acceleration,
            num_scans=args.num_scans,
            fractions=(0.25, 0.5, 0.75),
        )

    # Filter missing files once
    valid_records = []
    for rec in slice_records:
        h5_path = data_root / rec["scan_id"]
        if h5_path.exists():
            valid_records.append(rec)
        else:
            print(f"  SKIP (missing): {h5_path}")

    # Effective method tag (used for output dirs and parquet 'method' column)
    effective_method = args.ps_method_tag if (args.method == "PS" and args.ps_method_tag) else args.method

    mode = f"batch_size={args.batch_size}" if args.method in ("LES", "GEO", "GLES") else "serial"
    ps_info = (
        f", n_samples={args.ps_n_samples}, top_k={args.ps_top_k}, "
        f"T={args.ps_temperature}, top_p={args.ps_top_p}"
        if args.method == "PS" else ""
    )
    tds_info = (
        f", dist={args.tds_distance}, fg={args.tds_foreground_mode}@p{args.tds_foreground_percentile}"
        f", alpha={args.tds_alpha}, beta={args.tds_beta}, cand={args.tds_candidate_mode}"
        f", batch={args.tds_candidate_batch_size}"
        if args.method == "TDS" else ""
    )
    print(
        f"Entropy acquisition [{effective_method}]: dataset={args.dataset}, R={args.acceleration}, "
        f"{len(valid_records)} slices ({mode}{ps_info}{tds_info})"
        + (", +SSFD" if args.compute_ssfd else "")
    )

    all_rows: list[dict] = []

    if args.method in ("LES", "GEO", "GLES"):
        # Resume: separate done vs pending slices
        pending_recs, pending_h5s = [], []
        for rec in valid_records:
            sample_id = make_sample_id(args.dataset, rec["scan_id"], rec["slice_index"])
            done_file = (
                run_dir / "entropy_acquisition"
                / f"dataset={args.dataset}" / f"R={args.acceleration}"
                / f"method={effective_method}" / f"sample_id={sample_id}" / "done.json"
            )
            if done_file.exists():
                with open(done_file) as f:
                    all_rows.extend(json.load(f))
            else:
                pending_recs.append(rec)
                pending_h5s.append(data_root / rec["scan_id"])

        for i in tqdm(range(0, len(pending_recs), args.batch_size), desc="Batches"):
            batch_recs = pending_recs[i : i + args.batch_size]
            batch_h5s  = pending_h5s [i : i + args.batch_size]
            try:
                if args.method == "LES":
                    rows = _run_batch_les(
                        model=model, tokenizer=tokenizer,
                        slice_recs=batch_recs, h5_paths=batch_h5s,
                        dataset=args.dataset, acceleration=args.acceleration,
                        center_fraction=center_fraction,
                        run_dir=run_dir, device=device,
                        compute_ssfd=args.compute_ssfd,
                    )
                elif args.method == "GEO":
                    rows = _run_batch_geo(
                        model=model, tokenizer=tokenizer,
                        slice_recs=batch_recs, h5_paths=batch_h5s,
                        dataset=args.dataset, acceleration=args.acceleration,
                        center_fraction=center_fraction,
                        run_dir=run_dir, device=device,
                        compute_ssfd=args.compute_ssfd,
                    )
                else:  # GLES
                    rows = _run_batch_gles(
                        model=model, tokenizer=tokenizer,
                        slice_recs=batch_recs, h5_paths=batch_h5s,
                        dataset=args.dataset, acceleration=args.acceleration,
                        center_fraction=center_fraction,
                        run_dir=run_dir, device=device,
                        compute_ssfd=args.compute_ssfd,
                        gles_m=args.gles_m,
                        gles_use_foreground=args.gles_use_foreground,
                        gles_foreground_percentile=args.gles_foreground_percentile,
                    )
                all_rows.extend(rows)
            except Exception as e:
                print(f"  ERROR in batch starting at index {i}: {e}")
                raise

    else:  # PS / TDS / random — serial
        for rec in tqdm(valid_records, desc="Slices"):
            h5_path = data_root / rec["scan_id"]
            try:
                rows = run_slice(
                    model=model, tokenizer=tokenizer,
                    slice_rec=rec, h5_path=h5_path,
                    dataset=args.dataset, acceleration=args.acceleration,
                    method=effective_method,
                    center_fraction=center_fraction,
                    run_dir=run_dir, device=device,
                    compute_ssfd=args.compute_ssfd,
                    ps_n_samples=args.ps_n_samples,
                    ps_top_k=args.ps_top_k,
                    ps_temperature=args.ps_temperature,
                    ps_top_p=args.ps_top_p,
                    save_ps_samples=args.save_ps_samples,
                    save_per_step_recon=args.save_per_step_recon,
                    # TDS hyperparameters
                    tds_distance=args.tds_distance,
                    tds_foreground_mode=args.tds_foreground_mode,
                    tds_foreground_threshold=args.tds_foreground_threshold,
                    tds_foreground_percentile=args.tds_foreground_percentile,
                    tds_top_m=args.tds_top_m,
                    tds_alpha=args.tds_alpha,
                    tds_beta=args.tds_beta,
                    tds_candidate_mode=args.tds_candidate_mode,
                    tds_candidate_k=args.tds_candidate_k,
                    tds_entropy_normalization=args.tds_entropy_normalization,
                    tds_line_aggregation=args.tds_line_aggregation,
                    tds_candidate_batch_size=args.tds_candidate_batch_size,
                    gles_m=args.gles_m,
                    gles_use_foreground=args.gles_use_foreground,
                    gles_foreground_percentile=args.gles_foreground_percentile,
                    # ERSE hyperparameters
                    erse_residual_mode=args.erse_residual_mode,
                    erse_weight_mode=args.erse_weight_mode,
                    erse_top_m=args.erse_top_m,
                    erse_smoothing=args.erse_smoothing,
                    erse_smoothing_kernel=args.erse_smoothing_kernel,
                    erse_entropy_normalization=args.erse_entropy_normalization,
                    erse_line_aggregation=args.erse_line_aggregation,
                    erse_use_complex_residual=args.erse_use_complex_residual,
                )
                all_rows.extend(rows)
            except Exception as e:
                print(f"  ERROR on {rec['scan_id']} slice {rec['slice_index']}: {e}")
                raise

    out_dir = (
        run_dir / "entropy_acquisition"
        / f"dataset={args.dataset}" / f"R={args.acceleration}"
        / f"method={effective_method}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    save_parquet(out_dir / "metadata.parquet", all_rows)
    print(f"Saved {len(all_rows)} rows → {out_dir / 'metadata.parquet'}")

    # ── Assemble per-scan reconstruction H5 files ─────────────────────────────
    # Format: recon/<scan_id>.h5  with dataset "reconstruction" (N_slices, H, W) float32
    scans: dict[str, list[tuple[int, pathlib.Path]]] = collections.defaultdict(list)
    for rec in valid_records:
        sample_id = make_sample_id(args.dataset, rec["scan_id"], rec["slice_index"])
        npy = (
            out_dir / f"sample_id={sample_id}" / "recon_final.npy"
        )
        if npy.exists():
            scans[rec["scan_id"]].append((rec["slice_index"], npy))

    if scans:
        recon_dir = out_dir / "recon"
        recon_dir.mkdir(exist_ok=True)
        for scan_id, slices in scans.items():
            slices.sort(key=lambda x: x[0])
            volume = np.stack([np.load(str(p)) for _, p in slices], axis=0)  # (S, H, W)
            with h5py.File(str(recon_dir / scan_id), "w") as hf:
                hf.create_dataset("reconstruction", data=volume.astype(np.float32))
        print(f"Saved {len(scans)} scan recons → {recon_dir}")


if __name__ == "__main__":
    main()
