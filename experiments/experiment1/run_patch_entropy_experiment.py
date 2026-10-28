#!/usr/bin/env python
"""Patch-level entropy investigation experiment.

Runs LES or GEO acquisition and saves per-patch entropy data at every step,
including the unsummed per-codebook/per-channel entropy tensor, for diagnosing
why global entropy may not decrease during acquisition.

Saved per slice:
    gt_rss.npy                         (H, W)           GT RSS image
    entropy_map_t{t:03d}.npy           (H_p, W_p)       summed entropy
    entropy_per_codebook_t{t:03d}.npy  (H_p, W_p, 2, K) per-channel per-codebook
    patch_error_map_t{t:03d}.npy       (H_p, W_p)       per-patch MSE
    kspace_scores_t{t:03d}.npy         (W,)             line scores (steps 0..B-1)
    mask_t{t:03d}.npy                  (W,)             k-space mask
    done.json                          list[dict]        metadata per step

Output path:
    <run_dir>/patch_entropy/dataset=<d>/R=<r>/method=<m>/sample_id=<id>/

Usage:
    python experiments/experiment1/run_patch_entropy_experiment.py \\
        --config experiments/experiment1/config.yaml \\
        --run_dir results/patch_entropy \\
        --dataset brain --acceleration 16 --method LES \\
        --split_file fastmri/splits/suno_brain_test.txt \\
        --center_slices 15 --batch_size 16
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

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
    make_sample_id, select_all_slices_from_split, select_stratified_center_slices,
)
from experiments.experiment1.run_normal_trajectory import load_slice, make_center_mask, _geo_step

from data import transforms
from autoregressive.models.generate import reconstruct_far, data_consistency
from autoregressive.sample.adaptive_sampling import kspace_uncertainty_from_patch_logits


FAR_KWARGS = dict(
    n_steps=1,
    far_mask_mode="cartesian",
    active_sampling_mask_type="cartesian",
)


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def _compute_entropy(logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return (entropy_map, entropy_per_codebook) from logits.

    Args:
        logits: (B, H_p, W_p, 2, K, V_K)

    Returns:
        entropy_map:         (H_p, W_p)       — summed over 2 channels × K codebooks
        entropy_per_codebook:(H_p, W_p, 2, K) — raw Shannon entropy per channel/codebook
    """
    log_p = F.log_softmax(logits.float(), dim=-1)
    p = log_p.exp()
    H_full = -(p * log_p).sum(-1)           # (B, H_p, W_p, 2, K)
    entropy_map = H_full[0].sum(dim=(2, 3)).cpu().numpy()   # (H_p, W_p)
    entropy_per_cb = H_full[0].cpu().numpy()                # (H_p, W_p, 2, K)
    return entropy_map.astype(np.float32), entropy_per_cb.astype(np.float32)


def _compute_kl_from_ref(logits: torch.Tensor, log_p_ref: torch.Tensor) -> np.ndarray:
    """KL(p_t || p_t0) per patch per channel per codebook.

    Args:
        logits:    (B, H_p, W_p, 2, K, V_K) — current step logits
        log_p_ref: (H_p, W_p, 2, K, V_K)    — log-probs at step 0

    Returns:
        kl_per_cb: (H_p, W_p, 2, K)
    """
    log_p_t = F.log_softmax(logits.float(), dim=-1)   # (B, H_p, W_p, 2, K, V_K)
    p_t = log_p_t.exp()
    kl = (p_t * (log_p_t - log_p_ref.unsqueeze(0))).sum(-1)  # (B, H_p, W_p, 2, K)
    return kl[0].cpu().numpy().astype(np.float32)


def _patch_mse(rec_rss: np.ndarray, gt_rss: np.ndarray, H_p: int, W_p: int) -> np.ndarray:
    H, W = rec_rss.shape
    ph, pw = H // H_p, W // W_p
    rec_p = rec_rss.reshape(H_p, ph, W_p, pw)
    gt_p  = gt_rss.reshape(H_p, ph, W_p, pw)
    return ((rec_p - gt_p) ** 2).mean(axis=(1, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-slice loop — GEO (serial)
# ---------------------------------------------------------------------------

def run_slice(
    model, tokenizer,
    slice_rec: dict,
    h5_path: pathlib.Path,
    dataset: str,
    acceleration: int,
    method: str,
    center_fraction: float,
    run_dir: pathlib.Path,
    device: torch.device,
    les_zero_mean: bool = False,
) -> list[dict]:
    sample_id = make_sample_id(dataset, slice_rec["scan_id"], slice_rec["slice_index"])
    slice_dir = (
        run_dir / "patch_entropy"
        / f"dataset={dataset}" / f"R={acceleration}"
        / f"method={method}" / f"sample_id={sample_id}"
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
    gt_c_image = slice_data["gt_c_image"].to(device)
    _, H, W, _ = gt_c_image.shape
    gt_rss = transforms.complex_abs(gt_c_image[0]).cpu().numpy()

    np.save(str(slice_dir / "gt_rss.npy"), gt_rss.astype(np.float32))

    center_mask = make_center_mask(W, center_fraction, device)
    B_budget = max(1, W // acceleration - int(W * center_fraction))
    far_kwargs = {**FAR_KWARGS, "center_fractions": center_fraction}

    sample_mask = center_mask.clone()
    selected_line = -1
    rows = []
    log_p_t0 = None  # stored at step 0 for KL computation

    for t in range(B_budget + 1):
        masked_image = data_consistency(gt_c_image, torch.zeros_like(gt_c_image), sample_mask)

        with torch.no_grad():
            rec_img, aux = reconstruct_far(model, tokenizer, masked_image, sample_mask, **far_kwargs)

        logits = aux["logits"]
        H_p, W_p = logits.shape[1], logits.shape[2]

        rec_rss = transforms.complex_abs(rec_img[0]).cpu().numpy()
        emap, emap_cb = _compute_entropy(logits)
        pmap = _patch_mse(rec_rss, gt_rss, H_p, W_p)
        metrics = compute_all_metrics(rec_rss, gt_rss, device, compute_ssfd_flag=False)

        mask_vec = sample_mask[0, 0, :, 0].cpu().numpy().astype(np.int8)

        np.save(str(slice_dir / f"entropy_map_t{t:03d}.npy"),            emap)
        np.save(str(slice_dir / f"entropy_per_codebook_t{t:03d}.npy"),   emap_cb)
        np.save(str(slice_dir / f"patch_error_map_t{t:03d}.npy"),        pmap)
        np.save(str(slice_dir / f"mask_t{t:03d}.npy"),                   mask_vec)

        if t == 0:
            log_p_t0 = F.log_softmax(logits.float(), dim=-1)[0]  # (H_p, W_p, 2, K, V_K)
        else:
            kl_cb = _compute_kl_from_ref(logits, log_p_t0)
            np.save(str(slice_dir / f"kl_from_t0_t{t:03d}.npy"), kl_cb)

        rows.append({
            "dataset": dataset, "R": acceleration, "method": method,
            "scan_id": slice_rec["scan_id"], "slice_id": slice_rec["slice_index"],
            "step": t, "selected_line": selected_line,
            "mean_entropy": float(emap.mean()),
            "PSNR": metrics["psnr"], "SSIM": metrics["ssim"],
        })

        if t < B_budget:
            old_mask = sample_mask.clone()
            if method in ("LES", "LES_ZM"):
                mask_2d = sample_mask.permute(0, 3, 1, 2)
                with torch.no_grad():
                    new_mask_2d, _, freq_map = kspace_uncertainty_from_patch_logits(
                        logits, mask_2d,
                        vertical_mask=True, rank=1,
                        les_zero_mean=les_zero_mean, les_radial_weight_alpha=0.0,
                        mask_type="cartesian",
                    )
                sample_mask = new_mask_2d.permute(0, 2, 3, 1).float()
                scores = freq_map[0].mean(dim=0).cpu().numpy()
            elif method == "GEO":
                new_mask, geo_scores = _geo_step(
                    model, tokenizer, masked_image, sample_mask, 1, device
                )
                sample_mask = new_mask.detach()
                scores = geo_scores.cpu().numpy()

            diff = sample_mask[0, 0, :, 0] - old_mask[0, 0, :, 0]
            sel = diff.nonzero(as_tuple=True)[0]
            selected_line = int(sel[0].item()) if len(sel) > 0 else -1
            np.save(str(slice_dir / f"kspace_scores_t{t:03d}.npy"), scores.astype(np.float32))

    done_file.write_text(json.dumps(rows))
    return rows


# ---------------------------------------------------------------------------
# Batched LES loop
# ---------------------------------------------------------------------------

def _run_batch_les(
    model, tokenizer,
    slice_recs: list[dict],
    h5_paths: list[pathlib.Path],
    dataset: str,
    acceleration: int,
    center_fraction: float,
    run_dir: pathlib.Path,
    device: torch.device,
    method: str = "LES",
    les_zero_mean: bool = False,
) -> list[dict]:
    n = len(slice_recs)
    ckpt_args = getattr(model, "_ckpt_args", None)
    resolution = 256
    if ckpt_args is not None and hasattr(ckpt_args, "image_size"):
        resolution = ckpt_args.image_size

    gt_images, gt_rsses, sample_masks, slice_dirs = [], [], [], []
    for rec, h5_path in zip(slice_recs, h5_paths):
        sample_id = make_sample_id(dataset, rec["scan_id"], rec["slice_index"])
        sdir = (
            run_dir / "patch_entropy"
            / f"dataset={dataset}" / f"R={acceleration}"
            / f"method={method}" / f"sample_id={sample_id}"
        )
        sdir.mkdir(parents=True, exist_ok=True)
        slice_dirs.append(sdir)

        sd = load_slice(h5_path, rec["slice_index"], resolution)
        gt = sd["gt_c_image"].to(device)
        _, H, W, _ = gt.shape
        gt_images.append(gt)
        gt_rss_i = transforms.complex_abs(gt[0]).cpu().numpy()
        gt_rsses.append(gt_rss_i)
        np.save(str(sdir / "gt_rss.npy"), gt_rss_i.astype(np.float32))
        sample_masks.append(make_center_mask(W, center_fraction, device))

    _, H, W, _ = gt_images[0].shape
    B_budget = max(1, W // acceleration - int(W * center_fraction))
    far_kwargs = {**FAR_KWARGS, "center_fractions": center_fraction}

    gt_batch   = torch.cat(gt_images, dim=0)
    mask_batch = torch.cat(sample_masks, dim=0)

    selected_lines = [-1] * n
    slice_rows: list[list[dict]] = [[] for _ in range(n)]
    log_p_t0_list = [None] * n  # per-slice reference log-probs at step 0

    for t in range(B_budget + 1):
        masked_batch = data_consistency(gt_batch, torch.zeros_like(gt_batch), mask_batch)

        with torch.no_grad():
            rec_imgs, aux = reconstruct_far(model, tokenizer, masked_batch, mask_batch, **far_kwargs)

        logits_batch = aux["logits"]
        H_p, W_p = logits_batch.shape[1], logits_batch.shape[2]

        for i, rec in enumerate(slice_recs):
            log_p = F.log_softmax(logits_batch[i : i + 1].float(), dim=-1)
            p = log_p.exp()
            H_full = -(p * log_p).sum(-1)                        # (1, H_p, W_p, 2, K)
            emap    = H_full[0].sum(dim=(2, 3)).cpu().numpy().astype(np.float32)
            emap_cb = H_full[0].cpu().numpy().astype(np.float32)

            rec_rss = transforms.complex_abs(rec_imgs[i]).cpu().numpy()
            pmap    = _patch_mse(rec_rss, gt_rsses[i], H_p, W_p)
            metrics = compute_all_metrics(rec_rss, gt_rsses[i], device, compute_ssfd_flag=False)

            mask_vec = mask_batch[i, 0, :, 0].cpu().numpy().astype(np.int8)

            np.save(str(slice_dirs[i] / f"entropy_map_t{t:03d}.npy"),          emap)
            np.save(str(slice_dirs[i] / f"entropy_per_codebook_t{t:03d}.npy"), emap_cb)
            np.save(str(slice_dirs[i] / f"patch_error_map_t{t:03d}.npy"),      pmap)
            np.save(str(slice_dirs[i] / f"mask_t{t:03d}.npy"),                 mask_vec)

            if t == 0:
                log_p_t0_list[i] = log_p[0]   # (H_p, W_p, 2, K, V_K)
            else:
                kl_cb = _compute_kl_from_ref(logits_batch[i : i + 1], log_p_t0_list[i])
                np.save(str(slice_dirs[i] / f"kl_from_t0_t{t:03d}.npy"), kl_cb)

            slice_rows[i].append({
                "dataset": dataset, "R": acceleration, "method": method,
                "scan_id": rec["scan_id"], "slice_id": rec["slice_index"],
                "step": t, "selected_line": selected_lines[i],
                "mean_entropy": float(emap.mean()),
                "PSNR": metrics["psnr"], "SSIM": metrics["ssim"],
            })

        if t < B_budget:
            old_mask = mask_batch.clone()
            mask_2d = mask_batch.permute(0, 3, 1, 2)
            with torch.no_grad():
                new_mask_2d, _, freq_map = kspace_uncertainty_from_patch_logits(
                    logits_batch, mask_2d,
                    vertical_mask=True, rank=1,
                    les_zero_mean=les_zero_mean, les_radial_weight_alpha=0.0,
                    mask_type="cartesian",
                )
            mask_batch = new_mask_2d.permute(0, 2, 3, 1).float()

            for i in range(n):
                diff = mask_batch[i, 0, :, 0] - old_mask[i, 0, :, 0]
                sel  = diff.nonzero(as_tuple=True)[0]
                selected_lines[i] = int(sel[0].item()) if len(sel) > 0 else -1
                scores = freq_map[i].mean(dim=0).cpu().numpy()
                np.save(str(slice_dirs[i] / f"kspace_scores_t{t:03d}.npy"), scores.astype(np.float32))

    all_rows: list[dict] = []
    for i, sdir in enumerate(slice_dirs):
        (sdir / "done.json").write_text(json.dumps(slice_rows[i]))
        all_rows.extend(slice_rows[i])
    return all_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Patch-level entropy investigation experiment")
    p.add_argument("--config",       default="experiments/experiment1/config.yaml")
    p.add_argument("--run_dir",      required=True)
    p.add_argument("--dataset",      required=True, choices=["brain", "knee"])
    p.add_argument("--acceleration", required=True, type=int, choices=[8, 16])
    p.add_argument("--method",       required=True, choices=["LES", "LES_ZM", "GEO"])
    p.add_argument("--split_file",   required=True, help="Text file of H5 filenames")
    p.add_argument("--center_slices", default=None, type=int,
                   help="Take N center slices per scan (default: all slices)")
    p.add_argument("--batch_size",   default=16, type=int,
                   help="Slices per forward pass (LES only)")
    p.add_argument("--seed",         default=42, type=int)
    p.add_argument("--checkpoint",   default=None)
    p.add_argument("--meditok_config", default=None)
    p.add_argument("--meditok_ckpt",   default=None)
    p.add_argument("--data_path",      default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    root = _ROOT
    cfg_path = pathlib.Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    run_dir = pathlib.Path(args.run_dir)
    center_fraction = cfg.get("center_fraction", 0.04)

    ckpt_path = args.checkpoint or root / cfg["checkpoints"][args.dataset][args.acceleration]
    model, ckpt_args = load_singlecoil_model(ckpt_path, device)
    model._ckpt_args = ckpt_args

    mt_cfg  = args.meditok_config or root / cfg["meditok_config"]
    mt_ckpt = args.meditok_ckpt   or root / cfg["meditok_ckpt"]
    tokenizer = load_meditok(mt_cfg, mt_ckpt, device)

    data_root = pathlib.Path(args.data_path or root / cfg["data_paths"][args.dataset])
    sf = pathlib.Path(args.split_file)
    if not sf.is_absolute():
        sf = root / sf

    if args.center_slices is not None:
        slice_records = select_stratified_center_slices(
            data_root, sf,
            dataset=args.dataset,
            acceleration=args.acceleration,
            scans_per_contrast=999,   # no limit per contrast — take all scans
            n_center_slices=args.center_slices,
            seed=args.seed,
        )
    else:
        slice_records = select_all_slices_from_split(
            data_root, sf,
            dataset=args.dataset,
            acceleration=args.acceleration,
        )

    valid_records, valid_h5s = [], []
    for rec in slice_records:
        h5_path = data_root / rec["scan_id"]
        if h5_path.exists():
            valid_records.append(rec)
            valid_h5s.append(h5_path)
        else:
            print(f"  SKIP (missing): {h5_path}")

    is_les = args.method in ("LES", "LES_ZM")
    zero_mean = args.method == "LES_ZM"

    print(
        f"Patch entropy [{args.method}]: dataset={args.dataset}, R={args.acceleration}, "
        f"{len(valid_records)} slices"
        + (f", batch_size={args.batch_size}, zero_mean={zero_mean}" if is_les else " (GEO: serial)")
    )

    all_rows: list[dict] = []

    if is_les:
        pending_recs, pending_h5s = [], []
        for rec, h5_path in zip(valid_records, valid_h5s):
            sample_id = make_sample_id(args.dataset, rec["scan_id"], rec["slice_index"])
            done_file = (
                run_dir / "patch_entropy"
                / f"dataset={args.dataset}" / f"R={args.acceleration}"
                / f"method={args.method}" / f"sample_id={sample_id}" / "done.json"
            )
            if done_file.exists():
                with open(done_file) as f:
                    all_rows.extend(json.load(f))
            else:
                pending_recs.append(rec)
                pending_h5s.append(h5_path)

        print(f"  {len(all_rows)} already done, {len(pending_recs)} pending")
        for i in tqdm(range(0, len(pending_recs), args.batch_size), desc=f"{args.method} batches"):
            batch_recs = pending_recs[i : i + args.batch_size]
            batch_h5s  = pending_h5s [i : i + args.batch_size]
            rows = _run_batch_les(
                model=model, tokenizer=tokenizer,
                slice_recs=batch_recs, h5_paths=batch_h5s,
                dataset=args.dataset, acceleration=args.acceleration,
                center_fraction=center_fraction,
                run_dir=run_dir, device=device,
                method=args.method, les_zero_mean=zero_mean,
            )
            all_rows.extend(rows)

    else:  # GEO
        for rec in tqdm(valid_records, desc="GEO slices"):
            h5_path = data_root / rec["scan_id"]
            rows = run_slice(
                model=model, tokenizer=tokenizer,
                slice_rec=rec, h5_path=h5_path,
                dataset=args.dataset, acceleration=args.acceleration,
                method=args.method,
                center_fraction=center_fraction,
                run_dir=run_dir, device=device,
            )
            all_rows.extend(rows)

    out_dir = (
        run_dir / "patch_entropy"
        / f"dataset={args.dataset}" / f"R={args.acceleration}"
        / f"method={args.method}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    save_parquet(out_dir / "metadata.parquet", all_rows)
    print(f"Saved {len(all_rows)} rows → {out_dir / 'metadata.parquet'}")


if __name__ == "__main__":
    main()
