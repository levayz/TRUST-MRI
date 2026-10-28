#!/usr/bin/env python
"""GT reconstruction experiment.

Runs the FAR model with a fully-sampled k-space mask and saves results in the
same directory format as run_patch_entropy_experiment.py so that existing
notebook visualization cells work by simply changing METHOD='GT'.

Saved per slice:
    gt_rss.npy                         (H, W)           GT RSS image
    rec_rss.npy                        (H, W)           Reconstructed RSS image
    entropy_map_t000.npy               (H_p, W_p)       entropy (one step only)
    entropy_per_codebook_t000.npy      (H_p, W_p, 2, K) per-channel/codebook entropy
    patch_error_map_t000.npy           (H_p, W_p)       per-patch MSE (rec vs GT)
    mask_t000.npy                      (W,)             random R-mask used for DC (CF=0.04)
    done.json                          list[dict]        metrics (PSNR, SSIM, NMSE, SSFD)

Output path:
    <run_dir>/patch_entropy/dataset=<d>/R=<r>/method=GT/sample_id=<id>/

To reuse existing notebook visualization cells:
    PE_DIR  = '<run_dir>'
    METHOD  = 'GT'

Usage:
    python experiments/experiment1/run_gt_recon_experiment.py \\
        --config experiments/experiment1/config.yaml \\
        --run_dir results/gt_recon \\
        --dataset brain --acceleration 8 \\
        --split_file fastmri/splits/suno_brain_test.txt \\
        --center_slices 15 --batch_size 16 --device cuda:0
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
from experiments.experiment1.run_normal_trajectory import load_slice, make_center_mask

from data import transforms
from autoregressive.models.generate import reconstruct_far, data_consistency


FAR_KWARGS = dict(
    n_steps=1,
    far_mask_mode="cartesian",
    active_sampling_mask_type="cartesian",
)


def _compute_entropy(logits: torch.Tensor):
    log_p = F.log_softmax(logits.float(), dim=-1)
    p = log_p.exp()
    H_full = -(p * log_p).sum(-1)                           # (B, H_p, W_p, 2, K)
    entropy_map = H_full[0].sum(dim=(2, 3)).cpu().numpy()   # (H_p, W_p)
    entropy_per_cb = H_full[0].cpu().numpy()                # (H_p, W_p, 2, K)
    return entropy_map.astype(np.float32), entropy_per_cb.astype(np.float32)


def _patch_mse(rec: np.ndarray, gt: np.ndarray, H_p: int, W_p: int) -> np.ndarray:
    H, W = rec.shape
    ph, pw = H // H_p, W // W_p
    return ((rec.reshape(H_p, ph, W_p, pw) - gt.reshape(H_p, ph, W_p, pw)) ** 2
            ).mean(axis=(1, 3)).astype(np.float32)


def _make_random_dc_mask(W: int, acceleration: int, center_fraction: float,
                         device: torch.device, seed: int) -> torch.Tensor:
    """Center lines + randomly sampled additional lines to reach W//acceleration total."""
    center_mask = make_center_mask(W, center_fraction, device)   # (1, 1, W, 1)
    n_center = int(W * center_fraction)
    n_total  = W // acceleration
    n_random = max(0, n_total - n_center)

    non_center = (center_mask[0, 0, :, 0] == 0).nonzero(as_tuple=True)[0]
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    perm = non_center.cpu()[torch.randperm(len(non_center), generator=rng)[:n_random]].to(device)

    dc_mask = center_mask.clone()
    dc_mask[0, 0, perm, 0] = 1.0
    return dc_mask   # (1, 1, W, 1)


def _run_batch(
    model, tokenizer,
    slice_recs: list[dict],
    h5_paths: list[pathlib.Path],
    dataset: str,
    acceleration: int,
    center_fraction: float,
    run_dir: pathlib.Path,
    device: torch.device,
    resolution: int,
    use_undersampled_input: bool = False,
) -> list[dict]:
    method = "GT_US" if use_undersampled_input else "GT"
    gt_images, gt_rsses, slice_dirs, dc_masks = [], [], [], []

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
        gt = sd["gt_c_image"].to(device)   # (1, H, W, 2)
        gt_images.append(gt)
        gt_rss = transforms.complex_abs(gt[0]).cpu().numpy()
        gt_rsses.append(gt_rss)
        np.save(str(sdir / "gt_rss.npy"), gt_rss.astype(np.float32))

        # Fixed random DC mask per slice (seeded by scan+slice for reproducibility)
        seed = hash((rec["scan_id"], rec["slice_index"])) & 0xFFFFFFFF
        _, _, W, _ = gt.shape
        dc_masks.append(_make_random_dc_mask(W, acceleration, center_fraction, device, seed))

    gt_batch = torch.cat(gt_images, dim=0)   # (B, H, W, 2)
    _, H, W, _ = gt_batch.shape

    if use_undersampled_input:
        # Sanity check: model sees undersampled aliased image, same DC mask as GT experiment
        model_input = torch.cat([
            data_consistency(gt_images[i], torch.zeros_like(gt_images[i]), dc_masks[i])
            for i in range(len(slice_recs))
        ], dim=0)
        model_mask = torch.cat(dc_masks, dim=0)   # (B, 1, W, 1)
    else:
        # Standard GT experiment: model sees fully sampled GT
        model_mask = torch.ones(len(slice_recs), 1, W, 1, device=device)
        model_input = data_consistency(gt_batch, torch.zeros_like(gt_batch), model_mask)

    far_kwargs = {**FAR_KWARGS, "center_fractions": center_fraction}
    with torch.no_grad():
        _, aux = reconstruct_far(model, tokenizer, model_input, model_mask, **far_kwargs)

    logits_batch  = aux["logits"]           # (B, H_p, W_p, 2, K, V/K)
    decoded_no_dc = aux["decoded_no_dc"]    # (B, H, W, 2) — AR output before data consistency
    H_p, W_p = logits_batch.shape[1], logits_batch.shape[2]

    all_rows = []
    for i, rec in enumerate(slice_recs):
        log_p = F.log_softmax(logits_batch[i : i + 1].float(), dim=-1)
        p = log_p.exp()
        H_full = -(p * log_p).sum(-1)
        emap    = H_full[0].sum(dim=(2, 3)).cpu().numpy().astype(np.float32)
        emap_cb = H_full[0].cpu().numpy().astype(np.float32)

        # DC with random R-mask: acquired lines come from GT, rest from AR decoded output
        dc_mask = dc_masks[i]                                    # (1, 1, W, 1)
        rec_img = data_consistency(gt_images[i], decoded_no_dc[i:i+1], dc_mask)
        rec_rss = transforms.complex_abs(rec_img[0]).cpu().numpy()

        pmap    = _patch_mse(rec_rss, gt_rsses[i], H_p, W_p)
        metrics = compute_all_metrics(rec_rss, gt_rsses[i], device, compute_ssfd_flag=True)

        dc_mask_vec = dc_mask[0, 0, :, 0].cpu().numpy().astype(np.int8)

        np.save(str(slice_dirs[i] / "rec_rss.npy"),                    rec_rss.astype(np.float32))
        np.save(str(slice_dirs[i] / "entropy_map_t000.npy"),           emap)
        np.save(str(slice_dirs[i] / "entropy_per_codebook_t000.npy"),  emap_cb)
        np.save(str(slice_dirs[i] / "patch_error_map_t000.npy"),       pmap)
        np.save(str(slice_dirs[i] / "mask_t000.npy"),                  dc_mask_vec)

        row = {
            "dataset": dataset, "R": acceleration, "method": method,
            "scan_id": rec["scan_id"], "slice_id": rec["slice_index"],
            "step": 0, "selected_line": -1,
            "mean_entropy": float(emap.mean()),
            "PSNR": metrics["psnr"], "SSIM": metrics["ssim"],
            "NMSE": metrics.get("nmse", float("nan")),
            "SSFD": metrics.get("ssfd", float("nan")),
        }
        (slice_dirs[i] / "done.json").write_text(json.dumps([row]))
        all_rows.append(row)

    return all_rows


def parse_args():
    p = argparse.ArgumentParser(description="GT reconstruction experiment")
    p.add_argument("--config",        default="experiments/experiment1/config.yaml")
    p.add_argument("--run_dir",       required=True)
    p.add_argument("--dataset",       required=True, choices=["brain", "knee"])
    p.add_argument("--acceleration",  required=True, type=int, choices=[8, 16])
    p.add_argument("--split_file",    required=True)
    p.add_argument("--center_slices", default=None, type=int)
    p.add_argument("--batch_size",    default=16, type=int)
    p.add_argument("--seed",          default=42, type=int)
    p.add_argument("--checkpoint",    default=None)
    p.add_argument("--meditok_config", default=None)
    p.add_argument("--meditok_ckpt",   default=None)
    p.add_argument("--data_path",      default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use_undersampled_input", action="store_true",
                   help="Sanity check: use aliased undersampled image as model input (method=GT_US)")
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
    resolution = getattr(ckpt_args, "image_size", 256)

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
            scans_per_contrast=999,
            n_center_slices=args.center_slices,
            seed=args.seed,
        )
    else:
        slice_records = select_all_slices_from_split(
            data_root, sf,
            dataset=args.dataset,
            acceleration=args.acceleration,
        )

    method = "GT_US" if args.use_undersampled_input else "GT"

    # Filter to existing H5 files; skip already completed
    pending_recs, pending_h5s = [], []
    all_rows = []
    for rec in slice_records:
        h5_path = data_root / rec["scan_id"]
        if not h5_path.exists():
            print(f"  SKIP (missing): {h5_path}")
            continue
        sample_id = make_sample_id(args.dataset, rec["scan_id"], rec["slice_index"])
        done_file = (
            run_dir / "patch_entropy"
            / f"dataset={args.dataset}" / f"R={args.acceleration}"
            / f"method={method}" / f"sample_id={sample_id}" / "done.json"
        )
        if done_file.exists():
            with open(done_file) as f:
                all_rows.extend(json.load(f))
        else:
            pending_recs.append(rec)
            pending_h5s.append(h5_path)

    print(
        f"{method} recon: dataset={args.dataset}, R={args.acceleration}, "
        f"{len(all_rows)} already done, {len(pending_recs)} pending"
    )

    for i in tqdm(range(0, len(pending_recs), args.batch_size), desc=f"{method} batches"):
        rows = _run_batch(
            model=model, tokenizer=tokenizer,
            slice_recs=pending_recs[i : i + args.batch_size],
            h5_paths=pending_h5s[i : i + args.batch_size],
            dataset=args.dataset,
            acceleration=args.acceleration,
            center_fraction=center_fraction,
            run_dir=run_dir,
            device=device,
            resolution=resolution,
            use_undersampled_input=args.use_undersampled_input,
        )
        all_rows.extend(rows)

    out_dir = (
        run_dir / "patch_entropy"
        / f"dataset={args.dataset}" / f"R={args.acceleration}" / f"method={method}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    save_parquet(out_dir / "metadata.parquet", all_rows)
    print(f"Saved {len(all_rows)} rows → {out_dir / 'metadata.parquet'}")


if __name__ == "__main__":
    main()
