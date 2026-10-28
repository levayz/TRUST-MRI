#!/usr/bin/env python
"""Compute GT entropy for existing patch entropy samples.

For each completed sample directory under
    <run_dir>/patch_entropy/dataset=<d>/R=<R>/method=<m>/
runs reconstruct_far with a fully-sampled mask and saves:

    entropy_map_gt.npy              (H_p, W_p)       summed entropy
    entropy_per_codebook_gt.npy     (H_p, W_p, 2, K) per-channel/codebook

Skips any sample that already has entropy_per_codebook_gt.npy.

Usage:
    python experiments/experiment1/run_gt_entropy.py \\
        --config experiments/experiment1/config.yaml \\
        --run_dir results/patch_entropy \\
        --dataset brain --acceleration 8 --method LES_ZM \\
        --batch_size 16 --device cuda:0
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
from experiments.experiment1.run_normal_trajectory import load_slice

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


def _process_batch(
    model, tokenizer,
    scan_ids: list[str],
    slice_indices: list[int],
    sample_dirs: list[pathlib.Path],
    data_root: pathlib.Path,
    center_fraction: float,
    device: torch.device,
    resolution: int,
):
    gt_images = []
    valid_idx = []
    for i, (scan_id, slice_idx) in enumerate(zip(scan_ids, slice_indices)):
        h5_path = data_root / scan_id
        if not h5_path.exists():
            print(f"  SKIP (missing H5): {h5_path}")
            continue
        sd = load_slice(h5_path, slice_idx, resolution)
        gt_images.append(sd["gt_c_image"].to(device))   # (1, H, W, 2)
        valid_idx.append(i)

    if not gt_images:
        return

    gt_batch = torch.cat(gt_images, dim=0)   # (B, H, W, 2)
    _, H, W, _ = gt_batch.shape

    # Full mask — all k-space lines acquired
    full_mask = torch.ones(len(gt_images), 1, W, 1, device=device)
    masked_batch = data_consistency(gt_batch, torch.zeros_like(gt_batch), full_mask)

    far_kwargs = {**FAR_KWARGS, "center_fractions": center_fraction}
    with torch.no_grad():
        _, aux = reconstruct_far(model, tokenizer, masked_batch, full_mask, **far_kwargs)

    logits_batch = aux["logits"]   # (B, H_p, W_p, 2, K, V/K)

    for out_i, orig_i in enumerate(valid_idx):
        log_p = F.log_softmax(logits_batch[out_i : out_i + 1].float(), dim=-1)
        p = log_p.exp()
        H_full = -(p * log_p).sum(-1)                        # (1, H_p, W_p, 2, K)
        emap    = H_full[0].sum(dim=(2, 3)).cpu().numpy().astype(np.float32)
        emap_cb = H_full[0].cpu().numpy().astype(np.float32)
        sdir = sample_dirs[orig_i]
        np.save(str(sdir / "entropy_map_gt.npy"),             emap)
        np.save(str(sdir / "entropy_per_codebook_gt.npy"),    emap_cb)


def parse_args():
    p = argparse.ArgumentParser(description="Compute GT entropy for patch entropy samples")
    p.add_argument("--config",       default="experiments/experiment1/config.yaml")
    p.add_argument("--run_dir",      required=True)
    p.add_argument("--dataset",      required=True, choices=["brain", "knee"])
    p.add_argument("--acceleration", required=True, type=int, choices=[8, 16])
    p.add_argument("--method",       required=True)
    p.add_argument("--batch_size",   default=16, type=int)
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
    center_fraction = cfg.get("center_fraction", 0.04)

    ckpt_path = args.checkpoint or root / cfg["checkpoints"][args.dataset][args.acceleration]
    model, ckpt_args = load_singlecoil_model(ckpt_path, device)
    model._ckpt_args = ckpt_args
    resolution = getattr(ckpt_args, "image_size", 256)

    mt_cfg  = args.meditok_config or root / cfg["meditok_config"]
    mt_ckpt = args.meditok_ckpt   or root / cfg["meditok_ckpt"]
    tokenizer = load_meditok(mt_cfg, mt_ckpt, device)

    data_root = pathlib.Path(args.data_path or root / cfg["data_paths"][args.dataset])

    base = (
        pathlib.Path(args.run_dir) / "patch_entropy"
        / f"dataset={args.dataset}" / f"R={args.acceleration}"
        / f"method={args.method}"
    )
    if not base.exists():
        print(f"No samples found under {base}")
        return

    all_dirs = sorted(p for p in base.glob("sample_id=*") if (p / "done.json").exists())
    pending  = [d for d in all_dirs if not (d / "entropy_per_codebook_gt.npy").exists()]

    print(
        f"GT entropy [{args.method}]: dataset={args.dataset}, R={args.acceleration}, "
        f"{len(all_dirs)} total, {len(pending)} pending, batch_size={args.batch_size}"
    )

    # Collect (scan_id, slice_id) from done.json
    scan_ids, slice_indices = [], []
    for sdir in pending:
        with open(sdir / "done.json") as f:
            meta = json.load(f)
        scan_ids.append(meta[0]["scan_id"])
        slice_indices.append(int(meta[0]["slice_id"]))

    for i in tqdm(range(0, len(pending), args.batch_size), desc="GT entropy batches"):
        _process_batch(
            model, tokenizer,
            scan_ids=scan_ids[i : i + args.batch_size],
            slice_indices=slice_indices[i : i + args.batch_size],
            sample_dirs=pending[i : i + args.batch_size],
            data_root=data_root,
            center_fraction=center_fraction,
            device=device,
            resolution=resolution,
        )

    print(f"Done. GT entropy saved to {base}/sample_id=*/entropy_per_codebook_gt.npy")


if __name__ == "__main__":
    main()
