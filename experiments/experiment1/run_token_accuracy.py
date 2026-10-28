#!/usr/bin/env python
"""Token accuracy experiment.

For each slice computes top-1, top-5, top-10 accuracy and NLL of the model's
token predictions vs the GT tokens (argmax when given fully sampled k-space),
for three acquisition strategies: LES_ZM, GEO, and RANDOM.

LES_ZM + RANDOM are batched across slices (batch_size slices per forward pass).
GEO is serial (requires per-slice gradient backward).

RANDOM: at each budget step t, a random mask with exactly (center_lines + t)
acquired lines is generated once (global seed=42) and used directly.

Output path:
    <run_dir>/dataset=<d>/R=<R>/sample_id=<id>/
        gt_tokens.npy              (H_p, W_p, 2, K)  int32
        accuracy_<method>.json     list[{step, top1, top5, top10, nll}]

    <run_dir>/dataset=<d>/R=<R>/method=<m>/metadata.parquet

Usage:
    python experiments/experiment1/run_token_accuracy.py \\
        --config experiments/experiment1/config.yaml \\
        --run_dir results/token_accuracy \\
        --dataset brain --acceleration 8 \\
        --split_file fastmri/splits/suno_brain_test.txt \\
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
from experiments.experiment1.utils.storage import save_parquet
from experiments.experiment1.utils.slice_selector import (
    make_sample_id, select_all_slices_from_split,
)
from experiments.experiment1.run_normal_trajectory import load_slice, make_center_mask, _geo_step

from data import transforms
from autoregressive.models.generate import reconstruct_far, data_consistency
from autoregressive.sample.adaptive_sampling import kspace_uncertainty_from_patch_logits


FAR_KWARGS = dict(n_steps=1, far_mask_mode="cartesian", active_sampling_mask_type="cartesian")
TOP_KS = (1, 5, 10)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _token_metrics_batch(logits_batch: torch.Tensor,
                         gt_tokens_list: list[np.ndarray]) -> list[dict]:
    """Compute top-k accuracy and NLL for a batch of slices.

    Args:
        logits_batch:    (B, H_p, W_p, 2, K, V/K)
        gt_tokens_list:  list of B arrays, each (H_p, W_p, 2, K) int32

    Returns list of B dicts with keys top1, top5, top10, nll.
    """
    B = logits_batch.shape[0]
    max_k = max(TOP_KS)
    topk_indices = logits_batch.topk(max_k, dim=-1).indices  # (B, H_p, W_p, 2, K, max_k)
    log_p = F.log_softmax(logits_batch.float(), dim=-1)       # (B, H_p, W_p, 2, K, V/K)

    results = []
    for i in range(B):
        gt = torch.from_numpy(gt_tokens_list[i]).long().to(logits_batch.device)  # (H_p, W_p, 2, K)
        nll = -log_p[i].gather(-1, gt.unsqueeze(-1)).squeeze(-1)
        row = {"nll": float(nll.mean().item())}
        gt_exp = gt.unsqueeze(-1)
        for k in TOP_KS:
            hit = (topk_indices[i, ..., :k] == gt_exp).any(dim=-1)
            row[f"top{k}"] = float(hit.float().mean().item())
        results.append(row)
    return results


def _make_random_masks(W: int, B_budget: int, center_fraction: float,
                       device: torch.device, seed: int = 42) -> list[torch.Tensor]:
    """B_budget+1 masks: mask[t] = center_mask + t randomly pre-selected lines."""
    center_mask = make_center_mask(W, center_fraction, device)
    non_center = (center_mask[0, 0, :, 0] == 0).nonzero(as_tuple=True)[0].cpu()
    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = non_center[torch.randperm(len(non_center), generator=rng)]
    masks = []
    for t in range(B_budget + 1):
        m = center_mask.clone()
        if t > 0:
            m[0, 0, perm[:t].to(device), 0] = 1.0
        masks.append(m)
    return masks


def _make_random_per_step_masks(W: int, B_budget: int, center_fraction: float,
                                device: torch.device, seed: int = 42) -> list[torch.Tensor]:
    """B_budget+1 masks: at each step t, independently re-sample t random non-center lines.

    Unlike _make_random_masks (cumulative), here each step's lines are freshly drawn.
    Seed for step t = seed + t so each step is reproducible but independent.
    """
    center_mask = make_center_mask(W, center_fraction, device)
    non_center = (center_mask[0, 0, :, 0] == 0).nonzero(as_tuple=True)[0].cpu()
    masks = []
    for t in range(B_budget + 1):
        m = center_mask.clone()
        if t > 0:
            rng = torch.Generator()
            rng.manual_seed(seed + t)
            chosen = non_center[torch.randperm(len(non_center), generator=rng)[:t]]
            m[0, 0, chosen.to(device), 0] = 1.0
        masks.append(m)
    return masks


def _load_slice_data(h5_path, slice_idx, resolution, device):
    sd = load_slice(h5_path, slice_idx, resolution)
    gt = sd["gt_c_image"].to(device)   # (1, H, W, 2)
    return gt


# ---------------------------------------------------------------------------
# Batched LES_ZM
# ---------------------------------------------------------------------------

def run_batch_les(
    model, tokenizer,
    slice_recs: list[dict],
    h5_paths: list[pathlib.Path],
    gt_tokens_list: list[np.ndarray],
    sample_ids: list[str],
    sdirs: list[pathlib.Path],
    dataset: str, acceleration: int, center_fraction: float,
    device: torch.device, resolution: int,
) -> list[dict]:
    gt_images = [_load_slice_data(h, r["slice_index"], resolution, device)
                 for r, h in zip(slice_recs, h5_paths)]
    gt_batch = torch.cat(gt_images, dim=0)
    _, H, W, _ = gt_batch.shape
    B_budget = max(1, W // acceleration - int(W * center_fraction))
    far_kwargs = {**FAR_KWARGS, "center_fractions": center_fraction}

    n = len(slice_recs)
    mask_batch = torch.cat([make_center_mask(W, center_fraction, device)] * n, dim=0)
    slice_rows = [[] for _ in range(n)]

    for t in range(B_budget + 1):
        masked = data_consistency(gt_batch, torch.zeros_like(gt_batch), mask_batch)
        with torch.no_grad():
            _, aux = reconstruct_far(model, tokenizer, masked, mask_batch, **far_kwargs)
        logits = aux["logits"]   # (B, H_p, W_p, 2, K, V/K)

        metrics = _token_metrics_batch(logits, gt_tokens_list)
        for i in range(n):
            slice_rows[i].append({
                "dataset": dataset, "R": acceleration, "method": "LES_ZM",
                "sample_id": sample_ids[i], "step": t, **metrics[i],
            })

        if t < B_budget:
            mask_2d = mask_batch.permute(0, 3, 1, 2)
            with torch.no_grad():
                new_mask_2d, _, _ = kspace_uncertainty_from_patch_logits(
                    logits, mask_2d, vertical_mask=True, rank=1,
                    les_zero_mean=True, les_radial_weight_alpha=0.0,
                    mask_type="cartesian",
                )
            mask_batch = new_mask_2d.permute(0, 2, 3, 1).float()

    all_rows = []
    for i, sdir in enumerate(sdirs):
        (sdir / "accuracy_LES_ZM.json").write_text(json.dumps(slice_rows[i]))
        all_rows.extend(slice_rows[i])
    return all_rows


# ---------------------------------------------------------------------------
# Batched RANDOM
# ---------------------------------------------------------------------------

def run_batch_random(
    model, tokenizer,
    slice_recs: list[dict],
    h5_paths: list[pathlib.Path],
    gt_tokens_list: list[np.ndarray],
    sample_ids: list[str],
    sdirs: list[pathlib.Path],
    dataset: str, acceleration: int, center_fraction: float,
    device: torch.device, resolution: int,
) -> list[dict]:
    gt_images = [_load_slice_data(h, r["slice_index"], resolution, device)
                 for r, h in zip(slice_recs, h5_paths)]
    gt_batch = torch.cat(gt_images, dim=0)
    _, H, W, _ = gt_batch.shape
    B_budget = max(1, W // acceleration - int(W * center_fraction))
    far_kwargs = {**FAR_KWARGS, "center_fractions": center_fraction}

    n = len(slice_recs)
    random_masks = _make_random_masks(W, B_budget, center_fraction, device, seed=42)
    slice_rows = [[] for _ in range(n)]

    for t, mask_1 in enumerate(random_masks):
        mask_batch = mask_1.expand(n, -1, -1, -1)   # broadcast to batch
        masked = data_consistency(gt_batch, torch.zeros_like(gt_batch), mask_batch)
        with torch.no_grad():
            _, aux = reconstruct_far(model, tokenizer, masked, mask_batch, **far_kwargs)
        logits = aux["logits"]

        metrics = _token_metrics_batch(logits, gt_tokens_list)
        for i in range(n):
            slice_rows[i].append({
                "dataset": dataset, "R": acceleration, "method": "RANDOM",
                "sample_id": sample_ids[i], "step": t, **metrics[i],
            })

    all_rows = []
    for i, sdir in enumerate(sdirs):
        (sdir / "accuracy_RANDOM.json").write_text(json.dumps(slice_rows[i]))
        all_rows.extend(slice_rows[i])
    return all_rows


# ---------------------------------------------------------------------------
# Batched RANDOM_PER_STEP
# ---------------------------------------------------------------------------

def run_batch_random_per_step(
    model, tokenizer,
    slice_recs: list[dict],
    h5_paths: list[pathlib.Path],
    gt_tokens_list: list[np.ndarray],
    sample_ids: list[str],
    sdirs: list[pathlib.Path],
    dataset: str, acceleration: int, center_fraction: float,
    device: torch.device, resolution: int,
) -> list[dict]:
    gt_images = [_load_slice_data(h, r["slice_index"], resolution, device)
                 for r, h in zip(slice_recs, h5_paths)]
    gt_batch = torch.cat(gt_images, dim=0)
    _, H, W, _ = gt_batch.shape
    B_budget = max(1, W // acceleration - int(W * center_fraction))
    far_kwargs = {**FAR_KWARGS, "center_fractions": center_fraction}

    n = len(slice_recs)
    random_masks = _make_random_per_step_masks(W, B_budget, center_fraction, device, seed=42)
    slice_rows = [[] for _ in range(n)]

    for t, mask_1 in enumerate(random_masks):
        mask_batch = mask_1.expand(n, -1, -1, -1)
        masked = data_consistency(gt_batch, torch.zeros_like(gt_batch), mask_batch)
        with torch.no_grad():
            _, aux = reconstruct_far(model, tokenizer, masked, mask_batch, **far_kwargs)
        logits = aux["logits"]

        metrics = _token_metrics_batch(logits, gt_tokens_list)
        for i in range(n):
            slice_rows[i].append({
                "dataset": dataset, "R": acceleration, "method": "RANDOM_PER_STEP",
                "sample_id": sample_ids[i], "step": t, **metrics[i],
            })

    all_rows = []
    for i, sdir in enumerate(sdirs):
        (sdir / "accuracy_RANDOM_PER_STEP.json").write_text(json.dumps(slice_rows[i]))
        all_rows.extend(slice_rows[i])
    return all_rows


# ---------------------------------------------------------------------------
# Batched GEO (_geo_step handles (B, 1, W, 1) masks natively)
# ---------------------------------------------------------------------------

def run_batch_geo(
    model, tokenizer,
    slice_recs: list[dict],
    h5_paths: list[pathlib.Path],
    gt_tokens_list: list[np.ndarray],
    sample_ids: list[str],
    sdirs: list[pathlib.Path],
    dataset: str, acceleration: int, center_fraction: float,
    device: torch.device, resolution: int,
) -> list[dict]:
    gt_images = [_load_slice_data(h, r["slice_index"], resolution, device)
                 for r, h in zip(slice_recs, h5_paths)]
    gt_batch = torch.cat(gt_images, dim=0)
    _, H, W, _ = gt_batch.shape
    B_budget = max(1, W // acceleration - int(W * center_fraction))
    far_kwargs = {**FAR_KWARGS, "center_fractions": center_fraction}

    n = len(slice_recs)
    mask_batch = torch.cat([make_center_mask(W, center_fraction, device)] * n, dim=0)
    slice_rows = [[] for _ in range(n)]

    for t in range(B_budget + 1):
        masked = data_consistency(gt_batch, torch.zeros_like(gt_batch), mask_batch)
        with torch.no_grad():
            _, aux = reconstruct_far(model, tokenizer, masked, mask_batch, **far_kwargs)
        logits = aux["logits"]

        metrics = _token_metrics_batch(logits, gt_tokens_list)
        for i in range(n):
            slice_rows[i].append({
                "dataset": dataset, "R": acceleration, "method": "GEO",
                "sample_id": sample_ids[i], "step": t, **metrics[i],
            })

        if t < B_budget:
            new_mask, _ = _geo_step(model, tokenizer, masked, mask_batch, 1, device)
            mask_batch = new_mask.detach()

    all_rows = []
    for i, sdir in enumerate(sdirs):
        (sdir / "accuracy_GEO.json").write_text(json.dumps(slice_rows[i]))
        all_rows.extend(slice_rows[i])
    return all_rows


# ---------------------------------------------------------------------------
# GT token computation (batched)
# ---------------------------------------------------------------------------

def compute_gt_tokens_batch(
    model, tokenizer,
    slice_recs: list[dict],
    h5_paths: list[pathlib.Path],
    sdirs: list[pathlib.Path],
    center_fraction: float,
    device: torch.device, resolution: int,
) -> list[np.ndarray]:
    """Compute and cache GT tokens for a batch. Returns list of (H_p, W_p, 2, K) arrays."""
    gt_tokens_list = []
    pending_i, pending_recs, pending_h5s = [], [], []

    for i, (rec, h5, sdir) in enumerate(zip(slice_recs, h5_paths, sdirs)):
        path = sdir / "gt_tokens.npy"
        if path.exists():
            gt_tokens_list.append((i, np.load(str(path))))
        else:
            pending_i.append(i)
            pending_recs.append(rec)
            pending_h5s.append(h5)

    if pending_recs:
        gt_images = [_load_slice_data(h, r["slice_index"], resolution, device)
                     for r, h in zip(pending_recs, pending_h5s)]
        gt_batch = torch.cat(gt_images, dim=0)
        _, H, W, _ = gt_batch.shape
        full_mask = torch.ones(len(pending_recs), 1, W, 1, device=device)
        far_kwargs = {**FAR_KWARGS, "center_fractions": center_fraction}
        masked = data_consistency(gt_batch, torch.zeros_like(gt_batch), full_mask)
        with torch.no_grad():
            _, aux = reconstruct_far(model, tokenizer, masked, full_mask, **far_kwargs)
        logits = aux["logits"]   # (B, H_p, W_p, 2, K, V/K)
        for j, orig_i in enumerate(pending_i):
            tok = logits[j].argmax(dim=-1).cpu().numpy().astype(np.int32)
            np.save(str(sdirs[orig_i] / "gt_tokens.npy"), tok)
            gt_tokens_list.append((orig_i, tok))

    gt_tokens_list.sort(key=lambda x: x[0])
    return [t for _, t in gt_tokens_list]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Token accuracy experiment")
    p.add_argument("--config",        default="experiments/experiment1/config.yaml")
    p.add_argument("--run_dir",       required=True)
    p.add_argument("--dataset",       required=True, choices=["brain", "knee"])
    p.add_argument("--acceleration",  required=True, type=int, choices=[8, 16])
    p.add_argument("--split_file",    required=True)
    p.add_argument("--batch_size",     default=16, type=int)
    p.add_argument("--geo_batch_size", default=4,  type=int,
                   help="Batch size for GEO (smaller due to backward pass memory)")
    p.add_argument("--checkpoint",    default=None)
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
    resolution = getattr(ckpt_args, "image_size", 256)

    mt_cfg  = args.meditok_config or root / cfg["meditok_config"]
    mt_ckpt = args.meditok_ckpt   or root / cfg["meditok_ckpt"]
    tokenizer = load_meditok(mt_cfg, mt_ckpt, device)

    data_root = pathlib.Path(args.data_path or root / cfg["data_paths"][args.dataset])
    sf = pathlib.Path(args.split_file)
    if not sf.is_absolute():
        sf = root / sf

    all_records = select_all_slices_from_split(
        data_root, sf, dataset=args.dataset, acceleration=args.acceleration,
    )

    # Filter missing H5s; build sample dirs
    records, h5_paths, sample_ids, sdirs = [], [], [], []
    for rec in all_records:
        h5 = data_root / rec["scan_id"]
        if not h5.exists():
            continue
        sid = make_sample_id(args.dataset, rec["scan_id"], rec["slice_index"])
        sdir = run_dir / f"dataset={args.dataset}" / f"R={args.acceleration}" / f"sample_id={sid}"
        sdir.mkdir(parents=True, exist_ok=True)
        records.append(rec)
        h5_paths.append(h5)
        sample_ids.append(sid)
        sdirs.append(sdir)

    print(f"Token accuracy: dataset={args.dataset}, R={args.acceleration}, {len(records)} slices")

    bs = args.batch_size
    all_rows: list[dict] = []

    # ── LES_ZM + RANDOM + RANDOM_PER_STEP (batched) ───────────────────────
    les_pending_i   = [i for i, s in enumerate(sdirs) if not (s / "accuracy_LES_ZM.json").exists()]
    rand_pending_i  = [i for i, s in enumerate(sdirs) if not (s / "accuracy_RANDOM.json").exists()]
    rps_pending_i   = [i for i, s in enumerate(sdirs) if not (s / "accuracy_RANDOM_PER_STEP.json").exists()]
    # Load existing rows
    for i, sdir in enumerate(sdirs):
        for m in ("LES_ZM", "RANDOM", "RANDOM_PER_STEP"):
            p = sdir / f"accuracy_{m}.json"
            if p.exists():
                all_rows.extend(json.loads(p.read_text()))

    # Process pending LES batches
    for start in tqdm(range(0, len(les_pending_i), bs), desc="LES_ZM batches"):
        idx = les_pending_i[start : start + bs]
        gt_tokens = compute_gt_tokens_batch(
            model, tokenizer,
            [records[i] for i in idx], [h5_paths[i] for i in idx], [sdirs[i] for i in idx],
            center_fraction, device, resolution,
        )
        rows = run_batch_les(
            model, tokenizer,
            [records[i] for i in idx], [h5_paths[i] for i in idx],
            gt_tokens, [sample_ids[i] for i in idx], [sdirs[i] for i in idx],
            args.dataset, args.acceleration, center_fraction, device, resolution,
        )
        all_rows.extend(rows)

    # Process pending RANDOM batches
    for start in tqdm(range(0, len(rand_pending_i), bs), desc="RANDOM batches"):
        idx = rand_pending_i[start : start + bs]
        gt_tokens = compute_gt_tokens_batch(
            model, tokenizer,
            [records[i] for i in idx], [h5_paths[i] for i in idx], [sdirs[i] for i in idx],
            center_fraction, device, resolution,
        )
        rows = run_batch_random(
            model, tokenizer,
            [records[i] for i in idx], [h5_paths[i] for i in idx],
            gt_tokens, [sample_ids[i] for i in idx], [sdirs[i] for i in idx],
            args.dataset, args.acceleration, center_fraction, device, resolution,
        )
        all_rows.extend(rows)

    # Process pending RANDOM_PER_STEP batches
    for start in tqdm(range(0, len(rps_pending_i), bs), desc="RANDOM_PER_STEP batches"):
        idx = rps_pending_i[start : start + bs]
        gt_tokens = compute_gt_tokens_batch(
            model, tokenizer,
            [records[i] for i in idx], [h5_paths[i] for i in idx], [sdirs[i] for i in idx],
            center_fraction, device, resolution,
        )
        rows = run_batch_random_per_step(
            model, tokenizer,
            [records[i] for i in idx], [h5_paths[i] for i in idx],
            gt_tokens, [sample_ids[i] for i in idx], [sdirs[i] for i in idx],
            args.dataset, args.acceleration, center_fraction, device, resolution,
        )
        all_rows.extend(rows)

    # ── GEO (batched) ──────────────────────────────────────────────────────
    geo_pending_i = [i for i, s in enumerate(sdirs) if not (s / "accuracy_GEO.json").exists()]
    for i, sdir in enumerate(sdirs):
        if (sdir / "accuracy_GEO.json").exists():
            all_rows.extend(json.loads((sdir / "accuracy_GEO.json").read_text()))

    geo_bs = args.geo_batch_size
    for start in tqdm(range(0, len(geo_pending_i), geo_bs), desc="GEO batches"):
        idx = geo_pending_i[start : start + geo_bs]
        gt_tokens = compute_gt_tokens_batch(
            model, tokenizer,
            [records[i] for i in idx], [h5_paths[i] for i in idx], [sdirs[i] for i in idx],
            center_fraction, device, resolution,
        )
        rows = run_batch_geo(
            model, tokenizer,
            [records[i] for i in idx], [h5_paths[i] for i in idx],
            gt_tokens, [sample_ids[i] for i in idx], [sdirs[i] for i in idx],
            args.dataset, args.acceleration, center_fraction, device, resolution,
        )
        all_rows.extend(rows)

    # ── Save aggregate parquets ────────────────────────────────────────────
    for method in ("LES_ZM", "GEO", "RANDOM", "RANDOM_PER_STEP"):
        out_dir = run_dir / f"dataset={args.dataset}" / f"R={args.acceleration}" / f"method={method}"
        out_dir.mkdir(parents=True, exist_ok=True)
        method_rows = [r for r in all_rows if r["method"] == method]
        if method_rows:
            save_parquet(out_dir / "metadata.parquet", method_rows)
            print(f"  {method}: {len(method_rows)} rows → {out_dir}")


if __name__ == "__main__":
    main()
