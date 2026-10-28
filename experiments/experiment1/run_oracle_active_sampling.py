#!/usr/bin/env python
"""Run Full Greedy Oracle Active-Sampling for Experiment 1.

For each (dataset, acceleration, oracle_objective, selected_slice):
    At each step, evaluates every unsampled k-space column, picks the one
    that maximally improves the oracle objective, and permanently adds it.

Oracle objectives:
    psnr      — maximise PSNR (≡ minimise MSE; same column selected as mse)
    ssfd      — minimise SSFD between predicted and GT RSS
    mse       — minimise MSE between predicted and GT RSS
    token_acc — maximise token prediction accuracy

Writes:
    <run_dir>/oracle_active_sampling/dataset=<d>/R=<r>/objective=<obj>/
        sample_id=<id>/step_{t:03d}.npz
        candidate_scores_step_{t:03d}.parquet

Usage:
    # From a split file (all slices per scan):
    python run_oracle_active_sampling.py \\
        --config experiments/experiment1/config.yaml \\
        --run_dir results/exp1 \\
        --dataset brain --acceleration 16 --oracle_objective psnr \\
        --split_file fastmri/splits/suno_brain_test.txt \\
        --batch_size 32

    # From Stage-1 selected slices JSON (legacy):
    python run_oracle_active_sampling.py \\
        --config experiments/experiment1/config.yaml \\
        --run_dir results/exp1 \\
        --dataset brain --acceleration 8 --oracle_objective mse
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

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
    encode_gt_tokens, compute_token_accuracy, compute_token_nll,
    compute_ssfd, compute_all_metrics,
)
from experiments.experiment1.utils.storage import save_step_npz, save_parquet
from experiments.experiment1.utils.slice_selector import (
    make_sample_id, select_stratified_center_slices, select_all_slices_from_split,
)
from experiments.experiment1.run_normal_trajectory import load_slice, make_center_mask

from data import transforms
from autoregressive.models.generate import reconstruct_far, data_consistency


# ---------------------------------------------------------------------------
# Oracle objective functions
# ---------------------------------------------------------------------------

def _oracle_score(
    rec_rss: np.ndarray,
    gt_rss: np.ndarray,
    logits: torch.Tensor,
    gt_tokens_BKL: torch.Tensor | None,
    objective: str,
    device: torch.device,
) -> float:
    """Higher score = better acquisition. All objectives return higher-is-better."""
    if objective in ("mse", "psnr"):
        # psnr = 10*log10(max²/mse); argmax psnr ≡ argmin mse ≡ argmax(-mse)
        return -float(np.mean((rec_rss - gt_rss) ** 2))
    elif objective == "ssfd":
        return -compute_ssfd(rec_rss, gt_rss, device)
    elif objective == "token_acc":
        return compute_token_accuracy(logits, gt_tokens_BKL)
    else:
        raise ValueError(f"Unknown oracle objective: {objective}")


# ---------------------------------------------------------------------------
# Batched oracle step
# ---------------------------------------------------------------------------

@torch.no_grad()
def oracle_step_batched(
    model,
    tokenizer,
    gt_c_image: torch.Tensor,       # (1, H, W, 2)
    sample_mask: torch.Tensor,       # (1, H, W, 2)
    gt_rss: np.ndarray,              # (H, W) – precomputed to avoid repeating
    gt_tokens_BKL: torch.Tensor | None,
    objective: str,
    far_kwargs: dict,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[int, torch.Tensor, list[dict], np.ndarray, torch.Tensor]:
    """Evaluate all unsampled columns in batches and return the best one.

    Stacks `batch_size` candidate masks into a single forward pass, then scores
    each reconstruction individually. Caches the best candidate's output so the
    caller does not need an extra forward pass.

    Returns:
        best_col:       column index selected
        new_mask:       updated sample_mask after selecting best_col
        candidate_scores: list of {candidate_line, score} dicts for parquet
        best_rec_rss:   (H, W) float32 – reconstruction for the best column
        best_logits:    (1, H_p, W_p, 2, K, V_K) – logits for the best column
    """
    _, H, W, _ = gt_c_image.shape
    acquired = sample_mask[0, 0, :, 0].bool()
    unsampled_cols = [c for c in range(W) if not acquired[c].item()]

    candidate_scores: list[dict] = []
    best_score = -float("inf")
    best_col = -1
    best_rec_rss: np.ndarray | None = None
    best_logits: torch.Tensor | None = None

    for i in range(0, len(unsampled_cols), batch_size):
        chunk_cols = unsampled_cols[i : i + batch_size]
        n = len(chunk_cols)

        masks_batch = sample_mask.expand(n, -1, -1, -1).clone()
        for j, col in enumerate(chunk_cols):
            masks_batch[j, :, col, :] = 1.0

        gt_batch = gt_c_image.expand(n, -1, -1, -1).contiguous()
        cand_imgs = data_consistency(gt_batch, torch.zeros_like(gt_batch), masks_batch)
        rec_imgs, aux = reconstruct_far(model, tokenizer, cand_imgs, masks_batch, **far_kwargs)
        logits_batch = aux["logits"]  # (n, H_p, W_p, 2, K, V_K)

        for j, col in enumerate(chunk_cols):
            rec_rss_j = transforms.complex_abs(rec_imgs[j]).cpu().numpy()
            score = _oracle_score(
                rec_rss_j, gt_rss,
                logits_batch[j : j + 1], gt_tokens_BKL,
                objective, device,
            )
            candidate_scores.append({"candidate_line": col, "score": score})
            if score > best_score:
                best_score = score
                best_col = col
                best_rec_rss = rec_rss_j
                best_logits = logits_batch[j : j + 1].clone()

    new_mask = sample_mask.clone()
    new_mask[0, :, best_col, :] = 1.0
    return best_col, new_mask, candidate_scores, best_rec_rss, best_logits


# ---------------------------------------------------------------------------
# Per-slice oracle trajectory
# ---------------------------------------------------------------------------

def run_oracle_for_slice(
    model,
    tokenizer,
    slice_rec: dict,
    h5_path: pathlib.Path,
    dataset: str,
    acceleration: int,
    objective: str,
    run_dir: pathlib.Path,
    cfg: dict,
    device: torch.device,
    batch_size: int = 32,
) -> None:
    sample_id = make_sample_id(dataset, slice_rec["scan_id"], slice_rec["slice_index"])
    out_dir = (
        run_dir / "oracle_active_sampling"
        / f"dataset={dataset}" / f"R={acceleration}"
        / f"objective={objective}" / f"sample_id={sample_id}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "summary.json").exists():
        return

    resolution = cfg.get("resolution", 256)
    ckpt_args = getattr(model, "_ckpt_args", None)
    if ckpt_args is not None and hasattr(ckpt_args, "image_size"):
        resolution = ckpt_args.image_size
    center_fraction = cfg.get("center_fraction", 0.04)
    rank = cfg.get("rank", 1)

    slice_data = load_slice(h5_path, slice_rec["slice_index"], resolution)
    gt_c_image = slice_data["gt_c_image"].to(device)
    _, H, W, _ = gt_c_image.shape

    budget = max(1, W // acceleration - int(W * center_fraction))
    n_steps = budget // rank

    sample_mask = make_center_mask(W, center_fraction, device)

    far_kwargs = {
        "n_steps": 1,
        "far_mask_mode": "cartesian",
        "center_fractions": center_fraction,
        "active_sampling_mask_type": "cartesian",
    }

    gt_rss = transforms.complex_abs(gt_c_image[0]).cpu().numpy()
    np.save(str(out_dir / "gt_rss.npy"), gt_rss.astype(np.float32))

    gt_tokens_BKL = encode_gt_tokens(tokenizer, gt_c_image) if objective == "token_acc" else None

    def _make_step_data(step, selected_line, cand_scores, rec_rss_arr, logits_t, mask_t, runtime):
        metrics = compute_all_metrics(rec_rss_arr, gt_rss, device, compute_ssfd_flag=True)
        log_p = F.log_softmax(logits_t.float(), dim=-1)
        total_entropy = float((-(log_p.exp() * log_p).sum(-1)).sum().item())
        tok_acc = compute_token_accuracy(logits_t, gt_tokens_BKL) if gt_tokens_BKL is not None else float("nan")
        tok_nll = compute_token_nll(logits_t, gt_tokens_BKL) if gt_tokens_BKL is not None else float("nan")
        return {
            "sample_id":       sample_id,
            "dataset":         dataset,
            "acceleration":    acceleration,
            "oracle_objective": objective,
            "step":            step,
            "selected_line":   selected_line,
            "candidate_lines": np.array([r["candidate_line"] for r in cand_scores], dtype=np.int16),
            "candidate_objective_values": np.array([r["score"] for r in cand_scores], dtype=np.float32),
            "mask_after_selection": mask_t[0, 0, :, 0].cpu().numpy().astype(np.int8),
            "rec_rss": rec_rss_arr.astype(np.float32),
            "psnr":  metrics["psnr"],
            "ssim":  metrics["ssim"],
            "nmse":  metrics["nmse"],
            "ssfd":  metrics.get("ssfd", float("nan")),
            "token_accuracy": tok_acc,
            "token_nll":      tok_nll,
            "total_entropy":  total_entropy,
            "runtime_seconds": runtime,
        }

    # Step 0 — center-only baseline (no outer lines added yet).
    # Matches the step-0 convention used by run_entropy_acquisition.py.
    with torch.no_grad():
        center_img = data_consistency(gt_c_image, torch.zeros_like(gt_c_image), sample_mask)
        base_rec, base_aux = reconstruct_far(model, tokenizer, center_img, sample_mask, **far_kwargs)
    base_rss = transforms.complex_abs(base_rec[0]).cpu().numpy()
    save_step_npz(
        out_dir / "step_000.npz",
        _make_step_data(0, -1, [], base_rss, base_aux["logits"], sample_mask, 0.0),
    )

    rec_rss = base_rss  # used for final_rec_rss.npy if n_steps == 0

    for step in tqdm(range(1, n_steps + 1), desc=f"Oracle-{objective[:4]} {sample_id[:25]}", leave=False):
        t_start = time.perf_counter()

        best_col, new_mask, cand_scores, rec_rss, logits = oracle_step_batched(
            model, tokenizer, gt_c_image, sample_mask,
            gt_rss, gt_tokens_BKL,
            objective, far_kwargs, device, batch_size=batch_size,
        )
        sample_mask = new_mask
        runtime = time.perf_counter() - t_start

        step_data = _make_step_data(step, best_col, cand_scores, rec_rss, logits, sample_mask, runtime)
        save_step_npz(out_dir / f"step_{step:03d}.npz", step_data)

        for row in cand_scores:
            row["sample_id"] = sample_id
            row["step"] = step
            row["oracle_objective"] = objective
        save_parquet(out_dir / f"candidate_scores_step_{step:03d}.parquet", cand_scores)

    np.save(str(out_dir / "final_rec_rss.npy"), rec_rss.astype(np.float32))

    final_metrics = compute_all_metrics(rec_rss, gt_rss, device)
    summary = {
        "sample_id": sample_id,
        "n_steps": n_steps,
        **{f"final_{k}": v for k, v in final_metrics.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Volume assembly
# ---------------------------------------------------------------------------

def assemble_oracle_volumes(
    run_dir: pathlib.Path,
    dataset: str,
    acceleration: int,
    objective: str,
    slice_records: list,
) -> None:
    """Gather per-slice final_rec_rss.npy files, group by scan, write H5 volumes."""
    oracle_root = (
        run_dir / "oracle_active_sampling"
        / f"dataset={dataset}" / f"R={acceleration}" / f"objective={objective}"
    )
    recon_dir = (
        run_dir / "reconstructions" / "oracle"
        / f"dataset={dataset}" / f"R={acceleration}" / f"objective={objective}"
    )
    recon_dir.mkdir(parents=True, exist_ok=True)

    by_scan: dict = collections.defaultdict(list)
    for rec in slice_records:
        scan_stem = pathlib.Path(rec["scan_id"]).stem
        by_scan[scan_stem].append(rec)

    n_written = 0
    for scan_stem, recs in sorted(by_scan.items()):
        recs_sorted = sorted(recs, key=lambda r: r["slice_index"])
        slices = []
        for rec in recs_sorted:
            sample_id = make_sample_id(dataset, rec["scan_id"], rec["slice_index"])
            npy_path = oracle_root / f"sample_id={sample_id}" / "final_rec_rss.npy"
            if not npy_path.exists():
                continue
            slices.append(np.load(str(npy_path)))
        if not slices:
            continue
        vol = np.stack(slices, axis=0)
        h5_path = recon_dir / f"{scan_stem}.h5"
        with h5py.File(str(h5_path), "w") as hf:
            hf.create_dataset("reconstruction", data=vol, compression="gzip")
        n_written += 1

    print(f"  Volumes written: {n_written} scans → {recon_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Oracle active sampling")
    p.add_argument("--config", default="experiments/experiment1/config.yaml")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--dataset", required=True, choices=["brain", "knee"])
    p.add_argument("--acceleration", required=True, type=int, choices=[8, 16])
    p.add_argument("--oracle_objective", required=True, choices=["mse", "psnr", "ssfd", "token_acc"])
    p.add_argument("--split_file", default=None,
                   help="Text file with one H5 filename per line (uses all slices per scan). "
                        "If omitted, falls back to Stage-1 selected_slices JSON.")
    p.add_argument("--num_scans", default=None, type=int,
                   help="Limit to N scans from the split file (after --scan_offset).")
    p.add_argument("--scan_offset", default=0, type=int,
                   help="Skip the first N scans in the split file (for parallelising across GPUs).")
    p.add_argument("--stratified", action="store_true",
                   help="Select scans stratified by contrast (PD/PDFS) with center slices only.")
    p.add_argument("--scans_per_contrast", default=8, type=int,
                   help="Scans to sample per contrast group when --stratified is set.")
    p.add_argument("--center_slices", default=15, type=int,
                   help="Number of center slices per scan when --stratified is set.")
    p.add_argument("--seed", default=42, type=int,
                   help="Random seed for --stratified scan selection.")
    p.add_argument("--num_slices", default=None, type=int,
                   help="(Legacy) Limit total slices when using Stage-1 JSON.")
    p.add_argument("--batch_size", default=32, type=int,
                   help="Candidate columns evaluated per forward pass.")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--meditok_config", default=None)
    p.add_argument("--meditok_ckpt", default=None)
    p.add_argument("--data_path", default=None)
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

    data_root = pathlib.Path(args.data_path or root / cfg["data_paths"][args.dataset])

    if args.split_file is not None:
        split_file = pathlib.Path(args.split_file)
        if not split_file.is_absolute():
            split_file = root / split_file
        if args.stratified:
            slice_records = select_stratified_center_slices(
                data_root, split_file,
                dataset=args.dataset, acceleration=args.acceleration,
                scans_per_contrast=args.scans_per_contrast,
                n_center_slices=args.center_slices,
                seed=args.seed,
            )
            n_scans = len({r["scan_id"] for r in slice_records})
            print(f"Stratified selection: {n_scans} scans × {args.center_slices} center slices "
                  f"= {len(slice_records)} slices  (seed={args.seed})")
        else:
            slice_records = select_all_slices_from_split(
                data_root, split_file,
                dataset=args.dataset, acceleration=args.acceleration,
                num_scans=args.num_scans, scan_offset=args.scan_offset,
            )
            print(f"Split file: {split_file.name} → {len(slice_records)} slices "
                  f"({len({r['scan_id'] for r in slice_records})} scans)")
    else:
        slices_json = (
            run_dir / "metadata"
            / f"selected_slices_{args.dataset}_R{args.acceleration}.json"
        )
        with open(slices_json) as f:
            slice_records = json.load(f)
        if args.num_slices is not None:
            slice_records = slice_records[:args.num_slices]

    print(
        f"Oracle {args.oracle_objective}: dataset={args.dataset}, R={args.acceleration}, "
        f"{len(slice_records)} slices, batch_size={args.batch_size}"
    )

    for rec in tqdm(slice_records, desc="Slices"):
        h5_path = data_root / rec["scan_id"]
        if not h5_path.exists():
            print(f"  SKIP: {h5_path}")
            continue
        try:
            run_oracle_for_slice(
                model=model,
                tokenizer=tokenizer,
                slice_rec=rec,
                h5_path=h5_path,
                dataset=args.dataset,
                acceleration=args.acceleration,
                objective=args.oracle_objective,
                run_dir=run_dir,
                cfg=cfg,
                device=device,
                batch_size=args.batch_size,
            )
        except Exception as e:
            print(f"  ERROR on {rec['scan_id']} slice {rec['slice_index']}: {e}")
            raise

    print("Assembling oracle volumes...")
    assemble_oracle_volumes(
        run_dir, args.dataset, args.acceleration, args.oracle_objective, slice_records
    )
    print("Done.")


if __name__ == "__main__":
    main()
