"""
Latency benchmark for TRUST-MRI active k-space acquisition methods.

Methods:
  FAR  — baseline full reconstruction (no active acquisition)
  LES  — Logit-based Entropy Spectral uncertainty
  GLES — Gradient-weighted Latent Entropy Selection
  TDS  — Token Distribution Shift

Hardware:  CUDA Events (GPU) or time.perf_counter (CPU).
Protocol:  N_WARMUP warm-up iterations, N_TRIALS timed trials, batch_size=1.
Input:     One real dataset slice loaded once and reused every trial.

Metrics per method:
  step_latency_ms  — one [Infer → Score → Mask Update] cycle (budget==rank)
  total_runtime_s  — full acquisition sequence from center mask to final recon
  throughput_fps   — slices per second (1 / total_runtime_s)

Usage:
  python scripts/benchmark_timing.py \\
      --dataset knee --acceleration 8 \\
      --budget 8 --rank 1 \\
      --n-warmup 10 --n-trials 50 \\
      --output results/timing/knee_R8.txt
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import contextlib
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data import transforms
from autoregressive.models.generate import (
    reconstruct_far,
    data_consistency,
)
from autoregressive.sample.adaptive_sampling import (
    kspace_uncertainty_from_patch_logits,
    geo_step,
    gles_step,
    posterior_sampling_scores_kspace,
)
from autoregressive.sample.tds import tds_score_step
from experiments.experiment1.utils.model_loader import load_singlecoil_model, load_meditok
from experiments.experiment1.run_normal_trajectory import load_slice, make_center_mask
from experiments.experiment1.utils.slice_selector import select_slices_center_out


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

class CUDATimer:
    """CUDA-event timer as a context manager. Falls back to perf_counter on CPU."""

    def __init__(self, device: torch.device):
        self.device = device
        self._use_cuda = device.type == "cuda"
        if self._use_cuda:
            self._s = torch.cuda.Event(enable_timing=True)
            self._e = torch.cuda.Event(enable_timing=True)
        self._t0 = 0.0
        self._t1 = 0.0

    def __enter__(self):
        if self._use_cuda:
            torch.cuda.synchronize(self.device)
            self._s.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        if self._use_cuda:
            self._e.record()
            torch.cuda.synchronize(self.device)
        else:
            self._t1 = time.perf_counter()

    @property
    def ms(self) -> float:
        if self._use_cuda:
            return self._s.elapsed_time(self._e)
        return (self._t1 - self._t0) * 1e3


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_benchmark_slice(
    data_root: pathlib.Path,
    split_file: pathlib.Path,
    dataset: str,
    acceleration: int,
    center_fraction: float,
    resolution: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load one real slice and return (gt_c_image, zf_image, center_mask).

    Returns:
        gt:     (1, H, W, 2) fully-sampled complex image
        masked: (1, H, W, 2) zero-filled input from center mask only
        mask:   (1, H, W, 2) center-fraction acquisition mask
    """
    slices = select_slices_center_out(
        data_root=data_root, split_file=split_file,
        dataset=dataset, acceleration=acceleration,
        num_slices=1,
    )
    if not slices:
        raise RuntimeError(f"No slices found in {split_file}")

    rec = slices[0]
    h5_path = data_root / rec["scan_id"]
    sd = load_slice(h5_path, rec["slice_index"], resolution)
    gt = sd["gt_c_image"].to(device)                      # (1, H, W, 2)
    _, H, W, _ = gt.shape
    mask = make_center_mask(W, center_fraction, device)    # (1, H, W, 2)
    masked = data_consistency(gt, torch.zeros_like(gt), mask)
    print(f"Loaded: {rec['scan_id']}  slice={rec['slice_index']}  shape={tuple(gt.shape)}")
    return gt, masked, mask


# ---------------------------------------------------------------------------
# Shared forward-pass kwargs
# ---------------------------------------------------------------------------

def _far_kwargs(center_fraction: float) -> dict:
    return dict(
        n_steps=1,
        far_mask_mode="cartesian",
        center_fractions=center_fraction,
        active_sampling_mask_type="cartesian",
    )


# ---------------------------------------------------------------------------
# FAR benchmark
# ---------------------------------------------------------------------------

@torch.no_grad()
def _time_far(model, tok, masked, mask, kw, device) -> float:
    with CUDATimer(device) as t:
        reconstruct_far(model, tok, masked.clone(), mask.clone(), **kw)
    return t.ms


# ---------------------------------------------------------------------------
# LES benchmark
# ---------------------------------------------------------------------------

@torch.no_grad()
def _les_step(model, tok, masked, mask, kw, device, rank) -> torch.Tensor:
    """One LES step: infer → entropy → mask update. Returns updated mask."""
    _, aux = reconstruct_far(model, tok, masked, mask, **kw)
    logits = aux["logits"]                                  # (1, H_p, W_p, 2, K, V)
    new_mask_perm, _, _ = kspace_uncertainty_from_patch_logits(
        logits,
        mask.permute(0, 3, 1, 2),                          # (1, 2, H, W)
        vertical_mask=True,
        rank=rank,
        les_zero_mean=False,
        les_radial_weight_alpha=0.0,
        mask_type="cartesian",
    )
    return new_mask_perm.permute(0, 2, 3, 1).float()       # (1, H, W, 2)


@torch.no_grad()
def _time_les_step(model, tok, masked, mask, kw, device, rank) -> float:
    with CUDATimer(device) as t:
        _les_step(model, tok, masked.clone(), mask.clone(), kw, device, rank)
    return t.ms


@torch.no_grad()
def _time_les_total(model, tok, gt, masked, mask, budget, rank, kw, device) -> float:
    gt_kspace = transforms.fft2(gt)
    masked = masked.clone()
    mask = mask.clone()
    with CUDATimer(device) as t:
        for _ in range(budget // rank):
            mask = _les_step(model, tok, masked, mask, kw, device, rank)
            masked = transforms.ifft2(gt_kspace * mask)
        reconstruct_far(model, tok, masked, mask, **kw)
    return t.ms


# ---------------------------------------------------------------------------
# GLES benchmark
# ---------------------------------------------------------------------------

def _time_gles_step(model, tok, masked, mask, device, rank) -> float:
    with CUDATimer(device) as t:
        gles_step(model, tok, masked.clone(), mask.clone(), rank, device)
    return t.ms


def _time_gles_total(model, tok, gt, masked, mask, budget, rank, kw, device) -> float:
    gt_kspace = transforms.fft2(gt)
    masked = masked.clone()
    mask = mask.clone()
    with CUDATimer(device) as t:
        for _ in range(budget // rank):
            mask, _ = gles_step(model, tok, masked, mask, rank, device)
            masked = transforms.ifft2(gt_kspace * mask)
        with torch.no_grad():
            reconstruct_far(model, tok, masked, mask, **kw)
    return t.ms


# ---------------------------------------------------------------------------
# GEO benchmark
# ---------------------------------------------------------------------------

def _time_geo_step(model, tok, masked, mask, device, rank) -> float:
    with CUDATimer(device) as t:
        geo_step(model, tok, masked.clone(), mask.clone(), rank, device)
    return t.ms


def _time_geo_total(model, tok, gt, masked, mask, budget, rank, kw, device) -> float:
    gt_kspace = transforms.fft2(gt)
    masked = masked.clone()
    mask = mask.clone()
    with CUDATimer(device) as t:
        for _ in range(budget // rank):
            mask, _ = geo_step(model, tok, masked, mask, rank, device)
            masked = transforms.ifft2(gt_kspace * mask)
        with torch.no_grad():
            reconstruct_far(model, tok, masked, mask, **kw)
    return t.ms


# ---------------------------------------------------------------------------
# PS_kspace benchmark
# ---------------------------------------------------------------------------

_PS_SAMPLING_KWARGS = {"temperature": 1.0, "top_k": 100, "top_p": 1.0}


def _ps_kspace_step(model, tok, masked, mask, kw, device, n_samples, rank) -> torch.Tensor:
    masked_rep = masked.expand(n_samples, -1, -1, -1).contiguous()
    mask_rep   = mask.expand(n_samples, -1, -1, -1).contiguous()
    with torch.no_grad():
        rec_batch, _ = reconstruct_far(model, tok, masked_rep, mask_rep,
                                       sampling_kwargs=_PS_SAMPLING_KWARGS, **kw)
    sel_cols, _ = posterior_sampling_scores_kspace(rec_batch, mask, rank=rank)
    new_mask = mask.clone()
    for c in sel_cols:
        new_mask[0, :, c.item(), :] = 1.0
    return new_mask


def _time_ps_kspace_step(model, tok, masked, mask, kw, device, n_samples, rank) -> float:
    with CUDATimer(device) as t:
        _ps_kspace_step(model, tok, masked.clone(), mask.clone(), kw, device, n_samples, rank)
    return t.ms


def _time_ps_kspace_total(model, tok, gt, masked, mask, budget, rank, kw, device, n_samples) -> float:
    gt_kspace = transforms.fft2(gt)
    masked = masked.clone()
    mask = mask.clone()
    with CUDATimer(device) as t:
        for _ in range(budget // rank):
            mask = _ps_kspace_step(model, tok, masked, mask, kw, device, n_samples, rank)
            masked = transforms.ifft2(gt_kspace * mask)
        with torch.no_grad():
            reconstruct_far(model, tok, masked, mask, **kw)
    return t.ms


# ---------------------------------------------------------------------------
# TDS benchmark
# ---------------------------------------------------------------------------

_TDS_DEFAULTS = dict(
    tds_distance="js",
    tds_foreground_mode="percentile",
    tds_foreground_percentile=20.0,
    tds_alpha=1.0,
    tds_beta=0.0,
    tds_candidate_mode="all",
    tds_candidate_batch_size=16,
    tds_entropy_normalization=True,
    tds_line_aggregation="sum",
)


@torch.no_grad()
def _tds_step(model, tok, gt, masked, mask, kw, device) -> int:
    """One TDS step: infer → score candidates → select column. Returns chosen column."""
    _, aux = reconstruct_far(model, tok, masked, mask, **kw)
    x_hat = aux["decoded_no_dc"]
    logits = aux["logits"]
    y_omega = transforms.fft2(gt) * mask
    result = tds_score_step(
        model, tok, y_omega, mask, x_hat=x_hat, logits_old=logits,
        device=device, sampling_kwargs={"temperature": 1.0, "top_k": 0, "top_p": 1.0},
        **_TDS_DEFAULTS,
    )
    return result["selected_col"]


def _time_tds_step(model, tok, gt, masked, mask, kw, device) -> float:
    with CUDATimer(device) as t:
        _tds_step(model, tok, gt.clone(), masked.clone(), mask.clone(), kw, device)
    return t.ms


@torch.no_grad()
def _time_tds_total(model, tok, gt, masked, mask, budget, rank, kw, device) -> float:
    gt_kspace = transforms.fft2(gt)
    masked = masked.clone()
    mask = mask.clone()
    with CUDATimer(device) as t:
        for _ in range(budget // rank):
            col = _tds_step(model, tok, gt, masked, mask, kw, device)
            upd = torch.zeros_like(mask)
            upd[:, :, col, :] = 1.0
            mask = (mask + upd).clamp_(0.0, 1.0)
            masked = transforms.ifft2(gt_kspace * mask)
        reconstruct_far(model, tok, masked, mask, **kw)
    return t.ms


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(model, tok, gt, masked, mask, args, device) -> dict:
    budget = args.budget
    rank = args.rank
    kw = _far_kwargs(args.center_fraction)
    n_warmup = args.n_warmup
    n_trials = args.n_trials

    n_samples = args.ps_n_samples

    def _make_methods():
        return {
            "FAR": {
                "step": lambda: _time_far(model, tok, masked, mask, kw, device),
                "total": lambda: _time_far(model, tok, masked, mask, kw, device),
            },
            "LES": {
                "step":  lambda: _time_les_step(model, tok, masked, mask, kw, device, rank),
                "total": lambda: _time_les_total(model, tok, gt, masked, mask, budget, rank, kw, device),
            },
            "GEO": {
                "step":  lambda: _time_geo_step(model, tok, masked, mask, device, rank),
                "total": lambda: _time_geo_total(model, tok, gt, masked, mask, budget, rank, kw, device),
            },
            "GLES": {
                "step":  lambda: _time_gles_step(model, tok, masked, mask, device, rank),
                "total": lambda: _time_gles_total(model, tok, gt, masked, mask, budget, rank, kw, device),
            },
            "PS_kspace": {
                "step":  lambda: _time_ps_kspace_step(model, tok, masked, mask, kw, device, n_samples, rank),
                "total": lambda: _time_ps_kspace_total(model, tok, gt, masked, mask, budget, rank, kw, device, n_samples),
            },
            "TDS": {
                "step":  lambda: _time_tds_step(model, tok, gt, masked, mask, kw, device),
                "total": lambda: _time_tds_total(model, tok, gt, masked, mask, budget, rank, kw, device),
            },
        }

    results = {}
    for name, fns in _make_methods().items():
        if args.methods and name not in args.methods:
            continue

        print(f"\n{'─'*56}")
        print(f"  {name}  |  warm-up={n_warmup}  trials={n_trials}  "
              f"budget={budget}  rank={rank}")
        print(f"{'─'*56}")

        print("  [warm-up]", end="", flush=True)
        for _ in range(n_warmup):
            fns["step"]()
            fns["total"]()
        print(" done")

        step_ms, total_ms = [], []
        for i in range(n_trials):
            step_ms.append(fns["step"]())
            total_ms.append(fns["total"]())
            if (i + 1) % 10 == 0:
                print(f"  trial {i+1:3d}/{n_trials}"
                      f"  step={step_ms[-1]:.1f} ms"
                      f"  total={total_ms[-1]/1e3:.3f} s")

        s = np.array(step_ms)
        r = np.array(total_ms) / 1e3
        fps = 1.0 / r
        results[name] = {
            "step_latency_ms": {"mean": float(s.mean()),   "std": float(s.std()),
                                "p50": float(np.median(s)), "p95": float(np.percentile(s, 95))},
            "total_runtime_s": {"mean": float(r.mean()),   "std": float(r.std()),
                                "p50": float(np.median(r)), "p95": float(np.percentile(r, 95))},
            "throughput_fps":  {"mean": float(fps.mean()), "std": float(fps.std())},
        }

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_table(results: dict, args) -> str:
    import datetime
    lines = []
    lines.append(f"Timestamp    : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Dataset      : {args.dataset}  acceleration={args.acceleration}")
    lines.append(f"Checkpoint   : {args.checkpoint}")
    lines.append(f"Budget       : {args.budget}  rank={args.rank}  cf={args.center_fraction}")
    lines.append(f"Trials       : {args.n_trials}  warm-up={args.n_warmup}")
    lines.append(f"TDS cand.mode: {_TDS_DEFAULTS['tds_candidate_mode']}"
                 f"  batch={_TDS_DEFAULTS['tds_candidate_batch_size']}")
    lines.append("")

    col = 30
    header = (f"{'Method':<6} | "
              f"{'Step Latency (ms)':^{col}} | "
              f"{'Total Runtime (s)':^{col}} | "
              f"{'Throughput (fps)':^{col}}")
    sep = "─" * len(header)
    lines += [sep, header, sep]
    for method, m in results.items():
        sl = m["step_latency_ms"]
        tr = m["total_runtime_s"]
        tp = m["throughput_fps"]
        lines.append(
            f"{method:<6} | "
            f"{sl['mean']:>8.2f} ± {sl['std']:>6.2f} ms  "
            f"(p95={sl['p95']:>7.2f})   | "
            f"{tr['mean']:>8.4f} ± {tr['std']:>6.4f} s  "
            f"(p95={tr['p95']:>7.4f})   | "
            f"{tp['mean']:>7.3f} ± {tp['std']:>5.3f}"
        )
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Active sampling latency benchmark")
    p.add_argument("--dataset",       required=True, choices=["brain", "knee"])
    p.add_argument("--acceleration",  required=True, type=int, choices=[8, 16])
    p.add_argument("--checkpoint",    default=None)
    p.add_argument("--meditok-config", default=None)
    p.add_argument("--meditok-ckpt",   default=None)
    p.add_argument("--data-path",      default=None)
    p.add_argument("--split-file",     default=None)
    p.add_argument("--config",        default="experiments/experiment1/config.yaml")
    p.add_argument("--budget",        type=int, required=True,
                   help="Total number of k-space lines to acquire")
    p.add_argument("--rank",          type=int, default=1,
                   help="Lines acquired per step")
    p.add_argument("--n-trials",      type=int, default=50)
    p.add_argument("--n-warmup",      type=int, default=10)
    p.add_argument("--methods",       nargs="+", default=None,
                   choices=["FAR", "LES", "GEO", "GLES", "PS_kspace", "TDS"],
                   help="Subset of methods to benchmark (default: all)")
    p.add_argument("--ps-n-samples",  type=int, default=8,
                   help="Posterior samples per PS_kspace step (default: 8)")
    p.add_argument("--output",        default=None,
                   help="Path to write the text table (default: stdout only)")
    p.add_argument("--output-json",   default=None,
                   help="Path to write full results as JSON")
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main() -> None:
    args = build_parser().parse_args()

    # Load config
    cfg_path = pathlib.Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = _ROOT / cfg_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    args.center_fraction = float(cfg.get("center_fraction", 0.04))

    # Resolve paths from config
    ckpt_path = args.checkpoint or (_ROOT / cfg["checkpoints"][args.dataset][args.acceleration])
    args.checkpoint = str(ckpt_path)
    mt_cfg  = args.meditok_config or (_ROOT / cfg["meditok_config"])
    mt_ckpt = args.meditok_ckpt   or (_ROOT / cfg["meditok_ckpt"])
    data_root  = pathlib.Path(args.data_path   or (_ROOT / cfg["data_paths"][args.dataset]))
    split_file = pathlib.Path(args.split_file  or (_ROOT / cfg["split_files"][args.dataset]))

    # Load models
    model, ckpt_args = load_singlecoil_model(ckpt_path, device)
    resolution = getattr(ckpt_args, "image_size", 256)
    tok = load_meditok(mt_cfg, mt_ckpt, device)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load one slice
    gt, masked, mask = load_benchmark_slice(
        data_root, split_file, args.dataset, args.acceleration,
        args.center_fraction, resolution, device,
    )

    # Run benchmark
    results = run_benchmark(model, tok, gt, masked, mask, args, device)

    # Print and save
    table = format_table(results, args)
    print(f"\n{table}\n")

    if args.output:
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table + "\n")
        print(f"Saved table → {out_path}")

    if args.output_json:
        json_path = pathlib.Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results, indent=2))
        print(f"Saved JSON  → {json_path}")


if __name__ == "__main__":
    main()
