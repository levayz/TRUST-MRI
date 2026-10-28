#!/usr/bin/env python
"""Analysis 2: Token Probability Calibration.

Loads top-k compressed token probabilities from normal trajectory and computes:
    - ECE (Expected Calibration Error)
    - Token NLL
    - Token accuracy
    - Mean confidence
    - Reliability curves for initial / middle / final steps

Writes:
    <run_dir>/results/tables/calibration_{dataset}_R{acc}_{method}.csv
    <run_dir>/results/plots/reliability_{tag}_{step}.png

Usage:
    python analyze_calibration.py \\
        --run_dir /fast/lev/TRUST-MRI/results/exp1 \\
        --dataset brain --acceleration 8 --method LES
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.experiment1.utils.storage import load_step_npz


def _ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc  = correct[mask].mean()
        ece += mask.sum() / n * abs(bin_acc - bin_conf)
    return float(ece)


def _reliability_curve(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (bin_centers, mean_conf_per_bin, accuracy_per_bin)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_confs, accs = [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            mean_confs.append(float("nan"))
            accs.append(float("nan"))
        else:
            mean_confs.append(float(confidences[mask].mean()))
            accs.append(float(correct[mask].mean()))
    return bin_centers, np.array(mean_confs), np.array(accs)


def load_step_token_stats(sf: pathlib.Path) -> dict | None:
    """Load per-token confidence, correct, accuracy, nll from a step file."""
    data = load_step_npz(sf)
    pred_tokens = data.get("pred_tokens")
    gt_tokens   = data.get("gt_tokens")
    topk_probs  = data.get("topk_probs")   # (H_p, W_p, 2, K, top_k)
    topk_idx    = data.get("topk_indices") # (H_p, W_p, 2, K, top_k)

    if pred_tokens is None or gt_tokens is None:
        return None

    correct = (pred_tokens.astype(np.int32) == gt_tokens.astype(np.int32)).ravel()

    if topk_probs is not None and topk_idx is not None:
        # Confidence = max probability (first entry of topk is the highest)
        confidence = topk_probs[..., 0].ravel().astype(np.float32)

        # NLL: look up p(gt_token) from top-k; if gt not in top-k use p≈0
        # Shape: (N, top_k)
        probs_flat = topk_probs.reshape(-1, topk_probs.shape[-1]).astype(np.float32)
        idx_flat   = topk_idx.reshape(-1, topk_idx.shape[-1]).astype(np.int32)
        gt_flat    = gt_tokens.ravel().astype(np.int32)
        # Find gt token in top-k
        nll_vals = []
        for i in range(len(gt_flat)):
            match = np.where(idx_flat[i] == gt_flat[i])[0]
            if len(match) > 0:
                p_gt = float(probs_flat[i, match[0]])
            else:
                p_gt = 1e-9
            nll_vals.append(-np.log(p_gt + 1e-12))
        nll = np.array(nll_vals, dtype=np.float32)
    else:
        confidence = None
        nll = None

    return {
        "correct":    correct,
        "confidence": confidence,
        "nll":        nll,
        "token_accuracy": float(data.get("token_accuracy", correct.mean())),
        "token_nll":      float(data.get("token_nll", float("nan"))),
    }


def analyze(
    run_dir: pathlib.Path,
    dataset: str,
    acceleration: int,
    method: str,
) -> None:
    traj_root = (
        run_dir / "normal_trajectory"
        / f"dataset={dataset}" / f"R={acceleration}" / f"method={method}"
    )
    if not traj_root.exists():
        print(f"No trajectory at {traj_root}")
        return

    step_files = sorted(traj_root.glob("sample_id=*/step_*.npz"))
    if not step_files:
        print("No step files found.")
        return

    # Group by step
    by_step: dict[int, list] = defaultdict(list)
    for sf in step_files:
        step = int(sf.stem.split("_")[1])
        by_step[step].append(sf)

    n_steps = max(by_step.keys()) + 1
    target_for_curves = {0, n_steps // 2, max(by_step.keys())}

    rows = []
    for step in tqdm(sorted(by_step.keys()), desc="Steps"):
        all_conf, all_correct, all_nll = [], [], []
        tok_acc_vals, tok_nll_vals = [], []

        for sf in by_step[step]:
            stats = load_step_token_stats(sf)
            if stats is None:
                continue
            all_correct.append(stats["correct"])
            tok_acc_vals.append(stats["token_accuracy"])
            tok_nll_vals.append(stats["token_nll"])
            if stats["confidence"] is not None:
                all_conf.append(stats["confidence"])
            if stats["nll"] is not None:
                all_nll.append(stats["nll"])

        if not all_correct:
            continue

        correct_all = np.concatenate(all_correct)
        mean_tok_acc = float(np.mean(tok_acc_vals))
        mean_tok_nll = float(np.mean(tok_nll_vals))

        ece = float("nan")
        mean_conf = float("nan")
        if all_conf:
            conf_all = np.concatenate(all_conf)
            ece = _ece(conf_all, correct_all)
            mean_conf = float(conf_all.mean())

            # Reliability curves for selected steps
            if step in target_for_curves:
                bc, mc, ac = _reliability_curve(conf_all, correct_all)
                _plot_reliability(
                    bc, mc, ac,
                    run_dir, dataset, acceleration, method, step,
                )

        rows.append({
            "dataset":       dataset,
            "acceleration":  acceleration,
            "method":        method,
            "step":          step,
            "ece":           ece,
            "token_nll":     mean_tok_nll,
            "token_accuracy": mean_tok_acc,
            "mean_confidence": mean_conf,
        })

    df = pd.DataFrame(rows)
    out_dir = run_dir / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"calibration_{dataset}_R{acceleration}_{method}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")

    _plot_summary(df, run_dir, dataset, acceleration, method)


def _plot_reliability(bc, mc, ac, run_dir, dataset, acc, method, step):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plot_dir = run_dir / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    valid = ~np.isnan(mc) & ~np.isnan(ac)
    ax.plot(mc[valid], ac[valid], marker="o", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability — {dataset} R={acc} {method} step={step}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f"reliability_{dataset}_R{acc}_{method}_step{step:03d}.png", dpi=120)
    plt.close()


def _plot_summary(df, run_dir, dataset, acc, method):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plot_dir = run_dir / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].plot(df["step"], df["ece"], marker="o")
    axes[0].set_title("ECE vs step")
    axes[1].plot(df["step"], df["token_nll"], marker="o", color="red")
    axes[1].set_title("Token NLL vs step")
    axes[2].plot(df["step"], df["token_accuracy"], marker="o", color="green")
    axes[2].plot(df["step"], df["mean_confidence"], marker="^", color="blue", ls="--")
    axes[2].set_title("Accuracy vs Confidence")
    axes[2].legend(["Accuracy", "Confidence"])
    axes[3].axis("off")
    fig.suptitle(f"{dataset} R={acc} {method}")
    plt.tight_layout()
    tag = f"{dataset}_R{acc}_{method}"
    plt.savefig(plot_dir / f"calibration_summary_{tag}.png", dpi=120)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--acceleration", required=True, type=int)
    p.add_argument("--method", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    analyze(pathlib.Path(args.run_dir), args.dataset, args.acceleration, args.method)


if __name__ == "__main__":
    main()
