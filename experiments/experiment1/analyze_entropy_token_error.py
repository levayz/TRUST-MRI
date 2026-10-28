#!/usr/bin/env python
"""Analysis 1: Token Entropy vs Token Prediction Error.

Loads normal-trajectory .npz files and computes:
    - Spearman correlation between per-token entropy and token error (per step)
    - AUROC for entropy predicting token error (per step)
    - Token error rate by entropy quartile (per step)

Writes:
    <run_dir>/results/tables/entropy_token_error_{dataset}_R{acc}_{method}.csv
    <run_dir>/results/plots/entropy_token_error_*.png

Usage:
    python analyze_entropy_token_error.py \\
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


def load_all_steps(traj_root: pathlib.Path) -> list[dict]:
    """Load all step npz files under traj_root, returning flat list of records."""
    records = []
    step_files = sorted(traj_root.glob("sample_id=*/step_*.npz"))
    for sf in tqdm(step_files, desc="Loading steps"):
        data = load_step_npz(sf)
        step = int(sf.stem.split("_")[1])
        sample_id = sf.parent.name.replace("sample_id=", "")
        records.append({
            "sample_id": sample_id,
            "step": step,
            "token_entropy": data.get("token_entropy"),     # (H_p, W_p, 2, K) float
            "pred_tokens":   data.get("pred_tokens"),       # (H_p, W_p, 2, K) int
            "gt_tokens":     data.get("gt_tokens"),         # (H_p, W_p, 2, K) int
        })
    return records


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if len(x) < 2:
        return float("nan")
    r, _ = spearmanr(x, y)
    return float(r)


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


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
        print(f"No trajectory found at {traj_root}")
        return

    records = load_all_steps(traj_root)
    if not records:
        print("No step files found.")
        return

    # Group by step
    by_step: dict[int, list] = defaultdict(list)
    for rec in records:
        if rec["token_entropy"] is None or rec["pred_tokens"] is None:
            continue
        by_step[rec["step"]].append(rec)

    rows = []
    for step in sorted(by_step.keys()):
        step_recs = by_step[step]
        H_all, E_all = [], []
        for rec in step_recs:
            ent  = rec["token_entropy"].astype(np.float32).mean(axis=-1).ravel()  # mean over K
            err  = (rec["pred_tokens"] != rec["gt_tokens"]).astype(np.float32).mean(axis=-1).ravel()
            H_all.append(ent)
            E_all.append(err)

        H_all = np.concatenate(H_all)
        E_all = np.concatenate(E_all)

        spearman_r = _spearman(H_all, E_all)
        auroc = _auroc(H_all, (E_all > 0).astype(int))

        # Error rate by entropy quartile
        q_edges = np.quantile(H_all, [0.0, 0.25, 0.5, 0.75, 1.0])
        quartile_error_rates = []
        for q in range(4):
            lo, hi = q_edges[q], q_edges[q + 1]
            mask = (H_all >= lo) & (H_all <= hi)
            if mask.sum() > 0:
                quartile_error_rates.append(float(E_all[mask].mean()))
            else:
                quartile_error_rates.append(float("nan"))

        rows.append({
            "dataset":      dataset,
            "acceleration": acceleration,
            "method":       method,
            "step":         step,
            "n_tokens":     len(H_all),
            "spearman_r":   spearman_r,
            "auroc":        auroc,
            "err_q1":       quartile_error_rates[0],
            "err_q2":       quartile_error_rates[1],
            "err_q3":       quartile_error_rates[2],
            "err_q4":       quartile_error_rates[3],
            "mean_entropy": float(H_all.mean()),
            "mean_token_error": float(E_all.mean()),
        })

    df = pd.DataFrame(rows)
    out_dir = run_dir / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"entropy_token_error_{dataset}_R{acceleration}_{method}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows to {csv_path}")

    # Plot
    _plot(df, run_dir, dataset, acceleration, method)


def _plot(df: pd.DataFrame, run_dir: pathlib.Path, dataset: str, acc: int, method: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plot_dir = run_dir / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{dataset}_R{acc}_{method}"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(df["step"], df["spearman_r"], marker="o")
    axes[0].set_xlabel("Acquisition step")
    axes[0].set_ylabel("Spearman r")
    axes[0].set_title("Entropy vs Token Error (Spearman)")
    axes[0].axhline(0, ls="--", color="gray")

    axes[1].plot(df["step"], df["auroc"], marker="o", color="green")
    axes[1].set_xlabel("Acquisition step")
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("Entropy → Token Error AUROC")
    axes[1].axhline(0.5, ls="--", color="gray")

    for q_col, label in zip(
        ["err_q1", "err_q2", "err_q3", "err_q4"],
        ["Q1 (low H)", "Q2", "Q3", "Q4 (high H)"],
    ):
        axes[2].plot(df["step"], df[q_col], marker="o", label=label)
    axes[2].set_xlabel("Acquisition step")
    axes[2].set_ylabel("Token error rate")
    axes[2].set_title("Error rate by entropy quartile")
    axes[2].legend(fontsize=8)

    fig.suptitle(f"{dataset} R={acc} {method}")
    plt.tight_layout()
    plt.savefig(plot_dir / f"entropy_token_error_{tag}.png", dpi=120)
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
