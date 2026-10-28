#!/usr/bin/env python
"""Analysis 7: Oracle Active-Sampling Comparison.

Aggregates final-step metrics across:
    - Normal trajectories: LES, GEO
    - Oracle trajectories: oracle_mse, oracle_ssfd, oracle_token_acc

Computes gap to oracle and per-step metric curves.

Writes:
    <run_dir>/results/tables/oracle_comparison_{dataset}_R{acc}.csv
    <run_dir>/results/tables/oracle_per_step_{dataset}_R{acc}.csv
    <run_dir>/results/plots/oracle_comparison_*.png

Usage:
    python analyze_oracle_active_sampling.py \\
        --run_dir /fast/lev/TRUST-MRI/results/exp1 \\
        --dataset brain --acceleration 8
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.experiment1.utils.storage import load_step_npz


METRIC_COLS = ["psnr", "ssim", "nmse", "ssfd", "token_accuracy", "token_nll", "total_entropy"]
METHODS_NORMAL = ["LES", "GEO"]
ORACLE_OBJS    = ["mse", "ssfd", "token_acc"]


def _load_traj_per_step(
    base_dir: pathlib.Path,
    method_or_obj: str,
    kind: str,  # "normal" or "oracle"
) -> pd.DataFrame:
    """Load all step npz files and return a per-step dataframe."""
    rows = []
    step_files = sorted(base_dir.glob("sample_id=*/step_*.npz"))
    for sf in tqdm(step_files, desc=f"{kind}/{method_or_obj}", leave=False):
        try:
            data = load_step_npz(sf)
        except Exception:
            continue
        step = int(sf.stem.split("_")[1])
        sample_id = sf.parent.name.replace("sample_id=", "")
        row = {"sample_id": sample_id, "step": step, "method": method_or_obj}
        for m in METRIC_COLS:
            row[m] = float(data.get(m, float("nan")))
        rows.append(row)
    return pd.DataFrame(rows)


def analyze(
    run_dir: pathlib.Path,
    dataset: str,
    acceleration: int,
) -> None:
    all_per_step = []

    # Normal trajectories
    for method in METHODS_NORMAL:
        traj_root = (
            run_dir / "normal_trajectory"
            / f"dataset={dataset}" / f"R={acceleration}" / f"method={method}"
        )
        if not traj_root.exists():
            continue
        df = _load_traj_per_step(traj_root, method, "normal")
        all_per_step.append(df)

    # Oracle trajectories
    for obj in ORACLE_OBJS:
        oracle_root = (
            run_dir / "oracle_active_sampling"
            / f"dataset={dataset}" / f"R={acceleration}" / f"objective={obj}"
        )
        if not oracle_root.exists():
            continue
        df = _load_traj_per_step(oracle_root, f"oracle_{obj}", "oracle")
        all_per_step.append(df)

    if not all_per_step:
        print("No trajectories found.")
        return

    per_step_df = pd.concat(all_per_step, ignore_index=True)
    per_step_df["dataset"] = dataset
    per_step_df["acceleration"] = acceleration

    # Per-step mean metrics
    per_step_mean = (
        per_step_df
        .groupby(["method", "step"])[METRIC_COLS]
        .mean()
        .reset_index()
    )
    per_step_mean["dataset"] = dataset
    per_step_mean["acceleration"] = acceleration

    out_dir = run_dir / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_step_mean.to_csv(
        out_dir / f"oracle_per_step_{dataset}_R{acceleration}.csv", index=False
    )

    # Final-step summary table
    final_step = per_step_df.groupby("method")["step"].max().reset_index()
    final_rows = []
    for _, row in final_step.iterrows():
        method = row["method"]
        step   = row["step"]
        sub = per_step_df[(per_step_df["method"] == method) & (per_step_df["step"] == step)]
        means = sub[METRIC_COLS].mean()
        final_rows.append({"method": method, "step": step, **means.to_dict()})

    final_df = pd.DataFrame(final_rows)
    final_df["dataset"] = dataset
    final_df["acceleration"] = acceleration

    # Compute gap to best oracle for each metric
    oracle_methods = [f"oracle_{o}" for o in ORACLE_OBJS]
    for m in ["psnr", "ssim"]:  # higher is better
        oracle_vals = final_df[final_df["method"].isin(oracle_methods)][m]
        if oracle_vals.notna().any():
            best_oracle = oracle_vals.max()
            final_df[f"gap_to_oracle_{m}"] = best_oracle - final_df[m]
    for m in ["nmse", "ssfd"]:  # lower is better
        oracle_vals = final_df[final_df["method"].isin(oracle_methods)][m]
        if oracle_vals.notna().any():
            best_oracle = oracle_vals.min()
            final_df[f"gap_to_oracle_{m}"] = final_df[m] - best_oracle

    final_df.to_csv(
        out_dir / f"oracle_comparison_{dataset}_R{acceleration}.csv", index=False
    )
    print(f"Oracle comparison saved for {dataset} R={acceleration}")

    _plot(per_step_mean, final_df, run_dir, dataset, acceleration)


def _plot(per_step_mean, final_df, run_dir, dataset, acc):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plot_dir = run_dir / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{dataset}_R{acc}"

    metrics_to_plot = ["psnr", "ssim", "nmse", "ssfd", "token_accuracy"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 4))

    all_methods = per_step_mean["method"].unique()
    cmap = plt.get_cmap("tab10")
    color_map = {m: cmap(i) for i, m in enumerate(all_methods)}
    style_map = {
        "LES": "-",
        "GEO": "--",
        "oracle_mse": (0, (3, 1, 1, 1)),
        "oracle_ssfd": (0, (5, 1)),
        "oracle_token_acc": (0, (1, 1)),
    }

    for ax, metric in zip(axes, metrics_to_plot):
        for method in all_methods:
            sub = per_step_mean[per_step_mean["method"] == method].sort_values("step")
            if sub.empty:
                continue
            ls = style_map.get(method, "-")
            ax.plot(
                sub["step"], sub[metric],
                label=method, color=color_map[method],
                linestyle=ls, marker="o" if len(sub) < 30 else None, markersize=3,
            )
        ax.set_xlabel("Step")
        ax.set_ylabel(metric.upper())
        ax.set_title(metric)

    axes[-1].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{dataset} R={acc} — Oracle Comparison")
    plt.tight_layout()
    plt.savefig(plot_dir / f"oracle_comparison_{tag}.png", dpi=120)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--acceleration", required=True, type=int)
    return p.parse_args()


def main():
    args = parse_args()
    analyze(pathlib.Path(args.run_dir), args.dataset, args.acceleration)


if __name__ == "__main__":
    main()
