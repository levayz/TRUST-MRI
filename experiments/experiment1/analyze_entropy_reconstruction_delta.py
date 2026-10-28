#!/usr/bin/env python
"""Analysis: Entropy Reduction vs Reconstruction Improvement.

For each acquisition step transition (t → t+1), computes:
    Δentropy  = total_entropy[t] - total_entropy[t+1]   (positive = entropy dropped)
    ΔPSNR     = psnr[t+1]  - psnr[t]                   (positive = improved)
    ΔSSIM     = ssim[t+1]  - ssim[t]
    ΔNMSE     = nmse[t]    - nmse[t+1]                  (positive = improved)

Then measures Spearman ρ(Δentropy, Δreconstruction) across all slices × steps.
A high ρ means: acquiring a line that reduces entropy more also improves
reconstruction more — validating entropy as an acquisition signal.

Writes:
    <run_dir>/results/tables/entropy_recon_delta_{dataset}_R{acc}_{method}.csv
    <run_dir>/results/plots/entropy_recon_delta_{dataset}_R{acc}_{method}.png

Usage:
    python analyze_entropy_reconstruction_delta.py \\
        --run_dir results/exp1_100slices \\
        --dataset brain --acceleration 8 --method LES
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.experiment1.utils.storage import load_step_npz

RECON_METRICS = ["psnr", "ssim", "nmse", "ssfd"]


def build_delta_df(traj_root: pathlib.Path) -> pd.DataFrame:
    """Load all step files, group by slice, compute consecutive deltas."""
    step_files = sorted(traj_root.glob("sample_id=*/step_*.npz"))
    if not step_files:
        return pd.DataFrame()

    # Group by sample_id
    by_slice: dict[str, list] = {}
    for sf in tqdm(step_files, desc="loading", leave=False):
        sample_id = sf.parent.name.replace("sample_id=", "")
        try:
            data = load_step_npz(sf)
        except Exception:
            continue
        by_slice.setdefault(sample_id, []).append(data)

    rows = []
    for sample_id, steps in by_slice.items():
        steps_sorted = sorted(steps, key=lambda d: int(d["step"]))
        for i in range(len(steps_sorted) - 1):
            a, b = steps_sorted[i], steps_sorted[i + 1]
            d_entropy = float(a["total_entropy"]) - float(b["total_entropy"])
            row = {
                "sample_id": sample_id,
                "step": int(a["step"]),
                "delta_entropy": d_entropy,
            }
            # positive delta = improvement for all metrics after sign flip for nmse/ssfd
            row["delta_psnr"] = float(b["psnr"]) - float(a["psnr"])
            row["delta_ssim"] = float(b["ssim"]) - float(a["ssim"])
            row["delta_nmse"] = float(a["nmse"]) - float(b["nmse"])
            row["delta_ssfd"] = float(a.get("ssfd", float("nan"))) - float(
                b.get("ssfd", float("nan"))
            )
            rows.append(row)

    return pd.DataFrame(rows)


def analyze(run_dir: pathlib.Path, dataset: str, acceleration: int, method: str) -> None:
    traj_root = (
        run_dir / "normal_trajectory"
        / f"dataset={dataset}" / f"R={acceleration}" / f"method={method}"
    )
    if not traj_root.exists():
        print(f"No trajectory data at {traj_root}")
        return

    df = build_delta_df(traj_root)
    if df.empty:
        print("No step pairs found.")
        return

    delta_cols = ["delta_psnr", "delta_ssim", "delta_nmse", "delta_ssfd"]
    summary_rows = []
    for col in delta_cols:
        valid = df[["delta_entropy", col]].dropna()
        if len(valid) < 10:
            continue
        rho, pval = spearmanr(valid["delta_entropy"], valid[col])
        metric = col.replace("delta_", "")
        summary_rows.append({
            "metric": metric,
            "spearman_rho": rho,
            "p_value": pval,
            "n_pairs": len(valid),
            "dataset": dataset,
            "acceleration": acceleration,
            "method": method,
        })
        print(f"  Δentropy vs Δ{metric:6s}: ρ={rho:+.3f}  p={pval:.2e}  n={len(valid)}")

    out_dir = run_dir / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{dataset}_R{acceleration}_{method}"
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / f"entropy_recon_delta_{tag}.csv", index=False)

    _plot(df, summary_df, run_dir, dataset, acceleration, method, tag)


def _plot(df, summary_df, run_dir, dataset, acc, method, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plot_dir = run_dir / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    delta_cols = ["delta_psnr", "delta_ssim", "delta_nmse", "delta_ssfd"]
    metric_labels = {"delta_psnr": "ΔPSNR (dB)", "delta_ssim": "ΔSSIM",
                     "delta_nmse": "ΔNMSE (↑ = better)", "delta_ssfd": "ΔSSFD (↑ = better)"}

    fig, axes = plt.subplots(1, len(delta_cols), figsize=(5 * len(delta_cols), 4))
    for ax, col in zip(axes, delta_cols):
        valid = df[["delta_entropy", col]].dropna()
        if valid.empty:
            ax.set_visible(False)
            continue
        ax.scatter(valid["delta_entropy"], valid[col], alpha=0.15, s=5, rasterized=True)
        # regression line
        m, b = np.polyfit(valid["delta_entropy"], valid[col], 1)
        x_range = np.linspace(valid["delta_entropy"].min(), valid["delta_entropy"].max(), 100)
        ax.plot(x_range, m * x_range + b, "r-", linewidth=1.5)

        row = summary_df[summary_df["metric"] == col.replace("delta_", "")]
        rho_str = f"ρ={row['spearman_rho'].values[0]:+.3f}" if not row.empty else ""
        ax.set_xlabel("Δ total entropy (↑ = more reduced)")
        ax.set_ylabel(metric_labels.get(col, col))
        ax.set_title(f"{rho_str}")
        ax.axhline(0, color="k", linewidth=0.5, linestyle=":")
        ax.axvline(0, color="k", linewidth=0.5, linestyle=":")

    fig.suptitle(f"{dataset} R={acc} {method} — Entropy Reduction vs Reconstruction Improvement")
    plt.tight_layout()
    plt.savefig(plot_dir / f"entropy_recon_delta_{tag}.png", dpi=120)
    plt.close()
    print(f"  Plot saved: entropy_recon_delta_{tag}.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--dataset", required=True, choices=["brain", "knee"])
    p.add_argument("--acceleration", required=True, type=int)
    p.add_argument("--method", required=True, choices=["LES", "GEO"])
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Entropy-reconstruction delta: {args.dataset} R={args.acceleration} {args.method}")
    analyze(pathlib.Path(args.run_dir), args.dataset, args.acceleration, args.method)


if __name__ == "__main__":
    main()
