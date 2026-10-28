#!/usr/bin/env python
"""Analyze entropy validation experiments (A and B) and generate 4-panel figure.

Panel (a) Knee  — normalized patch MSE by entropy quintile (random masks, R=8/R=16)
Panel (b) Brain — same
Panel (c) Knee  — normalized entropy + normalized MSE over acquisition steps (LES, R=8/R=16)
Panel (d) Brain — same

Reads:
    <run_dir>/random_mask_diagnostic/dataset=<d>/R=<r>/metadata.parquet
    <run_dir>/entropy_acquisition/dataset=<d>/R=<r>/metadata.parquet

Writes:
    <run_dir>/results/tables/entropy_validation_summary.csv
    <run_dir>/results/plots/figure_entropy_validation.png

Usage:
    python analyze_entropy_validation.py --run_dir results/exp1_entropy
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


DATASETS = ["knee", "brain"]
ACCELS = [8, 16]
METHODS = ["LES", "GEO"]
BIN_COLS = [f"normalized_bin_error_Q{q}" for q in range(1, 6)]
BIN_LABELS = ["Q1\n(low)", "Q2", "Q3", "Q4", "Q5\n(high)"]
R_COLORS = {8: "#1f77b4", 16: "#ff7f0e"}   # used by bin panels (a)/(b)

# Acquisition panels (c)/(d): one distinct color per (R, method) combo
COMBO_COLORS = {
    (8,  "LES"): "#1f77b4",  # blue
    (8,  "GEO"): "#2ca02c",  # green
    (16, "LES"): "#ff7f0e",  # orange
    (16, "GEO"): "#d62728",  # red
}
ENTROPY_LW = 2.5   # bold solid for entropy
MSE_LW     = 1.5   # dashed for MSE


# ---------------------------------------------------------------------------
# Bootstrap CI helpers
# ---------------------------------------------------------------------------

def _scan_bootstrap(df: pd.DataFrame, col: str, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Scan-level bootstrap CI for the mean of `col`. Returns (mean, ci_low, ci_high)."""
    rng = np.random.default_rng(seed)
    scan_ids = df["scan_id"].unique()
    means = []
    for _ in range(n_boot):
        sampled = rng.choice(scan_ids, size=len(scan_ids), replace=True)
        subset = df[df["scan_id"].isin(sampled)]
        means.append(subset[col].mean())
    return float(df[col].mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _spearman_ci(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float, float]:
    """Mean Spearman rho with scan-level bootstrap CI. Returns (mean, ci_low, ci_high, p_value)."""
    rng = np.random.default_rng(seed)
    scan_ids = df["scan_id"].unique()
    rhos = []
    for _ in range(n_boot):
        sampled = rng.choice(scan_ids, size=len(scan_ids), replace=True)
        subset = df[df["scan_id"].isin(sampled)]
        rhos.append(subset["spearman_rho"].mean())
    mean_rho = float(df["spearman_rho"].mean())
    ci_lo = float(np.percentile(rhos, 2.5))
    ci_hi = float(np.percentile(rhos, 97.5))
    # One-sample t-test against rho=0
    _, p = ttest_1samp(df["spearman_rho"].dropna().values, 0)
    return mean_rho, ci_lo, ci_hi, float(p)


# ---------------------------------------------------------------------------
# Panel (a) / (b) — entropy-bin patch errors
# ---------------------------------------------------------------------------

def _bin_panel(ax, rmd_data: dict, dataset: str) -> list[dict]:
    """Plot entropy-bin normalized patch MSE for both R values on `ax` (None = stats only)."""
    summary_rows = []
    for R in ACCELS:
        df = rmd_data.get((dataset, R))
        if df is None or df.empty:
            continue
        means, lo, hi = [], [], []
        for col in BIN_COLS:
            m, l, h = _scan_bootstrap(df, col)
            means.append(m); lo.append(l); hi.append(h)
        means = np.array(means); lo = np.array(lo); hi = np.array(hi)

        if ax is not None:
            x = np.arange(len(BIN_COLS))
            ax.plot(x, means, color=R_COLORS[R], marker="o", linewidth=1.8, label=f"R={R}")
            ax.fill_between(x, lo, hi, color=R_COLORS[R], alpha=0.2)

        rho_mean, rho_lo, rho_hi, rho_p = _spearman_ci(df)
        summary_rows.append({
            "dataset": dataset, "R": R, "method": "random",
            "analysis": "random_mask",
            "mean_spearman_rho": rho_mean,
            "CI_low": rho_lo, "CI_high": rho_hi, "p_value": rho_p,
            "final_entropy_drop_pct": float("nan"),
            "final_MSE_drop_pct": float("nan"),
        })

    if ax is not None:
        ax.set_xticks(range(len(BIN_COLS)))
        ax.set_xticklabels(BIN_LABELS, fontsize=8)
        ax.set_xlabel("Entropy quintile")
        ax.set_ylabel("Normalized patch MSE")
        ax.set_title(f"{dataset.capitalize()} — patch entropy vs error (random masks)")
        ax.legend(fontsize=8)
    return summary_rows


# ---------------------------------------------------------------------------
# Panel (c) / (d) — acquisition curves
# ---------------------------------------------------------------------------

def _acq_panel(ax, acq_data: dict, dataset: str) -> list[dict]:
    """Plot normalized entropy (bold solid) and normalized MSE (dashed) on `ax`.

    Encoding:
        color     = (R, method) combo — 4 distinct colors
        entropy   = bold solid line
        MSE       = dashed line
    Two y-axes: left = normalized MSE, right = normalized entropy (zoomed).
    """
    summary_rows = []

    if ax is not None:
        ax_mse = ax
        ax_ent = ax.twinx()
        ax_ent.set_ylabel("Normalized entropy", fontsize=8)
    else:
        ax_mse = ax_ent = None

    all_entropy_vals = []

    for R in ACCELS:
        for method in METHODS:
            df = acq_data.get((dataset, R, method))
            if df is None or df.empty:
                continue

            gb = df.groupby("normalized_step")[
                ["normalized_mean_entropy", "normalized_MSE"]
            ].mean().reset_index().sort_values("normalized_step")

            if ax is not None:
                color = COMBO_COLORS[(R, method)]
                tau      = gb["normalized_step"].values
                ent_vals = gb["normalized_mean_entropy"].values
                mse_vals = gb["normalized_MSE"].values
                all_entropy_vals.extend(ent_vals.tolist())

                ax_ent.plot(tau, ent_vals,
                            color=color, linestyle="-", linewidth=ENTROPY_LW,
                            label=f"R={R} {method} entropy")
                ax_mse.plot(tau, mse_vals,
                            color=color, linestyle="--", linewidth=MSE_LW,
                            label=f"R={R} {method} MSE")

            per_step = df.groupby("step")[["normalized_mean_entropy", "normalized_MSE"]].mean()
            entropy_drop = 100.0 * (1.0 - float(per_step["normalized_mean_entropy"].iloc[-1]))
            mse_drop     = 100.0 * (1.0 - float(per_step["normalized_MSE"].iloc[-1]))
            summary_rows.append({
                "dataset": dataset, "R": R, "method": method,
                "analysis": "entropy_acquisition",
                "mean_spearman_rho": float("nan"),
                "CI_low": float("nan"), "CI_high": float("nan"), "p_value": float("nan"),
                "final_entropy_drop_pct": entropy_drop,
                "final_MSE_drop_pct": mse_drop,
            })

    if ax is not None:
        if all_entropy_vals:
            e_min, e_max = min(all_entropy_vals), max(all_entropy_vals)
            margin = max(0.005, (e_max - e_min) * 0.2)
            ax_ent.set_ylim(e_min - margin, e_max + margin)

        ax_mse.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
        ax_mse.set_xlabel("Normalized acquisition progress τ = t/B")
        ax_mse.set_ylabel("Normalized MSE")
        ax_mse.set_title(f"{dataset.capitalize()} — entropy + MSE during acquisition")

        lines1, labels1 = ax_mse.get_legend_handles_labels()
        lines2, labels2 = ax_ent.get_legend_handles_labels()
        ax_mse.legend(lines1 + lines2, labels1 + labels2, fontsize=6, ncol=2)
    return summary_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    args = p.parse_args()
    run_dir = pathlib.Path(args.run_dir)

    # Load data
    rmd_data: dict = {}
    acq_data: dict = {}
    for ds in DATASETS:
        for R in ACCELS:
            rmd_path = run_dir / "random_mask_diagnostic" / f"dataset={ds}" / f"R={R}" / "metadata.parquet"
            if rmd_path.exists():
                rmd_data[(ds, R)] = pd.read_parquet(rmd_path)
            else:
                print(f"  Missing: {rmd_path}")

            for method in METHODS:
                acq_path = (
                    run_dir / "entropy_acquisition"
                    / f"dataset={ds}" / f"R={R}" / f"method={method}" / "metadata.parquet"
                )
                if acq_path.exists():
                    acq_data[(ds, R, method)] = pd.read_parquet(acq_path)
                else:
                    print(f"  Missing: {acq_path}")

    if not rmd_data and not acq_data:
        print("No data found. Run the data generation scripts first.")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        plt = None

    plot_dir = run_dir / "results" / "plots"
    table_dir = run_dir / "results" / "tables"
    plot_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    all_summary = []

    if plt is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        panel_labels = ["(a)", "(b)", "(c)", "(d)"]
        ax_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

        # Panels (a) Knee, (b) Brain — entropy-bin errors
        for i, ds in enumerate(["knee", "brain"]):
            ax = ax_list[i]
            rows = _bin_panel(ax, rmd_data, ds)
            all_summary.extend(rows)
            ax.text(0.02, 0.98, panel_labels[i], transform=ax.transAxes,
                    fontsize=12, fontweight="bold", va="top")

        # Panels (c) Knee, (d) Brain — acquisition curves
        for i, ds in enumerate(["knee", "brain"]):
            ax = ax_list[i + 2]
            rows = _acq_panel(ax, acq_data, ds)
            all_summary.extend(rows)
            ax.text(0.02, 0.98, panel_labels[i + 2], transform=ax.transAxes,
                    fontsize=12, fontweight="bold", va="top")

        plt.tight_layout()
        out_fig = plot_dir / "figure_entropy_validation.png"
        plt.savefig(out_fig, dpi=150)
        plt.close()
        print(f"Figure saved: {out_fig}")
    else:
        # Still compute summary without plotting
        for ds in ["knee", "brain"]:
            all_summary.extend(_bin_panel(None, rmd_data, ds))
            all_summary.extend(_acq_panel(None, acq_data, ds))

    # Summary table
    if all_summary:
        summary_df = pd.DataFrame(all_summary)
        out_csv = table_dir / "entropy_validation_summary.csv"
        summary_df.to_csv(out_csv, index=False)
        print(f"Summary saved: {out_csv}")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
