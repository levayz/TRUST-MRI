#!/usr/bin/env python
"""Make paper-ready tables and plots for Experiment 1.

Aggregates all analysis CSV outputs and generates:
    - LaTeX tables
    - Publication-quality matplotlib figures

Reads: <run_dir>/results/tables/*.csv
Writes:
    <run_dir>/results/tables/paper_*.csv
    <run_dir>/results/tables/paper_*.tex
    <run_dir>/results/plots/paper_*.pdf  (and .png)

Usage:
    python make_experiment1_tables_and_plots.py --run_dir /fast/lev/TRUST-MRI/results/exp1
"""

from __future__ import annotations

import argparse
import glob
import pathlib
import sys

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


DATASETS       = ["brain", "knee"]
ACCELERATIONS  = [8, 16]
METHODS        = ["LES", "GEO"]
ORACLE_METHODS = ["oracle_mse", "oracle_ssfd", "oracle_token_acc"]


def _read_glob(run_dir: pathlib.Path, pattern: str) -> pd.DataFrame | None:
    files = sorted(run_dir.glob(pattern))
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else None


# ---------------------------------------------------------------------------
# Table 1: Entropy Calibration Summary
# ---------------------------------------------------------------------------

def table_entropy_calibration(run_dir: pathlib.Path) -> None:
    """Aggregate ECE and Spearman entropy↔token-error across all settings."""
    df_cal = _read_glob(run_dir, "results/tables/calibration_*.csv")
    df_ent = _read_glob(run_dir, "results/tables/entropy_token_error_*.csv")
    if df_cal is None and df_ent is None:
        print("[SKIP] Table 1: no calibration/entropy-token-error data found")
        return

    rows = []
    for ds in DATASETS:
        for acc in ACCELERATIONS:
            for method in METHODS:
                row = {"dataset": ds, "acceleration": acc, "method": method}
                if df_cal is not None:
                    sub = df_cal[
                        (df_cal["dataset"] == ds) &
                        (df_cal["acceleration"] == acc) &
                        (df_cal["method"] == method)
                    ]
                    if not sub.empty:
                        # Mean ECE and NLL at middle step
                        mid = sub["step"].median()
                        mid_sub = sub[sub["step"] == sub["step"].iloc[(sub.shape[0] - 1) // 2]]
                        row["ece_mid"] = float(mid_sub["ece"].mean()) if not mid_sub.empty else float("nan")
                        row["nll_mid"] = float(mid_sub["token_nll"].mean()) if not mid_sub.empty else float("nan")
                        row["ece_final"] = float(sub[sub["step"] == sub["step"].max()]["ece"].mean())
                if df_ent is not None:
                    sub2 = df_ent[
                        (df_ent["dataset"] == ds) &
                        (df_ent["acceleration"] == acc) &
                        (df_ent["method"] == method)
                    ]
                    if not sub2.empty:
                        row["spearman_mean"] = float(sub2["spearman_r"].mean())
                        row["auroc_mean"]    = float(sub2["auroc"].mean())
                rows.append(row)

    df_out = pd.DataFrame(rows)
    out_dir = run_dir / "results" / "tables"
    df_out.to_csv(out_dir / "paper_entropy_calibration.csv", index=False)
    df_out.to_latex(
        out_dir / "paper_entropy_calibration.tex",
        index=False, float_format="%.3f", na_rep="—",
    )
    print(f"  Table 1: entropy calibration → {out_dir / 'paper_entropy_calibration.csv'}")


# ---------------------------------------------------------------------------
# Table 2: Candidate-Line Utility
# ---------------------------------------------------------------------------

def table_candidate_utility(run_dir: pathlib.Path) -> None:
    """LES/GEO vs oracle Spearman, Recall@5, Regret@1."""
    df = _read_glob(run_dir, "results/tables/candidate_line_utility_*.csv")
    if df is None:
        print("[SKIP] Table 2: no candidate line utility data found")
        return

    # Pivot: rows = (dataset, acc, method), cols = (score, target, metric)
    pivot = df.pivot_table(
        index=["dataset", "acceleration", "base_method"],
        columns=["score", "target"],
        values=["spearman_r", "recall_at_5", "regret_at_1"],
        aggfunc="mean",
    )
    out_dir = run_dir / "results" / "tables"
    pivot.to_csv(out_dir / "paper_candidate_utility.csv")
    try:
        pivot.to_latex(out_dir / "paper_candidate_utility.tex", float_format="%.3f")
    except Exception:
        pass
    print(f"  Table 2: candidate utility → {out_dir / 'paper_candidate_utility.csv'}")


# ---------------------------------------------------------------------------
# Table 3: Oracle Comparison
# ---------------------------------------------------------------------------

def table_oracle_comparison(run_dir: pathlib.Path) -> None:
    """Final metrics for all methods vs oracle."""
    df = _read_glob(run_dir, "results/tables/oracle_comparison_*.csv")
    if df is None:
        print("[SKIP] Table 3: no oracle comparison data found")
        return

    metrics = ["psnr", "ssim", "nmse", "ssfd", "token_accuracy"]
    pivot = df.pivot_table(
        index="method",
        columns=["dataset", "acceleration"],
        values=metrics,
        aggfunc="mean",
    )
    out_dir = run_dir / "results" / "tables"
    pivot.to_csv(out_dir / "paper_oracle_comparison.csv")
    try:
        pivot.to_latex(out_dir / "paper_oracle_comparison.tex", float_format="%.3f")
    except Exception:
        pass
    print(f"  Table 3: oracle comparison → {out_dir / 'paper_oracle_comparison.csv'}")


# ---------------------------------------------------------------------------
# Figure 1: Entropy-token-error Spearman across methods
# ---------------------------------------------------------------------------

def figure_spearman_vs_step(run_dir: pathlib.Path) -> None:
    df = _read_glob(run_dir, "results/tables/entropy_token_error_*.csv")
    if df is None:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plot_dir = run_dir / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab10")
    style = {"LES": "-", "GEO": "--"}

    for ds in DATASETS:
        for acc in ACCELERATIONS:
            fig, ax = plt.subplots(figsize=(6, 4))
            for i, method in enumerate(METHODS):
                sub = df[
                    (df["dataset"] == ds) &
                    (df["acceleration"] == acc) &
                    (df["method"] == method)
                ].sort_values("step")
                if sub.empty:
                    continue
                ax.plot(
                    sub["step"], sub["spearman_r"],
                    color=cmap(i), ls=style.get(method, "-"),
                    label=method, marker="o", markersize=3,
                )
            ax.axhline(0, ls="--", color="gray", alpha=0.5)
            ax.set_xlabel("Acquisition step")
            ax.set_ylabel("Spearman r  (entropy vs token error)")
            ax.set_title(f"{ds}  R={acc}")
            ax.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(plot_dir / f"paper_spearman_{ds}_R{acc}.pdf")
            plt.savefig(plot_dir / f"paper_spearman_{ds}_R{acc}.png", dpi=150)
            plt.close()
    print("  Figure 1: Spearman vs step")


# ---------------------------------------------------------------------------
# Figure 2: Oracle comparison curves
# ---------------------------------------------------------------------------

def figure_oracle_comparison(run_dir: pathlib.Path) -> None:
    df = _read_glob(run_dir, "results/tables/oracle_per_step_*.csv")
    if df is None:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plot_dir = run_dir / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab10")
    all_methods = METHODS + [f"oracle_{o}" for o in ["mse", "ssfd", "token_acc"]]
    color_map = {m: cmap(i) for i, m in enumerate(all_methods)}

    for ds in DATASETS:
        for acc in ACCELERATIONS:
            sub = df[(df["dataset"] == ds) & (df["acceleration"] == acc)]
            if sub.empty:
                continue
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            for ax, metric in zip(axes, ["psnr", "ssim", "ssfd"]):
                for method in all_methods:
                    ms = sub[sub["method"] == method].sort_values("step")
                    if ms.empty or metric not in ms.columns:
                        continue
                    ls = "--" if method.startswith("oracle") else "-"
                    ax.plot(
                        ms["step"], ms[metric],
                        label=method, color=color_map.get(method),
                        ls=ls, linewidth=1.5,
                    )
                ax.set_xlabel("Step")
                ax.set_ylabel(metric.upper())
                ax.set_title(metric)
            axes[0].legend(fontsize=7, ncol=1, loc="lower right")
            fig.suptitle(f"{ds}  R={acc}")
            plt.tight_layout()
            plt.savefig(plot_dir / f"paper_oracle_{ds}_R{acc}.pdf")
            plt.savefig(plot_dir / f"paper_oracle_{ds}_R{acc}.png", dpi=150)
            plt.close()
    print("  Figure 2: Oracle comparison curves")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Make paper tables and plots for Experiment 1")
    p.add_argument("--run_dir", required=True)
    args = p.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    out_dir = run_dir / "results"
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    print("Generating paper tables and plots...")
    table_entropy_calibration(run_dir)
    table_candidate_utility(run_dir)
    table_oracle_comparison(run_dir)
    figure_spearman_vs_step(run_dir)
    figure_oracle_comparison(run_dir)
    print("Done.")


if __name__ == "__main__":
    main()
