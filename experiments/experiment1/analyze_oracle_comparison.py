#!/usr/bin/env python
"""Compare oracle vs entropy-guided acquisition for a given (dataset, R).

Reads:
    <oracle_dir>/oracle_active_sampling/dataset=<d>/R=<r>/objective=<obj>/
        sample_id=<id>/step_*.npz
    <entropy_dir>/entropy_acquisition/dataset=<d>/R=<r>/method=<m>/metadata.parquet

Produces:
    <out_dir>/figure_oracle_comparison_<d>_R<r>.png
    <out_dir>/oracle_comparison_<d>_R<r>.csv

Usage:
    python analyze_oracle_comparison.py \\
        --oracle_dir  results/oracle_benchmark \\
        --entropy_dir results/exp1_entropy \\
        --dataset brain --acceleration 16 \\
        --out_dir results/oracle_benchmark/results/plots
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ORACLE_OBJECTIVES = ["psnr", "ssfd"]
ENTROPY_METHODS   = ["LES", "GEO"]

# ── Colors & style ────────────────────────────────────────────────────────────
COLORS = {
    "oracle_psnr": "#d62728",   # red
    "oracle_ssfd": "#ff7f0e",   # orange
    "LES":         "#1f77b4",   # blue
    "GEO":         "#2ca02c",   # green
}
LABELS = {
    "oracle_psnr": "Oracle (PSNR)",
    "oracle_ssfd": "Oracle (SSFD)",
    "LES":         "LES (entropy)",
    "GEO":         "GEO (entropy)",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_oracle(oracle_dir: pathlib.Path, dataset: str, R: int) -> dict[str, pd.DataFrame]:
    """Return {objective: DataFrame} with columns [sample_id, step, psnr, ssim, ssfd, nmse]."""
    out = {}
    base = oracle_dir / "oracle_active_sampling" / f"dataset={dataset}" / f"R={R}"
    for obj in ORACLE_OBJECTIVES:
        obj_dir = base / f"objective={obj}"
        if not obj_dir.exists():
            print(f"  Missing: {obj_dir}")
            continue
        rows = []
        for sample_dir in sorted(obj_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            sample_id = sample_dir.name.replace("sample_id=", "")
            for npz_path in sorted(sample_dir.glob("step_*.npz")):
                d = dict(np.load(str(npz_path), allow_pickle=True))
                rows.append({
                    "sample_id": sample_id,
                    "step":  int(d["step"]),
                    "psnr":  float(d["psnr"]),
                    "ssim":  float(d["ssim"]),
                    "nmse":  float(d["nmse"]),
                    "ssfd":  float(d["ssfd"]) if not np.isnan(float(d["ssfd"])) else np.nan,
                })
        if rows:
            out[obj] = pd.DataFrame(rows)
    return out


def load_entropy(entropy_dir: pathlib.Path, dataset: str, R: int) -> dict[str, pd.DataFrame]:
    """Return {method: DataFrame} with columns [sample_id, step, PSNR, SSIM, NMSE]."""
    out = {}
    base = entropy_dir / "entropy_acquisition" / f"dataset={dataset}" / f"R={R}"
    for method in ENTROPY_METHODS:
        parquet = base / f"method={method}" / "metadata.parquet"
        if not parquet.exists():
            print(f"  Missing: {parquet}")
            continue
        df = pd.read_parquet(parquet)
        df["sample_id"] = df["scan_id"].astype(str) + "_sl" + df["slice_id"].astype(str)
        df = df.rename(columns={"PSNR": "psnr", "SSIM": "ssim", "NMSE": "nmse"})
        out[method] = df[["sample_id", "step", "psnr", "ssim", "nmse"]].copy()
    return out


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_mean(df: pd.DataFrame, col: str, n_boot: int = 1000, seed: int = 0):
    """Per-step bootstrap CI (resample sample_ids). Returns DataFrame [step, mean, lo, hi]."""
    rng = np.random.default_rng(seed)
    ids = df["sample_id"].unique()
    steps = sorted(df["step"].unique())
    records = []
    for step in steps:
        vals_by_id = df[df["step"] == step].groupby("sample_id")[col].mean()
        means = [vals_by_id.loc[rng.choice(ids, size=len(ids), replace=True)].mean()
                 for _ in range(n_boot)]
        records.append({
            "step": step,
            "mean": float(vals_by_id.mean()),
            "lo":   float(np.percentile(means, 2.5)),
            "hi":   float(np.percentile(means, 97.5)),
        })
    return pd.DataFrame(records)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_metric(ax, curves: dict[str, pd.DataFrame], metric: str, ylabel: str, title: str):
    """Plot mean ± 95% CI for each method on `ax`."""
    for key, df in curves.items():
        color = COLORS[key]
        label = LABELS[key]
        lw    = 2.2 if key.startswith("oracle") else 1.8
        ls    = "--" if key.startswith("oracle") else "-"

        stats = bootstrap_mean(df, metric)
        ax.plot(stats["step"], stats["mean"], color=color, linewidth=lw,
                linestyle=ls, label=label, marker="o", markersize=4)
        ax.fill_between(stats["step"], stats["lo"], stats["hi"],
                        color=color, alpha=0.15)

    ax.set_xlabel("Acquisition step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ── Summary table ─────────────────────────────────────────────────────────────

def build_summary(curves: dict[str, pd.DataFrame], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for key, df in curves.items():
        # Final step stats
        B = df["step"].max()
        final = df[df["step"] == B]
        row = {"method": key, "n_slices": df["sample_id"].nunique(), "budget_B": int(B)}
        for m in metrics:
            vals = final.groupby("sample_id")[m].mean()
            row[f"final_{m}_mean"] = float(vals.mean())
            row[f"final_{m}_std"]  = float(vals.std())
        # Gain from step 0
        step0 = df[df["step"] == 0]
        for m in metrics:
            v0 = step0.groupby("sample_id")[m].mean().mean()
            vB = final.groupby("sample_id")[m].mean().mean()
            row[f"gain_{m}"] = float(vB - v0)
        rows.append(row)
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oracle_dir",  required=True)
    p.add_argument("--entropy_dir", required=True)
    p.add_argument("--dataset",     required=True, choices=["brain", "knee"])
    p.add_argument("--acceleration", required=True, type=int)
    p.add_argument("--out_dir",     default=None,
                   help="Output directory (default: <oracle_dir>/results/plots)")
    args = p.parse_args()

    oracle_dir  = pathlib.Path(args.oracle_dir)
    entropy_dir = pathlib.Path(args.entropy_dir)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else oracle_dir / "results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{args.dataset}_R{args.acceleration}"

    print(f"Loading oracle data  ({tag})...")
    oracle_data = load_oracle(oracle_dir, args.dataset, args.acceleration)

    print(f"Loading entropy data ({tag})...")
    entropy_data = load_entropy(entropy_dir, args.dataset, args.acceleration)

    if not oracle_data and not entropy_data:
        print("No data found."); return

    # Merge into one curves dict: oracle methods first (dashed), entropy second (solid)
    curves: dict[str, pd.DataFrame] = {}
    for obj, df in oracle_data.items():
        curves[f"oracle_{obj}"] = df
    for method, df in entropy_data.items():
        curves[method] = df

    print(f"  Sources: {list(curves.keys())}")
    for k, df in curves.items():
        print(f"  {k}: {df['sample_id'].nunique()} slices, steps {df['step'].min()}–{df['step'].max()}")

    # ── Figure ────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available"); return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Oracle vs Entropy Acquisition — {args.dataset.capitalize()} R={args.acceleration}",
                 fontsize=13, fontweight="bold")

    plot_metric(axes[0], curves, "psnr", "PSNR (dB)",
                "Reconstruction quality per step")
    plot_metric(axes[1], curves, "ssim", "SSIM",
                "Structural similarity per step")

    # Add panel labels
    for ax, lbl in zip(axes, ["(a)", "(b)"]):
        ax.text(0.02, 0.98, lbl, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

    plt.tight_layout()
    fig_path = out_dir / f"figure_oracle_comparison_{tag}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {fig_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    summary = build_summary(curves, ["psnr", "ssim", "nmse"])
    csv_path = out_dir.parent / "tables" / f"oracle_comparison_{tag}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_path, index=False)
    print(f"\nSummary ({tag}):")
    print(summary[["method", "n_slices", "final_psnr_mean", "gain_psnr",
                    "final_ssim_mean", "gain_ssim"]].to_string(index=False))
    print(f"\nTable saved: {csv_path}")


if __name__ == "__main__":
    main()
