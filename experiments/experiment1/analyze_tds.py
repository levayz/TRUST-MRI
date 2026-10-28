#!/usr/bin/env python
"""Diagnostic analysis and plots for TDS active acquisition experiments.

Reads the saved .npy files and metadata.parquet produced by run_entropy_acquisition.py
and generates the five diagnostic plots:

  1. K-space lines selected over acquisition steps (heatmap)
  2. Mean entropy vs acquisition step (TDS vs LES vs GEO)
  3. TDS score vs acquisition step
  4. Runtime per method (bar chart)
  5. Simulated entropy drop vs actual entropy drop (scatter)

Usage:
    python experiments/experiment1/analyze_tds.py \
        --run_dir results/tds_sweep/TDS_base \
        --dataset brain --acceleration 8 \
        --compare_run_dirs results/exp1 \
        --out_dir results/tds_plots
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _try_import_plotting():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )


def _load_parquet(path: pathlib.Path):
    try:
        import pandas as pd
        return pd.read_parquet(path)
    except ImportError:
        raise ImportError("pandas is required. Install with: pip install pandas pyarrow")


# ---------------------------------------------------------------------------
# Per-slice data collectors
# ---------------------------------------------------------------------------

def collect_mask_trajectories(
    run_dir: pathlib.Path,
    dataset: str,
    acceleration: int,
    method: str,
) -> list[np.ndarray]:
    """Return list of (B_budget+1, W) mask arrays, one per slice."""
    base = (
        run_dir / "entropy_acquisition"
        / f"dataset={dataset}" / f"R={acceleration}" / f"method={method}"
    )
    trajectories = []
    for sample_dir in sorted(base.glob("sample_id=*")):
        masks = sorted(sample_dir.glob("mask_t*.npy"))
        if not masks:
            continue
        traj = np.stack([np.load(str(m)) for m in masks])  # (T+1, W)
        trajectories.append(traj)
    return trajectories


def collect_entropy_curves(
    run_dir: pathlib.Path,
    dataset: str,
    acceleration: int,
    method: str,
) -> list[np.ndarray]:
    """Return list of (T+1,) mean-entropy curves, one per slice."""
    base = (
        run_dir / "entropy_acquisition"
        / f"dataset={dataset}" / f"R={acceleration}" / f"method={method}"
    )
    curves = []
    for sample_dir in sorted(base.glob("sample_id=*")):
        emaps = sorted(sample_dir.glob("entropy_map_t*.npy"))
        if not emaps:
            continue
        vals = [np.load(str(e)).mean() for e in emaps]
        curves.append(np.array(vals))
    return curves


def collect_tds_diagnostics(
    run_dir: pathlib.Path,
    dataset: str,
    acceleration: int,
    method: str = "TDS",
) -> list[list[dict]]:
    """Return list-of-lists of per-step TDS diagnostic dicts."""
    base = (
        run_dir / "entropy_acquisition"
        / f"dataset={dataset}" / f"R={acceleration}" / f"method={method}"
    )
    all_diags = []
    for sample_dir in sorted(base.glob("sample_id=*")):
        diag_files = sorted(sample_dir.glob("tds_diag_t*.npy"))
        if not diag_files:
            continue
        diags = [np.load(str(f), allow_pickle=True).item() for f in diag_files]
        all_diags.append(diags)
    return all_diags


# ---------------------------------------------------------------------------
# Plot 1: K-space line selection heatmap
# ---------------------------------------------------------------------------

def plot_kspace_trajectories(
    trajectories: list[np.ndarray],
    out_path: pathlib.Path,
    method: str = "TDS",
):
    plt = _try_import_plotting()

    if not trajectories:
        print(f"  [skip] No trajectories found for {method}")
        return

    # Compute per-column selection frequency across slices
    T, W = trajectories[0].shape
    selection_map = np.zeros((len(trajectories), T - 1, W), dtype=np.float32)
    for i, traj in enumerate(trajectories):
        diff = np.diff(traj, axis=0)  # (T-1, W) — 1 where line newly acquired
        selection_map[i] = diff

    # Mean across slices
    mean_sel = selection_map.mean(axis=0)  # (T-1, W)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mean_sel.T, aspect="auto", origin="lower",
                   cmap="hot", interpolation="nearest")
    ax.set_xlabel("Acquisition step")
    ax.set_ylabel("k-space column")
    ax.set_title(f"{method}: Mean selection frequency per column and step")
    plt.colorbar(im, ax=ax, label="Fraction of slices selecting this column")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Entropy vs acquisition step
# ---------------------------------------------------------------------------

def plot_entropy_curves(
    curves_per_method: dict[str, list[np.ndarray]],
    out_path: pathlib.Path,
):
    plt = _try_import_plotting()

    if not any(curves_per_method.values()):
        print("  [skip] No entropy curves found.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"TDS": "#e74c3c", "LES": "#3498db", "GEO": "#2ecc71", "PS": "#9b59b6"}

    for method, curves in curves_per_method.items():
        if not curves:
            continue
        max_T = max(len(c) for c in curves)
        mat = np.full((len(curves), max_T), np.nan)
        for i, c in enumerate(curves):
            mat[i, : len(c)] = c / max(c[0], 1e-8)  # normalize to step-0
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        steps = np.arange(max_T)
        color = colors.get(method, None)
        ax.plot(steps, mean, label=method, color=color, linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Acquisition step")
    ax.set_ylabel("Normalized mean entropy")
    ax.set_title("Mean token entropy vs acquisition step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: TDS score vs acquisition step
# ---------------------------------------------------------------------------

def plot_tds_scores(
    all_diags: list[list[dict]],
    out_path: pathlib.Path,
):
    plt = _try_import_plotting()

    if not all_diags:
        print("  [skip] No TDS diagnostics found.")
        return

    scores_list = [
        [d["selected_tds_score"] for d in diags if "selected_tds_score" in d]
        for diags in all_diags
    ]
    scores_list = [s for s in scores_list if s]
    if not scores_list:
        print("  [skip] No TDS scores in diagnostics.")
        return

    max_T = max(len(s) for s in scores_list)
    mat = np.full((len(scores_list), max_T), np.nan)
    for i, s in enumerate(scores_list):
        mat[i, : len(s)] = s

    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    steps = np.arange(max_T)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, mean, color="#e74c3c", linewidth=2, label="TDS score (selected)")
    ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color="#e74c3c")
    ax.set_xlabel("Acquisition step")
    ax.set_ylabel("TDS score")
    ax.set_title("TDS score of selected line vs acquisition step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 4: Runtime per method
# ---------------------------------------------------------------------------

def plot_runtime_comparison(
    runtime_per_method: dict[str, list[float]],
    out_path: pathlib.Path,
):
    plt = _try_import_plotting()

    methods = [m for m, v in runtime_per_method.items() if v]
    if not methods:
        print("  [skip] No runtime data.")
        return

    means = [np.mean(runtime_per_method[m]) for m in methods]
    stds  = [np.std(runtime_per_method[m]) for m in methods]
    colors = {"TDS": "#e74c3c", "LES": "#3498db", "GEO": "#2ecc71", "PS": "#9b59b6"}

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        methods, means,
        yerr=stds, capsize=5,
        color=[colors.get(m, "#95a5a6") for m in methods],
    )
    ax.set_ylabel("Runtime per step (s)")
    ax.set_title("Mean runtime per acquisition step")
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, mean + 0.01,
            f"{mean:.2f}s", ha="center", va="bottom", fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 5: Simulated entropy drop vs actual entropy drop
# ---------------------------------------------------------------------------

def plot_entropy_drop_calibration(
    all_diags: list[list[dict]],
    entropy_curves: list[np.ndarray],
    out_path: pathlib.Path,
):
    plt = _try_import_plotting()

    sim_drops, actual_drops = [], []

    for diags, curve in zip(all_diags, entropy_curves):
        for t, d in enumerate(diags):
            if t + 1 >= len(curve):
                break
            sim_drop = d.get("entropy_drop_selected", float("nan"))
            actual_drop = curve[t] - curve[t + 1]
            if np.isfinite(sim_drop) and np.isfinite(actual_drop):
                sim_drops.append(sim_drop)
                actual_drops.append(actual_drop)

    if not sim_drops:
        print("  [skip] Not enough data for entropy-drop calibration plot.")
        return

    sim_drops    = np.array(sim_drops)
    actual_drops = np.array(actual_drops)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sim_drops, actual_drops, alpha=0.4, s=10, color="#e74c3c")
    lim = max(sim_drops.max(), actual_drops.max(), 1e-6)
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("Simulated entropy drop (TDS prediction)")
    ax.set_ylabel("Actual entropy drop (after acquiring real line)")
    ax.set_title("Entropy-drop calibration: simulated vs actual")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="TDS diagnostic analysis")
    p.add_argument("--run_dir", required=True,
                   help="Run directory containing TDS results (the root passed to --run_dir)")
    p.add_argument("--dataset", default="brain", choices=["brain", "knee"])
    p.add_argument("--acceleration", default=8, type=int)
    p.add_argument("--method", default="TDS",
                   help="TDS method tag (e.g. TDS, TDS_base, TDS_beta=0.5)")
    p.add_argument("--compare_run_dirs", nargs="*", default=[],
                   help="Additional run dirs (LES/GEO) to include in entropy comparison plot")
    p.add_argument("--compare_methods", nargs="*", default=["LES", "GEO"],
                   help="Method names for each compare_run_dir")
    p.add_argument("--out_dir", default=None,
                   help="Output directory for plots (default: <run_dir>/plots)")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = pathlib.Path(args.run_dir)

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    dataset = args.dataset
    accel   = args.acceleration
    method  = args.method

    # --- Collect TDS data ---
    trajs   = collect_mask_trajectories(run_dir, dataset, accel, method)
    curves  = collect_entropy_curves(run_dir, dataset, accel, method)
    diags   = collect_tds_diagnostics(run_dir, dataset, accel, method)

    print(f"Found {len(trajs)} slices with mask trajectories.")
    print(f"Found {len(curves)} slices with entropy curves.")
    print(f"Found {len(diags)} slices with TDS diagnostics.")

    # --- Collect comparison data ---
    curves_per_method: dict[str, list[np.ndarray]] = {method: curves}
    runtime_per_method: dict[str, list[float]] = {}

    for cmp_dir_str, cmp_method in zip(
        args.compare_run_dirs, args.compare_methods
    ):
        cmp_dir = pathlib.Path(cmp_dir_str)
        cmp_curves = collect_entropy_curves(cmp_dir, dataset, accel, cmp_method)
        curves_per_method[cmp_method] = cmp_curves

    # Runtime from TDS diagnostics
    if diags:
        tds_runtimes = [
            d.get("runtime_total_s", d.get("runtime_loop_s", float("nan")))
            for slice_diags in diags
            for d in slice_diags
            if np.isfinite(d.get("runtime_total_s", float("nan")))
               or np.isfinite(d.get("runtime_loop_s", float("nan")))
        ]
        if tds_runtimes:
            runtime_per_method[method] = tds_runtimes

    # --- Plot 1: K-space trajectories ---
    plot_kspace_trajectories(
        trajs,
        out_dir / f"kspace_trajectories_{method}.png",
        method=method,
    )

    # --- Plot 2: Entropy curves ---
    plot_entropy_curves(
        curves_per_method,
        out_dir / "entropy_vs_step.png",
    )

    # --- Plot 3: TDS score vs step ---
    plot_tds_scores(
        diags,
        out_dir / f"tds_score_vs_step_{method}.png",
    )

    # --- Plot 4: Runtime ---
    if runtime_per_method:
        plot_runtime_comparison(
            runtime_per_method,
            out_dir / "runtime_per_method.png",
        )
    else:
        print("  [skip] No runtime data available for comparison plot.")

    # --- Plot 5: Entropy-drop calibration ---
    if diags and curves:
        min_len = min(len(diags), len(curves))
        plot_entropy_drop_calibration(
            diags[:min_len],
            curves[:min_len],
            out_dir / f"entropy_drop_calibration_{method}.png",
        )

    print(f"\nAll plots written to {out_dir}")


if __name__ == "__main__":
    main()
