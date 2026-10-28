"""Visualization utilities for oracle vs entropy acquisition comparison.

Public API — Oracle vs entropy comparison
------------------------------------------
oracle  = load_oracle_steps(run_dir, dataset='brain', R=16)
entropy = load_entropy_steps(run_dir, dataset='brain', R=16)
fig = plot_psnr(oracle, entropy)
fig = plot_ssfd(oracle, entropy)
fig = plot_metric(curves, metric='psnr', ylabel='PSNR (dB)', title='...')

Public API — Entropy validation (Experiment A + B)
----------------------------------------------------
fig = plot_entropy_bins(run_dir, dataset='brain', accelerations=[8, 16])
fig = plot_acquisition_curves(run_dir, dataset='brain', accelerations=[8, 16])
"""

from __future__ import annotations

import pathlib
from collections import defaultdict
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd

# ── Paper-ready defaults ──────────────────────────────────────────────────────
PAPER_FIGSIZE = (5.5, 3.8)   # inches — consistent across all figures

matplotlib.rcParams.update({
    "font.size":        13,
    "axes.labelsize":   14,
    "axes.titlesize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  11,
    "lines.linewidth":  2.2,
    "axes.linewidth":   1.2,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "font.weight":      "normal",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})

COLORS = {
    "oracle_psnr": "#d62728",
    "oracle_ssfd": "#ff7f0e",
    "LES":         "#1f77b4",
    "GEO":         "#2ca02c",
    "PS":          "#9467bd",
    "PS_kspace":   "#7b2d8b",   # darker purple
    "TDS":         "#8c564b",   # brown
    "GLES":        "#17becf",   # teal
}
LINESTYLES = {
    "oracle_psnr": "--",
    "oracle_ssfd": "--",
    "LES":         "-",
    "GEO":         "-",
    "PS":          "-",
    "PS_kspace":   "--",
    "TDS":         "-",
    "GLES":        "-",
}
LINEWIDTHS = {
    "oracle_psnr": 2.2,
    "oracle_ssfd": 2.2,
    "LES":         1.8,
    "GEO":         1.8,
    "PS":          1.8,
    "PS_kspace":   1.8,
    "TDS":         1.8,
    "GLES":        1.8,
}
LABELS = {
    "oracle_psnr": "Oracle (PSNR obj.)",
    "oracle_ssfd": "Oracle (SSFD obj.)",
    "LES":         "LES (entropy)",
    "GEO":         "GEO (entropy)",
    "PS":          "Posterior Sampling",
    "PS_kspace":   "PS (k-space)",
    "TDS":         "TDS",
    "GLES":        "GLES",
}

ORACLE_OBJECTIVES = ["psnr", "ssfd"]
ENTROPY_METHODS   = ["LES", "GEO", "PS", "PS_kspace", "TDS", "GLES"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_oracle_steps(
    run_dir: str | pathlib.Path,
    dataset: str,
    R: int,
) -> dict[str, pd.DataFrame]:
    """Load oracle per-step npz files.

    Returns
    -------
    dict mapping "oracle_psnr" / "oracle_ssfd" →
        DataFrame[sample_id, step, psnr, ssim, nmse, ssfd]
    """
    run_dir = pathlib.Path(run_dir)
    base = run_dir / "oracle_active_sampling" / f"dataset={dataset}" / f"R={R}"
    out: dict[str, pd.DataFrame] = {}

    for obj in ORACLE_OBJECTIVES:
        obj_dir = base / f"objective={obj}"
        if not obj_dir.exists():
            print(f"  [load_oracle_steps] Missing: {obj_dir}")
            continue
        rows = []
        for sample_dir in sorted(obj_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            sample_id = sample_dir.name.replace("sample_id=", "")
            for npz_path in sorted(sample_dir.glob("step_*.npz")):
                d = dict(np.load(str(npz_path), allow_pickle=True))
                ssfd_val = float(d["ssfd"]) if "ssfd" in d and not np.isnan(float(d["ssfd"])) else np.nan
                rows.append({
                    "sample_id": sample_id,
                    "step":  int(d["step"]),
                    "psnr":  float(d["psnr"]),
                    "ssim":  float(d["ssim"]),
                    "nmse":  float(d["nmse"]),
                    "ssfd":  ssfd_val,
                })
        if rows:
            out[f"oracle_{obj}"] = pd.DataFrame(rows)

    return out


def load_entropy_steps(
    run_dir: str | pathlib.Path,
    dataset: str,
    R: int,
) -> dict[str, pd.DataFrame]:
    """Load entropy acquisition parquet files.

    Returns
    -------
    dict mapping "LES" / "GEO" →
        DataFrame[sample_id, step, psnr, ssim, nmse, ssfd]
    """
    run_dir = pathlib.Path(run_dir)
    base = run_dir / "entropy_acquisition" / f"dataset={dataset}" / f"R={R}"
    out: dict[str, pd.DataFrame] = {}

    for method in ENTROPY_METHODS:
        parquet = base / f"method={method}" / "metadata.parquet"
        if not parquet.exists():
            print(f"  [load_entropy_steps] Missing: {parquet}")
            continue
        df = pd.read_parquet(parquet)
        # Build a stable sample_id from scan_id + slice_id
        if "sample_id" not in df.columns:
            df["sample_id"] = df["scan_id"].astype(str) + "_sl" + df["slice_id"].astype(str)
        # Normalise column names to lower-case
        rename = {}
        for col in ["PSNR", "SSIM", "NMSE", "SSFD"]:
            if col in df.columns:
                rename[col] = col.lower()
        if rename:
            df = df.rename(columns=rename)
        # Keep only the columns we need; SSFD may be absent
        keep = ["sample_id", "step", "psnr", "ssim", "nmse"]
        if "ssfd" in df.columns:
            keep.append("ssfd")
        else:
            df["ssfd"] = float("nan")
            keep.append("ssfd")
        out[method] = df[keep].copy()

    return out


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    df: pd.DataFrame,
    col: str,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Scan-level bootstrap CI per step.

    Returns DataFrame[step, mean, lo, hi].
    """
    rng = np.random.default_rng(seed)
    ids = df["sample_id"].unique()
    steps = sorted(df["step"].unique())
    records = []
    for step in steps:
        step_df = df[df["step"] == step]
        vals_by_id = step_df.groupby("sample_id")[col].mean()
        valid_ids = vals_by_id.dropna().index
        if len(valid_ids) == 0:
            records.append({"step": step, "mean": np.nan, "lo": np.nan, "hi": np.nan})
            continue
        resample_pool = valid_ids.to_numpy()
        means = [
            vals_by_id.loc[rng.choice(resample_pool, size=len(resample_pool), replace=True)].mean()
            for _ in range(n_boot)
        ]
        records.append({
            "step": step,
            "mean": float(vals_by_id.mean()),
            "lo":   float(np.percentile(means, 2.5)),
            "hi":   float(np.percentile(means, 97.5)),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Core plot function
# ---------------------------------------------------------------------------

def plot_metric(
    curves: dict[str, pd.DataFrame],
    metric: str,
    ylabel: str,
    title: str,
    ax=None,
    n_boot: int = 1000,
    plot_names: dict[str, str] | None = None,
):
    """Plot mean ± 95% CI for each method.

    Parameters
    ----------
    curves      : dict of key → DataFrame[sample_id, step, <metric>]
    metric      : column name (e.g. "psnr", "ssfd")
    ylabel      : y-axis label
    title       : axes title
    ax          : matplotlib Axes; if None, a new figure is created
    n_boot      : bootstrap iterations
    plot_names  : optional dict overriding display labels, e.g.
                  {"PS_kspace": "Posterior Sampling", "GEO": "TRUST-MRI"}

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    else:
        fig = ax.get_figure()

    overrides = plot_names or {}

    for key, df in curves.items():
        if metric not in df.columns or df[metric].isna().all():
            continue
        color = COLORS.get(key, None)
        label = overrides.get(key, LABELS.get(key, key))
        ls    = LINESTYLES.get(key, "-")
        lw    = LINEWIDTHS.get(key, 1.8)

        stats = _bootstrap_ci(df, metric, n_boot=n_boot)
        valid = stats.dropna(subset=["mean"])
        ax.plot(valid["step"], valid["mean"], color=color, linewidth=lw,
                linestyle=ls, label=label, marker="o", markersize=4)
        ax.fill_between(valid["step"], valid["lo"], valid["hi"],
                        color=color, alpha=0.15)

    ax.set_xlabel("Acquisition step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return fig, ax


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def plot_psnr(
    oracle: dict[str, pd.DataFrame],
    entropy: dict[str, pd.DataFrame],
    dataset: str = "Brain",
    R: int = 16,
    n_boot: int = 1000,
    title: str | None = None,
    plot_names: dict[str, str] | None = None,
):
    """PSNR figure with optional title and per-curve label overrides.

    Parameters
    ----------
    plot_names : dict mapping curve key → display label, e.g.
                 {"PS_kspace": "Posterior Sampling", "GEO": "TRUST-MRI"}
    title      : override the default axes title
    """
    import matplotlib.pyplot as plt
    curves = {**oracle, **entropy}
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    plot_metric(curves, "psnr", "PSNR (dB)",
                title if title is not None else f"PSNR vs acquisition step — {dataset} R={R}",
                ax=ax, n_boot=n_boot, plot_names=plot_names)
    plt.tight_layout()
    return fig


def plot_ssfd(
    oracle: dict[str, pd.DataFrame],
    entropy: dict[str, pd.DataFrame],
    dataset: str = "Brain",
    R: int = 16,
    n_boot: int = 1000,
    title: str | None = None,
    plot_names: dict[str, str] | None = None,
):
    """SSFD figure with optional title and per-curve label overrides.

    Parameters
    ----------
    plot_names : dict mapping curve key → display label, e.g.
                 {"PS_kspace": "Posterior Sampling", "GEO": "TRUST-MRI"}
    title      : override the default axes title
    """
    import matplotlib.pyplot as plt
    curves = {**oracle, **entropy}
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    plot_metric(curves, "ssfd", "SSFD (lower = better)",
                title if title is not None else f"SSFD vs acquisition step — {dataset} R={R}",
                ax=ax, n_boot=n_boot, plot_names=plot_names)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entropy validation — Experiment A: patch entropy bins
# ---------------------------------------------------------------------------

# Colors per (method, R) for acquisition-curve plots
ACQ_COLORS = {
    ("LES",        8):  "#1f77b4",   # blue
    ("GEO",        8):  "#ff7f0e",   # orange
    ("PS",         8):  "#9467bd",   # purple
    ("PS_kspace",  8):  "#7b2d8b",   # darker purple
    ("TDS",        8):  "#8c564b",   # brown
    ("GLES",       8):  "#17becf",   # teal
    ("LES",       16):  "#2ca02c",   # green
    ("GEO",       16):  "#d62728",   # red
    ("PS",        16):  "#e377c2",   # pink
    ("PS_kspace", 16):  "#b5179e",   # darker pink
    ("TDS",       16):  "#bcbd22",   # yellow-green
    ("GLES",      16):  "#7f7f7f",   # grey
}
BIN_COLORS = {8: "#1f77b4", 16: "#d62728"}


def _bootstrap_ci_rows(
    df: pd.DataFrame,
    col: str,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Scalar bootstrap CI over all rows (no step grouping). Returns (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    scan_ids = df["scan_id"].unique()
    vals_by_scan = df.groupby("scan_id")[col].mean()
    means = [
        vals_by_scan.loc[rng.choice(scan_ids, size=len(scan_ids), replace=True)].mean()
        for _ in range(n_boot)
    ]
    return float(vals_by_scan.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def plot_entropy_bins(
    run_dir: str | pathlib.Path,
    dataset: str,
    accelerations: list[int] = (8, 16),
    n_boot: int = 1000,
):
    """Experiment A: normalized patch-MSE per entropy quintile.

    Reads random_mask_diagnostic/.../metadata.parquet.
    One line per R; horizontal dashed line at y=1 (no correlation baseline).

    Returns fig.
    """
    import matplotlib.pyplot as plt

    run_dir = pathlib.Path(run_dir)
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)

    quintiles = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    x = np.arange(1, 6)
    found_any = False

    for R in accelerations:
        parquet = (
            run_dir / "random_mask_diagnostic"
            / f"dataset={dataset}" / f"R={R}" / "metadata.parquet"
        )
        if not parquet.exists():
            print(f"  [plot_entropy_bins] Missing: {parquet}")
            continue
        df = pd.read_parquet(parquet)
        color = BIN_COLORS.get(R, None)

        means, los, his = [], [], []
        for q in quintiles:
            col = f"normalized_bin_error_{q}"
            if col not in df.columns:
                print(f"  [plot_entropy_bins] Column {col} not found")
                break
            m, lo, hi = _bootstrap_ci_rows(df, col, n_boot=n_boot)
            means.append(m); los.append(lo); his.append(hi)

        if len(means) == 5:
            ax.plot(x, means, marker="o", color=color, linewidth=2, label=f"R={R}", markersize=5)
            ax.fill_between(x, los, his, color=color, alpha=0.15)
            found_any = True

    ax.axhline(1.0, color="gray", linewidth=1, linestyle="--", label="flat (no correlation)")
    ax.set_xticks(x)
    ax.set_xticklabels(quintiles)
    ax.set_xlabel("Entropy quintile (Q1=low, Q5=high)")
    ax.set_ylabel("Normalized patch MSE")
    ax.set_title(f"Patch error vs entropy quintile — {dataset.capitalize()}")
    if found_any:
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entropy validation — Experiment B: acquisition curves
# ---------------------------------------------------------------------------

def plot_acquisition_curves(
    run_dir: str | pathlib.Path,
    dataset: str,
    accelerations: list[int] = (8, 16),
    methods: list[str] = ("LES", "GEO"),
    n_boot: int = 1000,
):
    """Experiment B: normalized entropy (solid) + normalized MSE (dashed) vs τ=t/B.

    One color per (method, R) combo; entropy = solid bold, MSE = dashed.
    Returns fig.
    """
    import matplotlib.pyplot as plt

    run_dir = pathlib.Path(run_dir)
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    found_any = False

    for R in accelerations:
        for method in methods:
            parquet = (
                run_dir / "entropy_acquisition"
                / f"dataset={dataset}" / f"R={R}"
                / f"method={method}" / "metadata.parquet"
            )
            if not parquet.exists():
                print(f"  [plot_acquisition_curves] Missing: {parquet}")
                continue

            df = pd.read_parquet(parquet)
            if "scan_id" not in df.columns:
                df["scan_id"] = df["scan_id"] if "scan_id" in df.columns else df.index.astype(str)

            color = ACQ_COLORS.get((method, R), None)
            base_label = f"{method} R={R}"

            # Group by normalized_step (round to avoid float drift)
            df["tau"] = df["normalized_step"].round(4)

            for col, ls, lw, suffix in [
                ("normalized_mean_entropy", "-",  2.2, "entropy"),
                ("normalized_MSE",          "--", 1.6, "MSE"),
            ]:
                if col not in df.columns:
                    continue
                steps = sorted(df["tau"].unique())
                means, los, his = [], [], []
                rng = np.random.default_rng(0)
                scan_ids = df["scan_id"].unique()

                for tau in steps:
                    sub = df[df["tau"] == tau]
                    vals = sub.groupby("scan_id")[col].mean()
                    if vals.empty or vals.isna().all():
                        means.append(np.nan); los.append(np.nan); his.append(np.nan)
                        continue
                    boot = [
                        vals.loc[rng.choice(vals.index, size=len(vals), replace=True)].mean()
                        for _ in range(n_boot)
                    ]
                    means.append(float(vals.mean()))
                    los.append(float(np.percentile(boot, 2.5)))
                    his.append(float(np.percentile(boot, 97.5)))

                steps_arr = np.array(steps)
                means_arr = np.array(means)
                los_arr   = np.array(los)
                his_arr   = np.array(his)
                valid = ~np.isnan(means_arr)

                label = f"{base_label} ({suffix})" if suffix == "entropy" else None
                ax.plot(steps_arr[valid], means_arr[valid],
                        color=color, linewidth=lw, linestyle=ls,
                        label=label, marker="o", markersize=3)
                ax.fill_between(steps_arr[valid], los_arr[valid], his_arr[valid],
                                color=color, alpha=0.12)
                found_any = True

    ax.axhline(1.0, color="gray", linewidth=1, linestyle=":", alpha=0.6)
    ax.set_xlabel("Normalized acquisition progress (τ = t/B)")
    ax.set_ylabel("Normalized value (relative to step 0)")
    ax.set_title(f"Entropy & MSE during acquisition — {dataset.capitalize()}")
    if found_any:
        # Add style legend entries
        from matplotlib.lines import Line2D
        handles, labels = ax.get_legend_handles_labels()
        handles += [
            Line2D([0], [0], color="k", lw=2.2, ls="-",  label="entropy (solid)"),
            Line2D([0], [0], color="k", lw=1.6, ls="--", label="MSE (dashed)"),
        ]
        ax.legend(handles=handles, fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ===========================================================================
# Patch Entropy Investigation — data loader + 9 visualization functions
# ===========================================================================

import json as _json_mod

# ---------------------------------------------------------------------------
# Entropy selector helper
# ---------------------------------------------------------------------------

def _select_entropy(entropy_per_cb: np.ndarray, entropy_sel: dict | None) -> np.ndarray:
    """Reduce (H_p, W_p, 2, K) → (H_p, W_p) according to entropy_sel.

    entropy_sel keys (all optional):
        'channel'   : int 0 (real) or 1 (imag)
        'codebook'  : int 0..K-1          — single codebook
        'codebooks' : list[int]           — sum over this subset of codebooks

    None or {} → sum over all channels and codebooks.
    """
    arr = entropy_per_cb  # (H_p, W_p, 2, K)
    if not entropy_sel:
        return arr.sum(axis=(2, 3))
    ch  = entropy_sel.get("channel",   None)
    cb  = entropy_sel.get("codebook",  None)
    cbs = entropy_sel.get("codebooks", None)

    # select channel axis
    if ch is not None:
        arr = arr[:, :, ch:ch+1, :]  # keep dim: (H_p, W_p, 1, K)
    # select codebook axis
    if cb is not None:
        arr = arr[:, :, :, cb:cb+1]  # (H_p, W_p, ch, 1)
    elif cbs is not None:
        arr = arr[:, :, :, cbs]      # (H_p, W_p, ch, len(cbs))
    return arr.sum(axis=(2, 3))


def _sel_label(entropy_sel: dict | None) -> str:
    if not entropy_sel:
        return "all ch+cb"
    parts = []
    if "channel" in entropy_sel:
        parts.append(["real", "imag"][entropy_sel["channel"]])
    if "codebook" in entropy_sel:
        parts.append(f"cb{entropy_sel['codebook']}")
    elif "codebooks" in entropy_sel:
        parts.append(f"cb{entropy_sel['codebooks']}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_patch_entropy_data(
    run_dir: str | pathlib.Path,
    dataset: str,
    R: int,
    method: str,
    sample_id: str,
) -> dict:
    """Load all saved .npy files and done.json for one sample."""
    run_dir = pathlib.Path(run_dir)
    sdir = (
        run_dir / "patch_entropy"
        / f"dataset={dataset}" / f"R={R}"
        / f"method={method}" / f"sample_id={sample_id}"
    )
    if not sdir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sdir}")

    with open(sdir / "done.json") as f:
        metadata = _json_mod.load(f)

    B = max(r["step"] for r in metadata)
    gt_rss = np.load(str(sdir / "gt_rss.npy"))

    entropy_maps, entropy_per_cb, patch_error_maps, masks = [], [], [], []
    for t in range(B + 1):
        entropy_maps.append(np.load(str(sdir / f"entropy_map_t{t:03d}.npy")))
        entropy_per_cb.append(np.load(str(sdir / f"entropy_per_codebook_t{t:03d}.npy")))
        pem = sdir / f"patch_error_map_t{t:03d}.npy"
        patch_error_maps.append(np.load(str(pem)) if pem.exists() else None)
        masks.append(np.load(str(sdir / f"mask_t{t:03d}.npy")))

    kspace_scores = []
    for t in range(B):
        ks = sdir / f"kspace_scores_t{t:03d}.npy"
        kspace_scores.append(np.load(str(ks)) if ks.exists() else None)

    kl_from_t0 = []   # list[np.ndarray (H_p, W_p, 2, K) or None], length B (steps 1..B)
    for t in range(1, B + 1):
        kl_path = sdir / f"kl_from_t0_t{t:03d}.npy"
        kl_from_t0.append(np.load(str(kl_path)) if kl_path.exists() else None)

    selected_lines = [r.get("selected_line", -1) for r in metadata]
    H_p, W_p = entropy_maps[0].shape
    K = entropy_per_cb[0].shape[-1]

    rec_rss_path = sdir / "rec_rss.npy"
    rec_rss = np.load(str(rec_rss_path)) if rec_rss_path.exists() else None

    gt_ecb_path = sdir / "entropy_per_codebook_gt.npy"
    entropy_per_cb_gt  = np.load(str(gt_ecb_path))                           if gt_ecb_path.exists() else None
    entropy_map_gt     = np.load(str(sdir / "entropy_map_gt.npy"))            if (sdir / "entropy_map_gt.npy").exists() else None

    return {
        "gt_rss": gt_rss,
        "rec_rss": rec_rss,                    # (H, W) or None
        "entropy_maps": entropy_maps,
        "entropy_per_cb": entropy_per_cb,
        "entropy_per_cb_gt": entropy_per_cb_gt,   # (H_p, W_p, 2, K) or None
        "entropy_map_gt": entropy_map_gt,          # (H_p, W_p) or None
        "patch_error_maps": patch_error_maps,
        "kspace_scores": kspace_scores,
        "kl_from_t0": kl_from_t0,         # steps 1..B; None if not yet computed
        "masks": masks,
        "selected_lines": selected_lines,
        "metadata": metadata,
        "H_p": H_p, "W_p": W_p, "K": K, "B": B,
        "sample_id": sample_id,
    }


def list_patch_entropy_samples(
    run_dir: str | pathlib.Path,
    dataset: str,
    R: int,
    method: str,
) -> list[str]:
    """Return sorted list of sample_ids with a completed done.json."""
    run_dir = pathlib.Path(run_dir)
    base = run_dir / "patch_entropy" / f"dataset={dataset}" / f"R={R}" / f"method={method}"
    if not base.exists():
        return []
    return sorted(
        d.name.replace("sample_id=", "")
        for d in base.iterdir()
        if d.is_dir() and (d / "done.json").exists()
    )


# ---------------------------------------------------------------------------
# Viz 0 — Reference grid
# ---------------------------------------------------------------------------

def show_patch_reference_grid(data: dict):
    """GT RSS image with H_p×W_p grid overlay labelled with (row, col)."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    gt = data["gt_rss"]
    H, W = gt.shape
    H_p, W_p = data["H_p"], data["W_p"]
    ph, pw = H // H_p, W // W_p

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(gt, cmap="gray", origin="upper")
    for r in range(H_p):
        for c in range(W_p):
            ax.add_patch(mpatches.Rectangle(
                (c * pw, r * ph), pw, ph,
                linewidth=0.8, edgecolor="yellow", facecolor="none"
            ))
            ax.text(c * pw + pw / 2, r * ph + ph / 2,
                    f"({r},{c})", color="yellow", fontsize=7, ha="center", va="center")
    sid = data['sample_id']
    short_id = sid.split("_slice")[0].split("_")[-1] + "_" + sid.split("_slice")[-1] if "_slice" in sid else sid[-30:]
    ax.set_title(f"Ref grid  {short_id}", fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Viz 1 — Per-patch entropy over steps
# ---------------------------------------------------------------------------

def plot_patch_entropy_over_steps(
    data: dict, patch_row: int, patch_col: int, entropy_sel: dict | None = None,
):
    """Entropy of patch (patch_row, patch_col) vs acquisition step."""
    import matplotlib.pyplot as plt

    steps = list(range(data["B"] + 1))
    vals = [float(_select_entropy(data["entropy_per_cb"][t], entropy_sel)[patch_row, patch_col])
            for t in steps]
    argmax_steps = [
        t for t in steps
        if np.unravel_index(
            np.argmax(_select_entropy(data["entropy_per_cb"][t], entropy_sel)),
            (data["H_p"], data["W_p"])
        ) == (patch_row, patch_col)
    ]

    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    ax.plot(steps, vals, marker="o", linewidth=2.2, color="#1f77b4",
            label=f"patch ({patch_row},{patch_col})")
    if argmax_steps:
        ax.scatter(argmax_steps, [vals[t] for t in argmax_steps],
                   marker="*", s=120, color="red", zorder=5, label="argmax step")
    ax.set_xlabel("Acquisition step")
    ax.set_ylabel(f"Entropy [{_sel_label(entropy_sel)}]")
    ax.set_title(f"Patch ({patch_row},{patch_col}) entropy — {data['sample_id']}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Viz 2 — Selection progression
# ---------------------------------------------------------------------------

def plot_selection_progression(data: dict, entropy_sel: dict | None = None):
    """B panels: GT image + entropy heatmap + red argmax patch + cyan selected k-space line."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.cm import get_cmap

    B = data["B"]
    gt = data["gt_rss"]
    H, W = gt.shape
    H_p, W_p = data["H_p"], data["W_p"]
    ph, pw = H // H_p, W // W_p
    cmap = get_cmap("hot")

    ncols = min(B, 6)
    nrows = (B + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 2.5, nrows * 2.8), squeeze=False)

    for idx in range(B):
        t = idx + 1
        ax = axes[idx // ncols][idx % ncols]
        emap = _select_entropy(data["entropy_per_cb"][t], entropy_sel)
        emap_full = np.repeat(np.repeat(emap, ph, axis=0), pw, axis=1)
        norm_e = (emap_full - emap_full.min()) / max(emap_full.max() - emap_full.min(), 1e-8)

        ax.imshow(gt, cmap="gray", origin="upper")
        ax.imshow(cmap(norm_e), alpha=0.35, origin="upper")

        for r in range(H_p):
            for c in range(W_p):
                ax.add_patch(mpatches.Rectangle(
                    (c * pw, r * ph), pw, ph,
                    linewidth=0.4, edgecolor="white", facecolor="none"
                ))
        amax = np.unravel_index(np.argmax(emap), emap.shape)
        ax.add_patch(mpatches.Rectangle(
            (amax[1] * pw, amax[0] * ph), pw, ph,
            linewidth=2.0, edgecolor="red", facecolor="none"
        ))
        sel_line = data["selected_lines"][t] if t < len(data["selected_lines"]) else -1
        if sel_line >= 0:
            ax.axvline(x=sel_line, color="cyan", linewidth=1.5, alpha=0.8)
        ax.set_title(f"step {t}  patch=({amax[0]},{amax[1]})", fontsize=8)
        ax.axis("off")

    for idx in range(B, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle(f"Selection progression [{_sel_label(entropy_sel)}] — {data['sample_id']}", fontsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Viz 3 — Entropy reduction grid
# ---------------------------------------------------------------------------

def plot_entropy_reduction_grid(
    data: dict, relative: bool = True, entropy_sel: dict | None = None,
):
    """Heatmap of entropy change step 0 → B per patch."""
    import matplotlib.pyplot as plt

    e0 = _select_entropy(data["entropy_per_cb"][0],        entropy_sel)
    eB = _select_entropy(data["entropy_per_cb"][data["B"]], entropy_sel)
    reduction = (e0 - eB) / np.maximum(e0, 1e-8) if relative else (e0 - eB)
    clabel = "Relative Δ (Δ/H₀)" if relative else "Absolute Δ entropy"

    H_p, W_p = data["H_p"], data["W_p"]
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    ext = abs(reduction).max()
    im = ax.imshow(reduction, cmap="RdYlGn", origin="upper", vmin=-ext, vmax=ext)
    plt.colorbar(im, ax=ax, label=clabel, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(W_p)); ax.set_xticklabels(np.arange(W_p), fontsize=7)
    ax.set_yticks(np.arange(H_p)); ax.set_yticklabels(np.arange(H_p), fontsize=7)
    ax.set_xlabel("Patch column"); ax.set_ylabel("Patch row")
    ax.set_title(f"Entropy reduction (0→{data['B']}) [{_sel_label(entropy_sel)}]\n{data['sample_id']}", fontsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Viz 4 — Argmax patch entropy over steps
# ---------------------------------------------------------------------------

def plot_selected_patch_entropy(data: dict, entropy_sel: dict | None = None):
    """Entropy of the argmax patch at each step + global mean entropy."""
    import matplotlib.pyplot as plt

    steps = list(range(data["B"] + 1))
    argmax_vals, mean_vals, argmax_patches = [], [], []
    for t in steps:
        emap = _select_entropy(data["entropy_per_cb"][t], entropy_sel)
        mean_vals.append(float(emap.mean()))
        amax = np.unravel_index(np.argmax(emap), emap.shape)
        argmax_patches.append(amax)
        argmax_vals.append(float(emap[amax]))

    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    ax.plot(steps, argmax_vals, marker="o", linewidth=2.2, color="#d62728", label="max-entropy patch")
    ax.plot(steps, mean_vals,   marker="s", linewidth=1.8, linestyle="--",
            color="#1f77b4", label="global mean")
    for t, (r, c) in enumerate(argmax_patches):
        ax.annotate(f"({r},{c})", (t, argmax_vals[t]),
                    textcoords="offset points", xytext=(0, 6), fontsize=6, ha="center", color="#d62728")
    ax.set_xlabel("Acquisition step")
    ax.set_ylabel(f"Entropy [{_sel_label(entropy_sel)}]")
    ax.set_title(f"Max-entropy patch vs mean — {data['sample_id']}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Viz 5 — Entropy map sequence
# ---------------------------------------------------------------------------

def plot_entropy_map_sequence(data: dict, entropy_sel: dict | None = None):
    """All B+1 entropy heatmaps side by side with selected line marked."""
    import matplotlib.pyplot as plt

    B = data["B"]
    n = B + 1
    emaps = [_select_entropy(data["entropy_per_cb"][t], entropy_sel) for t in range(n)]
    vmin = min(e.min() for e in emaps)
    vmax = max(e.max() for e in emaps)
    _, W = data["gt_rss"].shape
    W_p = data["W_p"]

    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.8), squeeze=False)
    for t, (ax, emap) in enumerate(zip(axes[0], emaps)):
        im = ax.imshow(emap, cmap="hot", origin="upper", vmin=vmin, vmax=vmax)
        sel = data["selected_lines"][t] if t < len(data["selected_lines"]) else -1
        if sel >= 0:
            ax.axvline(x=sel / W * W_p - 0.5, color="cyan", linewidth=1.2, linestyle="--")
        ax.set_title(f"t={t}", fontsize=8); ax.axis("off")
    fig.colorbar(im, ax=axes[0].tolist(), label=f"Entropy [{_sel_label(entropy_sel)}]",
                 fraction=0.015, pad=0.01)
    fig.suptitle(f"Entropy map sequence — {data['sample_id']}", fontsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Viz 6 — Per-codebook / per-channel heatmaps
# ---------------------------------------------------------------------------

def plot_per_codebook_entropy(
    data: dict, step: int = 0, entropy_sel: dict | None = None,
):
    """2×K entropy heatmaps (rows=channels, cols=codebooks) at a given step.

    entropy_sel filters which panels to show (same syntax as other functions).
    """
    import matplotlib.pyplot as plt

    epcb = data["entropy_per_cb"][step]   # (H_p, W_p, 2, K)
    K = data["K"]
    ch_sel  = entropy_sel.get("channel",   None) if entropy_sel else None
    cb_sel  = entropy_sel.get("codebook",  None) if entropy_sel else None
    cbs_sel = entropy_sel.get("codebooks", None) if entropy_sel else None
    channels  = [ch_sel] if ch_sel is not None else [0, 1]
    codebooks = [cb_sel] if cb_sel is not None else (cbs_sel if cbs_sel is not None else list(range(K)))
    ch_names  = ["real", "imag"]

    nrows, ncols = len(channels), len(codebooks)
    all_vals = [epcb[:, :, ch, cb] for ch in channels for cb in codebooks]
    vmin = min(v.min() for v in all_vals)
    vmax = max(v.max() for v in all_vals)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(max(2.0 * ncols, 4), max(2.0 * nrows, 2.5)),
                              squeeze=False)
    for ri, ch in enumerate(channels):
        for ci, cb in enumerate(codebooks):
            ax = axes[ri][ci]
            im = ax.imshow(epcb[:, :, ch, cb], cmap="hot", origin="upper", vmin=vmin, vmax=vmax)
            ax.set_title(f"{ch_names[ch]} cb{cb}", fontsize=8); ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Entropy (nats)", fraction=0.02, pad=0.01)
    fig.suptitle(f"Per-codebook entropy  step={step} — {data['sample_id']}", fontsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Viz 7 — Entropy delta sequence
# ---------------------------------------------------------------------------

def plot_entropy_delta_sequence(data: dict, entropy_sel: dict | None = None):
    """B heatmaps of entropy_t - entropy_{t-1}. Green=decrease, red=increase."""
    import matplotlib.pyplot as plt

    B = data["B"]
    deltas = [
        _select_entropy(data["entropy_per_cb"][t],     entropy_sel)
        - _select_entropy(data["entropy_per_cb"][t-1], entropy_sel)
        for t in range(1, B + 1)
    ]
    vext = max(abs(d).max() for d in deltas)

    fig, axes = plt.subplots(1, B, figsize=(2.2 * B, 2.8), squeeze=False)
    for t, (ax, delta) in enumerate(zip(axes[0], deltas), start=1):
        im = ax.imshow(delta, cmap="RdYlGn", origin="upper", vmin=-vext, vmax=vext)
        sel = data["selected_lines"][t] if t < len(data["selected_lines"]) else -1
        ax.set_title(f"Δt={t}  line={sel}", fontsize=7); ax.axis("off")
    fig.colorbar(im, ax=axes[0].tolist(), label=f"ΔEntropy [{_sel_label(entropy_sel)}]",
                 fraction=0.015, pad=0.01)
    fig.suptitle(f"Entropy delta per step — {data['sample_id']}", fontsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Viz 8 — Entropy vs patch MSE scatter
# ---------------------------------------------------------------------------

def plot_entropy_mse_scatter(
    data: dict, step: int = 0, entropy_sel: dict | None = None,
):
    """Scatter: patch entropy vs patch MSE. Validates entropy as an uncertainty signal."""
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    emap = _select_entropy(data["entropy_per_cb"][step], entropy_sel).ravel()
    pmap = data["patch_error_maps"][step]
    if pmap is None:
        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
        ax.text(0.5, 0.5, "patch_error_map not available", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    pmap = pmap.ravel()
    rho, pval = spearmanr(emap, pmap)
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    ax.scatter(emap, pmap, alpha=0.7, s=40, color="#1f77b4", edgecolors="none")
    ax.set_xlabel(f"Patch entropy [{_sel_label(entropy_sel)}]")
    ax.set_ylabel("Patch MSE")
    ax.set_title(f"Entropy vs MSE  step={step}  ρ={rho:.3f}  p={pval:.3g}\n{data['sample_id']}", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Viz 9 — Mean entropy vs step
# ---------------------------------------------------------------------------

def plot_mean_entropy_vs_coverage(data: dict, entropy_sel: dict | None = None):
    """Mean patch entropy vs acquisition step."""
    import matplotlib.pyplot as plt

    steps = list(range(data["B"] + 1))
    means = [float(_select_entropy(data["entropy_per_cb"][t], entropy_sel).mean()) for t in steps]

    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    ax.plot(steps, means, marker="o", linewidth=2.2, color="#1f77b4")
    ax.set_xlabel("Acquisition step")
    ax.set_ylabel(f"Mean entropy [{_sel_label(entropy_sel)}]")
    ax.set_title(f"Global mean entropy vs step — {data['sample_id']}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_psnr_vs_step(data: dict):
    """Line plot of PSNR vs acquisition step for one slice."""
    import matplotlib.pyplot as plt

    steps = list(range(data["B"] + 1))
    psnrs = [row["PSNR"] for row in data["metadata"]]

    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    ax.plot(steps, psnrs, marker="o", linewidth=2.2, color="#2ca02c")
    ax.set_xlabel("Acquisition step")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"PSNR vs step — {data['sample_id']}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_entropy_per_codebook_bar(data: dict, step: int = 0):
    """Bar chart of mean patch entropy per codebook at a given step.

    Shows both real and imag channels side by side so you can see which
    codebook dominates and whether cb0 (no AR conditioning) is an outlier.
    """
    import matplotlib.pyplot as plt

    ecb = data["entropy_per_cb"][step]   # (H_p, W_p, 2, K)
    K = ecb.shape[-1]
    mean_re = ecb[:, :, 0, :].mean(axis=(0, 1))  # (K,)
    mean_im = ecb[:, :, 1, :].mean(axis=(0, 1))  # (K,)

    x = np.arange(K)
    width = 0.38
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    ax.bar(x - width/2, mean_re, width, label="real",  color="#1f77b4", alpha=0.85)
    ax.bar(x + width/2, mean_im, width, label="imag",  color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"cb{i}" for i in range(K)])
    ax.set_xlabel("Codebook index")
    ax.set_ylabel("Mean patch entropy (nats)")
    ax.set_title(f"Entropy per codebook  step={step} — {data['sample_id']}", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def plot_entropy_per_codebook_bar_dataset(
    run_dir: str | pathlib.Path,
    dataset: str,
    R: int,
    method: str,
    step: int = 0,
):
    """Bar chart of mean entropy per codebook aggregated over all samples.

    Only loads the one .npy file needed per sample (fast). Error bars show
    the std across samples (one mean-per-sample, then std over samples).
    """
    import matplotlib.pyplot as plt

    run_dir = pathlib.Path(run_dir)
    base = run_dir / "patch_entropy" / f"dataset={dataset}" / f"R={R}" / f"method={method}"
    sample_dirs = sorted(p for p in base.glob("sample_id=*") if (p / "done.json").exists())
    if not sample_dirs:
        raise FileNotFoundError(f"No completed samples found under {base}")

    sample_means_re, sample_means_im = [], []
    for sdir in sample_dirs:
        fpath = sdir / f"entropy_per_codebook_t{step:03d}.npy"
        if not fpath.exists():
            continue
        ecb = np.load(str(fpath))          # (H_p, W_p, 2, K)
        sample_means_re.append(ecb[:, :, 0, :].mean(axis=(0, 1)))  # (K,)
        sample_means_im.append(ecb[:, :, 1, :].mean(axis=(0, 1)))  # (K,)

    sample_means_re = np.stack(sample_means_re)  # (N, K)
    sample_means_im = np.stack(sample_means_im)  # (N, K)
    K = sample_means_re.shape[1]

    mean_re = sample_means_re.mean(axis=0)
    mean_im = sample_means_im.mean(axis=0)
    std_re  = sample_means_re.std(axis=0)
    std_im  = sample_means_im.std(axis=0)

    x = np.arange(K)
    width = 0.38
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    ax.bar(x - width/2, mean_re, width, yerr=std_re, capsize=3,
           label="real", color="#1f77b4", alpha=0.85)
    ax.bar(x + width/2, mean_im, width, yerr=std_im, capsize=3,
           label="imag", color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"cb{i}" for i in range(K)])
    ax.set_xlabel("Codebook index")
    ax.set_ylabel("Mean patch entropy (nats)")
    ax.set_title(
        f"Entropy per codebook  step={step}  N={len(sample_means_re)}\n"
        f"{dataset.capitalize()}  R={R}  {method}",
        fontsize=9,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def plot_gt_recon_comparison(data: dict):
    """GT image | Reconstruction | Difference map + metrics table.

    Intended for data loaded from run_gt_recon_experiment.py (METHOD='GT').
    Requires rec_rss.npy to be present (loaded into data['rec_rss']).
    """
    import matplotlib.pyplot as plt

    if data["rec_rss"] is None:
        raise ValueError("rec_rss not found — run run_gt_recon_experiment.py first.")

    gt  = data["gt_rss"]
    rec = data["rec_rss"]
    diff = np.abs(gt - rec)

    meta = data["metadata"][0]
    psnr = meta.get("PSNR",  float("nan"))
    ssim = meta.get("SSIM",  float("nan"))
    nmse = meta.get("NMSE",  float("nan"))
    ssfd = meta.get("SSFD",  float("nan"))

    vmin = min(gt.min(), rec.min())
    vmax = max(gt.max(), rec.max())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].imshow(gt,   cmap="gray", origin="upper", vmin=vmin, vmax=vmax)
    axes[0].set_title("GT (fully sampled)", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(rec,  cmap="gray", origin="upper", vmin=vmin, vmax=vmax)
    axes[1].set_title(
        f"FAR reconstruction (full mask)\n"
        f"PSNR={psnr:.2f} dB  SSIM={ssim:.4f}  NMSE={nmse:.4f}  SSFD={ssfd:.4f}",
        fontsize=9,
    )
    axes[1].axis("off")

    im = axes[2].imshow(diff, cmap="hot", origin="upper")
    axes[2].set_title("|GT − Rec|", fontsize=10)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"GT reconstruction — {data['sample_id']}", fontsize=9)
    plt.tight_layout()
    return fig


def plot_kl_map_sequence(data: dict, entropy_sel: dict | None = None):
    """B heatmaps of KL(p_t || p_t0) per patch, one per acquisition step.

    Answers: does acquiring a k-space line actually shift any patch distributions?
    High KL = distribution changed a lot relative to step-0 baseline.
    Requires re-running run_patch_entropy_experiment.py (now saves kl_from_t0_t*.npy).
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as mcm

    kl_list = data["kl_from_t0"]   # list of (H_p, W_p, 2, K) or None, length B
    if all(k is None for k in kl_list):
        raise ValueError("No KL data found — re-run run_patch_entropy_experiment.py.")

    B = data["B"]
    # Reduce (H_p, W_p, 2, K) → (H_p, W_p) using same selector as entropy
    kl_maps = [_select_entropy(k, entropy_sel) if k is not None else None for k in kl_list]

    valid = [m for m in kl_maps if m is not None]
    vmin, vmax = 0.0, max(m.max() for m in valid)

    ncols = min(B, 8)
    nrows = (B + ncols - 1) // ncols
    cmap  = mcm.get_cmap("hot")

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.2 * ncols, 2.2 * nrows + 0.4),
                             squeeze=False)
    sel_lines = data["selected_lines"]   # length B+1; index t is line chosen before step t

    for idx in range(B):
        t = idx + 1
        ax = axes[idx // ncols][idx % ncols]
        kmap = kl_maps[idx]
        if kmap is not None:
            im = ax.imshow(kmap, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
        ax.set_title(f"step {t}  line={sel_lines[t] if t < len(sel_lines) else '?'}", fontsize=7)
        ax.axis("off")

    for idx in range(B, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.colorbar(im, ax=axes.ravel().tolist(),
                 label=f"KL(p_t||p_t0) [{_sel_label(entropy_sel)}] (nats)",
                 fraction=0.02, pad=0.01)
    fig.suptitle(f"KL divergence from step-0 — {data['sample_id']}", fontsize=9)
    plt.tight_layout()
    return fig


def plot_gt_comparison_heatmaps(data: dict, entropy_sel: dict | None = None):
    """Side-by-side entropy heatmaps: GT | step-0 | step-B.

    Requires run_gt_entropy.py to have been run first (entropy_per_codebook_gt.npy).
    All three panels share the same colour scale.
    """
    import matplotlib.pyplot as plt

    if data["entropy_per_cb_gt"] is None:
        raise ValueError("GT entropy not found — run run_gt_entropy.py first.")

    e_gt  = _select_entropy(data["entropy_per_cb_gt"],        entropy_sel)
    e_t0  = _select_entropy(data["entropy_per_cb"][0],        entropy_sel)
    e_tB  = _select_entropy(data["entropy_per_cb"][data["B"]], entropy_sel)

    vmin = min(e_gt.min(), e_t0.min(), e_tB.min())
    vmax = max(e_gt.max(), e_t0.max(), e_tB.max())

    titles = ["GT (full k-space)", f"Step 0 (center only)", f"Step {data['B']} (final)"]
    maps   = [e_gt, e_t0, e_tB]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, title, emap in zip(axes, titles, maps):
        im = ax.imshow(emap, cmap="hot", origin="upper", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.colorbar(im, ax=axes.tolist(), label=f"Entropy [{_sel_label(entropy_sel)}] (nats)",
                 fraction=0.02, pad=0.01)
    fig.suptitle(f"GT vs reconstruction entropy — {data['sample_id']}", fontsize=9)
    plt.tight_layout()
    return fig


def plot_gt_vs_recon_bar_dataset(
    run_dir: str | pathlib.Path,
    dataset: str,
    R: int,
    method: str,
):
    """Per-codebook bar chart comparing GT, step-0, and step-B entropy across all samples.

    Three grouped bars per codebook (GT / step-0 / step-B), averaged over dataset.
    Error bars = std across samples.
    Requires run_gt_entropy.py to have been run first.
    """
    import matplotlib.pyplot as plt

    run_dir = pathlib.Path(run_dir)
    base = run_dir / "patch_entropy" / f"dataset={dataset}" / f"R={R}" / f"method={method}"
    sample_dirs = sorted(
        p for p in base.glob("sample_id=*")
        if (p / "done.json").exists() and (p / "entropy_per_codebook_gt.npy").exists()
    )
    if not sample_dirs:
        raise FileNotFoundError(f"No samples with GT entropy found under {base}")

    gt_re, gt_im = [], []
    t0_re, t0_im = [], []
    tB_re, tB_im = [], []

    for sdir in sample_dirs:
        with open(sdir / "done.json") as f:
            meta = _json_mod.load(f)
        B = max(r["step"] for r in meta)

        ecb_gt = np.load(str(sdir / "entropy_per_codebook_gt.npy"))
        ecb_t0 = np.load(str(sdir / "entropy_map_t000.npy".replace("entropy_map", "entropy_per_codebook")))
        ecb_tB = np.load(str(sdir / f"entropy_per_codebook_t{B:03d}.npy"))

        gt_re.append(ecb_gt[:, :, 0, :].mean(axis=(0, 1)))
        gt_im.append(ecb_gt[:, :, 1, :].mean(axis=(0, 1)))
        t0_re.append(ecb_t0[:, :, 0, :].mean(axis=(0, 1)))
        t0_im.append(ecb_t0[:, :, 1, :].mean(axis=(0, 1)))
        tB_re.append(ecb_tB[:, :, 0, :].mean(axis=(0, 1)))
        tB_im.append(ecb_tB[:, :, 1, :].mean(axis=(0, 1)))

    def _agg(lst):
        arr = np.stack(lst)
        return arr.mean(0), arr.std(0)

    K = np.stack(gt_re).shape[1]
    x = np.arange(K)
    w = 0.25

    conditions = [
        ("GT",     _agg(gt_re), _agg(gt_im),  "#2ca02c"),
        ("step-0", _agg(t0_re), _agg(t0_im),  "#1f77b4"),
        (f"step-{B}", _agg(tB_re), _agg(tB_im), "#d62728"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, ch_idx, ch_name in zip(axes, [0, 1], ["real", "imag"]):
        for k, (label, (mean_re, std_re), (mean_im, std_im), color) in enumerate(conditions):
            vals = mean_re if ch_idx == 0 else mean_im
            errs = std_re  if ch_idx == 0 else std_im
            ax.bar(x + (k - 1) * w, vals, w, yerr=errs, capsize=3,
                   label=label, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"cb{i}" for i in range(K)])
        ax.set_xlabel("Codebook index")
        ax.set_ylabel("Mean entropy (nats)")
        ax.set_title(f"{ch_name} channel")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"GT vs step-0 vs step-B entropy per codebook  N={len(sample_dirs)}\n"
        f"{dataset.capitalize()}  R={R}  {method}",
        fontsize=9,
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Token accuracy
# ---------------------------------------------------------------------------

def plot_token_accuracy(
    run_dir: str | pathlib.Path,
    dataset: str,
    R: int,
    methods: list[str] | None = None,
    ks: list[int] | None = None,
):
    """Line plots of top-k accuracy and NLL vs acquisition step.

    Four panels: top-1, top-5, top-10, NLL.
    Each line = one method (LES_ZM, GEO, RANDOM).
    Shaded band = ±1 std across scans (scan-mean first, then std over scans).

    Requires run_token_accuracy.py output.
    """
    import matplotlib.pyplot as plt

    run_dir = pathlib.Path(run_dir)
    methods = methods or ["LES_ZM", "GEO", "RANDOM", "RANDOM_PER_STEP"]
    ks      = ks      or [1, 5, 10]

    METHOD_COLORS = {
        "LES_ZM":         "#1f77b4",
        "GEO":            "#ff7f0e",
        "RANDOM":         "#7f7f7f",
        "RANDOM_PER_STEP":"#bcbd22",
    }
    METHOD_LABELS = {
        "LES_ZM":         "LES (zero-mean)",
        "GEO":            "GEO",
        "RANDOM":         "Random (cumulative)",
        "RANDOM_PER_STEP":"Random (per step)",
    }

    metrics = [f"top{k}" for k in ks] + ["nll"]
    ylabels = [f"Top-{k} accuracy" for k in ks] + ["NLL (nats)"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4), squeeze=False)
    axes = axes[0]

    for method in methods:
        parquet = (
            run_dir
            / f"dataset={dataset}" / f"R={R}"
            / f"method={method}" / "metadata.parquet"
        )
        if parquet.exists():
            df = pd.read_parquet(str(parquet))
        else:
            # Fall back to per-slice JSON files (experiment may still be running)
            json_files = sorted(
                (run_dir / f"dataset={dataset}" / f"R={R}")
                .glob(f"sample_id=*/accuracy_{method}.json")
            )
            if not json_files:
                print(f"  No data for {method} (parquet missing, no JSON files found)")
                continue
            rows = []
            for jf in json_files:
                rows.extend(_json_mod.loads(jf.read_text()))
            df = pd.DataFrame(rows)
            print(f"  {method}: loaded {len(json_files)} JSON files ({len(df)} rows)")

        # Per-scan mean at each step, then mean/std across scans
        scan_step = df.groupby(["sample_id", "step"])[metrics].mean()
        step_stats = scan_step.groupby("step")[metrics].agg(["mean", "std"])

        steps = step_stats.index.tolist()
        color = METHOD_COLORS.get(method, None)
        label = METHOD_LABELS.get(method, method)

        for ax, metric, ylabel in zip(axes, metrics, ylabels):
            mean = step_stats[(metric, "mean")].values
            std  = step_stats[(metric, "std")].values
            ax.plot(steps, mean, label=label, color=color, linewidth=2.2)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)

    for ax, metric, ylabel in zip(axes, metrics, ylabels):
        ax.set_xlabel("Acquisition step")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Token accuracy vs acquisition step — {dataset.capitalize()}  R={R}",
        fontsize=10,
    )
    plt.tight_layout()
    return fig
