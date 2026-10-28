#!/usr/bin/env python
"""Analyses 4, 5, 6: Candidate-Line Utility (Entropy Reduction, Scores, Controls).

Loads candidate-counterfactual parquet files and computes:

  Analysis 4 — Entropy reduction vs reconstruction improvement:
    Spearman(delta_entropy, delta_mse/ssfd/token_acc) per step

  Analysis 5 — LES/GEO score vs actual line improvement:
    Spearman(LES/GEO score, delta_*), Recall@5, Regret@1

  Analysis 6 — Negative controls:
    Same metrics for random / shuffled_entropy / uniform / center_distance / kspace_mag scores

Writes:
    <run_dir>/results/tables/candidate_line_utility_{dataset}_R{acc}_{method}.csv
    <run_dir>/results/plots/candidate_line_utility_*.png

Usage:
    python analyze_candidate_line_utility.py \\
        --run_dir /fast/lev/TRUST-MRI/results/exp1 \\
        --dataset brain --acceleration 8 --base_method LES
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.experiment1.utils.storage import load_parquet


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return float("nan")
    r, _ = spearmanr(x[valid], y[valid])
    return float(r)


def _recall_at_k(scores: np.ndarray, oracle_scores: np.ndarray, k: int = 5) -> float:
    """Recall@k: fraction of oracle top-k lines that appear in model top-k."""
    if len(scores) <= k:
        return float("nan")
    top_model  = set(np.argsort(scores)[-k:])
    top_oracle = set(np.argsort(oracle_scores)[-k:])
    return len(top_model & top_oracle) / k


def _regret_at_1(scores: np.ndarray, improvement: np.ndarray) -> float:
    """Regret@1: oracle_best_improvement - improvement_of_model_top1_choice."""
    best_model_idx  = int(np.argmax(scores))
    oracle_best_imp = float(improvement.max())
    model_choice_imp = float(improvement[best_model_idx])
    return oracle_best_imp - model_choice_imp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze(
    run_dir: pathlib.Path,
    dataset: str,
    acceleration: int,
    base_method: str,
) -> None:
    cf_root = (
        run_dir / "candidate_counterfactuals"
        / f"dataset={dataset}" / f"R={acceleration}" / f"base_method={base_method}"
    )
    if not cf_root.exists():
        print(f"No candidate counterfactuals at {cf_root}")
        return

    parquet_files = sorted(cf_root.glob("sample_id=*/step_*.parquet"))
    if not parquet_files:
        print("No parquet files found.")
        return

    all_dfs = []
    for pf in tqdm(parquet_files, desc="Loading parquets"):
        try:
            df = load_parquet(pf)
            all_dfs.append(df)
        except Exception as e:
            print(f"  SKIP {pf}: {e}")
    if not all_dfs:
        return

    data = pd.concat(all_dfs, ignore_index=True)

    # Improvement = reduction (negative delta means improved for MSE/SSFD)
    data["imp_mse"]  = -data["delta_mse"]       # higher is better
    data["imp_ssfd"] = -data["delta_ssfd"]
    data["imp_tok"]  = data["delta_token_accuracy"]  # higher is better

    SCORE_COLS = [
        ("LES_score",             "LES"),
        ("GEO_score",             "GEO"),
        ("random_score",          "Random"),
        ("shuffled_entropy_score","ShuffledEntropy"),
        ("uniform_entropy_score", "Uniform"),
        ("center_distance_score", "CenterDist"),
        ("kspace_magnitude_score","KspaceMag"),
        ("delta_entropy",         "EntropyReduction"),  # Analysis 4
    ]

    TARGETS = [
        ("imp_mse",  "MSE improvement"),
        ("imp_ssfd", "SSFD improvement"),
        ("imp_tok",  "Token accuracy improvement"),
    ]

    rows = []
    for (score_col, score_name), (target_col, target_name) in (
        (s, t) for s in SCORE_COLS for t in TARGETS
    ):
        if score_col not in data.columns or target_col not in data.columns:
            continue
        grouped = data.groupby(["sample_id", "step"])
        spearman_vals, recall5_vals, regret1_vals = [], [], []

        for (sid, step), grp in grouped:
            sc = grp[score_col].values.astype(np.float32)
            tg = grp[target_col].values.astype(np.float32)
            # For delta_entropy: smaller (more negative) = better
            sc_for_recall = -sc if score_col == "delta_entropy" else sc

            spearman_vals.append(_spearman(sc, tg))
            oracle_target = tg  # oracle picks by improvement
            recall5_vals.append(_recall_at_k(sc_for_recall, oracle_target, k=5))
            regret1_vals.append(_regret_at_1(sc_for_recall, oracle_target))

        rows.append({
            "dataset":     dataset,
            "acceleration": acceleration,
            "base_method": base_method,
            "score":       score_name,
            "target":      target_name,
            "spearman_r":  float(np.nanmean(spearman_vals)),
            "recall_at_5": float(np.nanmean(recall5_vals)),
            "regret_at_1": float(np.nanmean(regret1_vals)),
            "n_groups":    len(spearman_vals),
        })

    df_out = pd.DataFrame(rows)
    out_dir = run_dir / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"candidate_line_utility_{dataset}_R{acceleration}_{base_method}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved {len(df_out)} rows to {out_path}")

    _plot(df_out, run_dir, dataset, acceleration, base_method)


def _plot(df, run_dir, dataset, acc, base_method):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plot_dir = run_dir / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{dataset}_R{acc}_{base_method}"

    scores = df["score"].unique()
    targets = df["target"].unique()
    for target in targets:
        sub = df[df["target"] == target].set_index("score")
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(scores))
        spr = [float(sub.loc[s, "spearman_r"]) if s in sub.index else float("nan") for s in scores]
        ax.bar(x, spr)
        ax.set_xticks(x)
        ax.set_xticklabels(scores, rotation=30, ha="right")
        ax.set_ylabel("Spearman r")
        ax.set_title(f"Score vs {target} — {dataset} R={acc} {base_method}")
        ax.axhline(0, ls="--", color="gray")
        plt.tight_layout()
        target_slug = target.replace(" ", "_").lower()
        plt.savefig(plot_dir / f"candidate_utility_{tag}_{target_slug}.png", dpi=120)
        plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--acceleration", required=True, type=int)
    p.add_argument("--base_method", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    analyze(pathlib.Path(args.run_dir), args.dataset, args.acceleration, args.base_method)


if __name__ == "__main__":
    main()
