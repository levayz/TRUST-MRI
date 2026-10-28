#!/usr/bin/env python
"""TDS hyperparameter sweep.

Generates (and optionally runs) all TDS experiment configurations described in
the initial experiment grid:

  1. Plain TDS baseline (distance=js, percentile fg, alpha=1, beta=0, all cands)
  2. TDS-ER: vary beta in [0.25, 0.5, 1.0]
  3. Foreground ablation: none / percentile / top_m in [32, 64, 128]
  4. Candidate shortlist: geo_topk with K in [16, 32]
  5. Distance ablation: symmetric_kl / topk_l1

Usage:
    # Dry-run: print commands only
    python experiments/experiment1/run_tds_sweep.py --dry_run

    # Generate a shell script
    python experiments/experiment1/run_tds_sweep.py --output_script run_tds_sweep.sh

    # Run all serially (slow)
    python experiments/experiment1/run_tds_sweep.py --run

    # Run a specific group
    python experiments/experiment1/run_tds_sweep.py --run --groups baseline tds_er
"""

from __future__ import annotations

import argparse
import pathlib
import shlex
import subprocess
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Sweep configurations
# ---------------------------------------------------------------------------

BASE = dict(
    config="experiments/experiment1/config.yaml",
    method="TDS",
    tds_distance="js",
    tds_foreground_mode="percentile",
    tds_foreground_percentile=20,
    tds_alpha=1.0,
    tds_beta=0.0,
    tds_candidate_mode="all",
    tds_entropy_normalization=True,
    tds_line_aggregation="sum",
    tds_candidate_batch_size=16,
)

SWEEP_GROUPS: dict[str, list[dict]] = {
    # 1. Plain TDS baseline (suggested first run)
    "baseline": [
        {**BASE},
    ],

    # 2. TDS-ER: entropy-reduction weight
    "tds_er": [
        {**BASE, "tds_beta": 0.25},
        {**BASE, "tds_beta": 0.50},
        {**BASE, "tds_beta": 1.00},
    ],

    # 3. Foreground ablation
    "foreground": [
        {**BASE, "tds_foreground_mode": "none"},
        {**BASE, "tds_foreground_mode": "percentile", "tds_foreground_percentile": 20},
        {**BASE, "tds_foreground_mode": "top_m_entropy_foreground", "tds_top_m": 32},
        {**BASE, "tds_foreground_mode": "top_m_entropy_foreground", "tds_top_m": 64},
        {**BASE, "tds_foreground_mode": "top_m_entropy_foreground", "tds_top_m": 128},
    ],

    # 4. Candidate shortlist
    "candidate_shortlist": [
        {**BASE, "tds_candidate_mode": "geo_topk", "tds_candidate_k": 16},
        {**BASE, "tds_candidate_mode": "geo_topk", "tds_candidate_k": 32},
    ],

    # 5. Distance metric ablation
    "distance": [
        {**BASE, "tds_distance": "symmetric_kl"},
        {**BASE, "tds_distance": "topk_l1"},
    ],
}


# ---------------------------------------------------------------------------
# Tag builder
# ---------------------------------------------------------------------------

def _config_tag(cfg: dict) -> str:
    """Human-readable tag derived from non-default TDS config values."""
    parts = []
    if cfg.get("tds_distance", "js") != "js":
        parts.append(f"dist={cfg['tds_distance']}")
    fg = cfg.get("tds_foreground_mode", "percentile")
    if fg != "percentile":
        parts.append(f"fg={fg}")
    if fg == "percentile":
        pct = cfg.get("tds_foreground_percentile", 20)
        if pct != 20:
            parts.append(f"pct={pct}")
    if fg == "top_m_entropy_foreground":
        parts.append(f"m={cfg.get('tds_top_m', 64)}")
    beta = cfg.get("tds_beta", 0.0)
    if beta != 0.0:
        parts.append(f"beta={beta}")
    cm = cfg.get("tds_candidate_mode", "all")
    if cm != "all":
        parts.append(f"cand={cm}_k{cfg.get('tds_candidate_k', 16)}")
    return "TDS_" + ("_".join(parts) if parts else "base")


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def build_command(
    cfg: dict,
    run_dir: str,
    dataset: str,
    acceleration: int,
    num_scans: int,
    extra: list[str] | None = None,
) -> list[str]:
    tag = _config_tag(cfg)
    effective_run_dir = str(pathlib.Path(run_dir) / tag)

    cmd = [
        sys.executable,
        str(_ROOT / "experiments/experiment1/run_entropy_acquisition.py"),
        "--config", str(_ROOT / cfg.get("config", "experiments/experiment1/config.yaml")),
        "--run_dir", effective_run_dir,
        "--dataset", dataset,
        "--acceleration", str(acceleration),
        "--method", cfg.get("method", "TDS"),
        "--num_scans", str(num_scans),
    ]

    # TDS-specific flags
    tds_keys = [
        "tds_distance", "tds_foreground_mode", "tds_foreground_threshold",
        "tds_foreground_percentile", "tds_top_m", "tds_alpha", "tds_beta",
        "tds_candidate_mode", "tds_candidate_k", "tds_entropy_normalization",
        "tds_line_aggregation", "tds_candidate_batch_size",
    ]
    for key in tds_keys:
        if key in cfg:
            cmd += [f"--{key}", str(cfg[key])]

    if extra:
        cmd += extra

    return cmd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="TDS sweep script")
    p.add_argument("--run_dir", default="results/tds_sweep",
                   help="Root directory for all sweep runs")
    p.add_argument("--dataset", default="brain", choices=["brain", "knee"])
    p.add_argument("--acceleration", default=8, type=int, choices=[8, 16])
    p.add_argument("--num_scans", default=10, type=int,
                   help="Scans per config (use small value for quick tests)")
    p.add_argument("--groups", nargs="*", choices=list(SWEEP_GROUPS.keys()),
                   default=list(SWEEP_GROUPS.keys()),
                   help="Which sweep groups to include")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without running")
    p.add_argument("--run", action="store_true",
                   help="Actually run the commands serially")
    p.add_argument("--output_script", default=None,
                   help="Write all commands to a shell script")
    p.add_argument("--extra", nargs="*", default=[],
                   help="Extra flags appended to every command")
    return p.parse_args()


def main():
    args = parse_args()

    all_commands: list[list[str]] = []
    for group in args.groups:
        for cfg in SWEEP_GROUPS[group]:
            cmd = build_command(
                cfg,
                run_dir=args.run_dir,
                dataset=args.dataset,
                acceleration=args.acceleration,
                num_scans=args.num_scans,
                extra=args.extra,
            )
            all_commands.append(cmd)

    print(f"Sweep: {len(all_commands)} configs across {len(args.groups)} group(s)")
    print()

    if args.output_script:
        script_path = pathlib.Path(args.output_script)
        lines = ["#!/usr/bin/env bash", "set -e", ""]
        for cmd in all_commands:
            lines.append(shlex.join(cmd))
        script_path.write_text("\n".join(lines) + "\n")
        script_path.chmod(0o755)
        print(f"Wrote {len(all_commands)} commands → {script_path}")

    if args.dry_run or not args.run:
        for cmd in all_commands:
            print(shlex.join(cmd))
        if not args.run:
            return

    for i, cmd in enumerate(all_commands, 1):
        print(f"\n[{i}/{len(all_commands)}] {shlex.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(_ROOT))
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode})")
            sys.exit(result.returncode)

    print(f"\nSweep complete: {len(all_commands)} configs finished.")


if __name__ == "__main__":
    main()
