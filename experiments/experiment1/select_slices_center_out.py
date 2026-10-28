#!/usr/bin/env python
"""Select slices for Experiment 1 using center-out ordering.

Writes:
    <run_dir>/metadata/selected_slices_{dataset}_R{acc}.csv
    <run_dir>/metadata/selected_slices_{dataset}_R{acc}.json
    <run_dir>/manifest.json

Usage:
    python select_slices_center_out.py \\
        --config experiments/experiment1/config.yaml \\
        --run_dir /fast/lev/TRUST-MRI/results/exp1
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.experiment1.utils.slice_selector import select_slices_center_out
from experiments.experiment1.utils.storage import write_manifest


def parse_args():
    p = argparse.ArgumentParser(description="Center-out slice selection for Experiment 1")
    p.add_argument("--config", default="experiments/experiment1/config.yaml",
                   help="Path to experiment config YAML (relative to TRUST-MRI root or absolute)")
    p.add_argument("--run_dir", required=True,
                   help="Output directory for this experiment run")
    p.add_argument("--data_path_brain", default=None,
                   help="Override brain singlecoil directory (default: from config)")
    p.add_argument("--data_path_knee", default=None,
                   help="Override knee singlecoil directory (default: from config)")
    p.add_argument("--split_file_brain", default=None,
                   help="Override brain split file (default: from config)")
    p.add_argument("--split_file_knee", default=None,
                   help="Override knee split file (default: from config)")
    return p.parse_args()


def _resolve(path: str, root: pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = root / p
    return p


def main():
    args = parse_args()
    root = _ROOT

    config_path = _resolve(args.config, root)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    run_dir = pathlib.Path(args.run_dir)
    meta_dir = run_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    data_paths = {
        "brain": args.data_path_brain or _resolve(cfg["data_paths"]["brain"], root),
        "knee":  args.data_path_knee  or _resolve(cfg["data_paths"]["knee"],  root),
    }
    split_files = {
        "brain": args.split_file_brain or _resolve(cfg["split_files"]["brain"], root),
        "knee":  args.split_file_knee  or _resolve(cfg["split_files"]["knee"],  root),
    }

    all_records: dict[str, list[dict]] = {}

    for dataset in cfg["datasets"]:
        for acc in cfg["accelerations"]:
            print(f"Selecting slices: dataset={dataset}, R={acc}")
            records = select_slices_center_out(
                data_root=data_paths[dataset],
                split_file=split_files[dataset],
                dataset=dataset,
                acceleration=acc,
                num_slices=cfg.get("num_slices_heavy"),
                seed=cfg.get("seed", 42),
            )
            key = f"{dataset}_R{acc}"
            all_records[key] = records
            print(f"  → {len(records)} slices selected from {data_paths[dataset]}")

            # Save CSV
            csv_path = meta_dir / f"selected_slices_{dataset}_R{acc}.csv"
            if records:
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                    writer.writeheader()
                    writer.writerows(records)

            # Save JSON
            json_path = meta_dir / f"selected_slices_{dataset}_R{acc}.json"
            with open(json_path, "w") as f:
                json.dump(records, f, indent=2)

    # Write manifest
    write_manifest(
        run_dir=run_dir,
        config=cfg,
        extra={
            "selected_slices": {
                k: str(meta_dir / f"selected_slices_{k.split('_')[0]}_R{k.split('R')[1]}.json")
                for k in all_records
            },
            "num_slices": {k: len(v) for k, v in all_records.items()},
        },
    )
    print(f"Manifest written to {run_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
