#!/usr/bin/env python
"""Score eval_varnet_baseline.py outputs with the TRUST-MRI metrics (per-volume PSNR etc.) against the oracle SENSE
image (reconstruction x oracle support vs |x0|) and against the RSS image (no support), and write a summary.json in
the results/sense_block format (plus summary['vs_rss']) so scripts/summarize_sense_block.py can tabulate it.
Run with /opt/miniconda3/envs/meditok/bin/python (piq)."""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoregressive.utils.metrics import compute_reconstruction_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon-dir", required=True, help="output dir of eval_varnet_baseline.py")
    ap.add_argument("--out-dir", required=True, help="results/sense_block/<run>")
    args = ap.parse_args()
    rd, out = Path(args.recon_dir), Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    info = json.load(open(rd / "info.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bufs, bufs_rss = {}, {}
    for p in sorted((rd / "h5").glob("*.h5")):
        with h5py.File(p, "r") as f:
            d = {k: f[k][()] for k in f.keys()}
        key = p.stem; sl = list(range(d["x0_ref"].shape[0])); sup = d["support"].astype(np.float32)
        arms = [k for k in d if k.endswith("_rss") or k.endswith("_sense")]
        for arm in arms:
            b = bufs.setdefault(arm, {}).setdefault(key, {"rec": [], "gt": [], "slice": []})
            b["rec"] += list(d[arm] * sup); b["gt"] += list(d["x0_ref"]); b["slice"] += sl
            b = bufs_rss.setdefault(arm, {}).setdefault(key, {"rec": [], "gt": [], "slice": []})
            b["rec"] += list(d[arm]); b["gt"] += list(d["rss_ref"]); b["slice"] += sl
    summary = {"acceleration": info["acceleration"], "center_fractions": info["center_fractions"], "split": info["split"],
               "gpt_ckpt": info["ckpt"], "encoder_ckpt": "varnet", "sens_maps": "varnet (combine: oracle)", "eval_support": "oracle",
               "gt_ref": "both", "model": info["model"], "n_volumes": info["n_volumes"], "vs_rss": {}}
    for arm in bufs:
        m = compute_reconstruction_metrics(bufs[arm], device)
        json.dump(m, open(out / f"metrics_{arm}.json", "w"), indent=2)
        summary[arm] = {k: float(v) for k, v in m.items() if k.endswith("_mean")}
        m2 = compute_reconstruction_metrics(bufs_rss[arm], device)
        json.dump(m2, open(out / f"metrics_{arm}_vs_rss.json", "w"), indent=2)
        summary["vs_rss"][arm] = {k: float(v) for k, v in m2.items() if k.endswith("_mean")}
        print(f"{arm:14s} vs SENSE: PSNR {m['psnr_mean']:.3f} SSIM {m['ssim_mean']:.4f} LPIPS {m['lpips_mean']:.4f} | "
              f"vs RSS: PSNR {m2['psnr_mean']:.3f} SSIM {m2['ssim_mean']:.4f} LPIPS {m2['lpips_mean']:.4f}")
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print("->", out / "summary.json")


if __name__ == "__main__":
    main()
