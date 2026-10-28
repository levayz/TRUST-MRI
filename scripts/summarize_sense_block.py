#!/usr/bin/env python
"""Tabulate results/sense_block/*/summary.json (eval_sense_block.py runs): one row per run x arm.

Usage: python scripts/summarize_sense_block.py [--root results/sense_block] [--glob 'suno_acc8_*'] [--md]
"""
import argparse
import json
from pathlib import Path

ARMS = ("zero_filled", "greedy_no_dc", "greedy_dc", "ps_mean_no_dc", "ps_mean_dc", "ps_single_dc", "gt_reim", "gt_reim_dc", "gt_mag")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/sense_block")
    p.add_argument("--glob", default="*")
    p.add_argument("--arms", nargs="+", default=["zero_filled", "greedy_no_dc", "greedy_dc", "ps_mean_dc", "gt_reim_dc"])
    p.add_argument("--md", action="store_true", help="markdown table")
    p.add_argument("--ref", default="sense", choices=["sense", "rss"], help="metrics vs the oracle SENSE image (default) or vs the RSS image (runs made with --gt-ref rss|both)")
    a = p.parse_args()
    rows = []
    for d in sorted(Path(a.root).glob(a.glob)):
        f = d / "summary.json"
        if not f.exists():
            continue
        s = json.load(open(f))
        gpt = Path(s.get("gpt_ckpt") or "none")
        gpt_tag = gpt.parent.parent.name if gpt.name == "best.pt" and gpt.parent.name == "checkpoints" else gpt.parent.name + "/" + gpt.name
        tok = Path(s.get("encoder_ckpt", "")).stem if Path(s.get("encoder_ckpt", "")).parent.name == "meditok" else Path(s.get("encoder_ckpt", "")).parent.name
        tok = f"{tok}@{s['token_size']}" if s.get("token_size") and s.get("token_size") != s.get("crop_size", 256) else tok
        ag = s.get("map_agreement_support", {}).get("mean")
        src = s if a.ref == "sense" else s.get("vs_rss", {})
        for arm in a.arms:
            if arm in src:
                rows.append((d.name, f"R{s['acceleration']}", s.get("sens_maps", "oracle"), (s.get("eval_support", "none") if a.ref == "sense" else "rss-ref"), gpt_tag, tok,
                             "-" if ag is None else f"{ag:.3f}", arm, src[arm]["psnr_mean"], src[arm]["ssim_mean"], src[arm].get("lpips_mean", float("nan"))))
    hdr = ("run", "R", "maps", "support", "gpt", "tokenizer", "agree", "arm", "PSNR", "SSIM", "LPIPS")
    if a.md:
        print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
        for r in rows:
            print("| " + " | ".join([*map(str, r[:8]), f"{r[8]:.2f}", f"{r[9]:.4f}", f"{r[10]:.4f}"]) + " |")
    else:
        w = [max(len(str(x)) for x in col) for col in zip(hdr, *[[*map(str, r[:8]), f"{r[8]:.2f}", f"{r[9]:.4f}", f"{r[10]:.4f}"] for r in rows])] if rows else [len(h) for h in hdr]
        print("  ".join(h.ljust(wi) for h, wi in zip(hdr, w)))
        for r in rows:
            cells = [*map(str, r[:8]), f"{r[8]:.2f}", f"{r[9]:.4f}", f"{r[10]:.4f}"]
            print("  ".join(c.ljust(wi) for c, wi in zip(cells, w)))


if __name__ == "__main__":
    main()
