#!/usr/bin/env python
"""Summarize a ps_recon top-k sweep.

Reads every ``cf_*_acc_<ACC>_val_cart_1d_random_ps_recon_top<K>_n<N>_<split>/`` directory under
--sweep-dir, collects metrics.json (PS-mean reconstruction), metrics_greedy.json and
metrics_single_sample.json, and writes sweep_summary.md and sweep_summary.json next to them.
Also prints the k with the best PS-mean PSNR averaged over the accelerations present.

usage: python scripts/summarize_ps_sweep.py --sweep-dir checkpoints/brain/cart_1D/eval/ps_recon_topk_sweep
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

PAT = re.compile(r"cf_(?P<cf>[\d.]+)_acc_(?P<acc>\d+)_val_.*_ps_recon_top(?P<k>\d+)_n(?P<n>\d+)_(?P<split>\w+)$")
ARMS = (("greedy", "metrics_greedy.json"), ("single_sample", "metrics_single_sample.json"), ("ps_mean", "metrics.json"))
METRICS = ("psnr", "ssim", "lpips", "dists", "nmse")


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--select-metric", default="psnr", choices=METRICS)
    ap.add_argument("--vary", default="k", choices=["k", "n"], help="swept parameter: top-k (k) or number of posterior samples (n)")
    args = ap.parse_args()
    sweep = Path(args.sweep_dir)

    rows = []
    for d in sorted(sweep.iterdir()):
        m = PAT.match(d.name)
        if not d.is_dir() or not m:
            continue
        row = {"dir": d.name, "acc": int(m["acc"]), "k": int(m["k"]), "n": int(m["n"]), "split": m["split"]}
        ps = load(d / "metrics.json")
        row["status"] = "ok" if ps and "psnr_mean" in ps else "incomplete"
        for arm, fn in ARMS:
            mm = load(d / fn) or {}
            for met in METRICS:
                row[f"{arm}_{met}"] = mm.get(f"{met}_mean")
        row["ps_config"] = (ps or {}).get("ps_config")
        rows.append(row)
    V = args.vary
    rows.sort(key=lambda r: (r["acc"], r[V]))

    ok = [r for r in rows if r["status"] == "ok"]
    accs = sorted({r["acc"] for r in ok})
    sel = args.select_metric
    higher_is_better = sel in ("psnr", "ssim")

    # per-k average of the PS-mean metric over accelerations (only k present for every acc)
    by_k = defaultdict(dict)
    for r in ok:
        by_k[r[V]][r["acc"]] = r[f"ps_mean_{sel}"]
    avg = {k: sum(v.values()) / len(v) for k, v in by_k.items() if len(v) == len(accs) and all(x is not None for x in v.values())}
    best_k = (max if higher_is_better else min)(avg, key=avg.get) if avg else None
    best_per_acc = {}
    for acc in accs:
        cand = [r for r in ok if r["acc"] == acc and r[f"ps_mean_{sel}"] is not None]
        if cand:
            best_per_acc[acc] = (max if higher_is_better else min)(cand, key=lambda r: r[f"ps_mean_{sel}"])[V]

    # markdown
    lines = [f"# ps_recon sweep over {V} ({'top-k' if V == 'k' else 'number of posterior samples'}): `{sweep}`", ""]
    for acc in accs:
        lines += [f"## acc x{acc}", "",
                  "| k | n | greedy PSNR | single-sample PSNR | PS-mean PSNR | Δ PS−greedy | greedy SSIM | single SSIM | PS-mean SSIM | PS-mean LPIPS | PS-mean NMSE |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in [r for r in ok if r["acc"] == acc]:
            f = lambda x, p=3: "–" if x is None else f"{x:.{p}f}"
            delta = None if r["ps_mean_psnr"] is None or r["greedy_psnr"] is None else r["ps_mean_psnr"] - r["greedy_psnr"]
            lines.append(f"| {r['k']} | {r['n']} | {f(r['greedy_psnr'])} | {f(r['single_sample_psnr'])} | **{f(r['ps_mean_psnr'])}** | {f(delta)} | "
                         f"{f(r['greedy_ssim'],4)} | {f(r['single_sample_ssim'],4)} | {f(r['ps_mean_ssim'],4)} | {f(r['ps_mean_lpips'],4)} | {f(r['ps_mean_nmse'],4)} |")
        lines.append("")
    inc = [r["dir"] for r in rows if r["status"] != "ok"]
    if inc:
        lines += ["Incomplete runs: " + ", ".join(inc), ""]
    lines += [f"## Selection ({sel}, PS-mean, averaged over acc {accs})", ""]
    for k in sorted(avg):
        lines.append(f"- {V}={k}: {avg[k]:.4f}" + ("  <-- best" if k == best_k else ""))
    lines += ["", f"best {V} per acc: {best_per_acc}", f"**best {V} (best average): {best_k}**", ""]
    (sweep / "sweep_summary.md").write_text("\n".join(lines))
    (sweep / "sweep_summary.json").write_text(json.dumps({"rows": rows, "avg_ps_mean_by_k": avg, "vary": V, "best": best_k, "best_per_acc": best_per_acc, "select_metric": sel}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
