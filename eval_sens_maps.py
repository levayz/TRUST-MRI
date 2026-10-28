"""
How good are sensitivity maps estimated from the *undersampled* data, compared
with the oracle ESPIRiT maps computed from fully sampled k-space?

Estimators compared (all from the acquired data only, except `oracle`):
    oracle         precomputed ESPIRiT maps (sigpy, 24-line calibration, fully sampled)
    varnet256      pretrained E2E-VarNet sensitivity U-Net on the 256-crop masked k-space
                   (10 ACS lines at cf=0.04)
    varnet_native  same U-Net on the native-resolution masked k-space (640×W, round(0.04·W)
                   ACS lines — the resolution it was trained at), maps cropped to 256
    espirit_acs    ESPIRiT from the 10 acquired ACS columns only, on the 256 grid
    varnet256_trained  the TRUST-MRI stage-1 sensitivity U-Net (train_sens_net.py, --sens_ckpt) on the
                   256-crop masked k-space (ACS width passed explicitly), coil-0 gauge

For every slice and estimator ŝ:
    map agreement    |<ŝ(p), s_oracle(p)>|  (unit vectors; 1 = same direction), mean over support
    combined image   x̂0 = Σ conj(ŝ_c) x_c / Σ|ŝ_c|²  from the FULLY sampled coils,
                     |x̂0| vs oracle |x0| inside the support, and vs RSS
    model residual   ‖x_c − ŝ_c x̂0‖² / ‖x_c‖²   (coil energy the map model cannot explain)
    tokenizer        MedITok round trip of x̂0 with the recommended recipe
                     (demodulate with the ACS phase from the same maps, real channel 2r−1,
                     Im dropped) and with the raw Re/Im recipe; magnitude vs oracle |x0| / RSS

Usage:
    python eval_sens_maps.py \\
        --data_path fastmri/brain/multicoil_val \\
        --sens_dir  fastmri/brain/multicoil_val_sens_maps_gpu \\
        --varnet_ckpt weights/varnet/brain_leaderboard_state_dict.pt \\
        --meditok_config ... --meditok_ckpt ... \\
        --split_file fastmri/splits/suno_brain_test.txt --n_extra_random 25 \\
        --output_dir results/sens_check
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from dataset.data.subsample import create_mask_for_mask_type            # noqa: E402
from eval_demod_ceiling import fft2c, ifft2c, center_crop_or_pad, acs_col_range, compute_metrics  # noqa: E402
from eval_meditok_upper_bound import load_meditok, encode_decode_batch  # noqa: E402
from models.varnet_sens import load_pretrained_sens_net, espirit_maps, load_sens_net   # noqa: E402
from models.sense_ops import phase_ref_coil0                              # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Sensitivity-map estimators vs oracle ESPIRiT")
    p.add_argument("--data_path", required=True)
    p.add_argument("--sens_dir", required=True)
    p.add_argument("--sens_dir_fallback", default=None)
    p.add_argument("--varnet_ckpt", required=True, help="fastMRI E2E-VarNet state dict (sens_net.* keys)")
    p.add_argument("--sens_ckpt", default=None, help="stage-1 checkpoint (train_sens_net.py best.pt) for the varnet256_trained estimator")
    p.add_argument("--meditok_config", required=True)
    p.add_argument("--meditok_ckpt", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split_file", default=None)
    p.add_argument("--n_extra_random", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--center_fraction", type=float, default=0.04)
    p.add_argument("--acceleration", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_vis_scans", type=int, default=4)
    p.add_argument("--max_slices_per_scan", type=int, default=0)
    p.add_argument("--no_tokenizer", action="store_true", help="skip the MedITok round-trip part")
    p.add_argument("--estimators", nargs="+", default=["varnet256", "varnet_native", "espirit_acs"])
    p.add_argument("--espirit_selfcheck", action="store_true",
                   help="also compute ESPIRiT from the FULL k-space with a 24×24 calibration on the 256 grid "
                        "and report its agreement with the oracle maps (validates the torch implementation)")
    return p.parse_args()


def combine(coils: torch.Tensor, maps: torch.Tensor):
    """SENSE combine. coils, maps: (C, H, W) complex. Returns (x0 complex, support bool)."""
    ss = (maps.abs() ** 2).sum(0)
    support = ss > 1e-6
    num = (maps.conj() * coils).sum(0)
    return torch.where(support, num / ss.clamp_min(1e-6), torch.zeros_like(num)), support


def save_vis(out_dir: Path, stem: str, oracle, est: dict, x0, xhat: dict, agree: dict, n_coils_show=4):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = ["oracle"] + list(est.keys())
    ncols = n_coils_show + 2
    fig, axes = plt.subplots(len(names), ncols, figsize=(2.6 * ncols, 2.7 * len(names)))
    for r, name in enumerate(names):
        m = oracle if name == "oracle" else est[name]
        for c in range(n_coils_show):
            ax = axes[r, c]
            if c < m.shape[0]:
                ax.imshow(m[c].abs().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{name} |s{c}|", fontsize=8); ax.axis("off")
        ax = axes[r, n_coils_show]
        img = (x0 if name == "oracle" else xhat[name]).abs().cpu().numpy()
        ax.imshow(img, cmap="gray", vmin=0, vmax=float(x0.abs().max()))
        ax.set_title("|x0| oracle" if name == "oracle" else f"|x̂0| {name}", fontsize=8); ax.axis("off")
        ax = axes[r, n_coils_show + 1]
        if name == "oracle":
            ax.imshow(np.angle(x0.cpu().numpy()), cmap="twilight", vmin=-np.pi, vmax=np.pi); ax.set_title("phase x0", fontsize=8)
        else:
            ax.imshow(agree[name].cpu().numpy(), cmap="viridis", vmin=0, vmax=1); ax.set_title(f"agreement {name}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    (out_dir / "vis").mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "vis" / f"{stem}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data_path)
    sens_dirs = [Path(args.sens_dir)] + ([Path(args.sens_dir_fallback)] if args.sens_dir_fallback else [])

    # ---------------- scan selection (identical to eval_demod_ceiling.py) ----------------
    all_files = sorted(p.name for p in data_path.glob("*.h5"))
    have_sens = {f for f in all_files if any((d / f).exists() for d in sens_dirs)}
    selected = []
    if args.split_file:
        split = [l.strip() for l in Path(args.split_file).read_text().splitlines() if l.strip()]
        selected += [f for f in split if f in have_sens]
    if args.n_extra_random > 0:
        rng = random.Random(args.seed)
        pool = sorted(have_sens - set(selected))
        selected += rng.sample(pool, min(args.n_extra_random, len(pool)))
    split_set = set(selected[: len(selected) - args.n_extra_random]) if args.split_file else set()
    print(f"scans: {len(selected)}  (split: {len(split_set)}, random extra: {len(selected) - len(split_set)})")

    sens_net = load_pretrained_sens_net(args.varnet_ckpt, chans=8, num_pools=4, device=device)
    print(f"sens_net loaded: {sum(p.numel() for p in sens_net.parameters()):,} params")
    sens_net_trained = load_sens_net(args.sens_ckpt, device=device) if args.sens_ckpt else None
    if "varnet256_trained" in args.estimators and sens_net_trained is None:
        raise SystemExit("estimator varnet256_trained needs --sens_ckpt")
    meditok = None if args.no_tokenizer else load_meditok(args.meditok_config, args.meditok_ckpt, device)

    mask_func = create_mask_for_mask_type("equispaced", center_fractions=[args.center_fraction],
                                          accelerations=[args.acceleration])
    cs = args.crop_size
    c0, c1 = acs_col_range(cs, args.center_fraction)
    acs_mask = torch.zeros(cs, device=device); acs_mask[c0:c1] = 1.0
    print(f"ACS on {cs} grid: cols [{c0}:{c1}] ({c1 - c0} lines); acceleration x{args.acceleration}")

    def run_meditok(channels: torch.Tensor) -> torch.Tensor:
        outs = []
        for chunk in channels.split(args.batch_size, dim=0):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outs.append(encode_decode_batch(meditok, chunk.clamp(-1, 1)).float())
        return torch.cat(outs, dim=0)

    bufs = defaultdict(lambda: defaultdict(lambda: {"rec": [], "gt": [], "slice": []}))   # key -> scan -> buffers
    diag = defaultdict(lambda: defaultdict(list))                                          # estimator -> metric -> per-slice
    scan_info = {}

    for si, fname in enumerate(selected):
        stem = Path(fname).stem
        with h5py.File(data_path / fname, "r") as f:
            ksp_np = f["kspace"][()]
        sens_np = None
        for d in sens_dirs:
            if (d / fname).exists():
                with h5py.File(d / fname, "r") as f:
                    sens_np = f["sens_maps"][()]
                break
        S, C, H, W = ksp_np.shape
        slices = list(range(S))
        if args.max_slices_per_scan > 0 and S > args.max_slices_per_scan:
            mid, half = S // 2, args.max_slices_per_scan // 2
            slices = list(range(max(0, mid - half), min(S, mid - half + args.max_slices_per_scan)))
        scan_info[fname] = {"n_coils": int(C), "n_slices": len(slices), "kspace_shape": list(ksp_np.shape),
                            "in_split": fname in split_set}

        # one mask per scan, seeded by filename exactly like MulticoilTransform
        seed = tuple(map(ord, fname))
        mask256, _ = mask_func((1, cs, cs, 2), seed=seed)             # (1, 1, cs, 1)
        mask256 = mask256.to(device).float()
        mask_nat, _ = mask_func((1, H, W, 2), seed=seed)              # native grid
        mask_nat = mask_nat.to(device).float()

        for s in slices:
            ksp = torch.from_numpy(ksp_np[s]).to(device)                 # (C, H, W) native
            s_or = torch.from_numpy(sens_np[s]).to(device)               # (C, Hr, Wr)
            x = center_crop_or_pad(center_crop_or_pad(ifft2c(ksp), s_or.shape[-2:]), (cs, cs))   # (C, cs, cs)
            s_or = center_crop_or_pad(s_or, (cs, cs))
            k = fft2c(x)                                                  # 256-grid k-space of the crop
            k_us = k * mask256[0, 0, :, 0][None, None, :]                # undersampled 256-grid k-space

            x0, support = combine(x, s_or)
            rss = (x.abs() ** 2).sum(0).sqrt()
            gt_mag = x0.abs()

            # ---------------- estimated maps ----------------
            est = {}
            if "varnet256" in args.estimators:
                with torch.no_grad():
                    m = sens_net(torch.view_as_real(k_us).unsqueeze(0).float(), mask256.unsqueeze(0))
                est["varnet256"] = torch.view_as_complex(m[0].contiguous())
            if "varnet_native" in args.estimators:
                k_nat_us = ksp * mask_nat[0, 0, :, 0][None, None, :]
                with torch.no_grad():
                    m = sens_net(torch.view_as_real(k_nat_us).unsqueeze(0).float(), mask_nat.unsqueeze(0))
                m = torch.view_as_complex(m[0].contiguous())              # (C, H, W) native grid
                m = center_crop_or_pad(center_crop_or_pad(m, sens_np.shape[-2:]), (cs, cs))
                est["varnet_native"] = m / (m.abs() ** 2).sum(0).sqrt().clamp_min(1e-12)   # re-normalise after crop
            if "espirit_acs" in args.estimators:
                m, _ = espirit_maps(k_us, calib_cols=(c0, c1))
                est["espirit_acs"] = m
            if "varnet256_trained" in args.estimators:
                with torch.no_grad():
                    m = sens_net_trained(torch.view_as_real(k_us).unsqueeze(0).float(), mask256.unsqueeze(0), num_low_frequencies=c1 - c0)
                est["varnet256_trained"] = phase_ref_coil0(torch.view_as_complex(m[0].contiguous())[None])[0]
            if args.espirit_selfcheck:
                r0, r1 = acs_col_range(cs, 24 / cs)
                m, _ = espirit_maps(k, calib_cols=(r0, r1), calib_rows=(r0, r1))
                est["espirit_full24"] = m

            xhat, agree = {}, {}
            for name, m in est.items():
                m_unit = m / (m.abs() ** 2).sum(0).sqrt().clamp_min(1e-12)
                ag = (m_unit.conj() * s_or).sum(0).abs()                 # (cs, cs), oracle unit-norm in support
                agree[name] = ag
                xh, sup_h = combine(x, m)
                xhat[name] = xh
                resid = ((x - m * xh.unsqueeze(0)).abs() ** 2).sum() / (x.abs() ** 2).sum()
                bright = support & (gt_mag > 0.1 * gt_mag.max())
                diag[name]["agree_support"].append(float(ag[support].mean()))
                diag[name]["agree_bright"].append(float(ag[bright].mean()) if bright.any() else float("nan"))
                diag[name]["model_resid"].append(float(resid))
                diag[name]["ratio_to_rss_support"].append(float((xh.abs()[support] / rss[support].clamp_min(1e-12)).mean()))
                diag[name]["support_frac_est"].append(float(sup_h.float().mean()))
                # combined-image fidelity (metrics via repo function, per volume)
                for gk, g, pred in (("oracle_support", gt_mag, xh.abs() * support),
                                    ("rss", rss, xh.abs())):
                    b = bufs[f"{name}__combined__vs_{gk}"][fname]
                    b["rec"].append(pred.cpu().numpy()); b["gt"].append(g.cpu().numpy()); b["slice"].append(s)

            # ---------------- tokenizer round trip with each map set ----------------
            if meditok is not None:
                sets = {"oracle": (s_or, x0)}
                sets.update({n: (est[n], xhat[n]) for n in est if n != "espirit_full24"})
                chans, tags = [], []
                meta = {}
                for name, (m, xh) in sets.items():
                    x_acs, _ = combine(ifft2c(k * acs_mask[None, None, :]), m)   # ACS phase with the SAME maps
                    phi = torch.angle(x_acs)
                    mx = float(xh.abs().max()) + 1e-12
                    z = xh / mx
                    zd = z * torch.exp(-1j * phi)
                    meta[name] = (mx, phi)
                    tags += [f"{name}:raw_re", f"{name}:raw_im", f"{name}:dem_re"]
                    chans += [z.real, z.imag, 2 * zd.real - 1]
                dec = run_meditok(torch.stack(chans, 0))
                d = {t: dec[i] for i, t in enumerate(tags)}
                for name, (mx, phi) in meta.items():
                    rec_raw = torch.complex(d[f"{name}:raw_re"], d[f"{name}:raw_im"]).abs() * mx
                    rec_real = ((d[f"{name}:dem_re"] + 1) / 2).clamp_min(0) * mx
                    for var, r in (("raw_reim", rec_raw), ("demod_realonly", rec_real)):
                        for gk, g, pred in (("oracle_support", gt_mag, r * support), ("rss", rss, r)):
                            b = bufs[f"{name}__{var}__vs_{gk}"][fname]
                            b["rec"].append(pred.cpu().numpy()); b["gt"].append(g.cpu().numpy()); b["slice"].append(s)

            if si < args.n_vis_scans and s == S // 2:
                save_vis(out_dir, stem, s_or, est, x0, xhat, agree)

        print(f"[{si + 1}/{len(selected)}] {fname}  coils={C}  slices={len(slices)}", flush=True)

    # ---------------- metrics ----------------
    results = {key: compute_metrics(buf, device) for key, buf in bufs.items()}

    def agg(key, scans):
        m = results[key]
        rows = [m[s] for s in scans if s in m]
        if not rows:
            return None
        keys = [k for k in ("psnr", "ssim", "lpips", "dists", "nmse") if k in rows[0]]
        return {k: float(np.mean([r[k] for r in rows])) for k in keys} | {"n_scans": len(rows)}

    groups = {"all": list(scan_info)}
    if split_set:
        groups["split"] = [s for s in scan_info if scan_info[s]["in_split"]]
    for s, inf in scan_info.items():
        c = inf["n_coils"]
        groups.setdefault("coils<=8" if c <= 8 else ("coils_10-14" if c <= 14 else "coils>=16"), []).append(s)
    summary = {g: {k: agg(k, scans) for k in results} for g, scans in groups.items()}
    diag_summary = {n: {k: {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v))} for k, v in d.items()}
                    for n, d in diag.items()}

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"args": vars(args), "scan_info": scan_info, "groups": {g: len(s) for g, s in groups.items()},
                   "summary": summary, "map_diagnostics": diag_summary, "per_scan": results}, f, indent=1)

    # ---------------- table ----------------
    lines = ["== map diagnostics (per-slice means; agreement 1.0 = identical direction to oracle) =="]
    lines.append(f"{'estimator':<16}{'agree(support)':>15}{'agree(bright)':>14}{'model resid':>13}{'|x̂0|/RSS':>10}{'support':>9}")
    for n, d in diag_summary.items():
        lines.append(f"{n:<16}{d['agree_support']['mean']:>15.4f}{d['agree_bright']['mean']:>14.4f}"
                     f"{d['model_resid']['mean']:>13.4f}{d['ratio_to_rss_support']['mean']:>10.4f}{d['support_frac_est']['mean']:>9.3f}")
    for g in groups:
        lines.append(f"\n== group: {g}  ({len(groups[g])} scans) ==")
        lines.append(f"{'maps':<16}{'stage':<16}{'vs':<16}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>8}{'DISTS':>8}{'NMSE':>9}")
        for key in sorted(results):
            a = summary[g][key]
            if a is None:
                continue
            name, stage, vs = key.split("__")
            lines.append(f"{name:<16}{stage:<16}{vs[3:]:<16}{a['psnr']:>8.2f}{a['ssim']:>8.4f}"
                         f"{a.get('lpips', float('nan')):>8.4f}{a.get('dists', float('nan')):>8.4f}{a['nmse']:>9.4f}")
    table = "\n".join(lines)
    print(table)
    (out_dir / "summary.txt").write_text(table + "\n")
    print(f"\nwritten: {out_dir / 'metrics.json'}, {out_dir / 'summary.txt'}, {out_dir / 'vis'}/")


if __name__ == "__main__":
    main()
