"""
MedITok round-trip ceiling for SENSE-combined multicoil images, with and
without ACS-based phase demodulation.

Zero-training check. For every slice:
    1. IFFT fully-sampled coil k-space, crop to the sensitivity-map grid, then
       to --crop_size (image-domain center crops, same as MulticoilTransform).
    2. Oracle SENSE combine with the precomputed ESPIRiT maps:
           x0 = sum_c conj(s_c) x_c / sum_c |s_c|^2      (0 outside support)
    3. Low-resolution phase from the ACS columns only (always acquired):
           phi = angle( SENSE_combine( IFFT( ACS-masked k-space ) ) )
    4. Encode -> quantize -> decode the real / imaginary channels through the
       frozen MedITok tokenizer under several conventions and compare the
       magnitude of the round-tripped image against |x0| and against RSS.

Variants (each channel is a separate MedITok pass, grayscale in [-1, 1]):
    raw_reim      Re/Im of x0 / max|x0|                      (current complex pipeline)
    demod_reim    Re/Im of x0 e^{-i phi}                      (1-D ACS phase)
    demod2d_reim  Re/Im of x0 e^{-i phi_2d}                   (ACS phase, also low-passed along H)
    demod_shift   as demod_reim, real channel mapped 2r-1     (uses the full [-1,1] range)
    demod_realonly  demod_shift real channel only, Im dropped (|x| ~ Re after demod)
    demod_imgain  demod_shift with Im scaled x4 before encode  (better use of Im range)
    mag_only      2|x0|-1, phase discarded                    (perfect-demodulation upper bound)
    esc_reim      Re/Im of the emulated single-coil image     (paper's single-coil setting)
    esc_mag       2|esc|-1                                    (single-coil magnitude ceiling)

Usage:
    python eval_demod_ceiling.py \\
        --data_path   fastmri/brain/multicoil_val \\
        --sens_dir    fastmri/brain/multicoil_val_sens_maps_gpu \\
        --esc_path    fastmri/brain/singlecoil_emulated_val \\
        --meditok_config /fast/lev/meditok/weights/meditok/config.json \\
        --meditok_ckpt   /fast/lev/meditok/weights/meditok/meditok_ckpt_v1.pth \\
        --split_file  fastmri/splits/suno_brain_test.txt --n_extra_random 25 \\
        --output_dir  results/demod_check
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
import torch.nn.functional as F

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from eval_meditok_upper_bound import load_meditok, encode_decode_batch  # noqa: E402


# ---------------------------------------------------------------------------
# FFT / crop helpers (complex tensors, centered, ortho — same convention as
# dataset/data/fftc.py which operates on (..., 2) real views)
# ---------------------------------------------------------------------------

def fft2c(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def ifft2c(k: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def center_crop_or_pad(x: torch.Tensor, shape) -> torch.Tensor:
    """Center crop (or zero-pad) the last two dims of a complex tensor to `shape`."""
    h, w = shape
    H, W = x.shape[-2:]
    if H < h or W < w:
        ph, pw = max(h - H, 0), max(w - W, 0)
        x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        H, W = x.shape[-2:]
    top, left = (H - h) // 2, (W - w) // 2
    return x[..., top:top + h, left:left + w]


def acs_col_range(W: int, center_fraction: float):
    n_acs = int(round(W * center_fraction))
    pad = (W - n_acs + 1) // 2
    return pad, pad + n_acs


# ---------------------------------------------------------------------------
# Metrics (repo convention: volume-level PSNR over the sorted slice stack with
# data_range = GT range; SSIM per slice averaged; NMSE per volume). Falls back
# to a piq-free implementation when piq is not installed.
# ---------------------------------------------------------------------------

def _metrics_fallback(scan_buf):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    out, ps, ss, nm = {}, [], [], []
    for k, v in scan_buf.items():
        order = sorted(range(len(v["slice"])), key=lambda i: v["slice"][i])
        rec = np.array([v["rec"][i] for i in order]).astype(np.float64)
        gt = np.array([v["gt"][i] for i in order]).astype(np.float64)
        dr = gt.max() - gt.min()
        dr = dr if dr > 0 else 1.0
        psnr = peak_signal_noise_ratio(gt, rec, data_range=dr)
        ssim = float(np.mean([structural_similarity(gt[i], rec[i], data_range=dr) for i in range(len(rec))]))
        nmse = float(np.linalg.norm(gt - rec) ** 2 / max(np.linalg.norm(gt) ** 2, 1e-12))
        out[k] = {"psnr": float(psnr), "ssim": ssim, "nmse": nmse, "num_slices": len(rec)}
        ps.append(psnr); ss.append(ssim); nm.append(nmse)
    out.update(psnr_mean=float(np.mean(ps)), ssim_mean=float(np.mean(ss)), nmse_mean=float(np.mean(nm)))
    return out


def compute_metrics(scan_buf, device):
    try:
        import piq  # noqa: F401
        from autoregressive.utils.metrics import compute_reconstruction_metrics
        return compute_reconstruction_metrics(scan_buf, device)
    except ImportError:
        return _metrics_fallback(scan_buf)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_vis(out_dir: Path, stem: str, panels):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    n = len(panels)
    ncols = min(n, 5)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    for ax, (img, title, kind) in zip(axes, panels):
        if kind == "signed":
            m = float(np.abs(img).max()) or 1.0
            ax.imshow(img, cmap="RdBu_r", vmin=-m, vmax=m)
        elif kind == "phase":
            ax.imshow(img, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        elif kind == "err":
            ax.imshow(img, cmap="magma", vmin=0, vmax=float(img.max()) or 1.0)
        else:
            ax.imshow(img, cmap="gray", vmin=0, vmax=float(img.max()) or 1.0)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    (out_dir / "vis").mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "vis" / f"{stem}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MedITok round-trip ceiling with/without phase demodulation")
    p.add_argument("--data_path", required=True, help="multicoil .h5 directory")
    p.add_argument("--sens_dir", required=True, help="ESPIRiT maps directory (one .h5 per scan, key sens_maps)")
    p.add_argument("--sens_dir_fallback", default=None, help="second maps directory to try if a file is missing")
    p.add_argument("--esc_path", default=None, help="emulated single-coil .h5 directory (optional)")
    p.add_argument("--meditok_config", required=True)
    p.add_argument("--meditok_ckpt", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split_file", default=None, help="one .h5 filename per line; these scans are always included")
    p.add_argument("--n_extra_random", type=int, default=0, help="add this many random extra scans from data_path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--center_fraction", type=float, default=0.04)
    p.add_argument("--batch_size", type=int, default=16, help="images per MedITok pass")
    p.add_argument("--n_vis_scans", type=int, default=4)
    p.add_argument("--max_slices_per_scan", type=int, default=0, help="0 = all slices")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path)
    sens_dirs = [Path(args.sens_dir)] + ([Path(args.sens_dir_fallback)] if args.sens_dir_fallback else [])
    esc_path = Path(args.esc_path) if args.esc_path else None

    # ---------------- scan selection ----------------
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
    if not selected:
        raise SystemExit("no scans selected (check --split_file / --n_extra_random / sens dirs)")
    split_set = set(selected[: len(selected) - args.n_extra_random]) if args.split_file else set()
    print(f"scans: {len(selected)}  (split: {len(split_set)}, random extra: {len(selected) - len(split_set)})")

    meditok = load_meditok(args.meditok_config, args.meditok_ckpt, device)

    cs = args.crop_size
    c0, c1 = acs_col_range(cs, args.center_fraction)
    acs_mask = torch.zeros(cs, device=device)
    acs_mask[c0:c1] = 1.0
    row_mask = torch.zeros(cs, device=device)
    r0, r1 = acs_col_range(cs, args.center_fraction)
    row_mask[r0:r1] = 1.0
    print(f"ACS columns [{c0}:{c1}] ({c1 - c0} lines) on a {cs}x{cs} grid")

    # variant -> gt_kind -> scan -> buffers
    bufs = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"rec": [], "gt": [], "slice": []})))
    diag = defaultdict(list)          # per-slice diagnostics
    scan_info = {}

    def run_meditok(channels: torch.Tensor) -> torch.Tensor:
        outs = []
        for chunk in channels.split(args.batch_size, dim=0):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outs.append(encode_decode_batch(meditok, chunk.clamp(-1, 1)).float())
        return torch.cat(outs, dim=0)

    for si, fname in enumerate(selected):
        stem = Path(fname).stem
        with h5py.File(data_path / fname, "r") as f:
            ksp_np = f["kspace"][()]                                   # (S, C, H, W) complex64
        sens_np = None
        for d in sens_dirs:
            if (d / fname).exists():
                with h5py.File(d / fname, "r") as f:
                    sens_np = f["sens_maps"][()]                       # (S, C, Hr, Wr) complex64
                break
        assert sens_np is not None
        S, C, H, W = ksp_np.shape
        assert sens_np.shape[:2] == (S, C), (sens_np.shape, ksp_np.shape)
        esc_np = None
        if esc_path is not None and (esc_path / fname).exists():
            with h5py.File(esc_path / fname, "r") as f:
                esc_np = f["kspace"][()]                               # (S, H, W) complex64

        slices = list(range(S))
        if args.max_slices_per_scan > 0 and S > args.max_slices_per_scan:
            mid = S // 2
            half = args.max_slices_per_scan // 2
            slices = list(range(max(0, mid - half), min(S, mid - half + args.max_slices_per_scan)))
        scan_info[fname] = {"n_coils": int(C), "n_slices": len(slices), "kspace_shape": list(ksp_np.shape),
                            "in_split": fname in split_set}

        for s in slices:
            ksp = torch.from_numpy(ksp_np[s]).to(device)                # (C, H, W)
            sm = torch.from_numpy(sens_np[s]).to(device)                # (C, Hr, Wr)
            x = ifft2c(ksp)
            x = center_crop_or_pad(x, sm.shape[-2:])                    # to sens-map grid
            x = center_crop_or_pad(x, (cs, cs))
            sm = center_crop_or_pad(sm, (cs, cs))

            ss = (sm.abs() ** 2).sum(0)
            support = ss > 1e-6

            def combine(coils):
                num = (sm.conj() * coils).sum(0)
                return torch.where(support, num / ss.clamp_min(1e-6), torch.zeros_like(num))

            x0 = combine(x)                                             # oracle SENSE image
            rss = (x.abs() ** 2).sum(0).sqrt()                          # physical RSS
            k = fft2c(x)
            x_acs = combine(ifft2c(k * acs_mask[None, None, :]))       # 1-D ACS (columns)
            x_acs2d = combine(ifft2c(k * acs_mask[None, None, :] * row_mask[None, :, None]))
            phi = torch.angle(x_acs)
            phi2d = torch.angle(x_acs2d)

            max_val = float(x0.abs().max()) + 1e-12
            z_raw = x0 / max_val
            z_dem = z_raw * torch.exp(-1j * phi)
            z_dem2 = z_raw * torch.exp(-1j * phi2d)
            mag = z_raw.abs()

            chans, tags = [], []

            def add(tag, t):
                tags.append(tag); chans.append(t)

            add("raw_re", z_raw.real);   add("raw_im", z_raw.imag)
            add("dem_re", z_dem.real);   add("dem_im", z_dem.imag)
            add("dem2_re", z_dem2.real); add("dem2_im", z_dem2.imag)
            add("shift_re", 2 * z_dem.real - 1); add("shift_im", z_dem.imag)
            add("im_gain", 4 * z_dem.imag)            # clamped to [-1, 1] inside run_meditok
            add("mag", 2 * mag - 1)

            esc_gt = None
            if esc_np is not None:
                xe = center_crop_or_pad(ifft2c(torch.from_numpy(esc_np[s]).to(device)), (cs, cs))
                me = float(xe.abs().max()) + 1e-12
                ze = xe / me
                add("esc_re", ze.real); add("esc_im", ze.imag); add("esc_mag", 2 * ze.abs() - 1)
                esc_gt = xe.abs()

            dec = run_meditok(torch.stack(chans, 0))
            d = {t: dec[i] for i, t in enumerate(tags)}

            rec = {
                "raw_reim":     torch.complex(d["raw_re"], d["raw_im"]).abs() * max_val,
                "demod_reim":   torch.complex(d["dem_re"], d["dem_im"]).abs() * max_val,
                "demod2d_reim": torch.complex(d["dem2_re"], d["dem2_im"]).abs() * max_val,
                "demod_shift":  torch.complex((d["shift_re"] + 1) / 2, d["shift_im"]).abs() * max_val,
                "demod_realonly": ((d["shift_re"] + 1) / 2).clamp_min(0) * max_val,
                "demod_imgain": torch.complex((d["shift_re"] + 1) / 2, d["im_gain"] / 4).abs() * max_val,
                "mag_only":     ((d["mag"] + 1) / 2).clamp_min(0) * max_val,
            }
            gts = {"sense": x0.abs(), "rss": rss}
            for var, r in rec.items():
                for gk, g in gts.items():
                    b = bufs[var][gk][fname]
                    b["rec"].append(r.cpu().numpy()); b["gt"].append(g.cpu().numpy()); b["slice"].append(s)
            if esc_gt is not None:
                rec_esc = {
                    "esc_reim": torch.complex(d["esc_re"], d["esc_im"]).abs() * me,
                    "esc_mag":  ((d["esc_mag"] + 1) / 2).clamp_min(0) * me,
                }
                for var, r in rec_esc.items():
                    b = bufs[var]["esc"][fname]
                    b["rec"].append(r.cpu().numpy()); b["gt"].append(esc_gt.cpu().numpy()); b["slice"].append(s)

            # ---- diagnostics: how "real" is each representation inside the support
            def imag_frac(z):
                zz = z[support]
                return float((zz.imag ** 2).sum() / ((zz.abs() ** 2).sum() + 1e-12))
            diag["imag_frac_raw"].append(imag_frac(z_raw))
            diag["imag_frac_demod"].append(imag_frac(z_dem))
            diag["imag_frac_demod2d"].append(imag_frac(z_dem2))
            diag["support_frac"].append(float(support.float().mean()))
            bright = support & (mag > 0.1)
            diag["phase_std_raw_bright"].append(float(torch.angle(z_raw[bright]).std()) if bright.any() else float("nan"))
            diag["phase_std_demod_bright"].append(float(torch.angle(z_dem[bright]).std()) if bright.any() else float("nan"))
            if esc_gt is not None:
                diag["imag_frac_esc"].append(float((ze.imag ** 2).sum() / ((ze.abs() ** 2).sum() + 1e-12)))

            if si < args.n_vis_scans and s == S // 2:
                a = lambda t: t.cpu().numpy()
                panels = [
                    (a(x0.abs()), "|x0| SENSE GT", "mag"),
                    (a(rss), "RSS GT", "mag"),
                    (a(phi), "ACS phase (1-D)", "phase"),
                    (a(torch.angle(x0)), "true phase", "phase"),
                    (a(torch.angle(z_dem)), "residual phase after demod", "phase"),
                    (a(z_raw.real), "Re raw", "signed"),
                    (a(z_raw.imag), "Im raw", "signed"),
                    (a(z_dem.real), "Re demod", "signed"),
                    (a(z_dem.imag), "Im demod", "signed"),
                    (a(2 * mag - 1), "mag only (2m-1)", "signed"),
                    (a((rec["raw_reim"] - x0.abs()).abs()), "|err| raw_reim", "err"),
                    (a((rec["demod_reim"] - x0.abs()).abs()), "|err| demod_reim", "err"),
                    (a((rec["demod_shift"] - x0.abs()).abs()), "|err| demod_shift", "err"),
                    (a((rec["demod_realonly"] - x0.abs()).abs()), "|err| demod_realonly", "err"),
                    (a((rec["mag_only"] - x0.abs()).abs()), "|err| mag_only", "err"),
                ]
                if esc_gt is not None:
                    panels.append((a((rec_esc["esc_reim"] - esc_gt).abs()), "|err| esc_reim (own GT)", "err"))
                save_vis(out_dir, stem, panels)

        print(f"[{si + 1}/{len(selected)}] {fname}  coils={C}  slices={len(slices)}", flush=True)

    # ---------------- metrics ----------------
    results = {}
    for var in bufs:
        for gk in bufs[var]:
            m = compute_metrics(bufs[var][gk], device)
            results[f"{var}__vs_{gk}"] = m

    def agg(var_gk, scans):
        m = results[var_gk]
        rows = [m[s] for s in scans if s in m]
        if not rows:
            return None
        keys = [k for k in ("psnr", "ssim", "lpips", "dists", "nmse") if k in rows[0]]
        return {k: float(np.mean([r[k] for r in rows])) for k in keys} | {"n_scans": len(rows)}

    groups = {"all": list(scan_info)}
    if split_set:
        groups["split"] = [s for s in scan_info if scan_info[s]["in_split"]]
    def bucket(c):
        return "coils<=8" if c <= 8 else ("coils_10-14" if c <= 14 else "coils>=16")
    for s, inf in scan_info.items():
        groups.setdefault(bucket(inf["n_coils"]), []).append(s)

    summary = {g: {vk: agg(vk, scans) for vk in results} for g, scans in groups.items()}
    diag_summary = {k: {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v))} for k, v in diag.items()}

    payload = {
        "args": vars(args),
        "scan_info": scan_info,
        "groups": {g: len(s) for g, s in groups.items()},
        "summary": summary,
        "diagnostics": diag_summary,
        "per_scan": results,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=1)

    # ---------------- print table ----------------
    order = ["raw_reim", "demod_reim", "demod2d_reim", "demod_shift", "demod_realonly", "demod_imgain",
             "mag_only", "esc_reim", "esc_mag"]
    lines = []
    for g in groups:
        lines.append(f"\n== group: {g}  ({len(groups[g])} scans) ==")
        lines.append(f"{'variant':<14}{'vs':<7}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>8}{'DISTS':>8}{'NMSE':>9}")
        for var in order:
            for gk in ("sense", "rss", "esc"):
                vk = f"{var}__vs_{gk}"
                if vk not in results:
                    continue
                a = summary[g][vk]
                if a is None:
                    continue
                lines.append(f"{var:<14}{gk:<7}{a['psnr']:>8.2f}{a['ssim']:>8.4f}"
                             f"{a.get('lpips', float('nan')):>8.4f}{a.get('dists', float('nan')):>8.4f}{a['nmse']:>9.4f}")
    lines.append("\n== diagnostics (per-slice, inside ESPIRiT support) ==")
    for k, v in diag_summary.items():
        lines.append(f"{k:<26} mean={v['mean']:.4f}  median={v['median']:.4f}")
    table = "\n".join(lines)
    print(table)
    (out_dir / "summary.txt").write_text(table + "\n")
    print(f"\nwritten: {out_dir / 'metrics.json'}, {out_dir / 'summary.txt'}, {out_dir / 'vis'}/")


if __name__ == "__main__":
    main()
