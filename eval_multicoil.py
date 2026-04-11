"""
Evaluation script for MulticoilRecon.

Usage:
    python eval_multicoil.py \\
        --data_path   fastmri/brain/multicoil_val \\
        --checkpoint  checkpoints/multicoil/0100000.pt \\
        --meditok_config  checkpoints/meditok_config.json \\
        --meditok_ckpt    checkpoints/meditok.pt \\
        --output_dir  eval_results \\
        [--save_images] [--save_per_coil] [--batch_size 4]

Outputs:
    eval_results/metrics.json        — PSNR / SSIM / LPIPS / DISTS / NMSE
    eval_results/images/<scan>.png   — grid: [US RSS | GT RSS | Pred RSS | Error]  (--save_images)
    eval_results/per_coil/           — per-coil re/im for first scan  (--save_per_coil)
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from autoregressive.models.multicoil_recon import (
    MulticoilRecon,
    build_multicoil_recon,
    decode_coil_latents,
    rss_from_coils,
)
from autoregressive.utils.metrics import compute_reconstruction_metrics
from dataset.data.subsample import create_mask_for_mask_type
from dataset.multicoil_data import build_multicoil_dataset
from models.meditok import MedITok


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )
    return logging.getLogger(name)


class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


def save_image_grid(us_rss, gt_rss, pred_rss, error, scan_key, output_dir):
    """Save a 1×4 PNG grid: [US RSS | GT RSS | Predicted RSS | Error map]."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ["US RSS (input)", "GT RSS", "Predicted RSS", "Error (|pred - gt|)"]
    imgs   = [us_rss, gt_rss, pred_rss, error]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=img.max() if img.max() > 0 else 1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    safe_key = scan_key.replace("/", "_").replace(".", "_")
    out_path = Path(output_dir) / "images" / f"{safe_key}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_per_coil_images(coil_re_list, coil_im_list, scan_key, output_dir):
    """Save per-coil re/im images for one scan."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    C = len(coil_re_list)
    fig, axes = plt.subplots(C, 2, figsize=(8, 3 * C))
    if C == 1:
        axes = axes[None]
    for c in range(C):
        for j, (img, lbl) in enumerate([(coil_re_list[c], "Re"), (coil_im_list[c], "Im")]):
            axes[c, j].imshow(img, cmap="gray")
            axes[c, j].set_title(f"Coil {c} {lbl}", fontsize=8)
            axes[c, j].axis("off")
    fig.tight_layout()
    safe_key = scan_key.replace("/", "_").replace(".", "_")
    out_path = Path(output_dir) / "per_coil" / f"{safe_key}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MulticoilRecon")

    p.add_argument("--data_path",       required=True)
    p.add_argument("--checkpoint",      required=True)
    p.add_argument("--meditok_config",  required=True)
    p.add_argument("--meditok_ckpt",    required=True)
    p.add_argument("--output_dir",      required=True)

    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--center_fractions",type=float, default=0.08)
    p.add_argument("--acceleration",    type=int,   default=4)
    p.add_argument("--crop_size",       type=int,   default=256)

    p.add_argument("--split_file",       default=None,
                   help="text file with one H5 filename per line; "
                        "restricts evaluation to those scans only")
    p.add_argument("--n_vis_scans",     type=int, default=4,
                   help="save center-slice images for up to this many scans (0 = disabled)")
    p.add_argument("--save_images",     action="store_true",
                   help="alias for --n_vis_scans 4 (kept for backwards compat)")
    p.add_argument("--save_recon",      action="store_true",
                   help="save per-volume reconstructions as HDF5 files under output_dir/recon/")
    p.add_argument("--n_recon_scans",   type=int, default=0,
                   help="max volumes to save as H5 (0 = all scans)")
    p.add_argument("--save_per_coil",   action="store_true",
                   help="Save per-coil re/im images for the first scan")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    logger = get_logger("eval_multicoil")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load checkpoint early — dataset must match training settings
    # ------------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", argparse.Namespace())

    # Pull training settings; fall back to eval-script args where absent
    ck_max_coils      = getattr(ckpt_args, "max_coils",      args.crop_size // 64)
    ck_n_virt         = getattr(ckpt_args, "n_virtual_coils", 8)
    ck_vcc_dir_val    = getattr(ckpt_args, "vcc_matrix_dir_val", None) \
                        or getattr(ckpt_args, "vcc_matrix_dir", None)
    ck_simple         = getattr(ckpt_args, "simple",          False)
    ck_crop           = getattr(ckpt_args, "crop_size",       args.crop_size)
    ck_enc_chunk      = getattr(ckpt_args, "enc_chunk_size",  64)
    ck_min_coils      = getattr(ckpt_args, "min_coils",       0)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    mask_func = create_mask_for_mask_type(
        "equispaced",
        center_fractions=[args.center_fractions],
        accelerations=[args.acceleration],
    )
    dataset = build_multicoil_dataset(
        root=args.data_path,
        mask_func=mask_func,
        crop_size=(ck_crop, ck_crop),
        use_seed=True,
        max_coils=ck_max_coils,
        vcc_matrix_dir=ck_vcc_dir_val,
        n_virtual_coils=ck_n_virt,
        min_coils=ck_min_coils,
    )
    if args.split_file is not None:
        from pathlib import Path as _Path
        allowed = {l.strip() for l in _Path(args.split_file).read_text().splitlines() if l.strip()}
        base = dataset.base if hasattr(dataset, "base") else dataset
        before = len(base.raw_samples)
        base.raw_samples = [s for s in base.raw_samples if _Path(s.fname).name in allowed]
        logger.info(f"split filter: {len(base.raw_samples)}/{before} slices from {args.split_file}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info(f"eval dataset: {len(dataset)} slices  (simple={ck_simple}, max_coils={ck_max_coils})")

    # ------------------------------------------------------------------
    # MedITok (frozen)
    # ------------------------------------------------------------------
    with open(args.meditok_config) as f:
        enc_cfg = json.load(f)
    meditok = MedITok(DotDict(enc_cfg)).to(device).eval()
    enc_state = torch.load(args.meditok_ckpt, map_location="cpu")
    if isinstance(enc_state, dict) and "trainer" in enc_state:
        enc_state = enc_state["trainer"]["model"]
    elif isinstance(enc_state, dict) and "model" in enc_state:
        enc_state = enc_state["model"]
    if any(k.startswith("core.") for k in enc_state):
        enc_state = {k[len("core."):]: v for k, v in enc_state.items() if k.startswith("core.")}
    meditok.load_state_dict(enc_state, strict=False)
    for p in meditok.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Multicoil model — rebuilt from checkpoint args
    # ------------------------------------------------------------------
    model = build_multicoil_recon(
        meditok        = meditok,
        D              = getattr(ckpt_args, "D",              1024),
        n_layer        = getattr(ckpt_args, "n_layer",        24),
        n_head         = getattr(ckpt_args, "n_head",         16),
        De             = getattr(ckpt_args, "De",             64),
        max_coils      = ck_max_coils,
        Dh             = 64,       # fixed by MedITok vocab_width
        num_codebooks  = 8,        # fixed by MedITok
        vocab_size     = 32768,    # fixed by MedITok
        crop_size      = ck_crop,
        patch_size     = 16,
        enc_chunk_size = ck_enc_chunk,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(f"checkpoint loaded: step {ckpt.get('steps', '?')}  simple={ck_simple}")

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    # scan_buf: fname → {"rec": [arr], "gt": [arr], "us": [arr], "slice": [int]}
    scan_buf = defaultdict(lambda: {"rec": [], "gt": [], "us": [], "slice": []})

    # For per-coil saving: capture first scan's coil images
    first_scan_done   = False
    first_scan_coil_re = []
    first_scan_coil_im = []
    first_scan_key    = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x_us      = batch["x_us"].to(device)       # (B, C, H, W, 2)
            x_gt      = batch["x_gt"].to(device)
            kspace_us = batch["kspace_us"].to(device)
            mask      = batch["mask"].to(device)
            num_coils = batch["num_coils"].to(device)
            fnames     = batch["fname"]
            slice_nums = batch["slice_num"]
            max_vals   = batch["max_val"]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    x_us, x_gt, kspace_us, mask, num_coils,
                    simple=ck_simple,
                )

            B = x_us.shape[0]

            # ---- decode predicted RSS ----------------------------------------
            if ck_simple:
                pred_idx = out["logits"].argmax(dim=-1)       # (B, K, L)
                pred_img = meditok.idx_to_img(pred_idx)       # (B, 3, H, W) in [-1,1]
                pred_rss = (pred_img.mean(dim=1) + 1.0) / 2.0  # (B, H, W) in [0,1]
                coil_re_imgs, coil_im_imgs = [], []            # unused in simple mode
            else:
                C_eff = int(num_coils.min().item())
                coil_re_imgs, coil_im_imgs = [], []
                for c in range(C_eff):
                    img_re, img_im = decode_coil_latents(
                        meditok,
                        out["h_hat_re"][:, c],
                        out["h_hat_im"][:, c],
                    )
                    coil_re_imgs.append(img_re)
                    coil_im_imgs.append(img_im)
                stacked_re = torch.stack(coil_re_imgs, dim=1)
                stacked_im = torch.stack(coil_im_imgs, dim=1)
                pred_rss = torch.sqrt((stacked_re ** 2 + stacked_im ** 2).sum(dim=1))

            gt_rss = rss_from_coils(x_gt)   # (B, H, W)
            us_rss = rss_from_coils(x_us)   # (B, H, W)

            # ---- accumulate per-scan -----------------------------------------
            for i in range(B):
                fname   = fnames[i]
                sl      = int(slice_nums[i])
                max_val = float(max_vals[i])

                gt_np   = (gt_rss[i]   * max_val).cpu().float().numpy()
                pred_np = (pred_rss[i] * max_val).cpu().float().numpy()
                us_np   = (us_rss[i]   * max_val).cpu().float().numpy()

                scan_buf[fname]["rec"].append(pred_np)
                scan_buf[fname]["gt"].append(gt_np)
                scan_buf[fname]["us"].append(us_np)
                scan_buf[fname]["slice"].append(sl)

                if args.save_per_coil and not ck_simple and not first_scan_done:
                    first_scan_key = fname
                    first_scan_coil_re.append([coil_re_imgs[c][i].cpu().numpy() for c in range(C_eff)])
                    first_scan_coil_im.append([coil_im_imgs[c][i].cpu().numpy() for c in range(C_eff)])

            if args.save_per_coil and not first_scan_done and batch_idx > 0:
                first_scan_done = True

            if (batch_idx + 1) % 20 == 0:
                logger.info(f"processed {(batch_idx + 1) * args.batch_size} slices …")

    # ------------------------------------------------------------------
    # Save center-slice images (up to n_vis_scans scans)
    # ------------------------------------------------------------------
    n_vis = args.n_vis_scans if not args.save_images else max(args.n_vis_scans, 4)
    if n_vis > 0:
        img_dir = Path(args.output_dir) / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for scan_idx, (fname, v) in enumerate(scan_buf.items()):
            if scan_idx >= n_vis:
                break
            order = sorted(range(len(v["slice"])), key=lambda i: v["slice"][i])
            ci = order[len(order) // 2]   # centre slice
            gt_np   = v["gt"][ci]
            pred_np = v["rec"][ci]
            us_np   = v["us"][ci]
            error   = np.abs(pred_np - gt_np)
            save_image_grid(us_np, gt_np, pred_np, error, fname, args.output_dir)
        logger.info(f"saved images for {min(n_vis, len(scan_buf))} scans → {img_dir}")

    # ------------------------------------------------------------------
    # Save per-coil images (first scan only)
    # ------------------------------------------------------------------
    if args.save_per_coil and first_scan_key and first_scan_coil_re:
        # Collect all slices for first scan and average across slices for display
        n_slices = len(first_scan_coil_re)
        C_       = len(first_scan_coil_re[0])
        avg_re   = [np.mean([first_scan_coil_re[s][c] for s in range(n_slices)], axis=0) for c in range(C_)]
        avg_im   = [np.mean([first_scan_coil_im[s][c] for s in range(n_slices)], axis=0) for c in range(C_)]
        save_per_coil_images(avg_re, avg_im, first_scan_key, args.output_dir)
        logger.info(f"per-coil images saved for scan: {first_scan_key}")

    # ------------------------------------------------------------------
    # Save per-volume H5 reconstructions
    # ------------------------------------------------------------------
    if args.save_recon:
        recon_dir = Path(args.output_dir) / "recon"
        recon_dir.mkdir(parents=True, exist_ok=True)
        max_to_save = args.n_recon_scans if args.n_recon_scans > 0 else len(scan_buf)
        saved_count = 0
        for fname, v in sorted(scan_buf.items()):
            if saved_count >= max_to_save:
                break
            order = sorted(range(len(v["slice"])), key=lambda i: v["slice"][i])
            rec_vol = np.stack([v["rec"][i] for i in order], axis=0)   # (S, H, W)
            gt_vol  = np.stack([v["gt"][i]  for i in order], axis=0)
            us_vol  = np.stack([v["us"][i]  for i in order], axis=0)
            stem = Path(fname).stem
            out_h5 = recon_dir / f"{stem}.h5"
            with h5py.File(out_h5, "w") as hf:
                hf.create_dataset("reconstruction", data=rec_vol, compression="gzip")
                hf.create_dataset("ground_truth",   data=gt_vol,  compression="gzip")
                hf.create_dataset("undersampled",   data=us_vol,  compression="gzip")
            saved_count += 1
        logger.info(f"saved {saved_count} H5 reconstructions → {recon_dir}")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    logger.info(f"computing metrics over {len(scan_buf)} scans …")
    metrics = compute_reconstruction_metrics(dict(scan_buf), device)

    # Print summary
    logger.info(
        f"PSNR={metrics['psnr_mean']:.3f}  "
        f"SSIM={metrics['ssim_mean']:.4f}  "
        f"LPIPS={metrics['lpips_mean']:.4f}  "
        f"DISTS={metrics['dists_mean']:.4f}  "
        f"NMSE={metrics['nmse_mean']:.6f}"
    )

    # Save to JSON (convert numpy scalars for JSON serialisation)
    def _to_py(v):
        return float(v) if hasattr(v, "item") else v

    metrics_out = {
        k: (_to_py(v) if not isinstance(v, dict) else {kk: _to_py(vv) for kk, vv in v.items()})
        for k, v in metrics.items()
    }
    out_json = Path(args.output_dir) / "metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info(f"metrics saved: {out_json}")


if __name__ == "__main__":
    main()
