"""
MedITok upper-bound baseline for multicoil MRI reconstruction.

Measures the quality ceiling when VCC-compressed GT coil images are fed
directly through MedITok (encode → quantize → decode) and combined with RSS.
No learned reconstruction model is involved — this is the best the pipeline
can do given VCC compression + tokenisation.

GT for comparison: physical-coil RSS *before* VCC (stored as target_rss in
the dataset).

Usage:
    python eval_meditok_upper_bound.py \\
        --data_path       fastmri/brain/multicoil_val \\
        --vcc_matrix_dir  /path/to/vcc_matrices \\
        --meditok_config  checkpoints/meditok_config.json \\
        --meditok_ckpt    checkpoints/meditok.pt \\
        --output_dir      results/meditok_upper_bound \\
        [--n_virtual_coils 8] \\
        [--split_file fastmri/splits/suno_brain_test.txt] \\
        [--enc_chunk_size 32]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from autoregressive.utils.metrics import compute_reconstruction_metrics
from dataset.data.subsample import create_mask_for_mask_type
from dataset.multicoil_data import build_multicoil_dataset
from eval_multicoil import DotDict, get_logger, save_image_grid
from models.meditok import MedITok


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MedITok upper-bound baseline")

    p.add_argument("--data_path",       required=True)
    p.add_argument("--vcc_matrix_dir",  required=True,
                   help="Directory of per-scan VCC matrices (.pt files)")
    p.add_argument("--meditok_config",  required=True)
    p.add_argument("--meditok_ckpt",    required=True)
    p.add_argument("--output_dir",      required=True)

    p.add_argument("--n_virtual_coils", type=int,   default=8)
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--center_fractions",type=float, default=0.08)
    p.add_argument("--acceleration",    type=int,   default=4)
    p.add_argument("--crop_size",       type=int,   default=256)
    p.add_argument("--enc_chunk_size",  type=int,   default=32,
                   help="Coil images per MedITok forward pass — reduce if OOM")

    p.add_argument("--no_quantize", action="store_true",
                   help="Skip VQ quantization during encode/decode (continuous upper bound)")
    p.add_argument("--eval_size", type=int, default=None,
                   help="If set, upsample coil images to this size before MedITok "
                        "encode/decode, then downsample back to crop_size for metrics. "
                        "Tests the effect of encoding at higher resolution.")
    p.add_argument("--split_file",   default=None,
                   help="Text file with one H5 filename per line; restricts "
                        "evaluation to those scans only")
    p.add_argument("--n_vis_scans",  type=int, default=4,
                   help="Save centre-slice grids for up to this many scans (0 = off)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_decode_batch(meditok: MedITok, imgs_bhw: torch.Tensor,
                        quantize: bool = True) -> torch.Tensor:
    """Encode→(quantize)→decode a batch of single-channel images.

    Args:
        imgs_bhw: (B, H, W) float in [-1, 1]
        quantize: if True, snap to nearest VQ codebook entry; if False, skip quantization

    Returns:
        (B, H, W) float in [-1, 1]
    """
    # MedITok expects (B, 3, H, W); repeat grayscale across colour channels
    img = imgs_bhw.unsqueeze(-1).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)
    if quantize:
        rec = meditok.img_to_reconstructed_img(img)          # (B, 3, H, W) in [-1, 1]
    else:
        rec = meditok.img_to_reconstructed_img_no_quant(img) # (B, 3, H, W) in [-1, 1]
    return rec.mean(dim=1)                                    # (B, H, W)


def load_meditok(config_path: str, ckpt_path: str, device: torch.device) -> MedITok:
    with open(config_path) as f:
        cfg = json.load(f)
    model = MedITok(DotDict(cfg)).to(device).eval()

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "trainer" in state:
        state = state["trainer"]["model"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    if any(k.startswith("core.") for k in state):
        state = {k[len("core."):]: v for k, v in state.items() if k.startswith("core.")}

    model.load_state_dict(state, strict=False)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    logger = get_logger("eval_meditok_ub")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset  (VCC applied; target_rss is the physical GT before VCC)
    # ------------------------------------------------------------------
    mask_func = create_mask_for_mask_type(
        "equispaced",
        center_fractions=[args.center_fractions],
        accelerations=[args.acceleration],
    )
    dataset = build_multicoil_dataset(
        root            = args.data_path,
        mask_func       = mask_func,
        crop_size       = (args.crop_size, args.crop_size),
        use_seed        = True,
        max_coils       = args.n_virtual_coils,
        vcc_matrix_dir  = args.vcc_matrix_dir,
        n_virtual_coils = args.n_virtual_coils,
    )
    if args.split_file is not None:
        allowed = {
            l.strip()
            for l in Path(args.split_file).read_text().splitlines()
            if l.strip()
        }
        base = dataset.base if hasattr(dataset, "base") else dataset
        before = len(base.raw_samples)
        base.raw_samples = [s for s in base.raw_samples
                            if Path(s.fname).name in allowed]
        logger.info(f"split filter: {len(base.raw_samples)}/{before} slices "
                    f"from {args.split_file}")

    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
    )
    logger.info(f"eval dataset: {len(dataset)} slices  "
                f"(n_virtual_coils={args.n_virtual_coils})")

    # ------------------------------------------------------------------
    # MedITok
    # ------------------------------------------------------------------
    meditok = load_meditok(args.meditok_config, args.meditok_ckpt, device)
    quantize = not args.no_quantize
    logger.info(f"MedITok loaded and frozen  (quantize={quantize})")

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    scan_buf = defaultdict(lambda: {"rec": [], "gt": [], "us": [], "slice": []})

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x_gt       = batch["x_gt"].to(device)        # (B, C', H, W, 2) virtual coils
            x_us       = batch["x_us"].to(device)
            target_rss = batch["target_rss"].to(device)  # (B, H, W) physical GT / max_val
            num_coils  = batch["num_coils"].to(device)
            fnames     = batch["fname"]
            slice_nums = batch["slice_num"]
            max_vals   = batch["max_val"]

            B     = x_gt.shape[0]
            C_eff = int(num_coils.min().item())
            H, W  = x_gt.shape[2], x_gt.shape[3]

            # Flatten (B, C_eff, H, W) → (B*C_eff, H, W) for chunked encoding
            flat_re = x_gt[:, :C_eff, :, :, 0].reshape(B * C_eff, H, W)
            flat_im = x_gt[:, :C_eff, :, :, 1].reshape(B * C_eff, H, W)

            # Optional: upsample to eval_size before encoding, downsample after
            if args.eval_size is not None and args.eval_size != H:
                enc_re = F.interpolate(flat_re.unsqueeze(1), size=args.eval_size,
                                       mode="bilinear", align_corners=False,
                                       antialias=True).squeeze(1)
                enc_im = F.interpolate(flat_im.unsqueeze(1), size=args.eval_size,
                                       mode="bilinear", align_corners=False,
                                       antialias=True).squeeze(1)
            else:
                enc_re, enc_im = flat_re, flat_im

            flat_rec_re = torch.empty_like(enc_re)
            flat_rec_im = torch.empty_like(enc_im)

            chunk = args.enc_chunk_size
            for i in range(0, B * C_eff, chunk):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    flat_rec_re[i:i+chunk] = encode_decode_batch(
                        meditok, enc_re[i:i+chunk], quantize=quantize)
                    flat_rec_im[i:i+chunk] = encode_decode_batch(
                        meditok, enc_im[i:i+chunk], quantize=quantize)

            # Downsample reconstructions back to crop size if upsampling was applied
            if args.eval_size is not None and args.eval_size != H:
                flat_rec_re = F.interpolate(flat_rec_re.unsqueeze(1), size=(H, W),
                                            mode="bilinear", align_corners=False,
                                            antialias=True).squeeze(1)
                flat_rec_im = F.interpolate(flat_rec_im.unsqueeze(1), size=(H, W),
                                            mode="bilinear", align_corners=False,
                                            antialias=True).squeeze(1)

            rec_re = flat_rec_re.reshape(B, C_eff, H, W)
            rec_im = flat_rec_im.reshape(B, C_eff, H, W)

            # RSS of reconstructed virtual coils: sqrt(Σ_c (re_c² + im_c²))
            pred_rss = torch.sqrt((rec_re ** 2 + rec_im ** 2).sum(dim=1))  # (B, H, W)

            # Undersampled RSS for visualisation
            us_rss = torch.sqrt(
                (x_us[:, :C_eff, :, :, 0] ** 2 +
                 x_us[:, :C_eff, :, :, 1] ** 2).sum(dim=1)
            )

            # Accumulate per-scan
            for i in range(B):
                fname   = fnames[i]
                sl      = int(slice_nums[i])
                max_val = float(max_vals[i])

                # Scale back to physical units
                gt_np   = (target_rss[i] * max_val).cpu().float().numpy()
                pred_np = (pred_rss[i]   * max_val).cpu().float().numpy()
                us_np   = (us_rss[i]     * max_val).cpu().float().numpy()

                scan_buf[fname]["rec"].append(pred_np)
                scan_buf[fname]["gt"].append(gt_np)
                scan_buf[fname]["us"].append(us_np)
                scan_buf[fname]["slice"].append(sl)

            if (batch_idx + 1) % 20 == 0:
                logger.info(f"processed {(batch_idx + 1) * args.batch_size} slices …")

    # ------------------------------------------------------------------
    # Centre-slice visualisations
    # ------------------------------------------------------------------
    if args.n_vis_scans > 0:
        img_dir = Path(args.output_dir) / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for scan_idx, (fname, v) in enumerate(scan_buf.items()):
            if scan_idx >= args.n_vis_scans:
                break
            order = sorted(range(len(v["slice"])), key=lambda i: v["slice"][i])
            ci    = order[len(order) // 2]
            error = np.abs(v["rec"][ci] - v["gt"][ci])
            save_image_grid(v["us"][ci], v["gt"][ci], v["rec"][ci], error,
                            fname, args.output_dir)
        logger.info(f"saved images for {min(args.n_vis_scans, len(scan_buf))} scans "
                    f"→ {img_dir}")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    logger.info(f"computing metrics over {len(scan_buf)} scans …")
    metrics = compute_reconstruction_metrics(dict(scan_buf), device)

    logger.info(
        f"PSNR={metrics['psnr_mean']:.3f}  "
        f"SSIM={metrics['ssim_mean']:.4f}  "
        f"LPIPS={metrics['lpips_mean']:.4f}  "
        f"DISTS={metrics['dists_mean']:.4f}  "
        f"NMSE={metrics['nmse_mean']:.6f}"
    )

    def _to_py(v):
        return float(v) if hasattr(v, "item") else v

    metrics_out = {
        k: (_to_py(v) if not isinstance(v, dict)
            else {kk: _to_py(vv) for kk, vv in v.items()})
        for k, v in metrics.items()
    }
    out_json = Path(args.output_dir) / "metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info(f"metrics saved: {out_json}")


if __name__ == "__main__":
    main()
