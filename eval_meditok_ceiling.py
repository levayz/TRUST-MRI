#!/usr/bin/env python
"""MedITok reconstruction ceiling on the single-coil ps_recon test sets.

The fully-sampled GT complex image is encoded (real and imaginary parts separately, exactly as the
reconstruction pipeline does), quantised and decoded with MedITok, and scored against the GT with the
same GT definition, per-slice scaling and metric code as autoregressive/sample/reconstruct_far.py.
Arms:
  zero_filled            IFFT of the undersampled k-space (lower reference)
  meditok_roundtrip      GT -> encode -> quantise -> decode, no data consistency (pure tokenizer ceiling)
  meditok_roundtrip_dc   same, then hard data consistency on the measured lines, which every
                         reconstruction in the pipeline also gets (ceiling for the pipeline under this mask)
  meditok_roundtrip_mag  |GT| mapped to [-1,1] (2|x|-1, the dataset's gt_in) -> encode -> decode (magnitude ceiling)
With --ps-masks-dir the masks are compared slice by slice against the masks saved by a ps_recon run
(masks/<scan>_mask.pt) so that every arm in the final table provably used the same mask.

usage: python eval_meditok_ceiling.py --organ brain --acceleration 8 --output-dir results/meditok_ceiling/brain_suno_acc8
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from autoregressive.sample.reconstruct_far import load_meditok, enc_dec_image, encode, decode_codes  # noqa: E402
from autoregressive.models.generate import data_consistency  # noqa: E402
from autoregressive.utils.metrics import compute_reconstruction_metrics  # noqa: E402
from dataset.build import build_dataset  # noqa: E402
from data import transforms  # noqa: E402
from utils.logger import create_logger, log_args  # noqa: E402

ORGANS = {
    "knee": dict(data_root="fastmri/knee/singlecoil_val", data_split="fastmri/splits/adasense_test_split.txt", image_size=320),
    "brain": dict(data_root="fastmri/brain/singlecoil_emulated_all", data_split="fastmri/splits/suno_brain_test.txt", image_size=256),
}
ARMS = ("zero_filled", "meditok_roundtrip", "meditok_roundtrip_dc", "meditok_roundtrip_mag")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--organ", required=True, choices=list(ORGANS))
    p.add_argument("--acceleration", type=int, required=True)
    p.add_argument("--center-fractions", type=float, default=0.04)
    p.add_argument("--data-root", default=None, help="override the organ default")
    p.add_argument("--data-split", default=None, help="override the organ default")
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--use-seed", action="store_true", default=True, help="mask seeded from filename (default on)")
    p.add_argument("--no-use-seed", dest="use_seed", action="store_false")
    p.add_argument("--encoder-ckpt", default="weights/meditok/meditok_simple_v1.pth")
    p.add_argument("--encoder-config", default="weights/meditok/config.json")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--ps-masks-dir", default=None, help="masks/ dir of a ps_recon run to verify identical masks")
    p.add_argument("--start-slice-skip", type=int, default=0, help="drop the first N slices of every volume (MediTok_official knee protocol uses 5)")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = ORGANS[args.organ]
    args.data_root = args.data_root or cfg["data_root"]
    args.data_split = args.data_split or cfg["data_split"]
    args.image_size = args.image_size or cfg["image_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(out_dir)
    log_args(args, logger)

    # same dataset construction as reconstruct_far.py (build_mri_code reads these attributes)
    ds_args = SimpleNamespace(dataset="mri_code", data_root=args.data_root, data_split=args.data_split,
                              image_size=args.image_size, center_fractions=args.center_fractions,
                              accelerations=args.acceleration, code_dir=None, use_seed=args.use_seed)
    dataset = build_dataset(ds_args)
    if args.start_slice_skip > 0:
        n0 = len(dataset.examples)
        dataset.examples = [(f, s) for (f, s) in dataset.examples if s >= args.start_slice_skip]
        logger.info(f"start_slice_skip={args.start_slice_skip}: {n0} -> {len(dataset.examples)} slices")
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers, pin_memory=True)
    meditok = load_meditok(args.encoder_config, args.encoder_ckpt, device)

    bufs = {arm: {} for arm in ARMS}
    mask_check = {"dir": args.ps_masks_dir, "slices_checked": 0, "mismatches": 0, "volumes_without_saved_mask": 0}
    saved_masks = {}
    for batch in tqdm(loader, "meditok ceiling"):
        gt = batch["gt_c_image"].to(device, non_blocking=True)          # [B,H,W,2], / max_val
        masked = batch["masked_c_image"].to(device, non_blocking=True)  # zero-filled, / max_val
        mask = batch["mask"].to(device, non_blocking=True)
        max_val = batch["max_val"].reshape(-1, 1, 1, 1).to(device)

        rt = enc_dec_image(meditok, gt)                     # encode re/im -> quantise -> decode -> grayscale
        rt_dc = data_consistency(masked, rt, mask)          # keep measured k-space lines (pipeline's final step)
        mag = transforms.complex_abs(gt)                    # [B,H,W], per-slice max = 1 (gt is / max_val)
        dec = decode_codes(meditok, encode(meditok, mag * 2 - 1))   # [B,3,H,W] in [-1,1]
        mag_rt = ((dec.mean(dim=1) + 1) / 2).clamp(0, 1)   # back to [0,1]
        rt_mag = torch.stack([mag_rt, torch.zeros_like(mag_rt)], dim=-1)   # complex_abs() below == mag_rt
        arms = {"zero_filled": masked, "meditok_roundtrip": rt, "meditok_roundtrip_dc": rt_dc, "meditok_roundtrip_mag": rt_mag}

        gt_np = transforms.complex_abs(gt * max_val).cpu().float().numpy()
        arm_np = {k: transforms.complex_abs(v * max_val).cpu().float().numpy() for k, v in arms.items()}
        for i in range(gt.shape[0]):
            key = Path(batch["fname"][i]).stem
            sl = int(batch["slice"][i])
            if args.ps_masks_dir is not None:
                if key not in saved_masks:
                    mp = Path(args.ps_masks_dir) / f"{key}_mask.pt"
                    saved_masks[key] = torch.load(mp) if mp.exists() else None
                    if saved_masks[key] is None:
                        mask_check["volumes_without_saved_mask"] += 1
                sm = saved_masks[key]
                if sm is not None:
                    mask_check["slices_checked"] += 1
                    if not torch.equal(sm[sl].bool().reshape(-1), mask[i].cpu().bool().reshape(-1)):
                        mask_check["mismatches"] += 1
            for arm in ARMS:
                b = bufs[arm].setdefault(key, {"rec": [], "gt": [], "slice": []})
                b["rec"].append(arm_np[arm][i]); b["gt"].append(gt_np[i]); b["slice"].append(sl)

    summary = {"organ": args.organ, "acceleration": args.acceleration, "center_fractions": args.center_fractions,
               "data_split": args.data_split, "mask_seeded_by_filename": bool(args.use_seed),
               "start_slice_skip": args.start_slice_skip, "n_volumes": len(bufs[ARMS[0]]), "n_slices": sum(len(v["slice"]) for v in bufs[ARMS[0]].values())}
    for arm in ARMS:
        m = compute_reconstruction_metrics(bufs[arm], device)
        with open(out_dir / f"metrics_{arm}.json", "w") as f:
            json.dump(m, f, indent=2)
        summary[arm] = {k: float(v) for k, v in m.items() if k.endswith("_mean")}
        logger.info(f"{arm:22s} PSNR {m['psnr_mean']:.3f}  SSIM {m['ssim_mean']:.4f}  LPIPS {m['lpips_mean']:.4f}  "
                    f"DISTS {m['dists_mean']:.4f}  NMSE {m['nmse_mean']:.4f}")
    if args.ps_masks_dir is not None:
        summary["mask_check"] = mask_check
        logger.info(f"mask check vs {args.ps_masks_dir}: {mask_check['slices_checked']} slices checked, "
                    f"{mask_check['mismatches']} mismatches, {mask_check['volumes_without_saved_mask']} volumes without saved mask")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"{summary['n_volumes']} volumes, {summary['n_slices']} slices -> {out_dir}")


if __name__ == "__main__":
    main()
