#!/usr/bin/env python
"""
E2E-VarNet baseline trained on the TRUST-MRI 256-grid SENSE setup (same data pipeline as the SENSE GPT):
coils = crop(crop(ifft2c(k native), map grid), 256) / max|x0|, random 1D Cartesian masks (cf, R) on the 256 grid,
coils zero-padded to max_coils (the sens-net skips padded coils). Target = RSS of the fully sampled 256 coils,
loss = 1 - SSIM(RSS_pred, RSS_gt) (+ lambda_l1 * L1), as in fastMRI. Validation on suno every epoch: RSS output vs RSS,
SENSE-combined output (oracle maps) vs |x0| (support) and RSS output vs |x0|; best.pt by PSNR vs RSS.
best.pt = {"state_dict": fastMRI-named VarNet weights, ...} -> loadable by models.varnet_sens.load_varnet and by
scripts/eval_varnet_baseline.py (bare "state_dict" path).  Usage: see train_varnet256.sh
"""
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import piq                                                                        # noqa: E402
from autoregressive.utils.metrics import compute_reconstruction_metrics            # noqa: E402
from dataset.data.subsample import create_mask_for_mask_type                    # noqa: E402
from dataset.sense_data import SenseSliceDataset                                 # noqa: E402
from models.sense_ops import acs_num_low_freqs, c2r, fft2c, ifft2c, mask_bhw, mask_for_sens_net, r2c, sense_combine  # noqa: E402
from models.varnet_sens import VarNet, load_varnet                               # noqa: E402
from models.feature_varnet import FEATURE_VARNET_VARIANTS, FeatureVarNetK         # noqa: E402
from utils.logger import create_logger, log_args                                 # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train E2E-VarNet on the 256-grid SENSE setup")
    p.add_argument("--data-root", default="fastmri/brain/multicoil_train")
    p.add_argument("--data-split", default="fastmri/splits/multicoil_brain_train.txt")
    p.add_argument("--sens-dirs", default="fastmri/brain/multicoil_train_sens_maps_gpu,fastmri/brain/multicoil_train_sens_maps")
    p.add_argument("--val-data-root", default="fastmri/brain/multicoil_val")
    p.add_argument("--val-data-split", default="fastmri/splits/suno_brain_test.txt")
    p.add_argument("--val-sens-dirs", default="fastmri/brain/multicoil_val_sens_maps_gpu,fastmri/brain/multicoil_val_sens_maps")
    p.add_argument("--val-max-volumes", type=int, default=0)
    p.add_argument("--max-coils", type=int, default=20)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--center-fractions", type=float, default=0.04)
    p.add_argument("--accelerations", type=int, default=8)
    p.add_argument("--mask-type", default="cart_1d", choices=["cart_1d", "random", "equispaced"])
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--init-ckpt", default="weights/varnet/brain_leaderboard_state_dict.pt", help="VarNet weights to start from; 'none' = scratch")
    p.add_argument("--model", default="e2e", choices=["e2e", "feature_varnet"],
                   help="e2e = E2E-VarNet (models/varnet_sens.py); feature_varnet = Giannakopoulos et al. 2024 (models/feature_varnet.py), "
                        "trained from scratch (no released weights) with --init-ckpt ignored")
    p.add_argument("--fv-variant", default="fi_varnet", choices=list(FEATURE_VARNET_VARIANTS))
    p.add_argument("--fv-init-ckpt", default="none", help="feature_varnet: warm start from a checkpoint of this script (e.g. the x8 best.pt "
                   "for the x16 model); the architecture comes from its model_config, the attention block size from --accelerations")
    p.add_argument("--kspace-mult-factor", type=float, default=1.0, help="feature_varnet: 1.0 for our max|x0|-normalised slices (official 1e6 is for raw k-space)")
    p.add_argument("--optimizer", default="adam", choices=["adam", "adamw"])
    p.add_argument("--num-cascades", type=int, default=12)
    p.add_argument("--chans", type=int, default=18)
    p.add_argument("--sens-chans", type=int, default=8)
    p.add_argument("--lambda-l1", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--schedule", default="cosine", choices=["cosine", "constant"])
    p.add_argument("--warmup-frac", type=float, default=0.02)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--val-batch-size", type=int, default=4)
    p.add_argument("--resume", default=None)
    return p.parse_args()


def run_varnet(model, coils_r, mask, num_coils, nlf):
    """coils_r (B,C,H,W,2) -> (k_us complex (B,C,H,W), kspace_pred real (B,C,H,W,2))"""
    coils = r2c(coils_r)
    H = coils.shape[-2]
    k_us = fft2c(coils) * mask_bhw(mask, H)[:, None]
    mask5 = mask_for_sens_net(mask).bool()
    if isinstance(model, FeatureVarNetK):
        kp = model.forward_padded(c2r(k_us), mask5, num_low_frequencies=nlf, num_coils=num_coils)
    else:
        kp = model(c2r(k_us), mask5, num_low_frequencies=nlf, num_coils=num_coils)
    return k_us, kp


def ssim_loss(pred, gt):
    """pred, gt (B,H,W) magnitudes; SSIM with per-sample data range = max of the target (fastMRI convention)."""
    dr = gt.flatten(1).max(1).values.clamp_min(1e-8)[:, None, None]
    return 1 - piq.ssim((pred / dr).clamp(0, 1)[:, None], (gt / dr).clamp(0, 1)[:, None], data_range=1.0)


@torch.no_grad()
def validate(model, loader, device, logger, nlf, epoch):
    model.eval()
    bufs = {k: defaultdict(lambda: {"rec": [], "gt": [], "slice": []}) for k in ("rss_vs_rss", "sense_vs_sense", "rss_vs_sense")}
    t0 = time.time()
    for batch in loader:
        coils_r = batch["coils"].to(device, non_blocking=True)
        maps = r2c(batch["maps_oracle"].to(device, non_blocking=True))
        support = batch["support"].to(device).float()
        x0 = r2c(batch["gt_c_image"].to(device))
        num_coils = batch["num_coils"].to(device)
        mask = batch["mask"].to(device)
        mv = batch["max_val"].to(device).float()[:, None, None]
        _, kp = run_varnet(model, coils_r, mask, num_coils, nlf)
        coils_pred = ifft2c(r2c(kp))
        rss_pred = (coils_pred.abs() ** 2).sum(1).sqrt() * mv
        sense_pred = sense_combine(coils_pred, maps).abs() * mv
        rss_ref = (r2c(coils_r).abs() ** 2).sum(1).sqrt() * mv
        x0_ref = x0.abs() * mv
        for b in range(coils_r.shape[0]):
            key, s = batch["fname"][b], int(batch["slice"][b])
            for name, rec, gt in (("rss_vs_rss", rss_pred[b], rss_ref[b]), ("sense_vs_sense", sense_pred[b] * support[b], x0_ref[b]),
                                  ("rss_vs_sense", rss_pred[b] * support[b], x0_ref[b])):
                bb = bufs[name][key]; bb["rec"].append(rec.cpu().numpy()); bb["gt"].append(gt.cpu().numpy()); bb["slice"].append(s)
    metrics = {"epoch": epoch, "val_time_s": time.time() - t0}
    for name, buf in bufs.items():
        m = compute_reconstruction_metrics(dict(buf), device)
        metrics[name] = {k: float(v) for k, v in m.items() if k.endswith("_mean")}
    metrics["psnr"] = metrics["rss_vs_rss"]["psnr_mean"]
    logger.info(f"[val epoch {epoch}] RSS vs RSS: PSNR {metrics['rss_vs_rss']['psnr_mean']:.2f} SSIM {metrics['rss_vs_rss']['ssim_mean']:.4f} | "
                f"SENSE vs SENSE: PSNR {metrics['sense_vs_sense']['psnr_mean']:.2f} SSIM {metrics['sense_vs_sense']['ssim_mean']:.4f} | "
                f"RSS vs SENSE: PSNR {metrics['rss_vs_sense']['psnr_mean']:.2f} SSIM {metrics['rss_vs_sense']['ssim_mean']:.4f} ({metrics['val_time_s']:.0f}s)")
    model.train()
    return metrics


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(out_dir); log_args(args, logger)
    nlf = acs_num_low_freqs(args.image_size, args.center_fractions)
    mask_func = create_mask_for_mask_type(args.mask_type, [args.center_fractions], [args.accelerations])
    train_ds = SenseSliceDataset(args.data_root, args.sens_dirs, mask_func, resolution=args.image_size, max_coils=args.max_coils,
                                 use_seed=False, custom_split=None if args.data_split in (None, "", "none") else args.data_split,
                                 num_samples=args.num_samples, seed=args.seed, return_oracle_maps=False)
    val_ds = SenseSliceDataset(args.val_data_root, args.val_sens_dirs, mask_func, resolution=args.image_size, max_coils=args.max_coils,
                               use_seed=True, custom_split=args.val_data_split, seed=args.seed, return_oracle_maps=True)
    if args.val_max_volumes > 0:
        keep = sorted({f for f, _ in val_ds.examples})[:args.val_max_volumes]
        val_ds.examples = [(f, s) for f, s in val_ds.examples if f in keep]
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False, num_workers=min(4, args.num_workers), pin_memory=True)
    logger.info(f"train: {len(train_ds)} slices, val: {len(val_ds)} slices; ACS width {nlf}")

    if args.model == "feature_varnet":
        if args.fv_init_ckpt and args.fv_init_ckpt.lower() != "none":
            ck = torch.load(args.fv_init_ckpt, map_location="cpu", weights_only=False)
            cfg = dict(ck["model_config"], acceleration=args.accelerations)   # AttentionPE block size has no weights
            model = FeatureVarNetK(**cfg)
            model.load_state_dict(ck["state_dict"], strict=True)
            model = model.to(device)
            logger.info(f"Feature VarNet initialised from {args.fv_init_ckpt} (epoch {ck.get('epoch')}): {model.config}")
        else:
            model = FeatureVarNetK(variant=args.fv_variant, num_cascades=args.num_cascades, chans=args.chans, sens_chans=args.sens_chans,
                                   acceleration=args.accelerations, kspace_mult_factor=args.kspace_mult_factor).to(device)
            logger.info(f"Feature VarNet ({args.fv_variant}) initialised from scratch: {model.config}")
    elif args.init_ckpt and args.init_ckpt.lower() != "none":
        model = load_varnet(args.init_ckpt, device=device, trainable=True)
        logger.info(f"initialised from {args.init_ckpt}")
    else:
        model = VarNet(num_cascades=args.num_cascades, sens_chans=args.sens_chans, chans=args.chans).to(device)
        logger.info("initialised from scratch")
    model.train()
    logger.info(f"{type(model).__name__}: {args.num_cascades if args.model == 'feature_varnet' else len(model.cascades)} cascades, "
                f"{sum(p.numel() for p in model.parameters()):,} params")

    opt_cls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = opt_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch if args.max_steps <= 0 else min(args.max_steps, args.epochs * steps_per_epoch)
    warmup = max(1, int(args.warmup_frac * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        if args.schedule == "constant":
            return 1.0
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total_steps - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step, start_epoch, best_psnr, history = 0, 0, -float("inf"), []
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["state_dict"]); optimizer.load_state_dict(ck["optimizer"]); scheduler.load_state_dict(ck["scheduler"])
        step, start_epoch, best_psnr, history = ck["step"], ck["epoch"], ck.get("best_psnr", best_psnr), ck.get("history", [])
        logger.info(f"resumed from {args.resume}: step {step}, epoch {start_epoch}")

    def save(path, epoch, val=None, with_optim=False):
        ck = {"state_dict": model.state_dict(), "args": vars(args), "epoch": epoch, "step": step, "val": val, "best_psnr": best_psnr, "history": history}
        if isinstance(model, FeatureVarNetK):
            ck["model_config"] = model.config
        if with_optim:
            ck["optimizer"] = optimizer.state_dict(); ck["scheduler"] = scheduler.state_dict()
        torch.save(ck, path)

    logger.info(f"training for {args.epochs} epochs x {steps_per_epoch} steps (total {total_steps}, warmup {warmup}), fp32")
    done = False
    for epoch in range(start_epoch, args.epochs):
        run, n_run, t0 = defaultdict(float), 0, time.time()
        for batch in train_loader:
            coils_r = batch["coils"].to(device, non_blocking=True)
            num_coils = batch["num_coils"].to(device)
            mask = batch["mask"].to(device, non_blocking=True)
            _, kp = run_varnet(model, coils_r, mask, num_coils, nlf)
            rss_pred = VarNet.rss(kp)
            rss_gt = (r2c(coils_r).abs() ** 2).sum(1).sqrt()
            l_ssim = ssim_loss(rss_pred, rss_gt)
            loss = l_ssim
            if args.lambda_l1 > 0:
                l_l1 = (rss_pred - rss_gt).abs().mean(); loss = loss + args.lambda_l1 * l_l1; run["l1"] += float(l_l1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) if args.grad_clip > 0 else torch.tensor(0.0)
            optimizer.step(); scheduler.step(); step += 1
            run["loss"] += float(loss); run["1-ssim"] += float(l_ssim); run["gnorm"] += float(gn); n_run += 1
            if step % args.log_every == 0:
                msg = "  ".join(f"{k} {v / n_run:.4f}" for k, v in run.items())
                logger.info(f"(epoch {epoch} step {step:07d}) {msg}  lr {scheduler.get_last_lr()[0]:.2e}  {n_run / (time.time() - t0):.2f} it/s")
                run, n_run, t0 = defaultdict(float), 0, time.time()
            if args.max_steps > 0 and step >= args.max_steps:
                done = True
                break
        if (epoch + 1) % args.val_every == 0 or done or epoch == args.epochs - 1:
            val = validate(model, val_loader, device, logger, nlf, epoch + 1)
            history.append(val)
            if val["psnr"] > best_psnr:
                best_psnr = val["psnr"]
                save(out_dir / "best.pt", epoch + 1, val)
                logger.info(f"new best: PSNR vs RSS {best_psnr:.3f} -> {out_dir / 'best.pt'}")
            with open(out_dir / "history.json", "w") as f:
                json.dump(history, f, indent=1)
        save(out_dir / "last.pt", epoch + 1, history[-1] if history else None, with_optim=True)
        if done:
            break
    logger.info(f"done. best PSNR vs RSS: {best_psnr:.3f}")


if __name__ == "__main__":
    main()
