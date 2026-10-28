#!/usr/bin/env python
"""
Stage 1: train the E2E-VarNet-style sensitivity U-Net (models/varnet_sens.SensitivityModel) on the
256-grid undersampled k-space of fastMRI brain multicoil scans, supervised by the oracle ESPIRiT maps.

Input   k_us = fft2c(x_c) * mask on the crop grid; the net only sees the ACS columns (num_low_frequencies
        = round(W * cf), passed explicitly) -> per-coil U-Net -> RSS-normalised maps -> coil-0 phase gauge.
Target  oracle sigpy ESPIRiT maps (sum|S|^2 = 1 on the support, 0 outside, coil-0 real-positive).
Loss    L_map  = weighted complex L1 on the maps inside the oracle support, w = clamp(|S0|/tau, max=1)
                 (the gauge is only enforced where coil 0 is well-defined)
        L_mag  = L1 on | combine(coils_full, S_hat) | vs |x0| inside the support   (gauge-invariant)
        L_cplx = weighted complex L1 on combine(coils_full, S_hat) vs x0            (optional)
        loss   = L_map + lambda_mag * L_mag + lambda_cplx * L_cplx
Val     map agreement |<S_hat, S>| over the support / bright region, model residual, combined-image
        PSNR/SSIM/LPIPS/DISTS vs oracle |x0| (support) and vs RSS (compute_reconstruction_metrics).
best.pt = {"model", "chans", "num_pools", "args", "epoch", "step", "val"}  (loadable by models.varnet_sens.load_sens_net)

Usage: see train_sens_net.sh
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

from autoregressive.utils.metrics import compute_reconstruction_metrics            # noqa: E402
from dataset.data.subsample import create_mask_for_mask_type                    # noqa: E402
from dataset.sense_data import SenseSliceDataset                                 # noqa: E402
from models.sense_ops import (acs_num_low_freqs, c2r, fft2c, mask_bhw, mask_for_sens_net, phase_ref_coil0, r2c,
                              sense_combine)                                      # noqa: E402
from models.varnet_sens import SensitivityModel, load_sens_net                   # noqa: E402
from utils.logger import create_logger, log_args                                 # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train the SENSE sensitivity-map U-Net on 256-grid undersampled k-space")
    # data
    p.add_argument("--data-root", default="fastmri/brain/multicoil_train")
    p.add_argument("--data-split", default="fastmri/splits/multicoil_brain_train.txt")
    p.add_argument("--sens-dirs", default="fastmri/brain/multicoil_train_sens_maps_gpu,fastmri/brain/multicoil_train_sens_maps")
    p.add_argument("--val-data-root", default="fastmri/brain/multicoil_val")
    p.add_argument("--val-data-split", default="fastmri/splits/suno_brain_test.txt")
    p.add_argument("--val-sens-dirs", default="fastmri/brain/multicoil_val_sens_maps_gpu,fastmri/brain/multicoil_val_sens_maps")
    p.add_argument("--val-max-volumes", type=int, default=0, help="0 = all volumes of the val split")
    p.add_argument("--max-coils", type=int, default=20)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--center-fractions", type=float, default=0.04)
    p.add_argument("--accelerations", type=int, default=8)
    p.add_argument("--mask-type", default="cart_1d", choices=["cart_1d", "random", "equispaced"])
    p.add_argument("--num-samples", type=int, default=None, help="random subset of training volumes (no split file)")
    # model
    p.add_argument("--chans", type=int, default=8)
    p.add_argument("--num-pools", type=int, default=4)
    p.add_argument("--init-ckpt", default="weights/varnet/brain_leaderboard_state_dict.pt",
                   help="fastMRI VarNet state dict / stage-1 checkpoint to initialise from; 'none' = from scratch")
    # loss
    p.add_argument("--lambda-mag", type=float, default=1.0)
    p.add_argument("--lambda-cplx", type=float, default=0.0)
    p.add_argument("--gauge-tau", type=float, default=0.05, help="|S0| below which the map/complex losses are down-weighted")
    # optimisation
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (0 = epochs only)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--schedule", default="cosine", choices=["cosine", "constant"])
    p.add_argument("--warmup-frac", type=float, default=0.02)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--mixed-precision", default="bf16", choices=["none", "bf16"])
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    # logging / checkpoints
    p.add_argument("--output-dir", required=True)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--val-every", type=int, default=1, help="epochs")
    p.add_argument("--val-batch-size", type=int, default=8)
    p.add_argument("--n-vis", type=int, default=2, help="volumes visualised at every validation")
    p.add_argument("--resume", default=None, help="last.pt to resume from")
    return p.parse_args()


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------

def sens_loss(maps_hat, maps_or, support, coils, x0, num_coils, tau, lambda_mag, lambda_cplx):
    """maps_hat / maps_or / coils (B,C,H,W) complex, support (B,H,W) bool, x0 (B,H,W) complex, num_coils (B,)"""
    B, C = maps_hat.shape[:2]
    valid = (torch.arange(C, device=maps_hat.device)[None, :] < num_coils[:, None]).float()[:, :, None, None]
    sup = support.float()
    w = (maps_or[:, 0].abs() / tau).clamp(max=1.0) * sup                         # (B,H,W) gauge weight
    d = maps_hat - maps_or
    l1_coils = ((d.real.abs() + d.imag.abs()) * valid).sum(1) / num_coils[:, None, None].float()
    L_map = (l1_coils * w).sum() / w.sum().clamp_min(1.0)
    x_hat = sense_combine(coils, maps_hat)
    L_mag = ((x_hat.abs() - x0.abs()).abs() * sup).sum() / sup.sum().clamp_min(1.0)
    out = {"L_map": L_map, "L_mag": L_mag}
    loss = L_map + lambda_mag * L_mag
    if lambda_cplx > 0:
        dx = x_hat - x0
        L_cplx = ((dx.real.abs() + dx.imag.abs()) * w).sum() / w.sum().clamp_min(1.0)
        out["L_cplx"] = L_cplx
        loss = loss + lambda_cplx * L_cplx
    out["loss"] = loss
    return out


def run_net(model, coils_r, mask, num_coils, nlf, autocast_dtype):
    """coils_r (B,C,H,W,2) float -> k_us -> maps_hat (B,C,H,W) complex, coil-0 gauge; also returns coils (complex)."""
    coils = r2c(coils_r)
    H = coils.shape[-2]
    k_us = fft2c(coils) * mask_bhw(mask, H)[:, None]
    with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
        out = model(c2r(k_us), mask_for_sens_net(mask), num_low_frequencies=nlf, num_coils=num_coils)
    return phase_ref_coil0(r2c(out.float())), coils


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def save_vis(out_path: Path, oracle, est, x0, x_hat, agree, n_coils_show=4):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ncols = n_coils_show + 2
    fig, axes = plt.subplots(2, ncols, figsize=(2.6 * ncols, 5.4))
    for r, (name, m) in enumerate((("oracle", oracle), ("net", est))):
        for c in range(n_coils_show):
            ax = axes[r, c]
            if c < m.shape[0]:
                ax.imshow(m[c].abs().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{name} |s{c}|", fontsize=8); ax.axis("off")
        img = (x0 if r == 0 else x_hat).abs().cpu().numpy()
        axes[r, n_coils_show].imshow(img, cmap="gray", vmin=0, vmax=float(x0.abs().max()))
        axes[r, n_coils_show].set_title("|x0| oracle" if r == 0 else "|x̂0| net", fontsize=8); axes[r, n_coils_show].axis("off")
        ax = axes[r, n_coils_show + 1]
        if r == 0:
            ax.imshow(np.angle(x0.cpu().numpy()), cmap="twilight", vmin=-np.pi, vmax=np.pi); ax.set_title("phase x0", fontsize=8)
        else:
            ax.imshow(agree.cpu().numpy(), cmap="viridis", vmin=0, vmax=1); ax.set_title("agreement", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def validate(model, loader, args, device, logger, out_dir: Path, nlf, autocast_dtype, epoch):
    model.eval()
    bufs = {"vs_oracle_support": defaultdict(lambda: {"rec": [], "gt": [], "slice": []}),
            "vs_rss": defaultdict(lambda: {"rec": [], "gt": [], "slice": []})}
    diag = defaultdict(list)
    vis_done = set()
    t0 = time.time()
    for batch in loader:
        coils_r = batch["coils"].to(device, non_blocking=True)
        maps_or = r2c(batch["maps_oracle"].to(device, non_blocking=True))
        support = batch["support"].to(device)
        x0 = r2c(batch["gt_c_image"].to(device))
        num_coils = batch["num_coils"].to(device)
        mask = batch["mask"].to(device)
        max_val = batch["max_val"].to(device).float()
        maps_hat, coils = run_net(model, coils_r, mask, num_coils, nlf, autocast_dtype)
        x_hat = sense_combine(coils, maps_hat)
        rss = (coils.abs() ** 2).sum(1).sqrt()
        agree = (maps_hat.conj() * maps_or).sum(1).abs()                                   # (B,H,W)
        resid = ((coils - maps_hat * x_hat[:, None]).abs() ** 2).flatten(1).sum(1) / (coils.abs() ** 2).flatten(1).sum(1).clamp_min(1e-12)
        for b in range(coils.shape[0]):
            sup = support[b]
            bright = sup & (x0[b].abs() > 0.1 * x0[b].abs().max())
            diag["agree_support"].append(float(agree[b][sup].mean()))
            diag["agree_bright"].append(float(agree[b][bright].mean()) if bright.any() else float("nan"))
            diag["model_resid"].append(float(resid[b]))
            diag["ratio_to_rss_support"].append(float((x_hat[b].abs()[sup] / rss[b][sup].clamp_min(1e-12)).mean()))
            key, s, mv = batch["fname"][b], int(batch["slice"][b]), float(max_val[b])
            gt_phys = (x0[b].abs() * mv).cpu().numpy()
            bufs["vs_oracle_support"][key]["rec"].append((x_hat[b].abs() * sup * mv).cpu().numpy())
            bufs["vs_oracle_support"][key]["gt"].append(gt_phys)
            bufs["vs_oracle_support"][key]["slice"].append(s)
            bufs["vs_rss"][key]["rec"].append((x_hat[b].abs() * mv).cpu().numpy())
            bufs["vs_rss"][key]["gt"].append((rss[b] * mv).cpu().numpy())
            bufs["vs_rss"][key]["slice"].append(s)
            if len(vis_done) < args.n_vis and key not in vis_done and s == 8:
                vis_done.add(key)
                C = int(num_coils[b])
                save_vis(out_dir / "vis" / f"epoch{epoch:03d}_{Path(key).stem}.png", maps_or[b, :C], maps_hat[b, :C], x0[b], x_hat[b], agree[b])
    metrics = {"epoch": epoch, "n_volumes": len(bufs["vs_oracle_support"]), "n_slices": len(diag["agree_support"]),
               "val_time_s": time.time() - t0}
    for k, v in diag.items():
        metrics[k] = float(np.nanmean(v))
    for name, buf in bufs.items():
        m = compute_reconstruction_metrics(dict(buf), device)
        metrics[name] = {k: float(v) for k, v in m.items() if k.endswith("_mean")}
    metrics["psnr"] = metrics["vs_oracle_support"]["psnr_mean"]
    logger.info(f"[val epoch {epoch}] agree(support) {metrics['agree_support']:.4f}  agree(bright) {metrics['agree_bright']:.4f}  "
                f"resid {metrics['model_resid']:.4f}  |x̂0|/RSS {metrics['ratio_to_rss_support']:.4f}  |  "
                f"vs oracle: PSNR {metrics['vs_oracle_support']['psnr_mean']:.2f} SSIM {metrics['vs_oracle_support']['ssim_mean']:.4f}  |  "
                f"vs RSS: PSNR {metrics['vs_rss']['psnr_mean']:.2f} SSIM {metrics['vs_rss']['ssim_mean']:.4f}  "
                f"({metrics['n_volumes']} vols, {metrics['n_slices']} slices, {metrics['val_time_s']:.0f}s)")
    model.train()
    return metrics


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(out_dir); log_args(args, logger)
    autocast_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else None
    nlf = acs_num_low_freqs(args.image_size, args.center_fractions)
    logger.info(f"ACS width passed to the net: {nlf} columns (cf {args.center_fractions} on {args.image_size})")

    mask_func = create_mask_for_mask_type(args.mask_type, [args.center_fractions], [args.accelerations])
    train_ds = SenseSliceDataset(args.data_root, args.sens_dirs, mask_func, resolution=args.image_size, max_coils=args.max_coils,
                                 use_seed=False, custom_split=None if args.data_split in (None, "", "none") else args.data_split,
                                 num_samples=args.num_samples, seed=args.seed, return_oracle_maps=True)
    val_ds = SenseSliceDataset(args.val_data_root, args.val_sens_dirs, mask_func, resolution=args.image_size, max_coils=args.max_coils,
                               use_seed=True, custom_split=args.val_data_split, seed=args.seed, return_oracle_maps=True)
    if args.val_max_volumes > 0:
        keep = sorted({f for f, _ in val_ds.examples})[:args.val_max_volumes]
        val_ds.examples = [(f, s) for f, s in val_ds.examples if f in keep]
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False, num_workers=min(4, args.num_workers), pin_memory=True)
    logger.info(f"train: {len(train_ds)} slices, val: {len(val_ds)} slices")

    if args.init_ckpt and args.init_ckpt.lower() != "none":
        model = load_sens_net(args.init_ckpt, device=device, trainable=True, chans=args.chans, num_pools=args.num_pools)
        logger.info(f"initialised from {args.init_ckpt}")
    else:
        model = SensitivityModel(chans=args.chans, num_pools=args.num_pools).to(device)
        logger.info("initialised from scratch")
    model.train()
    logger.info(f"SensitivityModel chans={args.chans} pools={args.num_pools}: {sum(p.numel() for p in model.parameters()):,} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch if args.max_steps <= 0 else min(args.max_steps, args.epochs * steps_per_epoch)
    warmup = max(1, int(args.warmup_frac * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        if args.schedule == "constant":
            return 1.0
        prog = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step, start_epoch, best_psnr, history = 0, 0, -float("inf"), []
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"]); scheduler.load_state_dict(ck["scheduler"])
        step, start_epoch, best_psnr, history = ck["step"], ck["epoch"], ck.get("best_psnr", best_psnr), ck.get("history", [])
        logger.info(f"resumed from {args.resume}: step {step}, epoch {start_epoch}, best {best_psnr:.2f}")

    def save(path, epoch, val=None, with_optim=False):
        ck = {"model": model.state_dict(), "chans": args.chans, "num_pools": args.num_pools, "args": vars(args),
              "epoch": epoch, "step": step, "val": val, "best_psnr": best_psnr, "history": history}
        if with_optim:
            ck["optimizer"] = optimizer.state_dict(); ck["scheduler"] = scheduler.state_dict()
        torch.save(ck, path)

    logger.info(f"training for {args.epochs} epochs x {steps_per_epoch} steps (total {total_steps}, warmup {warmup})")
    done = False
    for epoch in range(start_epoch, args.epochs):
        run, n_run, t0 = defaultdict(float), 0, time.time()
        for batch in train_loader:
            coils_r = batch["coils"].to(device, non_blocking=True)
            maps_or = r2c(batch["maps_oracle"].to(device, non_blocking=True))
            support = batch["support"].to(device, non_blocking=True)
            x0 = r2c(batch["gt_c_image"].to(device, non_blocking=True))
            num_coils = batch["num_coils"].to(device)
            mask = batch["mask"].to(device, non_blocking=True)
            maps_hat, coils = run_net(model, coils_r, mask, num_coils, nlf, autocast_dtype)
            losses = sens_loss(maps_hat, maps_or, support, coils, x0, num_coils, args.gauge_tau, args.lambda_mag, args.lambda_cplx)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) if args.grad_clip > 0 else torch.tensor(0.0)
            optimizer.step(); scheduler.step(); step += 1
            for k, v in losses.items():
                run[k] += float(v)
            run["gnorm"] += float(gn); n_run += 1
            if step % args.log_every == 0:
                msg = "  ".join(f"{k} {v / n_run:.4f}" for k, v in run.items())
                logger.info(f"(epoch {epoch} step {step:07d}) {msg}  lr {scheduler.get_last_lr()[0]:.2e}  {n_run / (time.time() - t0):.2f} it/s")
                run, n_run, t0 = defaultdict(float), 0, time.time()
            if args.max_steps > 0 and step >= args.max_steps:
                done = True
                break
        if (epoch + 1) % args.val_every == 0 or done or epoch == args.epochs - 1:
            val = validate(model, val_loader, args, device, logger, out_dir, nlf, autocast_dtype, epoch + 1)
            history.append(val)
            if val["psnr"] > best_psnr:
                best_psnr = val["psnr"]
                save(out_dir / "best.pt", epoch + 1, val)
                logger.info(f"new best: PSNR vs oracle {best_psnr:.3f} -> {out_dir / 'best.pt'}")
            with open(out_dir / "history.json", "w") as f:
                json.dump(history, f, indent=1)
        save(out_dir / "last.pt", epoch + 1, history[-1] if history else None, with_optim=True)
        if done:
            break
    logger.info(f"done. best PSNR vs oracle (support): {best_psnr:.3f}")


if __name__ == "__main__":
    main()
