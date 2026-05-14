"""
Training script for MulticoilRecon.

Usage:
    python train_multicoil.py \\
        --train_path  fastmri/brain/multicoil_train \\
        --val_path    fastmri/brain/multicoil_val \\
        --meditok_config  checkpoints/meditok_config.json \\
        --meditok_ckpt    checkpoints/meditok.pt \\
        --output_dir      checkpoints/multicoil \\
        --batch_size 4  --lr 1e-4  --lambda_coil 0.1 \\
        --max_steps 100000  --val_every 1000  --save_every 5000 \\
        --center_fractions 0.08  --acceleration 4

Checkpoint format (matches existing train.py):
    {"model": state_dict, "optimizer": ..., "steps": int, "args": Namespace}
"""

import os
# Must be set before h5py is imported (by any transitive import).
# Prevents POSIX file-lock contention when 2 DDP ranks × 4 workers open the
# same HDF5 file simultaneously, which causes an indefinite hang on step 1.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import inspect
import json
import math
import warnings
from collections import defaultdict

# Suppress noisy checkpoint warnings from frozen MedITok encoder (called under no_grad)
warnings.filterwarnings("ignore", message=".*use_reentrant.*", module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", message=".*None of the inputs have requires_grad.*", module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
import sys
from pathlib import Path

import logging

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from autoregressive.models.multicoil_recon import (
    build_multicoil_recon,
    compute_loss,
    decode_coil_latents,
    image_dc,
    rss_from_coils,
)
from autoregressive.utils.metrics import compute_reconstruction_metrics
from dataset.data.subsample import create_mask_for_mask_type
from dataset.multicoil_data import build_multicoil_dataset
from models.meditok import MedITok
from utils.logger import create_logger


# ---------------------------------------------------------------------------
# Helpers copied / adapted from autoregressive/train/train.py
# ---------------------------------------------------------------------------

def create_optimizer(model, weight_decay, learning_rate, betas, logger):
    param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
    decay_params   = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params,   "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    nd = sum(p.numel() for p in decay_params)
    nn_ = sum(p.numel() for p in nodecay_params)
    logger.info(f"decay params: {len(decay_params)} tensors, {nd:,}")
    logger.info(f"no-decay params: {len(nodecay_params)} tensors, {nn_:,}")
    fused_ok = "fused" in inspect.signature(torch.optim.AdamW).parameters
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas,
        **({"fused": True} if fused_ok else {}),
    )
    logger.info(f"fused AdamW: {fused_ok}")
    return optimizer


def get_cosine_scheduler(optimizer, num_training_steps, base_lr=1e-4, end_lr=0.0):
    num_warmup = int(num_training_steps * 0.03)

    def lr_lambda(step):
        if step < num_warmup:
            return float(step) / max(1, num_warmup)
        progress = float(step - num_warmup) / max(1, num_training_steps - num_warmup)
        ratio = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return (end_lr + (base_lr - end_lr) * ratio) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)



class DotDict(dict):
    """dict with attribute access (missing keys raise AttributeError for getattr defaults)."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_train_vis(meditok, out: dict, x_us: torch.Tensor, x_gt: torch.Tensor,
                   kspace_us: torch.Tensor, mask: torch.Tensor,
                   output_dir: Path, step: int):
    """Save a 4-panel image reusing the already-computed training step output:
        [Undersampled RSS | GT RSS | Predicted RSS | |Pred − GT|]

    Uses the first sample in the batch.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    simple_mode = "h_hat_re" not in out

    if simple_mode:
        # Use teacher-forced argmax for vis (fast; good enough for monitoring).
        # logits: (B, K, L, vpc) — argmax over last dim gives (B, K, L) indices.
        pred_idx = out["logits"].argmax(dim=-1)                     # (B, K, L)
        pred_img = meditok.idx_to_img(pred_idx)                     # (B, 3, H, W) in [-1,1]
        pred_rss = (pred_img.mean(dim=1) + 1.0) / 2.0              # (B, H, W) in [0,1]
        # GT roundtrip: decode GT tokens to verify MedITok pipeline is correct
        gt_idx    = out["targets"]["rss_tokens"]                    # (B, K, L)
        gt_rt_img = meditok.idx_to_img(gt_idx)                      # (B, 3, H, W) in [-1,1]
        gt_roundtrip = (gt_rt_img.mean(dim=1) + 1.0) / 2.0         # (B, H, W) in [0,1]
        C_eff = x_us.shape[1]
    else:
        B_vis, C_eff, L, Dh = out["h_hat_re"].shape
        _, _, H_vis, W_vis, _ = x_us.shape
        N_vis = B_vis * C_eff
        if "pred_rss_dc" in out:
            # DC-corrected prediction already computed by the image-domain loss path — use it.
            pred_rss = out["pred_rss_dc"]                               # (B, H, W)
        else:
            img_re_flat, img_im_flat = decode_coil_latents(
                meditok,
                out["h_hat_re"].reshape(N_vis, L, Dh),
                out["h_hat_im"].reshape(N_vis, L, Dh),
            )
            stacked_re     = img_re_flat.view(B_vis, C_eff, H_vis, W_vis)
            stacked_im     = img_im_flat.view(B_vis, C_eff, H_vis, W_vis)
            pred_coil_imgs = torch.stack([stacked_re, stacked_im], dim=-1)
            dc_imgs        = image_dc(pred_coil_imgs, kspace_us, mask)
            mag            = torch.sqrt((dc_imgs ** 2).sum(dim=-1))
            pred_rss       = torch.sqrt((mag ** 2).sum(dim=1))

    us_rss = rss_from_coils(x_us[:, :C_eff])   # (B, H, W)
    gt_rss = rss_from_coils(x_gt[:, :C_eff])

    # First sample → numpy
    def _to_np(t):
        return t[0].cpu().float().numpy()

    us_np   = _to_np(us_rss)
    gt_np   = _to_np(gt_rss)
    pred_np = _to_np(pred_rss)

    # Normalise each panel by its own max so low-quality early predictions are visible
    def _norm(x):
        m = x.max()
        return x / m if m > 0 else x

    us_np_n   = _norm(us_np)
    gt_np_n   = _norm(gt_np)
    pred_np_n = _norm(pred_np)
    # Error relative to GT scale
    gt_scale  = gt_np.max() if gt_np.max() > 0 else 1.0
    diff_np   = np.abs(pred_np / gt_scale - gt_np / gt_scale)

    extra_panels = []
    if simple_mode:
        extra_panels = [(_norm(_to_np(gt_roundtrip)), "GT Roundtrip (encode→decode)")]

    ncols = 4 + len(extra_panels)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    panels = [
        (us_np_n,   "Undersampled RSS"),
        (gt_np_n,   "GT RSS"),
        (pred_np_n, "Predicted RSS"),
        (diff_np,   "|Pred − GT| (GT scale)"),
    ] + extra_panels
    for ax, (img, title) in zip(axes, panels):
        vmax = 1.0 if "GT scale" not in title else max(diff_np.max(), 1e-6)
        ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle(f"step {step}", fontsize=10)
    fig.tight_layout()

    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(vis_dir / f"{step:07d}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Val-split filtering
# ---------------------------------------------------------------------------

def _filter_dataset_by_split(dataset, split_file: str):
    """Restrict a multicoil dataset to the filenames listed in split_file.

    Works with both plain SliceDataset and TokenCachedDataset (the latter has
    the raw SliceDataset stored at .base).
    """
    if split_file is None:
        return dataset
    allowed = {
        line.strip()
        for line in Path(split_file).read_text().splitlines()
        if line.strip()
    }
    base = dataset.base if hasattr(dataset, "base") else dataset
    before = len(base.raw_samples)
    base.raw_samples = [s for s in base.raw_samples if Path(s.fname).name in allowed]
    after = len(base.raw_samples)
    logging.getLogger(__name__).info(
        f"val split filter: kept {after}/{before} slices from {split_file}"
    )
    return dataset


# ---------------------------------------------------------------------------
# Val visualisation
# ---------------------------------------------------------------------------

def _save_val_images(scan_buf: dict, output_dir: Path, step: int, max_scans: int = 4):
    """Save a 4-panel [US | GT | Pred | Error] PNG for the center slice of up
    to max_scans scans.  Written to output_dir/val_vis/{step}_{scan}.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    vis_dir = output_dir / "val_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for scan_idx, (fname, v) in enumerate(scan_buf.items()):
        if scan_idx >= max_scans:
            break
        order = sorted(range(len(v["slice"])), key=lambda i: v["slice"][i])
        ci = order[len(order) // 2]   # center slice index

        gt_np   = v["gt"][ci]
        pred_np = v["rec"][ci]
        us_np   = v["us"][ci]
        error   = np.abs(pred_np - gt_np)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, img, title in zip(
            axes,
            [us_np, gt_np, pred_np, error],
            ["US RSS", "GT RSS", "Pred RSS", "|Pred − GT|"],
        ):
            ax.imshow(img, cmap="gray", vmin=0, vmax=max(float(img.max()), 1e-6))
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        safe = Path(fname).stem
        fig.suptitle(f"step {step}  |  {safe}", fontsize=8)
        fig.tight_layout()
        fig.savefig(vis_dir / f"{step:07d}_{safe}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train MulticoilRecon")

    # data
    p.add_argument("--train_path",        required=True)
    p.add_argument("--val_path",          required=True)
    p.add_argument("--val_split_file",    default=None,
                   help="text file with one H5 filename per line; "
                        "restricts val set to those scans only")
    p.add_argument("--center_fraction",   type=float,  default=0.04)
    p.add_argument("--acceleration",      type=int,    default=4)
    p.add_argument("--vcc_matrix_dir",     default=None,
                   help="dir of per-scan VCC matrices for train set (from precompute_vcc.py)")
    p.add_argument("--vcc_matrix_dir_val", default=None,
                   help="dir of per-scan VCC matrices for val set; "
                        "defaults to --vcc_matrix_dir if not set")
    p.add_argument("--n_virtual_coils",    type=int, default=8,
                   help="C' virtual coils to load from VCC matrices")
    p.add_argument("--token_cache_train", default=None,
                   help="dir with pre-computed tokens for train set (from precompute_tokens.py)")
    p.add_argument("--token_cache_val",   default=None,
                   help="dir with pre-computed tokens for val set")
    p.add_argument("--use_cache",         action="store_true",
                   help="activate token cache dirs (--token_cache_train/val must also be set)")

    # MedITok
    p.add_argument("--meditok_config",    required=True, help="path to meditok config JSON")
    p.add_argument("--meditok_ckpt",      required=True, help="path to meditok checkpoint")

    # model
    p.add_argument("--D",           type=int,   default=1024, help="fusion/transformer dim")
    p.add_argument("--n_layer",     type=int,   default=24,   help="SharedTransformer depth")
    p.add_argument("--n_head",      type=int,   default=16,   help="attention heads")
    p.add_argument("--De",          type=int,   default=64,   help="coil embedding dim")
    p.add_argument("--max_coils",      type=int,   default=20)
    p.add_argument("--min_coils",      type=int,   default=0,
                   help="skip scans with fewer than this many physical coils (0 = no filter)")
    p.add_argument("--crop_size",      type=int,   default=256,  help="spatial crop (square)")
    p.add_argument("--enc_chunk_size", type=int,   default=64,
                   help="images per MedITok forward pass during coil encoding")

    # mode
    p.add_argument("--simple",      action="store_true",
                   help="simple mode: CE loss on RSS tokens only, no coil decoder")
    p.add_argument("--overfit",     action="store_true",
                   help="overfit to a single batch — sanity check that the model can learn")

    # training
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=0.05)
    p.add_argument("--lambda_coil", type=float, default=0.1,  help="weight for latent L1 loss")
    p.add_argument("--lambda_dc",   type=float, default=1.0,  help="weight for data-consistency image L1 loss")
    p.add_argument("--lambda_img",  type=float, default=0.0,
                   help="weight for image-domain RSS loss (L1 + 1-SSIM); "
                        "0 disables the loss and skips the decoder forward pass")
    p.add_argument("--max_steps",   type=int,   default=100_000)
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--num_workers", type=int,   default=4)

    # checkpointing / logging
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--val_every",   type=int,   default=1000)
    p.add_argument("--save_every",  type=int,   default=5000)
    p.add_argument("--log_every",   type=int,   default=50)
    p.add_argument("--vis_every",   type=int,   default=2000,
                   help="save a visualisation grid every N steps (0 = disabled)")
    p.add_argument("--val_max_images", type=int, default=4,
                   help="max number of scan images to save per validation run")
    p.add_argument("--no_scheduler", action="store_true",
                   help="disable cosine LR scheduler; use constant LR throughout")
    p.add_argument("--resume",      default=None, help="path to checkpoint to resume from")
    p.add_argument("--pretrain_backbone", default=None,
                   help="load backbone weights from a (simple-mode) checkpoint "
                        "without restoring optimizer state or step counter; "
                        "use with --freeze_backbone to train the coil decoder only")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="freeze coil_fuser / cross_coil_fusion / shared_transformer / "
                        "token_head and train only the coil_decoder")
    p.add_argument("--mixed_precision", default="bf16", choices=["none", "bf16", "fp16"],
                   help="mixed precision mode: none=fp32, bf16=bfloat16, fp16=float16+GradScaler")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(
    model, meditok, loader, lambda_coil, lambda_dc, lambda_img, lambda_ce, ptdtype, device, logger, step,
    simple=False, output_dir=None, max_vis_scans=4,
):
    model.eval()
    compute_img = False   # image-domain decoder is training-only; validation uses quantized decode + DC
    total, n_ce, n_lat, n_dc, n_img, count = 0.0, 0.0, 0.0, 0.0, 0.0, 0

    # per-scan accumulators: fname → {"rec", "gt", "us", "slice"}
    scan_buf = defaultdict(lambda: {"rec": [], "gt": [], "us": [], "slice": []})

    for batch in loader:
        x_us      = batch["x_us"].to(device)
        x_gt      = batch["x_gt"].to(device)
        kspace_us = batch["kspace_us"].to(device)
        mask      = batch["mask"].to(device)
        num_coils = batch["num_coils"].to(device)
        tok_us_re  = batch["tok_us_re"].to(device)  if "tok_us_re"  in batch else None
        tok_us_im  = batch["tok_us_im"].to(device)  if "tok_us_im"  in batch else None
        tok_gt_rss = batch["tok_gt_rss"].to(device) if "tok_gt_rss" in batch else None
        fnames     = batch["fname"]
        slice_nums = batch["slice_num"]
        max_vals   = batch["max_val"]

        with torch.cuda.amp.autocast(dtype=ptdtype):
            out = model(x_us, x_gt, kspace_us, mask, num_coils, simple=simple,
                        compute_img_loss=compute_img,
                        tok_us_re=tok_us_re, tok_us_im=tok_us_im, tok_gt_rss=tok_gt_rss)
            loss, l_ce, l_lat, l_dc, l_img = compute_loss(out, lambda_coil, lambda_dc, lambda_img, lambda_ce)
        total += loss.item()
        n_ce  += l_ce.item()
        n_lat += l_lat.item()
        n_dc  += l_dc.item()
        n_img += l_img.item()
        count += 1

        # ---- decode predictions → RSS images --------------------------------
        if simple:
            # Teacher-forced argmax gives a fast approximation (no AR loop).
            pred_idx = out["logits"].argmax(dim=-1)          # (B, K, L)
            pred_img = meditok.idx_to_img(pred_idx)          # (B, 3, H, W) in [-1, 1]
            pred_rss = (pred_img.mean(dim=1) + 1.0) / 2.0   # (B, H, W) in [0, 1]
        else:
            B_loc, _, L, Dh = out["h_hat_re"].shape
            C_eff  = int(num_coils.min().item())
            _, _, H, W, _  = x_us.shape
            N      = B_loc * C_eff
            # Flatten batch×coil for a single batched decode call
            img_re_flat, img_im_flat = decode_coil_latents(
                meditok,
                out["h_hat_re"][:, :C_eff].reshape(N, L, Dh),
                out["h_hat_im"][:, :C_eff].reshape(N, L, Dh),
            )                                                               # (N, H, W) each
            stacked_re     = img_re_flat.view(B_loc, C_eff, H, W)
            stacked_im     = img_im_flat.view(B_loc, C_eff, H, W)
            pred_coil_imgs = torch.stack([stacked_re, stacked_im], dim=-1) # (B, C_eff, H, W, 2)
            dc_imgs        = image_dc(pred_coil_imgs, kspace_us, mask)     # (B, C_eff, H, W, 2)
            mag            = torch.sqrt((dc_imgs ** 2).sum(dim=-1))        # (B, C_eff, H, W)
            pred_rss       = torch.sqrt((mag ** 2).sum(dim=1))             # (B, H, W)

        # zero-padded coils contribute 0 to RSS, so using full x_gt is correct
        gt_rss = rss_from_coils(x_gt)   # (B, H, W)
        us_rss = rss_from_coils(x_us)   # (B, H, W)

        B = x_us.shape[0]
        for i in range(B):
            max_val = float(max_vals[i])
            gt_np   = (gt_rss[i]   * max_val).cpu().float().numpy()
            pred_np = (pred_rss[i] * max_val).cpu().float().numpy()
            us_np   = (us_rss[i]   * max_val).cpu().float().numpy()
            fname   = fnames[i]
            sl      = int(slice_nums[i])
            scan_buf[fname]["rec"].append(pred_np)
            scan_buf[fname]["gt"].append(gt_np)
            scan_buf[fname]["us"].append(us_np)
            scan_buf[fname]["slice"].append(sl)

    n = max(count, 1)
    logger.info(
        f"[val @ step {step}]  "
        f"loss={total/n:.4f}  ce={n_ce/n:.4f}  lat={n_lat/n:.4f}  "
        f"dc_img={n_dc/n:.4f}  img={n_img/n:.4f}"
    )

    # ---- image-quality metrics ----------------------------------------------
    metrics_input = {
        k: {"rec": v["rec"], "gt": v["gt"], "slice": v["slice"]}
        for k, v in scan_buf.items()
    }
    metrics = compute_reconstruction_metrics(metrics_input, device)
    logger.info(
        f"[val @ step {step}]  "
        f"PSNR={metrics['psnr_mean']:.3f}  SSIM={metrics['ssim_mean']:.4f}  "
        f"LPIPS={metrics['lpips_mean']:.4f}  NMSE={metrics['nmse_mean']:.6f}"
    )

    # ---- save centre-slice images -------------------------------------------
    if output_dir is not None and max_vis_scans > 0:
        _save_val_images(scan_buf, Path(output_dir), step, max_vis_scans)

    model.train()
    return total / n, metrics["psnr_mean"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # DDP init — transparent: single-GPU when run with `python`,
    # multi-GPU when launched with `torchrun --nproc_per_node N`.
    # ------------------------------------------------------------------
    is_ddp     = "LOCAL_RANK" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if is_ddp:
        os.environ.setdefault("NCCL_DEBUG", "WARN")   # shows NCCL errors without flooding
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device    = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main   = (not is_ddp) or (dist.get_rank() == 0)
    world_size = dist.get_world_size() if is_ddp else 1

    # Silence all loggers and warnings on non-main ranks so only rank 0 prints
    if not is_main:
        logging.root.setLevel(logging.ERROR)
        import warnings as _w
        _w.filterwarnings("ignore")

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if is_main:
        logger = create_logger(args.output_dir, args)
    else:
        logger = logging.getLogger("null")
        logger.addHandler(logging.NullHandler())

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    mask_func = create_mask_for_mask_type(
        "equispaced",
        center_fractions=[args.center_fraction],
        accelerations=[args.acceleration],
    )
    train_dataset = build_multicoil_dataset(
        root=args.train_path,
        mask_func=mask_func,
        crop_size=(args.crop_size, args.crop_size),
        max_coils=args.max_coils,
        token_cache_dir=args.token_cache_train if args.use_cache else None,
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        vcc_matrix_dir=args.vcc_matrix_dir,
        n_virtual_coils=args.n_virtual_coils,
        min_coils=args.min_coils,
    )
    vcc_val_dir = args.vcc_matrix_dir_val or args.vcc_matrix_dir
    val_dataset = build_multicoil_dataset(
        root=args.val_path,
        mask_func=mask_func,
        crop_size=(args.crop_size, args.crop_size),
        use_seed=True,
        max_coils=args.max_coils,
        token_cache_dir=args.token_cache_val if args.use_cache else None,
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        vcc_matrix_dir=vcc_val_dir,
        n_virtual_coils=args.n_virtual_coils,
        min_coils=args.min_coils,
    )
    val_dataset = _filter_dataset_by_split(val_dataset, args.val_split_file)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_ddp else None
    _use_workers = args.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        multiprocessing_context="spawn" if _use_workers else None,
    )
    # Validation runs on rank 0 only — don't spawn idle workers on other ranks.
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers if is_main else 0,
        pin_memory=True,
        multiprocessing_context="spawn" if (_use_workers and is_main) else None,
    )
    logger.info(f"train: {len(train_dataset)} slices  |  val: {len(val_dataset)} slices  |  world_size: {world_size}")

    # ------------------------------------------------------------------
    # Frozen MedITok
    # ------------------------------------------------------------------
    with open(args.meditok_config) as f:
        enc_cfg = json.load(f)
    meditok = MedITok(DotDict(enc_cfg)).to(device).eval()
    enc_state = torch.load(args.meditok_ckpt, map_location="cpu")
    # Checkpoint stores weights under trainer.model, not a top-level 'model' key
    if isinstance(enc_state, dict) and "trainer" in enc_state:
        enc_state = enc_state["trainer"]["model"]
    elif isinstance(enc_state, dict) and "model" in enc_state:
        enc_state = enc_state["model"]
    # Strip 'core.' prefix — checkpoint wraps everything under a 'core' submodule
    if any(k.startswith("core.") for k in enc_state):
        enc_state = {k[len("core."):]: v for k, v in enc_state.items() if k.startswith("core.")}
    result = meditok.load_state_dict(enc_state, strict=False)
    if result.missing_keys:
        logger.warning(f"MedITok missing keys ({len(result.missing_keys)}): {result.missing_keys[:5]}")
    if result.unexpected_keys:
        logger.warning(f"MedITok unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]}")
    for p in meditok.parameters():
        p.requires_grad_(False)
    logger.info("MedITok loaded and frozen")

    # ------------------------------------------------------------------
    # Multicoil model
    # ------------------------------------------------------------------
    model = build_multicoil_recon(
        meditok       = meditok,
        D             = args.D,
        n_layer       = args.n_layer,
        n_head        = args.n_head,
        De            = args.De,
        max_coils     = args.max_coils,
        Dh            = 64,        # fixed by MedITok vocab_width
        num_codebooks = 8,         # fixed by MedITok
        vocab_size    = 32768,     # fixed by MedITok
        crop_size      = args.crop_size,
        patch_size     = 16,        # ViT patch size, always 16 for MedITok
        enc_chunk_size = args.enc_chunk_size,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable params: {n_params:,}")

    # ------------------------------------------------------------------
    # Optimizer + scheduler
    # ------------------------------------------------------------------
    optimizer = create_optimizer(
        model,
        weight_decay  = args.weight_decay,
        learning_rate = args.lr,
        betas         = (0.9, 0.95),
        logger        = logger,
    )
    scheduler = None if args.no_scheduler else get_cosine_scheduler(optimizer, args.max_steps, base_lr=args.lr)

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("steps", 0)
        logger.info(f"resumed from {args.resume} at step {start_step}")

    if args.pretrain_backbone:
        ckpt = torch.load(args.pretrain_backbone, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        logger.info(f"loaded backbone weights from {args.pretrain_backbone} "
                    f"(step {ckpt.get('steps', '?')}); optimizer/step reset to 0")

    if args.freeze_backbone:
        # model is still unwrapped here (DDP wrapping happens below)
        frozen_modules = [
            model.coil_fuser,
            model.cross_coil_fusion,
            model.shared_transformer,
            model.token_head,
        ]
        for m in frozen_modules:
            for p in m.parameters():
                p.requires_grad_(False)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"backbone frozen — only coil_decoder trains; "
            f"trainable params after freeze: {trainable:,}"
        )
        # Re-build optimizer so it only covers the now-unfrozen coil_decoder params.
        optimizer = create_optimizer(
            model,
            weight_decay  = args.weight_decay,
            learning_rate = args.lr,
            betas         = (0.9, 0.95),
            logger        = logger,
        )
        if not args.no_scheduler:
            scheduler = get_cosine_scheduler(optimizer, args.max_steps, base_lr=args.lr)

    # CE loss is irrelevant when backbone is frozen (token_head has no grad path
    # to coil_decoder). Zero it out so the logged total reflects actual training signal.
    lambda_ce = 0.0 if args.freeze_backbone else 1.0

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    # GradScaler is a no-op for bf16/fp32; only active for fp16
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == 'fp16'))

    # ------------------------------------------------------------------
    # Wrap in DDP (after optimizer creation so param references stay valid)
    # ------------------------------------------------------------------
    if is_ddp:
        # Barrier: ensure both ranks finish dataset scanning + checkpoint loading
        # before entering DDP init. Without this, the faster rank blocks inside
        # dist.broadcast while the slower rank is still reading H5 files.
        dist.barrier()
        # Temporarily detach the frozen MedITok from the module tree so DDP's
        # initial parameter broadcast only covers the ~468M trainable params
        # (not the additional ~310M frozen MedITok params).
        # Both ranks already loaded the same MedITok checkpoint, so broadcasting
        # it via NCCL is wasteful and can cause multi-minute hangs on PCIe setups.
        _meditok = model.meditok
        model.meditok = None
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True,
                    broadcast_buffers=False)
        model.module.meditok = _meditok   # restore after DDP init
    model_raw = model.module if is_ddp else model   # unwrapped for internal access

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    import time
    model.train()
    step            = start_step
    data_iter       = iter(train_loader)
    t_last          = time.perf_counter()
    steps_since_log = 0
    best_val_psnr   = -float("inf")
    _last_period_ckpt = None   # Path of the most recently saved save_every checkpoint
    _last_best_ckpt   = None   # Path of the most recently saved best-PSNR checkpoint

    # Overfit mode: grab one batch and reuse it every step
    _epoch = 0
    if args.overfit:
        _overfit_batch = next(iter(train_loader))
        logger.info(
            f"OVERFIT MODE — repeating single batch forever  "
            f"(fname={_overfit_batch['fname'][0]}  slice={int(_overfit_batch['slice_num'][0])})"
        )

    while step < args.max_steps:
        if args.overfit:
            batch = _overfit_batch
        else:
            try:
                batch = next(data_iter)
            except StopIteration:
                _epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(_epoch)
                data_iter = iter(train_loader)
                batch     = next(data_iter)

        x_us      = batch["x_us"].to(device, non_blocking=True)
        x_gt      = batch["x_gt"].to(device, non_blocking=True)
        kspace_us = batch["kspace_us"].to(device, non_blocking=True)
        mask      = batch["mask"].to(device, non_blocking=True)
        num_coils = batch["num_coils"].to(device, non_blocking=True)
        tok_us_re  = batch["tok_us_re"].to(device, non_blocking=True)  if "tok_us_re"  in batch else None
        tok_us_im  = batch["tok_us_im"].to(device, non_blocking=True)  if "tok_us_im"  in batch else None
        tok_gt_rss = batch["tok_gt_rss"].to(device, non_blocking=True) if "tok_gt_rss" in batch else None

        # All-reduce C_eff so every rank uses the same coil count (global MIN).
        if is_ddp:
            c_min = torch.tensor([int(num_coils.min().item())], device=device, dtype=torch.long)
            dist.all_reduce(c_min, op=dist.ReduceOp.MIN)
            num_coils = num_coils.clamp(max=int(c_min.item()))

        compute_img = (not args.simple) and (args.lambda_img > 0)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=ptdtype):
            out = model(x_us, x_gt, kspace_us, mask, num_coils,
                        simple=args.simple,
                        compute_img_loss=compute_img,
                        tok_us_re=tok_us_re, tok_us_im=tok_us_im, tok_gt_rss=tok_gt_rss)
            loss, l_ce, l_lat, l_dc, l_img = compute_loss(
                out, args.lambda_coil, args.lambda_dc, args.lambda_img, lambda_ce)
        scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip,
            )

        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        step += 1
        steps_since_log += 1

        # Logging
        if step % args.log_every == 0 and is_main:
            t_now  = time.perf_counter()
            sps    = steps_since_log / max(t_now - t_last, 1e-6)
            t_last = t_now
            steps_since_log = 0
            lr_now = scheduler.get_last_lr()[0] if scheduler is not None else args.lr
            logger.info(
                f"step {step}/{args.max_steps}  "
                f"loss={loss.item():.4f}  ce={l_ce.item():.4f}  "
                f"lat={l_lat.item():.4f}  dc_img={l_dc.item():.4f}  "
                f"img={l_img.item():.4f}  "
                f"lr={lr_now:.2e}  {sps:.2f} steps/s"
            )

        # Validation (rank 0 only; barrier keeps other ranks alive)
        if step % args.val_every == 0:
            if is_main:
                _, val_psnr = run_validation(
                    model_raw, meditok, val_loader,
                    args.lambda_coil, args.lambda_dc, args.lambda_img, lambda_ce, ptdtype,
                    device, logger, step,
                    simple=args.simple,
                    output_dir=args.output_dir,
                    max_vis_scans=args.val_max_images,
                )
                t_last = time.perf_counter()
                steps_since_log = 0

                # Save best-PSNR checkpoint (rolling — delete previous)
                if val_psnr > best_val_psnr:
                    best_val_psnr = val_psnr
                    _ckpt = {
                        "model":     model_raw.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "steps":     step,
                        "args":      args,
                        "val_psnr":  val_psnr,
                    }
                    best_path = Path(args.output_dir) / f"{step:07d}_best_psnr.pt"
                    torch.save(_ckpt, best_path)
                    if _last_best_ckpt is not None and _last_best_ckpt.exists():
                        _last_best_ckpt.unlink()
                    _last_best_ckpt = best_path
                    logger.info(f"new best PSNR {val_psnr:.3f} dB → {best_path}")
            if is_ddp:
                dist.barrier()

        # Visualisation (rank 0 only)
        if is_main and args.vis_every > 0 and step % args.vis_every == 0:
            save_train_vis(meditok, out, x_us, x_gt, kspace_us, mask, Path(args.output_dir), step)

        # Periodic checkpoint (rank 0 only) — rolling, deletes previous
        if is_main and step % args.save_every == 0:
            ckpt_path = Path(args.output_dir) / f"{step:07d}.pt"
            torch.save(
                {
                    "model":     model_raw.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "steps":     step,
                    "args":      args,
                },
                ckpt_path,
            )
            if _last_period_ckpt is not None and _last_period_ckpt.exists():
                _last_period_ckpt.unlink()
            _last_period_ckpt = ckpt_path
            logger.info(f"checkpoint saved: {ckpt_path}")

    # Final checkpoint (rank 0 only)
    if is_main:
        final_path = Path(args.output_dir) / f"{step:07d}_final.pt"
        torch.save(
            {"model": model_raw.state_dict(), "optimizer": optimizer.state_dict(),
             "steps": step, "args": args},
            final_path,
        )
        logger.info(f"training complete. final checkpoint: {final_path}")
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
