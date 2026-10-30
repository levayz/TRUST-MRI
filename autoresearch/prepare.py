"""
autoresearch harness (FROZEN: the agent does not edit this file).

Everything an experiment is measured against lives here: the problem constants, the data (fastMRI brain multicoil, SENSE
pipeline of TRUST-MRI), the frozen tokenizer and sensitivity net, the GPU-side input preparation, the budget, the evaluation
and the summary block. Experiments edit model.py, train.py and sample.py only.

Problem: random 1D Cartesian masks, acceleration 8, centre fraction 0.04, acquisition grid 256 (crop of the ESPIRiT-map grid),
tokenizer / GPT grid 512 (bicubic upsample of the SENSE image; the res512 cplxpair SENSE tokenizer), sensitivity maps from the
trained sens-net on the ACS lines, oracle (ESPIRiT) maps for the ground-truth SENSE image, per-coil hard data consistency at
inference. Ground-truth tokens and latents come from the precomputed cache (precompute_gt_cache.py).

Metric: val_psnr = mean over the validation volumes of the volume PSNR (data range of the ground-truth volume) of the
magnitude reconstruction after data consistency, on the autoresearch validation split (15 volumes of multicoil_val, disjoint
from the suno test split which is reserved for the human's confirmation runs). Higher is better.

Budget: SAMPLE_BUDGET training samples per experiment (steps x batch), MAX_MINUTES wall-clock cap on the training loop.
"""
import os
import sys
import time
import json
import math
import pathlib
import random

REPO = pathlib.Path(__file__).resolve().parents[1]
HERE = pathlib.Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.sense_data import SenseSliceDataset
from dataset.data.subsample import create_mask_for_mask_type
from models.sense_ops import (prepare_sense_batch, sense_combine, ifft2c, c2r, mask_bhw, tok_upsample, tok_downsample,
                              wrap_dc_fn_resized, latent_target)
from models.varnet_sens import load_sens_net as _load_sens_net
from autoregressive.sample.reconstruct_far import load_meditok
from autoregressive.utils.metrics import compute_reconstruction_metrics
from data import transforms

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ----------------------------------------------------------------------------------------------------------------------
# problem constants
# ----------------------------------------------------------------------------------------------------------------------
CROP_SIZE = 256                      # acquisition grid
TOKEN_SIZE = 512                     # tokenizer / GPT grid
GRID = TOKEN_SIZE // 16              # 32 x 32 token positions
NUM_POS = GRID * GRID                # 1024
ACCELERATION = 8
CENTER_FRACTION = 0.04
MASK_TYPE = "random"                 # random 1D Cartesian columns
MAX_COILS = 20
VOCAB_SIZE = 32768
NUM_CODEBOOKS = 8
LATENT_DIM = 64                      # tokenizer latent width per stream
MASK_WIDTH = CROP_SIZE               # sampling-mask column vector length
RESIZE_MODE = "bicubic"

# budget (per experiment). Env overrides exist for the calibration scripts only; the agent never sets them.
# One epoch: every one of the 56,228 training slices once (7,028 steps of 8; the user's choice, Oct 30 2026). The GPT-B
# baseline at batch 8 runs 2.5 steps/s on an RTX A6000 with cached inputs, about 46 min; the cap leaves room for slower
# recipes and a slower machine.
SAMPLE_BUDGET = int(os.environ.get("AR_SAMPLE_BUDGET", 56224))   # training samples (steps x batch)
MAX_MINUTES = float(os.environ.get("AR_MAX_MINUTES", 75))        # wall-clock cap on the training loop
SKIP_EVAL = os.environ.get("AR_SKIP_EVAL", "0") == "1"           # calibration: timing runs only
VRAM_SOFT_CAP_GB = 40

# data
TRAIN_ROOT = REPO / "fastmri/brain/multicoil_train"
TRAIN_SPLIT = REPO / "fastmri/splits/multicoil_brain_train.txt"
TRAIN_SENS_DIRS = f"{REPO}/fastmri/brain/multicoil_train_sens_maps_gpu,{REPO}/fastmri/brain/multicoil_train_sens_maps"
GT_CACHE_DIR = pathlib.Path("/fast/lev/datasets/fastmri/brain/multicoil_train_gt_cache_crop256_tok512")
COND_CACHE_DIR = pathlib.Path("/fast/lev/datasets/fastmri/brain/multicoil_train_cond_cache_crop256_tok512_random_acc8_cf0.04")
VAL_ROOT = REPO / "fastmri/brain/multicoil_val"
VAL_SPLIT = HERE / "splits/autoresearch_val.txt"
VAL_SENS_DIRS = f"{REPO}/fastmri/brain/multicoil_val_sens_maps_gpu,{REPO}/fastmri/brain/multicoil_val_sens_maps"
SUNO_SPLIT = REPO / "fastmri/splits/suno_brain_test.txt"         # the test split: confirmation runs only, never the loop
EVAL_BATCH_SIZE = 8
NUM_WORKERS = 8

# frozen networks
ENCODER_CKPT = REPO / "weights/meditok/brain_mc_sense_cplxpair_res512_ep5_frozen.pth"
ENCODER_CONFIG = REPO / "weights/meditok/config_512.json"
SENS_CKPT = REPO / "results/sens_net/run1/best.pt"

RUNS_DIR = HERE / "runs"


# ----------------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------------
def _mask_func():
    return create_mask_for_mask_type(MASK_TYPE, [CENTER_FRACTION], [ACCELERATION])


def train_dataset():
    """Training slices with the k-space, a fresh random mask per sample and the cached ground-truth tokens / latents
    (the slow path: sense_inputs encodes the input on the GPU; the trainer uses cached_train_loader by default)."""
    return SenseSliceDataset(str(TRAIN_ROOT), TRAIN_SENS_DIRS, _mask_func(), resolution=CROP_SIZE, max_coils=MAX_COILS,
                             use_seed=False, custom_split=str(TRAIN_SPLIT), gt_cache_dir=str(GT_CACHE_DIR),
                             gt_cache_token_size=TOKEN_SIZE)


def _shuffled_loader(ds, batch_size, seed, num_workers, rank, world_size):
    """Seeded shuffle; with world_size > 1 a DistributedSampler gives each rank a disjoint shard of the same permutation
    (batch_size is then the per-rank batch; call loader.sampler.set_epoch(e) before every pass)."""
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True,
                                                                  seed=seed, drop_last=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True,
                          drop_last=True, persistent_workers=num_workers > 0)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g, num_workers=num_workers, pin_memory=True,
                      drop_last=True, persistent_workers=num_workers > 0)


def train_loader(batch_size: int, seed: int = 0, num_workers: int = NUM_WORKERS, rank: int = 0, world_size: int = 1):
    """The slow training loader (k-space, fresh masks): consume its batches with sense_inputs()."""
    return _shuffled_loader(train_dataset(), batch_size, seed, num_workers, rank, world_size)


def cond_cache_seed(fname: str, s: int):
    """Seed of the fixed sampling mask of training slice (fname, s) in the conditioning cache."""
    return tuple(map(ord, fname)) + (int(s),)


class CondCacheDataset(torch.utils.data.Dataset):
    """Training slices served from the two caches, without k-space or networks: the conditioning latent of the zero-filled
    SENSE input (net maps) under one fixed random mask per slice, its mask columns and the ground-truth image on the
    acquisition grid (build_cond_cache.py), plus the ground-truth tokens / latents (precompute_gt_cache.py).
    An experiment sees fewer samples than one epoch, so no slice is drawn twice and the fixed mask loses nothing against a
    fresh draw; experiments also see identical (sample, mask) pairs, which lowers the noise between them."""

    def __init__(self, cond_dir=COND_CACHE_DIR, gt_dir=GT_CACHE_DIR, split=TRAIN_SPLIT, allow_partial: bool = False):
        import h5py
        self.cond_dir, self.gt_dir = pathlib.Path(cond_dir), pathlib.Path(gt_dir)
        assert allow_partial or (self.cond_dir / "_complete").exists(), f"conditioning cache under {self.cond_dir} is incomplete: run build_cond_cache.py"
        names = [ln.strip() for ln in open(split) if ln.strip()]
        names = [n for n in names if (self.cond_dir / n).exists()]
        assert names, f"no conditioning cache under {self.cond_dir}: run build_cond_cache.py"
        self.examples = []
        for n in names:
            with h5py.File(self.cond_dir / n, "r") as f:
                sl = f["slice"][:]
                assert (sl == np.arange(len(sl))).all(), f"{n}: cache slices are not 0..S-1"
                self.examples += [(n, int(s)) for s in sl]
        print(f"[CondCacheDataset] {len(names)} volumes, {len(self.examples)} slices from {self.cond_dir}", flush=True)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        import h5py
        n, s = self.examples[i]
        with h5py.File(self.cond_dir / n, "r") as f:
            cond = torch.from_numpy(f["cond_latent"][s])                  # (2, L, 64) f16
            cols = torch.from_numpy(f["mask_cols"][s])                    # (W,) bool
            img = torch.from_numpy(f["gt_image"][s])                      # (H, W, 2) f16, max-normalised SENSE GT
        with h5py.File(self.gt_dir / n, "r") as f:
            tok = torch.from_numpy(f["tokens"][s].astype("int64"))        # (2, K, L)
            lat = torch.from_numpy(f["latent"][s])                        # (2, L, 64) f16, unit-norm chunks
        return dict(cond_latent=cond, mask_cols=cols, gt_image=img, gt_tokens=tok, gt_latent=lat, fname=n, slice=s)


def cached_train_loader(batch_size: int, seed: int = 0, num_workers: int = NUM_WORKERS, rank: int = 0, world_size: int = 1):
    """The fast training loader (CondCacheDataset): consume its batches with cached_inputs()."""
    ds = CondCacheDataset(allow_partial=os.environ.get("AR_ALLOW_PARTIAL_CACHE", "0") == "1")   # tests of the harness only
    return _shuffled_loader(ds, batch_size, seed, num_workers, rank, world_size)


def cached_inputs(batch, device):
    """CondCacheDataset batch -> the same dict as sense_inputs (cond_latent, mask_cols, gt_tokens, gt_latent, gt_image);
    dc_fn is None because the cached path carries no k-space."""
    return dict(cond_latent=batch["cond_latent"].to(device, non_blocking=True).float(),
                mask_cols=batch["mask_cols"].to(device, non_blocking=True).float(),
                gt_tokens=batch["gt_tokens"].to(device, non_blocking=True),
                gt_latent=batch["gt_latent"].to(device, non_blocking=True).float(),
                gt_image=batch["gt_image"].to(device, non_blocking=True).float(),
                dc_fn=None)


def val_loader(split=VAL_SPLIT, batch_size: int = EVAL_BATCH_SIZE, num_workers: int = NUM_WORKERS):
    """Validation slices; the mask of each volume is seeded by its filename, so the evaluation is deterministic."""
    ds = SenseSliceDataset(str(VAL_ROOT), VAL_SENS_DIRS, _mask_func(), resolution=CROP_SIZE, max_coils=MAX_COILS,
                           use_seed=True, custom_split=str(split))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)


def load_tokenizer(device):
    return load_meditok(str(ENCODER_CONFIG), str(ENCODER_CKPT), device)


def load_sens_net(device):
    return _load_sens_net(str(SENS_CKPT), device=device, trainable=False)


@torch.no_grad()
def sense_inputs(batch, device, sens_net, tokenizer, extra_mask=None):
    """GPU-side input preparation of a training batch (the physics; frozen).
    extra_mask: optional (B, H, W, 1) bool of additional k-space revealed to the model (training-time augmentation; the mask
    columns the model is conditioned on are the union).
    Returns cond_latent (B, 2, L, 64) fp32, mask_cols (B, W) float, gt_tokens (B, 2, K, L) long, gt_latent (B, 2, L, 64)
    fp32, gt_image (B, H, W, 2) the max-normalised SENSE ground truth on the acquisition grid, and the per-coil DC closure
    of the union mask (crop grid)."""
    sample_mask = batch["mask"].to(device, non_blocking=True)                                        # (B, H, W, 1)
    H = sample_mask.shape[1]
    sb = prepare_sense_batch(batch, device, sens_net=sens_net, mode="net", center_fractions=CENTER_FRACTION, mask=sample_mask)
    full_mask = sample_mask.bool() if extra_mask is None else (sample_mask.bool() | extra_mask.bool())
    k_in = sb["kc"] * mask_bhw(full_mask, H)[:, None]
    x_in = tok_upsample(c2r(sense_combine(ifft2c(k_in), sb["maps"])), TOKEN_SIZE, RESIZE_MODE)        # (B, 512, 512, 2)
    cond_latent = torch.stack([latent_target(tokenizer, x_in[..., c]) for c in range(2)], dim=1)       # (B, 2, L, 64)
    assert bool(batch["gt_cached"][0]), "training batches must carry the cached ground truth (gt_cache_dir)"
    from models.sense_ops import make_sense_dc_fn
    return dict(cond_latent=cond_latent,
                mask_cols=full_mask[:, 0, :, 0].float(),
                gt_tokens=batch["gt_tokens"].to(device, non_blocking=True).long(),
                gt_latent=batch["gt_latent"].to(device, non_blocking=True).float(),
                gt_image=batch["gt_c_image"].to(device, non_blocking=True),
                dc_fn=make_sense_dc_fn(k_in, sb["maps"], full_mask))


# ----------------------------------------------------------------------------------------------------------------------
# budget
# ----------------------------------------------------------------------------------------------------------------------
class Budget:
    """samples: SAMPLE_BUDGET training samples; wall clock: MAX_MINUTES on the training loop (a run over the cap is a failure)."""

    def __init__(self):
        self.t0 = time.time()

    def exhausted(self, samples_seen: int) -> bool:
        return samples_seen >= SAMPLE_BUDGET

    def timed_out(self) -> bool:
        return (time.time() - self.t0) / 60 > MAX_MINUTES

    def check_time(self):
        minutes = (time.time() - self.t0) / 60
        if minutes > MAX_MINUTES:
            raise TimeoutError(f"training exceeded the wall-clock cap: {minutes:.1f} min > {MAX_MINUTES} min")

    def elapsed(self) -> float:
        return time.time() - self.t0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, tokenizer, sens_net, reconstruct, device, loader=None, log=print):
    """Score `reconstruct` (sample.py) on the validation split with the protocol of autoregressive/sample/reconstruct_far.validate
    (SENSE mode): zero-filled SENSE input from the sens-net maps -> upsample to the token grid -> reconstruct (which applies the
    per-coil DC through dc_fn) -> downsample to the acquisition grid -> x max_val -> magnitude -> per-volume PSNR / SSIM /
    LPIPS / DISTS. reconstruct(model, tokenizer, x_in, mask_cols, dc_fn) -> (rec (B, 512, 512, 2), extra arms {name: image}).
    Returns {"psnr_mean", "ssim_mean", ..., per-volume entries, "arms": {name: metrics}}."""
    was_training = model.training
    model.eval()
    loader = loader if loader is not None else val_loader()
    bufs = {}

    def push(arm, key, sl, rec_np, gt_np):
        d = bufs.setdefault(arm, {}).setdefault(key, {"rec": [], "gt": [], "slice": []})
        d["rec"].append(rec_np); d["gt"].append(gt_np); d["slice"].append(sl)

    for batch in tqdm(loader, "eval", leave=False):
        gt = batch["gt_c_image"].to(device, non_blocking=True)                                       # (B, 256, 256, 2)
        sample_mask = batch["mask"].to(device, non_blocking=True)
        sb = prepare_sense_batch(batch, device, sens_net=sens_net, mode="net", center_fractions=CENTER_FRACTION, mask=sample_mask)
        x_in = tok_upsample(sb["masked_c_image"], TOKEN_SIZE, RESIZE_MODE)
        dc_fn = wrap_dc_fn_resized(sb["dc_fn"], CROP_SIZE, TOKEN_SIZE, RESIZE_MODE)
        mask_cols = sample_mask[:, 0, :, 0].float()
        rec, arms = reconstruct(model, tokenizer, x_in, mask_cols, dc_fn)
        max_val = batch["max_val"].reshape(-1, 1, 1, 1).to(device).float()
        gt_np = transforms.complex_abs(gt * max_val).cpu().float().numpy()
        for arm, img in [("rec", rec)] + list(arms.items()):
            img = tok_downsample(img, CROP_SIZE) if img.shape[1] != CROP_SIZE else img
            rec_np = transforms.complex_abs(img.float() * max_val).cpu().numpy()
            for i in range(gt_np.shape[0]):
                push(arm, pathlib.Path(batch["fname"][i]).stem, int(batch["slice"][i]), rec_np[i], gt_np[i])
    metrics = compute_reconstruction_metrics(bufs["rec"], device)
    metrics["arms"] = {arm: compute_reconstruction_metrics(buf, device) for arm, buf in bufs.items() if arm != "rec"}
    if was_training:
        model.train()
    return metrics


def write_summary(metrics, *, train_steps, samples_seen, training_seconds, eval_seconds, peak_vram_mb, num_params, batch_size,
                  status="ok", out_dir=None, log=print):
    """The block the agent greps (grep '^val_psnr:' run.log). val_psnr 0 marks a crash / timeout."""
    psnr = float(metrics["psnr_mean"]) if metrics else 0.0
    ssim = float(metrics["ssim_mean"]) if metrics else 0.0
    lines = ["---",
             f"val_psnr:         {psnr:.4f}",
             f"val_ssim:         {ssim:.4f}",
             f"status:           {status}",
             f"train_steps:      {train_steps}",
             f"samples_seen:     {samples_seen}",
             f"batch_size:       {batch_size}",
             f"training_seconds: {training_seconds:.1f}",
             f"eval_seconds:     {eval_seconds:.1f}",
             f"peak_vram_mb:     {peak_vram_mb:.1f}",
             f"num_params_M:     {num_params / 1e6:.1f}"]
    for arm, m in (metrics or {}).get("arms", {}).items():
        lines.append(f"arm_{arm}_psnr:   {float(m['psnr_mean']):.4f}")
    for line in lines:
        log(line)
    if out_dir is not None:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "summary.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        if metrics:
            with open(out_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2, default=float)
