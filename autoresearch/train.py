"""
autoresearch trainer (EDITABLE): hyperparameters, model size, optimizer, losses, training loop.

One experiment = `bash run.sh` (python train.py): train from scratch for prepare.SAMPLE_BUDGET samples on one GPU, then
prepare.evaluate scores sample.reconstruct on the validation split and the summary block is printed (grep '^val_psnr:').

Baseline = the recipe of TRUST-MRI run19 (the current best full-scale line) at GPT-B size and from scratch:
token cross-entropy on the Re and Im code chains + LAMBDA_LATENT x latent regression (L2 between unit-norm chunks of the
regressed and the ground-truth tokenizer latents), AdamW, cosine schedule with a short warmup, bf16 autocast, the radial
"far mask" augmentation of the input with probability 1 - P_EMPTY (timestep > 0 tells the model extra k-space was revealed).
The optional image loss (LAMBDA_IMG) decodes the expected codebook embedding of the teacher-forced logits through the frozen
decoder and penalises the L1 of the magnitude against the ground truth on the acquisition grid.

Inputs come from the conditioning cache (prepare.cached_train_loader / cached_inputs: the tokenizer latent of the zero-filled
SENSE input under one fixed random mask per slice, about 3x faster than encoding on the fly). CACHED_INPUTS = False switches
to the slow path (prepare.train_loader / sense_inputs), which encodes on the GPU and supports extra revealed k-space.

run.sh runs one process per GPU it was given (torchrun); the global batch is split over them and DDP averages the gradients.

A few constants take env overrides (AR_*) so the calibration scripts can sweep them; the agent edits the constants instead.
"""
import os
import sys
import math
import time
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare                                   # frozen harness (adds the repo root to sys.path)
import sample
from model import build_model, num_params

import datetime

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models.sense_ops import tok_downsample, decode_expected
from autoregressive.utils.mask_utils import radial_circle_radii, radial_circle_mask, sample_circle_radius_idx_weighted
from data import transforms


def _env(name, default, cast):
    v = os.environ.get(name)
    return default if v is None else cast(v)


class Diverged(RuntimeError):
    """raised after MAX_SKIPPED updates were dropped for non-finite losses or gradients"""


# Data parallel: run.sh starts one process per GPU of AR_GPU through torchrun, which sets these (one GPU: a plain process).
# The global batch BATCH_SIZE is split over the ranks; gradients are averaged by DDP, so one experiment is the same
# optimisation on 1 or 2 GPUs. Only rank 0 logs, evaluates and writes the summary.
RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
DISTRIBUTED = WORLD_SIZE > 1


def log(*args, **kwargs):
    if RANK == 0:
        print(*args, flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# hyperparameters (edit freely)
# ----------------------------------------------------------------------------------------------------------------------
DEPTH = _env("AR_DEPTH", 12, int)                 # body layers      (full scale: 24)
DIM = _env("AR_DIM", 768, int)                    # model width      (full scale: 1024)
N_HEAD = _env("AR_NHEAD", 12, int)                # attention heads  (full scale: 16)
N_OUTPUT_LAYER = 1                                # transformer blocks of the token head's chain
HEAD_INTERLEAVED = _env("AR_HEAD_INTERLEAVED", 0, int) == 1
LATENT_DIM = 64 if _env("AR_LAMBDA_LATENT", 5.0, float) > 0 else 0

BATCH_SIZE = _env("AR_BATCH_SIZE", 16, int)            # global batch (split over the GPUs: 8 per GPU on two); run19 used 8
LR = _env("AR_LR", 1e-4, float)
WEIGHT_DECAY = 0.05
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0
WARMUP_FRAC = 0.03                                # fraction of the steps with linear warmup, then cosine to 0
DROPOUT = 0.1                                     # residual / FFN dropout
TOKEN_DROPOUT = 0.1                               # dropout on the input embeddings

LAMBDA_LATENT = _env("AR_LAMBDA_LATENT", 5.0, float)
LAMBDA_IMG = _env("AR_LAMBDA_IMG", 0.0, float)
IMG_LOSS_BATCH = 2                                # samples per step through the decoder for the image loss (memory)
IMG_TEMPERATURE = 1.0
CACHED_INPUTS = True                                   # cached conditioning (fast) vs on-the-fly encoding (prepare.sense_inputs)
P_EMPTY = 0.5                                     # probability of the plain input (timestep 0); else a radial far mask
FAR_NUM_MASKS = 1                                 # radii of the far mask (with 1: a 5-pixel central disk, i.e. inside the ACS)

COMPILE = _env("AR_COMPILE", 0, int) == 1
SEED = _env("AR_SEED", 0, int)
LOG_EVERY = 50
MAX_SKIPPED = 20                                       # updates dropped for non-finite values before the run counts as diverged
SAVE_CHECKPOINT = _env("AR_SAVE_CKPT", 0, int) == 1   # runs/<tag>/model.pt (model weights only)
RUN_TAG = os.environ.get("AR_TAG", time.strftime("%m%d_%H%M%S"))


# ----------------------------------------------------------------------------------------------------------------------
# training-time augmentation: radial far mask (extra low-frequency k-space revealed with timestep >= 1)
# ----------------------------------------------------------------------------------------------------------------------
def far_mask(B, H, W, device):
    """-> (extra_mask (B, H, W, 1) bool or None, timestep (B,) long)."""
    if torch.rand(1).item() < P_EMPTY:
        return None, torch.zeros(B, dtype=torch.long, device=device)
    radii = radial_circle_radii(H, W, num_masks=FAR_NUM_MASKS, center_fraction=prepare.CENTER_FRACTION, device=device)[:-1]
    idx = sample_circle_radius_idx_weighted(radii, device=device)
    m = radial_circle_mask(H, W, radii[idx].item(), device=device)                                   # (H, W) bool
    return m[None, :, :, None].expand(B, H, W, 1), torch.full((B,), idx + 1, dtype=torch.long, device=device)


def latent_loss(pred, target):
    """L2 between unit-norm codebook chunks (= 2 - 2 cos), averaged per chunk. pred / target (B, 2, L, 64)."""
    return ((pred.float() - target) ** 2).sum(-1).mean() * (prepare.NUM_CODEBOOKS / target.shape[-1])


def image_loss(tokenizer, logits, gt_image):
    """Expected-embedding decode of the teacher-forced logits -> acquisition grid -> L1 on the magnitude vs the ground truth."""
    nb = min(IMG_LOSS_BATCH, logits.shape[0])
    x_tok = decode_expected(tokenizer, logits[:nb], IMG_TEMPERATURE)                                 # (nb, 512, 512, 2)
    x_crop = tok_downsample(x_tok.float(), prepare.CROP_SIZE)
    mag = (x_crop.pow(2).sum(-1) + 1e-8).sqrt()                      # eps: the gradient of |z| is undefined at z = 0
    return (mag - transforms.complex_abs(gt_image[:nb].float())).abs().mean()


def cosine_lr(step, total):
    warmup = int(total * WARMUP_FRAC)
    if step < warmup:
        return LR * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return LR * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


# ----------------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------------
def main():
    if DISTRIBUTED:
        torch.cuda.set_device(LOCAL_RANK)
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
    device = torch.device("cuda", LOCAL_RANK)
    assert BATCH_SIZE % WORLD_SIZE == 0, "BATCH_SIZE must split evenly over the GPUs"
    per_rank_batch = BATCH_SIZE // WORLD_SIZE
    prepare.set_seed(SEED + RANK)                      # DDP broadcasts rank 0's initial weights; dropout etc. differ per rank
    out_dir = prepare.RUNS_DIR / RUN_TAG
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"run {RUN_TAG}: budget {prepare.SAMPLE_BUDGET} samples, cap {prepare.MAX_MINUTES} min, batch {BATCH_SIZE}, "
          f"model depth {DEPTH} dim {DIM} heads {N_HEAD}, lr {LR}, lambda_latent {LAMBDA_LATENT}, lambda_img {LAMBDA_IMG}, "
          f"seed {SEED}, cached_inputs {CACHED_INPUTS}, gpus {WORLD_SIZE}",
          flush=True)

    tokenizer = prepare.load_tokenizer(device)
    sens_net = prepare.load_sens_net(device)
    model = build_model(DEPTH, DIM, N_HEAD, n_output_layer=N_OUTPUT_LAYER, latent_dim=LATENT_DIM, head_interleaved=HEAD_INTERLEAVED,
                        resid_dropout_p=DROPOUT, ffn_dropout_p=DROPOUT, token_dropout_p=TOKEN_DROPOUT,
                        block_size=prepare.NUM_POS, mask_width=prepare.MASK_WIDTH, vocab_size=prepare.VOCAB_SIZE,
                        num_codebooks=prepare.NUM_CODEBOOKS, cont_in_dim=prepare.LATENT_DIM).to(device)
    n_params = num_params(model)
    log(f"model: {n_params / 1e6:.1f}M parameters", flush=True)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}],
                                  lr=LR, betas=BETAS, fused=True)
    total_steps = prepare.SAMPLE_BUDGET // BATCH_SIZE
    if CACHED_INPUTS:
        # the far mask of FAR_NUM_MASKS = 1 lies inside the ACS and reveals nothing: with cached inputs only the timestep flips
        assert FAR_NUM_MASKS == 1, "extra revealed k-space needs the on-the-fly path: CACHED_INPUTS = False"
        loader = prepare.cached_train_loader(per_rank_batch, seed=SEED, rank=RANK, world_size=WORLD_SIZE)
    else:
        loader = prepare.train_loader(per_rank_batch, seed=SEED, rank=RANK, world_size=WORLD_SIZE)
    log(f"training set: {len(loader.dataset):,} slices; {total_steps} steps", flush=True)
    # find_unused_parameters=False: every parameter must receive a gradient each step (DDP raises otherwise)
    train_model = DDP(model, device_ids=[LOCAL_RANK], find_unused_parameters=False) if DISTRIBUTED else model
    train_model = torch.compile(train_model) if COMPILE else train_model

    budget = prepare.Budget()
    step, samples, skipped = 0, 0, 0
    run_loss = run_ce = run_lat = run_img = 0.0
    t_log = time.time()
    status = "ok"
    model.train()
    try:
        epoch = 0
        while not budget.exhausted(samples):
            if DISTRIBUTED:
                loader.sampler.set_epoch(epoch)
            epoch += 1
            for batch in loader:
                if budget.exhausted(samples):
                    break
                for g in optimizer.param_groups:
                    g["lr"] = cosine_lr(step, total_steps)
                B = batch["gt_tokens"].shape[0]
                extra, t = far_mask(B, prepare.CROP_SIZE, prepare.CROP_SIZE, device)
                if CACHED_INPUTS:
                    inp = prepare.cached_inputs(batch, device)
                else:
                    inp = prepare.sense_inputs(batch, device, sens_net, tokenizer, extra_mask=extra)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, loss_ce, latent = train_model(cond_latent=inp["cond_latent"], sample_mask_cols=inp["mask_cols"],
                                                          timestep=t, targets=inp["gt_tokens"], return_latent=LAMBDA_LATENT > 0)
                loss = loss_ce
                l_lat = l_img = torch.zeros((), device=device)
                if LAMBDA_LATENT > 0:
                    l_lat = latent_loss(latent, inp["gt_latent"])
                    loss = loss + LAMBDA_LATENT * l_lat
                if LAMBDA_IMG > 0:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        l_img = image_loss(tokenizer, logits, inp["gt_image"])
                    loss = loss + LAMBDA_IMG * l_img
                finite = bool(torch.isfinite(loss))
                timed_out = budget.timed_out()
                if DISTRIBUTED:                        # backward is a collective: every rank must take the same branch
                    flags = torch.tensor([not finite, timed_out], device=device, dtype=torch.int32)
                    dist.all_reduce(flags, op=dist.ReduceOp.MAX)
                    finite, timed_out = not bool(flags[0]), bool(flags[1])
                if timed_out:
                    raise TimeoutError(f"training exceeded the wall-clock cap: {budget.elapsed() / 60:.1f} min > {prepare.MAX_MINUTES} min")
                if finite:
                    loss.backward()
                    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP if GRAD_CLIP > 0 else float("inf"))
                    finite = bool(torch.isfinite(gnorm))
                if finite:
                    optimizer.step()
                else:                                  # a non-finite loss or gradient would poison every weight: drop the update
                    skipped += 1
                    log(f"step {step}: non-finite loss or gradient, update skipped ({skipped} so far)", flush=True)
                    if skipped > MAX_SKIPPED:
                        raise Diverged(f"{skipped} updates skipped for non-finite values")
                optimizer.zero_grad(set_to_none=True)
                step += 1
                samples += B * WORLD_SIZE
                if finite:
                    run_loss += loss.item(); run_ce += loss_ce.item(); run_lat += l_lat.item(); run_img += l_img.item()
                if step % LOG_EVERY == 0:
                    torch.cuda.synchronize()
                    dt = time.time() - t_log
                    log(f"step {step:5d}/{total_steps} loss {run_loss / LOG_EVERY:.4f} ce {run_ce / LOG_EVERY:.4f} "
                          f"lat {run_lat / LOG_EVERY:.4f} img {run_img / LOG_EVERY:.4f} lr {cosine_lr(step, total_steps):.2e} "
                          f"{LOG_EVERY / dt:.2f} steps/s  {budget.elapsed() / 60:.1f} min" + (f"  skipped {skipped}" if skipped else ""),
                          flush=True)
                    run_loss = run_ce = run_lat = run_img = 0.0
                    t_log = time.time()
    except TimeoutError as e:
        log(f"TIMEOUT: {e}", flush=True)
        status = "timeout"
    except Diverged as e:
        log(f"DIVERGED: {e}", flush=True)
        status = "diverged"
    training_seconds = budget.elapsed()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 2 ** 20

    metrics, eval_seconds = None, 0.0
    if RANK == 0:
        if status == "ok" and not prepare.SKIP_EVAL:
            t0 = time.time()
            metrics = prepare.evaluate(model, tokenizer, sens_net, sample.reconstruct, device)
            eval_seconds = time.time() - t0
        if SAVE_CHECKPOINT:
            torch.save({"model": model.state_dict(), "step": step}, out_dir / "model.pt")
        print(f"skipped_updates:  {skipped}")
        prepare.write_summary(metrics, train_steps=step, samples_seen=samples, training_seconds=training_seconds,
                              eval_seconds=eval_seconds, peak_vram_mb=peak_vram_mb, num_params=n_params, batch_size=BATCH_SIZE,
                              status=status, out_dir=out_dir)
    if DISTRIBUTED:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
