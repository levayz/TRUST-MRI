
# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import os

import sys
from pathlib import Path
# ensure evaluation/generation is on sys.path for utils imports
gen_root = Path(__file__).resolve().parents[2]
if str(gen_root) not in sys.path:
    sys.path.append(str(gen_root))
import torch
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from glob import glob
from copy import deepcopy
import random
import re

import math
import time
import torch
import inspect
import argparse
import json
import subprocess

from utils.logger import create_logger, log_args
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.build import build_dataset
from autoregressive.models.gpt_kar import KAR_GPT_models, PROBE_GPT_B, remap_v1_to_v2, adapt_mask_width, codebook_from_quantizer
from autoregressive.models.commit import expected_embedding_view, commit_keep_mask, commit_timestep
from autoregressive.sample.reconstruct_far import validate
from autoregressive.models.generate import reconstruct_far as _reconstruct_far
from autoregressive.utils.mask_utils import (
    make_centered_radius_grid,
    radial_circle_mask,
    radial_circle_radii,
    sample_circle_radius_idx,
    sample_circle_radius_idx_weighted
)
from models.meditok import MedITok, DotDict
from models.sense_ops import prepare_sense_batch, make_sense_dc_fn, sense_combine, ifft2c, c2r, mask_bhw, tok_upsample, tok_downsample, wrap_dc_fn_resized, decode_expected, latent_target, encode_features, features_to_tokens, features_to_latent
from models.varnet_sens import load_sens_net
from data import transforms
import torchvision.utils as vutils
from PIL import Image


def save_pngs(out_path: Path, images_bchw_list: list[torch.Tensor]):
    """
    Columns = list items, rows = batch index.
    Expects each tensor to be BCHW floats in [0,1], all same shape.
    """
    if len(images_bchw_list) == 0:
        raise ValueError("images_bchw_list must be non-empty")

    ref = images_bchw_list[0]
    if ref.ndim != 4:
        raise ValueError(f"Expected BCHW (4D) tensor, got shape {tuple(ref.shape)}")

    B, C, H, W = ref.shape
    for i, x in enumerate(images_bchw_list):
        if x.ndim != 4:
            raise ValueError(f"Item {i} expected BCHW (4D) tensor, got shape {tuple(x.shape)}")
        if tuple(x.shape) != (B, C, H, W):
            raise ValueError(f"Item {i} shape {tuple(x.shape)} != {(B, C, H, W)}")

    K = len(images_bchw_list)
    panel = torch.stack(images_bchw_list, dim=0)     # [K,B,C,H,W]
    panel = panel.permute(1, 0, 2, 3, 4)            # [B,K,C,H,W]
    panel = panel.reshape(B * K, C, H, W)           # [B*K,C,H,W] interleaved by batch

    grid = vutils.make_grid(panel, nrow=K, padding=2)
    grid_img = (grid.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(grid_img).save(out_path)


def rgb_to_grayscale(img):
    # return img[:,0] * 0.2126 + img[:,1] * 0.7152 + img[:,2] * 0.0722
    return (img[:,0] + img[:,1] + img[:,2]) / 3

#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def create_optimizer(model, weight_decay, learning_rate, betas, logger):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer


def get_cosine_scheduler(
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
        base_lr: float = 1e-4,
        end_lr: float = 0.0,
):
    num_warmup_steps = int(num_training_steps * 0.03)

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
                   float(max(1, num_training_steps - num_warmup_steps))
        ratio = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return (end_lr + (base_lr - end_lr) * ratio) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def sample_far_cartesian_mask(image, curr_iter, total_iters, num_masks=32, center_fraction=0.08):
    """
    input: GT image
    output: masked image
    masking is symmetric, True indicates sampled positions
    """
    B, H, W, _ = image.shape
    kspace = transforms.fft2(image)
    end_idx =  W // 2 - int(W * center_fraction) // 2
    if num_masks == None:
        step = 1
    else:
        step = int(end_idx / num_masks + 1)
    mask_levels = list(range(0, end_idx, step))
    mask_levels.append(end_idx)

    p = float(curr_iter) / float(max(1, total_iters))
    p = max(0.0, min(1.0, p))

    min_level = int((1.0 - p**2) * (len(mask_levels) - 1))
    min_level = max(1, min(min_level, len(mask_levels) - 1))  # keep slice non-empty; avoid 0-mask

    chosen_mask_level = random.choice(mask_levels[min_level:])

    sample_mask = torch.ones_like(kspace)
    sample_mask[:,:, :chosen_mask_level] = 0
    sample_mask[:, :, -chosen_mask_level:] = 0

    sample_mask[:,:chosen_mask_level] = 0
    sample_mask[:,-chosen_mask_level:] = 0
    
    masked_kspace = kspace * sample_mask
    image = transforms.ifft2(masked_kspace)

    return image, sample_mask, torch.tensor(chosen_mask_level).to(image.device)


def sample_far_radial_mask(image, curr_iter, total_iters, num_masks=8, center_fraction=0.04, p_empty=0.5):
    B, H, W, _ = image.shape
    kspace = transforms.fft2(image)

    radii = radial_circle_radii(
        H,
        W,
        num_masks=num_masks,
        center_fraction=center_fraction,
        device=image.device,
        dtype=torch.float32,
    )
    # remove the last mask which should cover the full kspace
    radii = radii[:-1]

    # Empty mask case: t=0 means no additional k-space beyond the initial undersampling mask.
    # This is consistent with inference (reconstruct_far step idx=0 also uses t=0 with no far mask).
    if torch.rand(1).item() < p_empty:
        empty_mask = torch.zeros(B, H, W, kspace.shape[-1], device=image.device, dtype=kspace.dtype)
        return torch.zeros_like(image), empty_mask, torch.tensor(0).to(image.device)

    # Radial mask case: shift t by 1 to leave t=0 reserved for the empty mask.
    # idx = sample_circle_radius_idx(curr_iter, total_iters, num_masks=len(radii), exponent=2.0, device=image.device)
    idx = sample_circle_radius_idx_weighted(radii, device=image.device)
    radius = radii[idx].item()

    sample_mask_2d = radial_circle_mask(H, W, radius, device=image.device, dtype=torch.float32).to(kspace.dtype)
    sample_mask = sample_mask_2d[None, :, :, None].expand(B, H, W, kspace.shape[-1])

    masked_kspace = kspace * sample_mask
    image = transforms.ifft2(masked_kspace)

    return image, sample_mask, torch.tensor(idx + 1).to(image.device)

def sample_far_mask(mode, *args, **kwargs):
    if mode == 'cartesian':
        return sample_far_cartesian_mask(*args, **kwargs)
    elif mode =='radial':
        return sample_far_radial_mask(*args, **kwargs)
    else:
        raise ValueError("unknown mask mode")


def enrich_mask_cols(cols: torch.Tensor, acc_lo: float, acc_hi: float) -> torch.Tensor:
    """Privileged-teacher masks. cols (B, W) bool = the sampled columns -> (B, W) bool with extra random columns so that each
    sample has round(W / R) columns in total, R ~ U[acc_lo, acc_hi] per sample (never fewer than it already has)."""
    B, W = cols.shape
    R = acc_lo + (acc_hi - acc_lo) * torch.rand(B, device=cols.device)
    extra = ((W / R).round().long() - cols.sum(1)).clamp_min(0)                                    # (B,)
    scores = torch.rand(B, W, device=cols.device).masked_fill(cols, -1.0)                           # sampled columns never chosen
    order = scores.argsort(dim=1, descending=True)                                                  # unsampled columns, random order
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(W, device=cols.device).expand(B, W))
    return cols | (rank < extra[:, None])


def kl_to_teacher(logits_s: torch.Tensor, logits_t: torch.Tensor, temp: float, weights: torch.Tensor) -> torch.Tensor:
    """KL(teacher || student) of the teacher-forced token distributions, logits (B, 2, K, L, V), averaged over positions, streams
    and codebooks and weighted per sample by weights (B,) (x temp^2, the usual distillation scaling). Chunked over the codebooks
    so the fp32 softmaxes never hold the full (B, 2, K, L, V) tensor."""
    K = logits_s.shape[2]
    tot = logits_s.new_zeros((), dtype=torch.float32)
    wsum = weights.sum().clamp_min(1.0)
    for k in range(K):
        ls = F.log_softmax(logits_s[:, :, k].float() / temp, dim=-1)
        lt = F.log_softmax(logits_t[:, :, k].float() / temp, dim=-1)
        kl = (lt.exp() * (lt - ls)).sum(-1).mean(dim=(1, 2))                                        # (B,)
        tot = tot + (kl * weights).sum() / wsum
    return tot / K * (temp ** 2)

def data_consistency(in_c_image, sample_mask, output_c_image):
    in_kspace = transforms.fft2(in_c_image)
    output_kspace = transforms.fft2(output_c_image) 

    output_kspace = torch.where(sample_mask.bool(), in_kspace, output_kspace)

    output_c_image = transforms.ifft2(output_kspace)

    return output_c_image


def create_validation_args(train_args, experiment_dir, epoch, train_steps):
    """
    Create validation args from training args.
    
    Args:
        train_args: Training argument namespace
        experiment_dir: Base experiment directory
        epoch: Current epoch number (1-indexed for display)
        train_steps: Current training step number
    
    Returns:
        Namespace with validation arguments
    """
    val_args = argparse.Namespace()
    
    # Output directory
    val_args.output_dir = f"{experiment_dir}/val_epoch_{epoch:03d}_step_{train_steps:07d}"
    
    # Model & data configuration (from training args)
    val_args.image_size = train_args.image_size
    val_args.downsample_size = train_args.downsample_size
    val_args.num_classes = train_args.num_classes
    val_args.cls_token_num = train_args.cls_token_num
    val_args.num_codebooks = train_args.num_codebooks
    val_args.cond_token_num = train_args.cond_token_num
    val_args.far_mask_mode = train_args.far_mask_mode
    val_args.complex_mode = train_args.complex_mode
    val_args.radial_weighting = train_args.radial_weighting
    val_args.data_split = train_args.val_data_split
    
    # Validation-specific settings
    val_args.n_steps = train_args.val_n_steps
    val_args.n_scans_to_save = train_args.val_n_scans_to_save
    val_args.active_sampling = train_args.val_active_sampling
    val_args.active_sampling_fn = train_args.val_active_sampling_fn

    val_args.center_fractions = train_args.center_fractions
    val_args.accelerations = train_args.accelerations
    val_args.n_steps = train_args.val_n_steps if train_args.val_n_steps else train_args.num_masks
    val_args.refine_rounds = getattr(train_args, 'refine_rounds', 0)
    val_args.rec_mode = getattr(train_args, 'val_rec_mode', 'tokens')
    val_args.latent_dim = getattr(train_args, 'latent_dim', 0)
    val_args.model_version = getattr(train_args, 'model_version', 1)
    val_args.mask_width = getattr(train_args, 'mask_width', 256)
    val_args.draft_cond = getattr(train_args, 'draft_cond', False)
    val_args.gt_cache_dir = None   # validation scores images; no ground-truth tokens needed
    val_args.commit_schedule = getattr(train_args, 'commit_schedule', None) if getattr(train_args, 'commit', 'none') != 'none' else None
    val_args.commit_mode = getattr(train_args, 'commit', 'conf')
    val_args.head_interleaved = getattr(train_args, 'head_interleaved', 1)
    
    # Sampling settings (fixed for validation)
    val_args.viz_gen_steps = False
    val_args.temperature = 1.0
    val_args.top_p = 1.0
    val_args.top_k = 0
    val_args.denoise = False
    
    return val_args


def find_latest_checkpoint(experiment_dir):
    """
    Find the latest checkpoint in the experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Tuple of (checkpoint_path, step_number) or None if no checkpoints found
    """
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = []
    for fname in os.listdir(checkpoint_dir):
        if fname.endswith('.pt') and fname[:-3].isdigit():
            step_num = int(fname[:-3])
            checkpoint_files.append((step_num, os.path.join(checkpoint_dir, fname)))
    
    if not checkpoint_files:
        return None
    
    checkpoint_files.sort(key=lambda x: x[0], reverse=True)
    latest_step, latest_path = checkpoint_files[0]
    
    return latest_path, latest_step


#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        experiment_dir = f"{args.results_dir}/{args.exp_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        
        # Auto-resume: check for latest checkpoint if not explicitly provided
        if not args.gpt_ckpt and os.path.exists(experiment_dir):
            result = find_latest_checkpoint(experiment_dir)
            if result is not None:
                args.gpt_ckpt, found_step = result
                logger.info(f"=== RESUME MODE ===")
                logger.info(f"Found checkpoint at step {found_step}: {args.gpt_ckpt}")
                logger.info(f"Resuming training from this checkpoint")
                logger.info("=" * 50)
    else:
        logger = create_logger(None)
    
    # Broadcast checkpoint path from rank 0 to all other ranks
    checkpoint_path_list = [args.gpt_ckpt] if rank == 0 else [None]
    dist.broadcast_object_list(checkpoint_path_list, src=0)
    args.gpt_ckpt = checkpoint_path_list[0]

    # training args
    log_args(args, logger)

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup model
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
    latent_size = args.image_size // args.downsample_size
    model = KAR_GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        cond_token_num=args.cond_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
        num_codebooks=args.num_codebooks,
        n_output_layer=args.num_output_layer,
        vq_embed_path=args.vq_embed_path,
        latent_dim=args.latent_dim,
        latent_head_type=args.latent_head_type,
        latent_mlp_width=args.latent_mlp_width,
        model_version=args.model_version,
        head_interleaved=bool(args.head_interleaved),
        mask_width=(args.sense_crop_size if args.dataset == 'sense' else args.image_size),
        draft_cond=args.draft_cond,
    )
    if args.probe_model:
        model = PROBE_GPT_B(
            resolution=args.image_size,
            backbone=model,
            backbone_checkpoint=args.backbone_ckpt,
            freeze_backbone=args.freeze_backbone
        )

    model = model.to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load MedITok encoder for masked-image conditioning
    with open(args.encoder_config, "r") as f:
        enc_cfg = json.load(f)
    meditok = MedITok(DotDict(enc_cfg)).to(device).eval()
    enc_state = torch.load(args.encoder_ckpt, map_location="cpu")
    if isinstance(enc_state, dict) and "model" in enc_state:
        enc_state = enc_state["model"]
    meditok.load_state_dict(enc_state, strict=False)
    requires_grad(meditok, False)
    if getattr(model, 'latent_head_type', 'linear') == 'fused':   # the fused latent head reads the codebook rows of the token head's codes
        model.set_latent_codebook(meditok.quantizer)
        logger.info("fused latent head: tokenizer codebook registered (token head rolled out greedily during training)")

    # SENSE mode: frozen sensitivity net (maps from the ACS lines of the undersampled k-space)
    sens_net = None
    if args.dataset == 'sense' and args.sens_maps == 'net':
        commit_codebook = codebook_from_quantizer(meditok.quantizer).to(device) if getattr(args, 'commit', 'none') != 'none' else None   # (K, V, d) for the expected-embedding view
        sens_net = load_sens_net(args.sens_ckpt, device=device, trainable=False)
        logger.info(f"Sensitivity net loaded from {args.sens_ckpt}: {sum(p.numel() for p in sens_net.parameters()):,} params (frozen)")

    if args.ema:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    # Setup optimizer
    optimizer = create_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    # Setup data:
    dataset = build_dataset(args)
    if args.overfit_n_samples > 0:
        reps = args.overfit_n_samples // len(dataset)
        print(reps)
        dataset.examples *= reps
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images (from {args.tokenizer_name}: {args.code_dir}) ")
    
    # Setup validation
    val_loader = None
    best_val_psnr = -float('inf')
    if args.val_every > 0 and args.val_data_root and rank == 0:
        logger.info(f"Setting up validation dataset...")
        # Create validation args
        val_args = argparse.Namespace(**vars(args))
        val_args.data_root = args.val_data_root
        val_args.code_dir = args.val_code_dir if args.val_code_dir else args.code_dir
        if args.val_data_split:
            val_args.data_split = args.val_data_split
        if args.val_num_samples:
            val_args.num_samples = args.val_num_samples
        val_args.use_seed=True
        if args.dataset == 'sense':
            val_args.sens_dirs = args.val_sens_dirs or args.sens_dirs
            val_args.return_oracle_maps = (args.sens_maps == 'oracle')
        
        # Build validation dataset
        val_dataset = build_dataset(val_args)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        logger.info(f"Validation dataset contains {len(val_dataset):,} images")
        logger.info(f"Validation enabled: running every {args.val_every} epochs")
    
    # Prepare models for training:
    if args.gpt_ckpt:
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
        if args.load_weights_only:   # warm start: new heads / conditioning parameters (latent head, gates, mask embedding) start fresh
            sd = checkpoint["model"]
            if args.model_version >= 2.1 and "output.codebooks.0.weight" in sd:   # v1/v2 checkpoint -> v2.1 model (body + head layer + head tables)
                sd = remap_v1_to_v2(sd)
                logger.info("warm start from a v1/v2 checkpoint: input token tables dropped (v2.1 projects continuous latents), head tables copied per stream")
            if args.model_version >= 2 and "mask_embed.weight" in sd and int(sd["mask_embed.weight"].shape[1]) != int(model.mask_width):
                logger.info(f"warm start: mask embedding resampled from width {int(sd['mask_embed.weight'].shape[1])} to {int(model.mask_width)}")
                sd = adapt_mask_width(sd, model.mask_width)
            msd = model.state_dict()
            mismatched = [k for k, v in sd.items() if k in msd and tuple(msd[k].shape) != tuple(v.shape)]
            if mismatched:
                logger.info(f"warm start: {len(mismatched)} parameter(s) with a different shape start fresh: {mismatched}")
                sd = {k: v for k, v in sd.items() if k not in mismatched}
            res = model.load_state_dict(sd, strict=False)
            assert not res.unexpected_keys, f"checkpoint has keys the model does not: {res.unexpected_keys[:8]}"
            if res.missing_keys:
                logger.info(f"warm start: {len(res.missing_keys)} parameter(s) not in the checkpoint start fresh: {res.missing_keys}")
        else:
            model.load_state_dict(checkpoint["model"], strict=True)
        if args.load_weights_only:
            # warm start: model weights only; optimizer, EMA and the step counter start fresh
            if args.ema:
                update_ema(ema, model, decay=0)
            train_steps, start_epoch = 0, 0
            logger.info("--load-weights-only: model weights loaded, optimizer/EMA state not restored")
        else:
            if args.ema:
                ema.load_state_dict(checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"], strict=False)
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except:
                logger.info("Optimizer state loading failed, continuing without loading optimizer state...")
            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.gpt_ckpt.split('/')[-1].split('.')[0])
            start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
            train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model)  # requires PyTorch 2.0

    # Sync before DDP: dataset scanning and checkpoint loading are asymmetric across ranks
    # (rank 0 also builds val dataset). Without this barrier, rank 0 can enter DDP init
    # while rank 1 is still scanning H5 files → NCCL spin-wait.
    dist.barrier()
    model = DDP(model.to(device), device_ids=[args.gpu], find_unused_parameters=True)
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == 'fp16'))
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    running_ce = 0.0
    running_img = 0.0
    running_ref = 0.0
    running_lat = 0.0
    running_kl = 0.0
    running_dlat = 0.0
    start_time = time.time()

    if args.schedule == 'cosine':
        total_train_steps = math.ceil(args.epochs * len(dataset) / args.global_batch_size)
        scheduler = get_cosine_scheduler(optimizer, total_train_steps, base_lr=args.lr)
        if train_steps > 0:   # resume: fast-forward the schedule to the restored step (otherwise warmup restarts)
            for _ in range(train_steps):
                scheduler.step()
            logger.info(f"LR schedule fast-forwarded to step {train_steps}: lr={scheduler.get_last_lr()[0]:.3e}")
    else:
        scheduler = None

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for idx, batch in enumerate(loader):
            gt_c_image = batch["gt_c_image"].to(device, non_blocking=True)   # [B,H,W,2],
            gt_tokens = batch["gt_tokens"].to(device, non_blocking=True)   # saved as [B,2,K,256]
            sample_mask = batch["mask"].to(device, non_blocking=True)

            if args.dataset == 'sense':
                # SENSE path: maps from the ACS lines of the measurement mask (sens net or oracle); the GPT input is
                # the SENSE combine of the coils under mask ∪ radial far mask. No data consistency during training.
                _, far_sample_mask, t = sample_far_mask(args.far_mask_mode, gt_c_image, epoch * len(loader) + idx, 20 * len(loader), num_masks=args.num_masks, p_empty=args.p_empty)
                t = t.repeat(gt_c_image.shape[0])
                full_mask = (far_sample_mask.bool() | sample_mask.bool())[..., :1]            # (B,H,W,1)
                distill = args.distill_teacher != 'none'
                if distill:
                    # privileged teacher: the EMA teacher sees the student's lines plus extra random columns (acceleration drawn
                    # from --distill-acc); a fraction of the students gets that richer input itself (ground-truth loss only), so
                    # the EMA teacher is competent on such inputs
                    cols_t = enrich_mask_cols(sample_mask[:, 0, :, 0].bool(), *args.distill_acc)                  # (B, W)
                    teacher_mask = full_mask | cols_t[:, None, :, None].expand_as(full_mask)                       # (B,H,W,1)
                    rich = torch.rand(full_mask.shape[0], device=device) < args.p_mixed_acc                         # (B,)
                    full_mask = torch.where(rich[:, None, None, None], teacher_mask, full_mask)
                with torch.no_grad():
                    sb = prepare_sense_batch(batch, device, sens_net=sens_net, mode=args.sens_maps,
                                             center_fractions=args.center_fractions, mask=sample_mask)
                    k_in = sb["kc"] * mask_bhw(full_mask, gt_c_image.shape[1])[:, None]
                    masked_in = c2r(sense_combine(ifft2c(k_in), sb["maps"]))
                    if distill:
                        k_t = sb["kc"] * mask_bhw(teacher_mask, gt_c_image.shape[1])[:, None]
                        masked_t = tok_upsample(c2r(sense_combine(ifft2c(k_t), sb["maps"])), args.image_size, args.sense_resize_mode)
                        cond_latent_t = torch.stack([latent_target(meditok, masked_t[..., c]) for c in range(2)], dim=1)   # (B, 2, L, D)
                        mask_cols_t = teacher_mask[:, 0, :, 0].float()
                        del k_t, masked_t
                    if args.gt_gauge == 'net':
                        gt_c_image = c2r(sense_combine(sb["coils"], sb["maps"]) * batch["support"].to(device))
                    nb_img = min(args.img_loss_batch, k_in.shape[0])
                    sense_dc_fn_img = make_sense_dc_fn(k_in[:nb_img], sb["maps"][:nb_img], full_mask[:nb_img])   # per-coil DC for the image-loss sub-batch
                    sense_dc_fn = wrap_dc_fn_resized(make_sense_dc_fn(k_in, sb["maps"], full_mask), args.sense_crop_size, args.image_size, args.sense_resize_mode)  # viz only
                    nb_ref = args.refine_batch if args.refine_batch > 0 else k_in.shape[0]
                    sense_dc_fn_ref = wrap_dc_fn_resized(make_sense_dc_fn(k_in[:nb_ref], sb["maps"][:nb_ref], full_mask[:nb_ref]), args.sense_crop_size, args.image_size, args.sense_resize_mode)  # teacher drafts
                    gt_crop = gt_c_image                                                            # (B,H,W,2) on the acquisition grid, for the image loss
                    # tokenizer / GPT grid may be larger than the acquisition grid (e.g. 512 tokens of a 256 slice)
                    masked_in = tok_upsample(masked_in, args.image_size, args.sense_resize_mode)
                    gt_c_image = tok_upsample(gt_c_image, args.image_size, args.sense_resize_mode)
                    del sb, k_in
                sample_mask = full_mask
            else:
                masked_c_image = batch["masked_c_image"].to(device, non_blocking=True)   # [B,H,W,2]
                masked_in, far_sample_mask, t = sample_far_mask(args.far_mask_mode, gt_c_image, epoch * len(loader) + idx, 20 * len(loader), num_masks=args.num_masks, p_empty=args.p_empty)
                t = t.repeat(masked_in.shape[0])

                masked_in = data_consistency(gt_c_image, sample_mask, masked_in)
                sample_mask = far_sample_mask.bool() | sample_mask.bool()

            mask_cols = sample_mask[:, 0, :, 0].float()   # (B, W) sampling-mask columns on the acquisition grid (1D Cartesian masks; model v2 conditioning)

            if args.complex_mode == 'polar':
                masked_in = transforms.cartesian_to_polar(masked_in)
                gt_c_image = transforms.cartesian_to_polar(gt_c_image)

            k_gt = gt_tokens.shape[2]

            # Encode masked image -> indices; align codebooks to GT count
            with torch.no_grad():
                def encode(img_bhw):
                    img = img_bhw.unsqueeze(-1).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)
                    enc_tokens = meditok.encoder(img)           # [B,256,1024]
                    enc_tokens = meditok.quant_proj(enc_tokens)       # [B,256,64]
                    start_tokens = meditok.quantizer.f_to_idx(enc_tokens)  # [B,K_enc,256]
                    if start_tokens.shape[1] < k_gt:
                        raise ValueError(f"start_tokens has {start_tokens.shape[1]} codebooks; need {k_gt}.")
                    start_tokens = start_tokens[:, :k_gt, :]
                    return start_tokens
                if args.model_version >= 2.1:   # v2.1 input = continuous encoder latents (unit-norm chunks), no input tokens
                    cond_latent = torch.stack([latent_target(meditok, masked_in[..., c]) for c in range(2)], dim=1)   # (B, 2, L, D)
                    start_tokens_re = start_tokens_im = None
                else:
                    cond_latent = None
                    start_tokens_re = encode(masked_in[..., 0])
                    start_tokens_im = encode(masked_in[..., 1])
                if batch.get("gt_cached", None) is not None and bool(batch["gt_cached"][0]):   # ground-truth tokens and latents from the per-slice cache
                    gt_tokens = batch["gt_tokens"].to(device, non_blocking=True).long()[:, :, :k_gt]
                    gt_latent = batch["gt_latent"].to(device, non_blocking=True).float() if args.lambda_latent > 0 else None
                elif args.encode_gt:   # one encoder pass per stream gives both the GT tokens and the latent-regression target
                    f_gt = [encode_features(meditok, gt_c_image[..., c]) for c in range(2)]                                # 2 x (B, L, D)
                    gt_tokens = torch.stack([features_to_tokens(meditok, f)[:, :k_gt] for f in f_gt], dim=1)                   # (B, 2, K, L)
                    gt_latent = torch.stack([features_to_latent(f) for f in f_gt], dim=1) if args.lambda_latent > 0 else None   # (B, 2, L, D)
                    del f_gt

            cond_tokens = None if start_tokens_re is None else torch.cat((start_tokens_re.unsqueeze(1), start_tokens_im.unsqueeze(1)), dim=1) # B, 2, K, L

            if args.dataset == 'sense' and args.distill_teacher != 'none':
                # teacher-forced pass of the EMA teacher on its richer input: soft token targets and the teacher's latent
                ema_was_training = ema.training
                ema.eval()
                with torch.no_grad(), torch.cuda.amp.autocast(dtype=ptdtype):
                    out_t = ema(targets=gt_tokens.long(), cond_latent=cond_latent_t, timestep=t.long(), sample_mask_cols=mask_cols_t,
                                target_image=gt_c_image, return_latent=(args.lambda_dlat > 0))
                logits_t = out_t[0].detach()
                latent_t = out_t[2].detach() if len(out_t) > 2 and out_t[2] is not None else None
                del out_t
                if ema_was_training:
                    ema.train()
            with torch.cuda.amp.autocast(dtype=ptdtype):
                logits, *rest = model(
                    targets=gt_tokens.long(),
                    cond_tokens=None if cond_tokens is None else cond_tokens.long(),
                    cond_latent=cond_latent,
                    timestep=t.long(),
                    sample_mask_cols=mask_cols,
                    target_image=gt_c_image,
                    return_latent=(args.lambda_latent > 0),
                ) # logits shape: [B, 2, 1+(K-1), 1+(Li-1), V/K]
            if args.probe_model:
                loss, aux = rest
            else:
                loss = rest[0]
            loss_ce_only_val = loss.item()
            loss_lat_val = 0.0
            if args.lambda_latent > 0:
                # latent regression: L2 between unit-norm codebook chunks of the predicted and GT encoder latents (= 2 - 2 cos)
                loss_lat = ((rest[1].float() - gt_latent) ** 2).sum(-1).mean() * (args.num_codebooks / gt_latent.shape[-1])
                loss_lat_val = loss_lat.item()
                loss = loss + args.lambda_latent * loss_lat
            loss_ce_val = loss_ce_only_val
            loss_img_val = 0.0
            loss_ref_val = 0.0
            loss_kl_val = loss_dlat_val = 0.0
            if args.dataset == 'sense' and args.distill_teacher != 'none':
                w = (~rich).float()                                            # students that got the richer input: ground-truth loss only
                if args.lambda_kl > 0:
                    loss_kl = kl_to_teacher(logits, logits_t, args.distill_temp, w)
                    loss_kl_val = loss_kl.item()
                    loss = loss + args.lambda_kl * loss_kl
                if args.lambda_dlat > 0:
                    d2 = ((rest[1].float() - latent_t.float()) ** 2).sum(-1).mean(dim=(1, 2)) * (args.num_codebooks / latent_t.shape[-1])   # (B,)
                    loss_dlat = (d2 * w).sum() / w.sum().clamp_min(1.0)
                    loss_dlat_val = loss_dlat.item()
                    loss = loss + args.lambda_dlat * loss_dlat
                del logits_t
            if args.dataset == 'sense' and args.lambda_img > 0:
                # image-domain loss through the frozen tokenizer decoder: softmax-weighted codebook vectors (expected
                # embedding) of the teacher-forced logits -> decoder -> downsample to the acquisition grid [-> per-coil DC]
                # -> L1 (+ 1-SSIM) on the magnitude vs the SENSE GT. Gradients reach the GPT logits; the decoder stays frozen.
                nb = min(args.img_loss_batch, logits.shape[0])
                with torch.cuda.amp.autocast(dtype=ptdtype):
                    x_tok = decode_expected(meditok, logits[:nb], args.img_temperature)        # (nb, tok, tok, 2)
                x_crop = tok_downsample(x_tok.float(), args.sense_crop_size)
                if args.img_loss_dc:
                    x_crop = sense_dc_fn_img(x_crop, full_mask[:nb])
                mag = transforms.complex_abs(x_crop)
                gt_mag = transforms.complex_abs(gt_crop[:nb].float())
                loss_img = (mag - gt_mag).abs().mean()
                if args.img_loss == 'l1ssim':
                    import piq
                    loss_img = loss_img + (1 - piq.ssim(mag.clamp(0, 1)[:, None], gt_mag.clamp(0, 1)[:, None], data_range=1.0))
                loss_img_val = loss_img.item()
                loss = loss + args.lambda_img * loss_img
            loss_total_val = loss.item()
            if args.refine_rounds > 0:
                # Teacher-student refinement: the (EMA) teacher drafts a reconstruction of the zero-filled input
                # (pass 1, timestep 0; sampled or greedy), it is decoded -> per-coil DC -> re-tokenised, and the student
                # learns to predict the clean tokens from that draft with timestep 1. Loss = CE(pass 1) + w * CE(refine).
                # The pass-1 graph is backpropagated first (no DDP sync) so both passes never coexist in memory.
                with model.no_sync():
                    scaler.scale(loss).backward()
                logits = logits.detach()   # keep the values for the periodic viz; the graph is already freed
                del rest
                teacher = ema if args.refine_teacher == 'ema' else (model.module._orig_mod if not args.no_compile else model.module)
                teacher_was_training = teacher.training
                teacher.eval()
                draft_mode = args.refine_draft if args.refine_draft != 'mixed' else ('sample' if torch.rand(1).item() < 0.5 else 'greedy')   # mixed: per step
                draft_kw = ({"temperature": 1.0, "top_k": args.vocab_size // args.num_codebooks, "top_p": 1.0} if draft_mode == 'sample'
                            else {"temperature": 1.0, "top_k": 0, "top_p": 1.0})
                nb_ref = args.refine_batch if args.refine_batch > 0 else masked_in.shape[0]
                draft_keep = None
                t_ref = torch.ones_like(t[:nb_ref]).long()
                with torch.no_grad(), torch.cuda.amp.autocast(dtype=ptdtype):
                    x_draft, aux_d = _reconstruct_far(teacher, meditok, masked_in[:nb_ref], sample_mask[:nb_ref], sampling_kwargs=draft_kw, n_steps=1,
                                                      far_mask_mode=args.far_mask_mode, complex_mode=args.complex_mode,
                                                      center_fractions=args.center_fractions, dc_fn=sense_dc_fn_ref, timestep_offset=0,
                                                      latent_decode=(args.commit == 'none'))
                with torch.no_grad():   # draft features in fp32 like the input (bf16 flips ~30% of the near-tie codes)
                    if args.commit != 'none':
                        # progressive commitment: the second view is the expected codebook embedding of the teacher's pass-1 chunk
                        # distributions at the committed positions (a random fraction per sample, chosen by the latent spread or at
                        # random) and zero elsewhere; the timestep encodes the committed fraction
                        draft_latent, conf = expected_embedding_view(aux_d["logits_1d"].float(), commit_codebook)
                        frac = torch.empty(nb_ref, device=device).uniform_(args.commit_keep[0], args.commit_keep[1])
                        draft_keep = commit_keep_mask(conf, frac, args.commit)
                        t_ref = commit_timestep(frac).to(device)
                        draft_tokens = None
                    elif args.model_version >= 2.1:
                        draft_latent = torch.stack([latent_target(meditok, x_draft[..., c].float()) for c in range(2)], dim=1)
                        draft_tokens = None
                    else:
                        draft_latent = None
                        draft_tokens = torch.cat((encode(x_draft[..., 0].float()).unsqueeze(1), encode(x_draft[..., 1].float()).unsqueeze(1)), dim=1)
                    del x_draft, aux_d
                if teacher_was_training:
                    teacher.train()
                with torch.cuda.amp.autocast(dtype=ptdtype):
                    if args.model_version >= 2.1:
                        if args.draft_cond:   # two-view: zero-filled latents + draft latents as the second view
                            _, *rest_ref = model(targets=gt_tokens[:nb_ref].long(), cond_latent=cond_latent[:nb_ref], draft_latent=draft_latent, draft_keep=draft_keep, timestep=t_ref, sample_mask_cols=mask_cols[:nb_ref], target_image=gt_c_image[:nb_ref])
                        else:                 # single view: the draft replaces the zero-filled input
                            _, *rest_ref = model(targets=gt_tokens[:nb_ref].long(), cond_latent=draft_latent, timestep=t_ref, sample_mask_cols=mask_cols[:nb_ref], target_image=gt_c_image[:nb_ref])
                    elif args.draft_cond:   # two-view: zero-filled tokens + draft tokens as the second view
                        _, *rest_ref = model(targets=gt_tokens[:nb_ref].long(), cond_tokens=cond_tokens[:nb_ref].long(), draft_tokens=draft_tokens.long(), timestep=t_ref, sample_mask_cols=mask_cols[:nb_ref], target_image=gt_c_image[:nb_ref])
                    else:                 # single view: the draft replaces the zero-filled input
                        _, *rest_ref = model(targets=gt_tokens[:nb_ref].long(), cond_tokens=draft_tokens.long(), timestep=t_ref, sample_mask_cols=mask_cols[:nb_ref], target_image=gt_c_image[:nb_ref])
                loss_ref = rest_ref[0]
                loss_ref_val = loss_ref.item()
                scaler.scale(args.refine_weight * loss_ref).backward()
                loss_total_val += args.refine_weight * loss_ref_val
                del rest_ref, loss_ref
            else:
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
            if args.ema:
                update_ema(ema, model.module._orig_mod if not args.no_compile else model.module, decay=args.ema_decay)

            # Log loss values:
            running_loss += loss_total_val
            running_ce += loss_ce_val
            running_lat += loss_lat_val
            running_kl += loss_kl_val
            running_dlat += loss_dlat_val
            running_img += loss_img_val
            running_ref += loss_ref_val
            log_steps += 1
            train_steps += 1
            if (idx + 1) % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                lr = scheduler.get_last_lr()[0] if scheduler is not None else args.lr
                extra = f" CE: {running_ce / log_steps:.4f}, Img: {running_img / log_steps:.4f}," if args.lambda_img > 0 else ""
                extra += f" CE1: {running_ce / log_steps:.4f}, CEref: {running_ref / log_steps:.4f}," if args.refine_rounds > 0 else ""
                _m = model.module._orig_mod if not args.no_compile else model.module
                extra += f" t_gate: {float(_m.t_gate):.4f}," if args.model_version >= 2 else ""
                extra += f" m_gate: {float(_m.m_gate):.4f}," if args.model_version >= 2 else ""
                extra += f" d_gate: {float((model.module._orig_mod if not args.no_compile else model.module).draft_gate):.4f}," if args.draft_cond else ""
                extra += f" CE: {running_ce / log_steps:.4f}, Lat: {running_lat / log_steps:.4f}," if args.lambda_latent > 0 else ""
                extra += f" KL: {running_kl / log_steps:.4f}, DLat: {running_dlat / log_steps:.4f}," if args.distill_teacher != 'none' else ""
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f},{extra} "
                            f"Train Steps/Sec: {steps_per_sec:.2f}, LR: {lr}")
                # Reset monitoring variables:
                running_loss = 0
                running_ce = 0.0
                running_img = 0.0
                running_ref = 0.0
                running_lat = 0.0
                running_kl = 0.0
                running_dlat = 0.0
                log_steps = 0
                start_time = time.time()

            if train_steps % args.viz_every == 0 and rank == 0:
                with torch.no_grad():
                    def decode_codes(codes_for_decode):
                        f = meditok.quantizer.idx_to_f(codes_for_decode)          # [B,256,64]
                        f = meditok.post_quant_proj(f)                            # [B,256,1024]
                        dec = meditok.decoder(f)                                  # [B,3,H,W] in [-1,1]
                        dec = dec.clamp_(-1, 1)
                        return dec
                    def decode_complex_codes(codes_for_decode):
                        re_codes = codes_for_decode[:, 0]
                        im_codes = codes_for_decode[:, 1]
                        re_img = decode_codes(re_codes)
                        im_img = decode_codes(im_codes)
                        img = torch.stack([re_img, im_img], dim=-1)
                        return img

                if args.complex_mode == "polar":
                    gt_c_image = transforms.polar_to_cartesian(gt_c_image)
                gt_img = transforms.complex_abs(gt_c_image)

                indices = logits.argmax(dim=-1)
                rec_c_img = decode_complex_codes(indices)
                if args.complex_mode == "polar":
                    rec_c_img = transforms.polar_to_cartesian(rec_c_img)
                rec_c_img = rgb_to_grayscale(rec_c_img)
                if args.dataset == 'sense':
                    rec_c_img = sense_dc_fn(rec_c_img, sample_mask)
                else:
                    rec_c_img = data_consistency(gt_c_image, sample_mask, rec_c_img)
                rec_img = transforms.complex_abs(rec_c_img)
                if args.probe_model:
                    rec_img = aux["image_pred"]
                    rec_img = transforms.complex_abs(rec_img)

                if args.complex_mode == "polar":
                    masked_in = transforms.polar_to_cartesian(masked_in)
                masked_in = transforms.complex_abs(masked_in)

                def norm_0_1(img):
                    return (img - img.min()) / (img.max() - img.min())

                gt_img = norm_0_1(gt_img)
                rec_img = norm_0_1(rec_img)
                masked_in = norm_0_1(masked_in)

                diff_img = (rec_img - gt_img).abs()

                def add_rgb_channels(img):
                    return img.unsqueeze(1).repeat(1,3,1,1)
                
                gt_img = add_rgb_channels(gt_img)[:8]
                rec_img = add_rgb_channels(rec_img)[:8]
                masked_in = add_rgb_channels(masked_in)[:8]
                diff_img = add_rgb_channels(diff_img)[:8]
                sample_mask_img = add_rgb_channels(sample_mask[...,0]).float()[:8]
                if sample_mask_img.shape[-2:] != rec_img.shape[-2:]:   # mask on the acquisition grid, images on the token grid
                    sample_mask_img = torch.nn.functional.interpolate(sample_mask_img, size=rec_img.shape[-2:], mode='nearest')

                save_path = f"{experiment_dir}/viz/epoch{epoch}/step{train_steps:07d}.png"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_pngs(save_path, [sample_mask_img, masked_in, rec_img, gt_img, diff_img])


            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    if not args.no_compile:
                        model_weight = model.module._orig_mod.state_dict()
                    else:
                        model_weight = model.module.state_dict()
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    if not args.no_local_save:
                        # Find old numbered checkpoints to delete (before saving new one)
                        old_checkpoint = None
                        for fname in os.listdir(checkpoint_dir):
                            if re.match(r'^\d{7}\.pt$', fname):
                                old_checkpoint = os.path.join(checkpoint_dir, fname)
                                break
                        
                        # Save new checkpoint
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                        
                        # Delete old checkpoint after successful save
                        if old_checkpoint and os.path.exists(old_checkpoint):
                            os.remove(old_checkpoint)
                            logger.info(f"Deleted old checkpoint: {old_checkpoint}")

                dist.barrier()
        
        # Validation at end of epoch
        if args.val_every > 0 and (epoch + 1) % args.val_every == 0:
            if rank == 0 and val_loader is not None:
                logger.info("=" * 60)
                logger.info(f"Running validation at epoch {epoch + 1}, step {train_steps}...")
                logger.info("=" * 60)
                
                # Get the model to validate (prefer EMA if available)
                eval_model = ema if args.ema else model
                
                # Unwrap DDP and compiled model
                if hasattr(eval_model, 'module'):
                    if hasattr(eval_model.module, '_orig_mod'):
                        eval_model = eval_model.module._orig_mod
                    else:
                        eval_model = eval_model.module
                elif hasattr(eval_model, '_orig_mod'):
                    eval_model = eval_model._orig_mod
                
                # Save training state
                was_training = eval_model.training
                eval_model.eval()
                
                # Create validation args
                val_eval_args = create_validation_args(args, experiment_dir, epoch + 1, train_steps)
                
                try:
                    # Run validation
                    val_metrics = validate(
                        gpt=eval_model,
                        meditok=meditok,
                        loader=val_loader,
                        args=val_eval_args,
                        device=device,
                        logger=logger,
                        out_dir=val_eval_args.output_dir,
                        n_imgs_to_save=args.val_n_scans_to_save if args.val_save_viz else 0,
                        save_volumes=False,
                        sense_ctx=(dict(sens_net=sens_net, mode=args.sens_maps, center_fractions=args.center_fractions,
                                        token_size=args.image_size, resize_mode=args.sense_resize_mode) if args.dataset == 'sense' else None),
                    )
                    
                    # Log summary
                    logger.info("=" * 60)
                    logger.info(f"Validation Results at epoch {epoch + 1}, step {train_steps}:")
                    logger.info(f"  PSNR: {val_metrics['psnr_mean']:.4f}")
                    logger.info(f"  SSIM: {val_metrics['ssim_mean']:.4f}")
                    logger.info(f"  LPIPS: {val_metrics['lpips_mean']:.4f}")
                    logger.info(f"  DISTS: {val_metrics['dists_mean']:.4f}")
                    logger.info("=" * 60)
                    
                    # Track and save best model
                    if val_metrics['psnr_mean'] > best_val_psnr:
                        best_val_psnr = val_metrics['psnr_mean']
                        if not args.no_local_save:
                            if not args.no_compile:
                                model_weight = model.module._orig_mod.state_dict()
                            else:
                                model_weight = model.module.state_dict()
                            checkpoint = {
                                "model": model_weight,
                                "optimizer": optimizer.state_dict(),
                                "steps": train_steps,
                                "epoch": epoch + 1,
                                "args": args,
                                "val_metrics": val_metrics,
                            }
                            if args.ema:
                                checkpoint["ema"] = ema.state_dict()
                            checkpoint_path = f"{checkpoint_dir}/best.pt"
                            torch.save(checkpoint, checkpoint_path)
                            logger.info(f"New best model! PSNR: {val_metrics['psnr_mean']:.4f}")
                            logger.info(f"Saved best checkpoint to {checkpoint_path}")
                    
                finally:
                    # Restore training state
                    if was_training:
                        eval_model.train()
                    
                    # Clear cache
                    torch.cuda.empty_cache()
            
            # All ranks wait here
            dist.barrier()
                
        if epoch == args.epochs - 1:
            if rank == 0:
                if not args.no_compile:
                    model_weight = model.module._orig_mod.state_dict()
                else:
                    model_weight = model.module.state_dict()
                checkpoint = {
                    "model": model_weight,
                    "optimizer": optimizer.state_dict(),
                    "steps": train_steps,
                    "args": args
                }
                if args.ema:
                    checkpoint["ema"] = ema.state_dict()
                if not args.no_local_save:
                    checkpoint_path = f"{experiment_dir}/latest.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")


            dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Experiment & Logging
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--no-local-save", action='store_true', default=False, help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--viz-every", type=int, default=2000)
    
    # Data
    parser.add_argument("--dataset", type=str, default='mri_code')
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--data-split", type=str, default=None, help="path to a .txt file containing filename to test on")
    parser.add_argument("--code-dir", type=str, default=None)
    parser.add_argument("--image-size", type=int, choices=[256, 320, 384, 448, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--global-seed", type=int, default=0)
    
    # MRI-specific
    parser.add_argument("--center-fractions", type=float, default=0.08)
    parser.add_argument("--accelerations", type=int, default=4)
    parser.add_argument("--mask-type", type=str, default="cart_1d",
        choices=["cart_1d", "equispaced", "vds2d", "poisson"],
        help="Initial k-space undersampling mask type for training")
    parser.add_argument("--vds-acs-fraction", type=float, default=0.0,
        help="Side length of the hard ACS center square as a fraction of min(H,W) for VDS2D masks. "
             "E.g. 0.025 gives an 8x8 ACS on a 320x320 image. 0.0 = no hard ACS zone.")
    parser.add_argument("--acquisition-type", type=str, default=None)
    parser.add_argument("--far-mask-mode", type=str, default='cartesian')
    parser.add_argument("--complex-mode", type=str, default='cartesian', choices=['cartesian', 'polar'])
    parser.add_argument("--num-masks", type=int, default=None)
    parser.add_argument("--p-empty", type=float, default=0.5, help="probability of sampling an empty far mask (t=0, no additional k-space lines)")
    parser.add_argument("--radial-weighting", action='store_true')
    parser.add_argument("--end-slice-skip", type=int, default=0, help="number of end slices to skip (0 for knee, 5 for brain)")
    
    # Encoder/Tokenizer
    parser.add_argument("--encoder-ckpt", type=str, required=True)
    parser.add_argument("--encoder-config", type=str, default="weights/meditok/config.json")
    parser.add_argument("--tokenizer-name", type=str, default='unitok_unitok_tokenizer')
    parser.add_argument("--vocab-size", type=int, default=32768, help="vocabulary size of visual tokenizer")
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--encode-gt", action='store_true', help="instead of using precomputed codes, compute them on the fly")
    parser.add_argument("--gt-cache-dir", type=str, default=None, help="sense dataset: directory of per-volume ground-truth token/latent caches (precompute_gt_cache.py); skips the 2-4 tokenizer encoder passes per step on the ground truth")
    parser.add_argument("--vq-embed-path", type=str, default=None)

    # SENSE multicoil mode (--dataset sense; batches from dataset/sense_data.py)
    parser.add_argument("--sens-dirs", type=str, default=None, help="comma-separated dirs with oracle ESPIRiT maps for --data-root")
    parser.add_argument("--val-sens-dirs", type=str, default=None, help="same for --val-data-root (defaults to --sens-dirs)")
    parser.add_argument("--sens-ckpt", type=str, default=None, help="sensitivity net: train_sens_net.py best.pt or a fastMRI VarNet state dict")
    parser.add_argument("--sens-maps", type=str, default="net", choices=["net", "oracle"], help="maps for the zero-filled input (and DC at validation)")
    parser.add_argument("--gt-gauge", type=str, default="oracle", choices=["oracle", "net"], help="GT SENSE image combined with oracle maps, or with the net maps (x oracle support)")
    parser.add_argument("--max-coils", type=int, default=20, help="coil padding; volumes with more coils are skipped")
    parser.add_argument("--sense-crop-size", type=int, default=None, help="acquisition grid (default = --image-size); when smaller than --image-size the SENSE images are resampled to the token grid")
    parser.add_argument("--sense-resize-mode", type=str, default="bicubic", help="upsampling mode acquisition grid -> token grid (the res512 SENSE tokenizer used bicubic)")
    parser.add_argument("--lambda-img", type=float, default=0.0, help="weight of the image-domain loss (expected-embedding decode -> [DC] -> L1/SSIM on |x| vs SENSE GT); 0 = CE only")
    parser.add_argument("--img-loss", type=str, default="l1", choices=["l1", "l1ssim"])
    parser.add_argument("--img-loss-dc", action="store_true", help="apply per-coil DC before the image loss")
    parser.add_argument("--img-loss-batch", type=int, default=2, help="samples per step used for the image loss (decoder memory)")
    parser.add_argument("--img-temperature", type=float, default=1.0, help="softmax temperature of the expected embedding")
    parser.add_argument("--refine-rounds", type=int, default=0, help="teacher-student refinement: >0 adds a refinement pass (timestep 1) on the DC'd draft of the (EMA) teacher; validation runs this many refinement rounds")
    parser.add_argument("--refine-draft", type=str, default="sample", choices=["sample", "greedy", "mixed"], help="how the teacher draws the draft tokens (mixed: greedy or sampled, chosen per step)")
    parser.add_argument("--refine-teacher", type=str, default="ema", choices=["ema", "self"])
    parser.add_argument("--refine-weight", type=float, default=1.0, help="weight of the refinement CE")
    parser.add_argument("--refine-batch", type=int, default=0, help="samples per step used for the refinement pass (0 = full batch); lower it to save memory")
    parser.add_argument("--latent-dim", type=int, default=0, help=">0: add a latent-regression head (tokenizer latent width, 64 for MedITok)")
    parser.add_argument("--draft-cond", action="store_true", help="two-view refinement: the refine pass sees the zero-filled tokens plus the DC'd draft's tokens as a second view (needs --refine-rounds)")
    parser.add_argument("--commit", type=str, default="none", choices=["none", "conf", "random"], help="progressive commitment fine-tune (with --refine-rounds 1 --draft-cond, model v2.1): the refine pass sees the expected-embedding view of the teacher's pass 1 at a committed fraction of positions, chosen by confidence (latent spread) or at random")
    parser.add_argument("--commit-keep", type=float, nargs=2, default=[0.3, 0.9], help="range of the committed fraction drawn per sample in training")
    parser.add_argument("--commit-schedule", type=float, nargs="*", default=[0.55, 0.8], help="validation: committed fractions of the passes after pass 1")
    parser.add_argument("--head-interleaved", type=int, default=1, choices=[0, 1], help="v2.1 head: 1 = interleaved Re/Im chain, 0 = parallel asymmetric head")
    parser.add_argument("--model-version", type=float, default=1, choices=[1, 2, 2.1], help="GPT variant: 1 = legacy; 2 = + timestep & sampling-mask conditioning; 2.1 = v2 + continuous input + interleaved Re/Im head (see gpt_kar.ModelArgs)")
    parser.add_argument("--lambda-latent", type=float, default=0.0, help="weight of the latent regression loss (L2 on unit-norm codebook chunks, mean per chunk)")
    parser.add_argument("--latent-head-type", type=str, default="linear", choices=["linear", "mlp", "fused"], help="latent head: linear readout; + residual MLP on the hidden state; + residual MLP on [hidden state, token head's chain state, codebook latents of the token head's greedy codes]")
    parser.add_argument("--latent-mlp-width", type=int, default=0, help="hidden width of the latent MLP (0 = 2 x model dim)")
    parser.add_argument("--distill-teacher", type=str, default="none", choices=["none", "ema"], help="privileged-teacher distillation: the EMA teacher sees the student's lines plus extra random columns (acceleration from --distill-acc) and the student is trained toward its teacher-forced token distribution (KL) and its latent (L2) next to the ground-truth losses")
    parser.add_argument("--distill-acc", type=float, nargs=2, default=[2.0, 4.0], help="teacher acceleration range, uniform per sample")
    parser.add_argument("--lambda-kl", type=float, default=1.0, help="weight of KL(teacher || student) on the token logits")
    parser.add_argument("--lambda-dlat", type=float, default=1.0, help="weight of the L2 between the student's and the teacher's latent (needs the latent head)")
    parser.add_argument("--distill-temp", type=float, default=1.0, help="softmax temperature of the distillation KL")
    parser.add_argument("--p-mixed-acc", type=float, default=0.25, help="probability per sample that the student itself gets the teacher's richer input (ground-truth loss only), so the EMA teacher learns those inputs")
    parser.add_argument("--val-rec-mode", type=str, default="tokens", choices=["tokens", "latent"], help="which decoding path is the main validation output; the other is scored as alt_rec")
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    
    # Model Architecture
    parser.add_argument("--gpt-model", type=str, choices=list(KAR_GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="t2i")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--load-weights-only", action='store_true', help="only load ckpt weights, start epoch is set to 0")
    parser.add_argument("--num-classes", type=int, default=6, help="number of potential masks")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--cond-token-num", type=int, default=0)
    parser.add_argument("--num-output-layer", type=int, default=1)
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--no-compile", action='store_true')
    
    # Training
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 parameter for the Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="beta2 parameter for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--schedule", type=str, default='linear')
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--overfit-n-samples", type=int, default=0)

    # probe model
    parser.add_argument("--probe-model", action='store_true', help="wrap model with learnable kspace query tokens for active sampling")
    parser.add_argument("--backbone-ckpt", type=str, default=None, help="path to checkpoint for backbone model of probe model")
    parser.add_argument("--freeze-backbone", action='store_true', help="only train the active sampling layers")

    # Validation arguments
    parser.add_argument("--val-every", type=int, default=0, help="Run validation every N epochs (0 to disable)")
    parser.add_argument("--val-data-root", type=str, default=None, help="Path to validation h5 files")
    parser.add_argument("--val-data-split", type=str, default=None, help="path to a .txt file containing filename to test on")
    parser.add_argument("--val-code-dir", type=str, default=None, help="Path to validation tokenized codes")
    parser.add_argument("--val-num-samples", type=int, default=None, help="Limit validation set size for faster runs")
    parser.add_argument("--val-batch-size", type=int, default=16, help="Validation batch size")
    parser.add_argument("--val-n-steps", type=int, default=None, help="Number of generation steps for validation (None=single-step)")
    parser.add_argument("--val-save-viz", action='store_true', help="Save visualizations during validation")
    parser.add_argument("--val-n-scans-to-save", type=int, default=0, help="Number of validation scan volumes to save")
    parser.add_argument("--val-active-sampling", action='store_true', help="")
    parser.add_argument("--val-active-sampling-fn", type=str, default=None)


    args = parser.parse_args()

    if args.dataset == 'sense':
        assert args.encode_gt, "--dataset sense needs --encode-gt (GT SENSE tokens are computed on the fly)"
        assert args.mask_type in ("cart_1d", "equispaced"), "--dataset sense needs a 1D Cartesian mask"
        assert args.sens_maps == "oracle" or args.sens_ckpt, "--sens-ckpt is required with --sens-maps net"
        assert args.sens_dirs, "--sens-dirs is required with --dataset sense"
        assert not args.probe_model and not args.val_active_sampling, "probe/active sampling are not supported with --dataset sense"
        args.return_oracle_maps = (args.sens_maps == 'oracle')
        args.sense_crop_size = args.sense_crop_size or args.image_size
        assert args.lambda_latent == 0 or args.latent_dim > 0, "--lambda-latent needs --latent-dim"
        assert args.latent_head_type == 'linear' or args.latent_dim > 0, "--latent-head-type mlp/fused needs --latent-dim"
        if args.distill_teacher != 'none':
            assert args.ema, "--distill-teacher ema needs --ema"
            assert args.model_version >= 2.1, "privileged-teacher distillation is implemented for model v2.1 (continuous input)"
            assert args.lambda_dlat == 0 or args.lambda_latent > 0, "--lambda-dlat needs the latent head (--latent-dim, --lambda-latent > 0)"
            assert args.refine_rounds == 0, "distillation and refinement are not combined"
            assert args.distill_acc[0] <= args.distill_acc[1] and args.distill_acc[0] >= 1.0, "--distill-acc lo hi with 1 <= lo <= hi"
        if args.refine_rounds > 0:
            assert args.p_empty >= 1.0, "--refine-rounds needs --p-empty 1.0 (timestep 0 = zero-filled input, timesteps >= 1 = refinement)"
            assert args.refine_teacher != 'ema' or args.ema, "--refine-teacher ema needs --ema"
        assert not args.draft_cond or args.refine_rounds > 0, "--draft-cond needs --refine-rounds"
        if args.commit != "none":
            assert args.refine_rounds > 0 and args.draft_cond and args.model_version >= 2.1, "--commit needs --refine-rounds 1 --draft-cond and model v2.1"
            assert args.distill_teacher == "none" and args.lambda_img == 0, "--commit is not combined with distillation or the image loss"
        assert args.refine_rounds == 0 or args.model_version >= 2, "--refine-rounds needs --model-version 2 (v1 cannot see the timestep)"
        assert args.model_version < 2 or args.mask_type in ('cart_1d', 'equispaced'), "model v2 (mask conditioning) needs 1D Cartesian masks"
        assert args.sense_crop_size <= args.image_size, "--sense-crop-size must not exceed --image-size"
        assert args.sense_crop_size == args.image_size or (args.val_n_steps or 1) == 1, "multi-step generation is not supported on a resampled token grid"

    if args.num_masks is not None:
        assert args.num_classes > args.num_masks, \
            "num_classes must be strictly > num_masks (t=0 is reserved for the empty far mask, t=1..num_masks for radial levels)"
    if args.data_split in ["", " ", "none"]:
        args.data_split=None
    if args.val_data_split in ["", " ", "none"]:
        args.val_data_split=None
    main(args)
