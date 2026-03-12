
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
from autoregressive.models.gpt_kar import KAR_GPT_models
from autoregressive.sample.reconstruct_far import validate
from autoregressive.utils.mask_utils import (
    make_centered_radius_grid,
    radial_circle_mask,
    radial_circle_radii,
    sample_circle_radius_idx,
    sample_circle_radius_idx_weighted
)
from models.meditok import MedITok, DotDict
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
        model.load_state_dict(checkpoint["model"], strict=True)
        if args.ema:
            ema.load_state_dict(checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"], strict=False)
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except:
            logger.info("Optimizer state loading failed, continuing without loading optimizer state...")
        if args.load_weights_only:
            train_steps, start_epoch = 0, 0
        else:
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
    start_time = time.time()

    if args.schedule == 'cosine':
        total_train_steps = math.ceil(args.epochs * len(dataset) / args.global_batch_size)
        scheduler = get_cosine_scheduler(optimizer, total_train_steps, base_lr=args.lr)
    else:
        scheduler = None

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for idx, batch in enumerate(loader):
            gt_c_image = batch["gt_c_image"].to(device, non_blocking=True)   # [B,H,W,2],
            masked_c_image = batch["masked_c_image"].to(device, non_blocking=True)   # [B,H,W,2]
            gt_tokens = batch["gt_tokens"].to(device, non_blocking=True)   # saved as [B,2,K,256]
            sample_mask = batch["mask"].to(device, non_blocking=True) 
            
            masked_in, far_sample_mask, t = sample_far_mask(args.far_mask_mode, gt_c_image, epoch * len(loader) + idx, 20 * len(loader), num_masks=args.num_masks, p_empty=args.p_empty)
            t = t.repeat(masked_in.shape[0])

            masked_in = data_consistency(gt_c_image, sample_mask, masked_in)
            sample_mask = far_sample_mask.bool() | sample_mask.bool()
            # masked_in = masked_c_image
            # sample_mask = batch["mask"].to(device, non_blocking=True)

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
                start_tokens_re = encode(masked_in[..., 0])
                start_tokens_im = encode(masked_in[..., 1])
                if args.encode_gt:
                    gt_tokens_re = encode(gt_c_image[..., 0])
                    gt_tokens_im = encode(gt_c_image[...,1])
                    gt_tokens = torch.cat((gt_tokens_re.unsqueeze(1), gt_tokens_im.unsqueeze(1)), dim=1)

            cond_tokens = torch.cat((start_tokens_re.unsqueeze(1), start_tokens_im.unsqueeze(1)), dim=1) # B, 2, K, L

            with torch.cuda.amp.autocast(dtype=ptdtype):
                logits, *rest = model(
                    targets=gt_tokens.long(),
                    cond_tokens=cond_tokens.long(),
                    timestep=t.long(),
                    target_image=gt_c_image,
                ) # logits shape: [B, 2, 1+(K-1), 1+(Li-1), V/K]
            loss = rest[0]
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
                update_ema(ema, model.module._orig_mod if not args.no_compile else model.module)

            # Log loss values:
            running_loss += loss.item()
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
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, "
                            f"Train Steps/Sec: {steps_per_sec:.2f}, LR: {lr}")
                # Reset monitoring variables:
                running_loss = 0
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
                rec_c_img = data_consistency(gt_c_image, sample_mask, rec_c_img)
                rec_img = transforms.complex_abs(rec_c_img)
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
                        save_visualizations=args.val_save_viz,
                        save_volumes=False,
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
    parser.add_argument("--vq-embed-path", type=str, default=None)
    
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

    if args.num_masks is not None:
        assert args.num_classes > args.num_masks, \
            "num_classes must be strictly > num_masks (t=0 is reserved for the empty far mask, t=1..num_masks for radial levels)"
    if args.data_split in ["", " ", "none"]:
        args.data_split=None
    if args.val_data_split in ["", " ", "none"]:
        args.val_data_split=None
    main(args)
