
import os, sys, argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torchvision.utils as vutils
from PIL import Image
from skimage.restoration import denoise_nl_means, estimate_sigma
import h5py

# ensure eval/generation is on path
gen_root = Path(__file__).resolve().parents[2]
if str(gen_root) not in sys.path:
    sys.path.append(str(gen_root))

from autoregressive.models.gpt_kar import KAR_GPT_models as GPT_models
from autoregressive.models.generate import reconstruct_far, active_sampling, active_sampling_patch_logits, data_consistency, image_entropy_to_freq_entropy, active_sampling_entropy_gradient
from autoregressive.utils.metrics import compute_reconstruction_metrics
from models.meditok import MedITok, DotDict
from dataset.build import build_dataset  # expects args.dataset == 'mri_code'
from dataset.mri_code import inverse_radial_weighting_transform_image, radial_weighting_transform_image
from data import transforms
from utils.logger import create_logger, log_args


def _to_chw(x: torch.Tensor) -> torch.Tensor:
    """Convert (H, W), (H, W, 1), or (H, W, 2) to (1, H, W)"""
    if x.ndim == 3:
        if x.shape[-1] == 2:
            x = transforms.complex_abs(x)
        elif x.shape[-1] == 1:
            x = x.squeeze(-1)
    if x.ndim == 2:
        return x.unsqueeze(0).float()
    raise ValueError(f"Expected (H, W), (H, W, 1), or (H, W, 2), got {tuple(x.shape)}")


def _norm01_per_row(x: torch.Tensor) -> torch.Tensor:
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + 1e-8)


def _apply_colormap(x: torch.Tensor, cmap_name: str = 'viridis', use_log: bool = False) -> torch.Tensor:
    """Apply colormap to grayscale tensor (1, H, W) -> (3, H, W)"""
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(cmap_name)
    x_np = x.squeeze(0).cpu().numpy()
    if use_log:
        x_np = np.log1p(x_np)
        x_np = (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-8)
    colored = cmap(x_np)[:, :, :3]
    return torch.from_numpy(colored).permute(2, 0, 1).float().to(x.device)


def save_entropy_maps(
    *,
    out_dir: Path,
    name: str,
    i: int,
    sample_masks: list,          # List[Tensor(B, H, W)] or List[Tensor(B, H, W, 1)] - mask at each sampling step
    sample_progress: list,        # List[Tensor(B, H, W, 2)] - complex reconstruction at each step
    entropy_maps: list,           # List[Tensor(B, H, W, 2)] - complex entropy/uncertainty map at each step
    freq_maps: list,              # List[Tensor(B, H, W)] - pre-computed freq entropy (patch-space FFT)
    reconstructed_image: torch.Tensor,  # (B, H, W) - final reconstruction magnitude
    target_image: torch.Tensor,         # (B, H, W) - target magnitude
    scan_buf: dict,
    args,
) -> None:
    if entropy_maps is None or len(scan_buf) >= args.n_scans_to_save:
        return

    from autoregressive.sample.adaptive_sampling import image_entropy_to_freq_entropy

    device = reconstructed_image.device

    to_chw = _to_chw
    norm01_per_row = _norm01_per_row
    apply_colormap = _apply_colormap

    T = min(len(sample_masks), len(sample_progress), len(entropy_maps))
    if T == 0:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    col0_rows = []
    col1_rows = []
    col2_rows = []
    col3_rows = []
    col4_rows = []

    prev_mask = None
    prev_prog = None
    
    for t in range(T):
        m_t = to_chw(sample_masks[t][i].to(device))       # (B, H, W) -> (1, H, W)
        
        # Entropy: complex (H, W, 2) -> image entropy (1, H, W) and freq entropy (1, H, W)
        entropy_complex_t = entropy_maps[t][i].unsqueeze(0).to(device)  # (1, H, W, 2)

        # Use pre-computed patch-space freq entropy when available (avoids the
        # bandwidth-mismatch that occurs when FFT-ing the bilinear-upsampled map).
        # Fall back to recomputing from entropy_complex if freq_maps not provided.
        if freq_maps is not None and t < len(freq_maps):
            freq_entropy_t = freq_maps[t][i].unsqueeze(0).to(device)  # (1, H, W)
        else:
            freq_entropy_t = image_entropy_to_freq_entropy(
                entropy_complex_t,
                zero_mean=getattr(args, 'les_zero_mean', False),
                radial_weight_alpha=getattr(args, 'les_radial_weight_alpha', 0.0),
            ).squeeze(0).unsqueeze(0)  # (1, H, W)

        # Mask out already measured k-space positions using PREVIOUS mask
        # entropy_maps[t] was computed with sample_masks[t-1], so mask with that
        prev_mask_idx = max(0, t - 1)  # For t=0, use sample_masks[0] (initial mask)
        mask_for_entropy = sample_masks[prev_mask_idx][i].to(device)  # (H, W) or (H, W, C)
        if mask_for_entropy.ndim == 3:
            mask_for_entropy = mask_for_entropy[..., 0]  # Take first channel (k-space mask)
        # Multiply by (1 - mask) to zero out measured positions
        freq_entropy_t = freq_entropy_t * (1.0 - mask_for_entropy.unsqueeze(0).int())  # (1, H, W)
        
        image_entropy_t = to_chw(entropy_maps[t][i].to(device))  # (H, W, 2) -> (1, H, W) via complex_abs
        
        p_t = to_chw(sample_progress[t][i].to(device))    # (B, H, W, 2) -> (1, H, W)

        if t == 0:
            dm = m_t
            dp = p_t
        else:
            dm = (m_t - prev_mask).clamp(min=0.0, max=1.0).to(device)
            dp = torch.abs(p_t - prev_prog).to(device)

        col0_rows.append(dm)
        col1_rows.append(freq_entropy_t)
        col2_rows.append(image_entropy_t)
        col3_rows.append(p_t)
        col4_rows.append(dp)

        prev_mask = m_t
        prev_prog = p_t

    rec_abs = to_chw(reconstructed_image[i])  # (B, H, W) -> (1, H, W)
    tgt_abs = to_chw(target_image[i])         # (B, H, W) -> (1, H, W)
    l1 = torch.abs(rec_abs - tgt_abs)

    col0 = torch.stack(col0_rows + [prev_mask.to(device)], dim=0)
    col1 = torch.stack(col1_rows + [freq_entropy_t], dim=0)
    col2 = torch.stack(col2_rows + [rec_abs.to(device)], dim=0)
    col3 = torch.stack(col3_rows + [tgt_abs.to(device)], dim=0)
    col4 = torch.stack(col4_rows + [l1.to(device)], dim=0)

    col1 = norm01_per_row(col1)
    col2 = norm01_per_row(col2)
    col3 = norm01_per_row(col3)
    col4 = norm01_per_row(col4)
    
    # Apply heatmap colormap to entropy columns (hot: dark=low entropy, bright=high entropy)
    # Use log transform for frequency entropy to handle large dynamic range
    col1_colored = torch.stack([apply_colormap(col1[i], 'hot', use_log=True) for i in range(col1.shape[0])], dim=0)
    col2_colored = torch.stack([apply_colormap(col2[i], 'hot', use_log=False) for i in range(col2.shape[0])], dim=0)
    
    # Convert grayscale columns to RGB (3 channels) to match colored columns
    col0 = col0.repeat(1, 3, 1, 1)
    col3 = col3.repeat(1, 3, 1, 1)
    col4 = col4.repeat(1, 3, 1, 1)

    out_path = out_dir / f"{name}_entropy.png"
    save_pngs(
        out_path, 
        [col0, col1_colored, col2_colored, col3, col4],
        labels=["Δ Mask", "Freq Entropy (log)", "Image Entropy / Rec", "Progress / Target", "Δ Progress / L1"]
    )


def save_gradient_map(
    *,
    out_dir: Path,
    name: str,
    i: int,
    grad_magnitudes: list,              # List[Tensor(B, H, W)] - gradient magnitude at each GEO step
    sample_masks: list,                 # List[Tensor(B, H, W, C)] - mask evolution (initial + after each step)
    reconstructed_image: torch.Tensor,  # (B, H, W) - final reconstruction magnitude
    target_image: torch.Tensor,         # (B, H, W) - target magnitude
    scan_buf: dict,
    args,
) -> None:
    if grad_magnitudes is None or len(grad_magnitudes) == 0 or len(scan_buf) >= args.n_scans_to_save:
        return

    device = reconstructed_image.device
    T = len(grad_magnitudes)
    H, W = reconstructed_image.shape[-2:]
    out_dir.mkdir(parents=True, exist_ok=True)

    dm_rows   = []  # Δ Mask per step
    grad_rows = []  # Raw gradient magnitude per step (1, H, W)

    for t in range(T):
        # Δ mask = mask after step t+1 minus mask before step t
        # sample_masks[0] is the initial mask; sample_masks[t+1] is after step t
        if sample_masks is not None and len(sample_masks) > t + 1:
            m_before = _to_chw(sample_masks[t][i].to(device))      # (1, H, W)
            m_after  = _to_chw(sample_masks[t + 1][i].to(device))  # (1, H, W)
            dm = (m_after - m_before).clamp(0.0, 1.0)
        else:
            dm = torch.zeros(1, H, W, device=device)
        dm_rows.append(dm)

        g_t = grad_magnitudes[t][i].to(device).unsqueeze(0).float()  # (1, H, W)
        grad_rows.append(g_t)

    rec_abs = _to_chw(reconstructed_image[i].to(device))  # (1, H, W)
    tgt_abs = _to_chw(target_image[i].to(device))         # (1, H, W)
    l1      = (rec_abs - tgt_abs).abs()

    # Normalize gradients per-row then apply viridis with log and linear scales
    grad_stack = _norm01_per_row(torch.stack(grad_rows, dim=0))  # (T, 1, H, W)
    grad_log    = torch.stack([_apply_colormap(grad_stack[t], 'hot', use_log=True)  for t in range(T)], dim=0)  # (T, 3, H, W)
    grad_linear = torch.stack([_apply_colormap(grad_stack[t], 'hot', use_log=False) for t in range(T)], dim=0)  # (T, 3, H, W)

    # Build final summary row: final mask | reconstruction | GT | L1
    final_mask = _to_chw(sample_masks[-1][i].to(device)) if sample_masks else rec_abs.clone()
    summary_row = [
        _norm01_per_row(final_mask).repeat(3, 1, 1).unsqueeze(0),  # (1, 3, H, W)
        _norm01_per_row(rec_abs).repeat(3, 1, 1).unsqueeze(0),     # (1, 3, H, W)
        _norm01_per_row(tgt_abs).repeat(3, 1, 1).unsqueeze(0),     # (1, 3, H, W)
        _norm01_per_row(l1).repeat(3, 1, 1).unsqueeze(0),          # (1, 3, H, W)
    ]

    # Each column: T step-rows + 1 summary row = (T+1, 3, H, W)
    dm_rgb = torch.stack(dm_rows, dim=0).repeat(1, 3, 1, 1)  # (T, 3, H, W)

    col0 = torch.cat([dm_rgb,    summary_row[0]], dim=0)  # Δ Mask | final mask
    col1 = torch.cat([grad_log,  summary_row[1]], dim=0)  # Grad (log)   | rec
    col2 = torch.cat([grad_linear, summary_row[2]], dim=0)  # Grad (linear)| GT
    col3 = torch.cat([grad_linear.clone(), summary_row[3]], dim=0)  # Grad (linear)| L1

    out_path = out_dir / f"{name}_gradient.png"
    save_pngs(
        out_path,
        [col0, col1, col2, col3],
        labels=["Δ Mask / Final mask", "Grad (log) / Rec", "Grad (linear) / GT", "Grad (linear) / L1"],
    )


def save_slice_viz(
    *,
    args,
    out_dir: Path,
    i: int,
    fname,
    slice_idx,
    rec_vis: torch.Tensor,
    gt_vis: torch.Tensor,
    masked_in_vis: torch.Tensor,
    sample_mask: torch.Tensor,
    scan_buf: dict,
    gen_prog,
    gt_c_image: torch.Tensor,
) -> str:
    name = "sample"
    if fname is not None:
        name = Path(fname[i]).stem
    if slice_idx is not None:
        name = f"{name}_slice{int(slice_idx[i])}"

    out_path = out_dir / f"{name}.png"
    os.makedirs(out_dir, exist_ok=True)

    rec_i = rec_vis[i].unsqueeze(0)
    gt_i = gt_vis[i].unsqueeze(0)
    delta_i = (rec_i - gt_i).abs()
    delta_i = (delta_i - delta_i.min()) / (delta_i.max() - delta_i.min())
    masked_in_i = masked_in_vis[i].unsqueeze(0)
    sample_mask_i = sample_mask[i, ..., 0].unsqueeze(0)

    if len(scan_buf) < args.n_scans_to_save:
        save_pngs(
            out_path, 
            [sample_mask_i, masked_in_i, rec_i, gt_i, delta_i],
            labels=["Mask", "Masked Input", "Reconstruction", "Ground Truth", "|Rec - GT|"]
        )

        if args.viz_gen_steps and gen_prog is not None and slice_idx is not None:
            s = slice_idx[i]
            if s > 15 and s < 27:
                _gen_step = []
                _gen_mask = []
                _gen_step_ti = []
                for t, (gen_step, gen_mask) in enumerate(gen_prog):
                    _gen_step.append(gen_step[i])
                    _gen_mask.append(gen_mask[i][...,0])
                    _gen_step_ti.append(transforms.ifft2(transforms.fft2(gen_step[i]) * gen_mask[i]))
                _gen_step = torch.stack(_gen_step, dim=0) # N, H, W,2
                _gen_mask = torch.stack(_gen_mask, dim=0)
                _gen_step_ti = torch.stack(_gen_step_ti, dim=0) 

                _gen_step_abs = transforms.complex_abs(_gen_step).unsqueeze(1)
                _gen_step_ti = transforms.complex_abs(_gen_step_ti).unsqueeze(1)
                _first_gen_step_abs = _gen_step_abs[1].clone().unsqueeze(0).repeat(_gen_step_abs.shape[0], 1, 1, 1)
                _gen_mask = _gen_mask.unsqueeze(1)

                delta_t_t0 = torch.abs(_gen_step_abs - _first_gen_step_abs)
                delta_t_t0 = norm_0_1(delta_t_t0)

                delta_t_tprev = torch.abs(_gen_step_ti[1:] - _gen_step_ti[:-1])
                delta_t_tprev = torch.cat([torch.zeros_like(delta_t_tprev)[:1], delta_t_tprev], dim=0)
                delta_t_tprev = norm_0_1(delta_t_tprev)

                gt_i= gt_i.repeat(_gen_step.shape[0], 1, 1, 1)
                delta_gt_t = torch.abs(gt_i - _gen_step_abs)
                delta_gt_t = norm_0_1(delta_gt_t)

                gt_i_t = transforms.ifft2(transforms.fft2(gt_c_image[i].unsqueeze(0)) * _gen_mask[:,0].unsqueeze(-1))
                gt_i_t = transforms.complex_abs(gt_i_t).unsqueeze(1)
                delta_t_gt_it = torch.abs(gt_i_t - _gen_step_ti)
                delta_t_gt_it = norm_0_1(delta_t_gt_it)

                out_path = out_dir / f"{name}_gen_steps.png"
                save_pngs(
                    out_path, 
                    [
                        _gen_mask,
                        norm_0_1(_gen_step_abs),
                        norm_0_1(_gen_step_ti),
                        norm_0_1(gt_i_t),
                        norm_0_1(gt_i),
                        norm_0_1(delta_t_tprev),
                        norm_0_1(delta_gt_t),
                    ],
                    labels=[
                        "Mask",
                        "i_t (before HF mask)",
                        "i_t (masked)",
                        "GT_t (masked)",
                        "GT",
                        "|i_t - i_{t-1}|",
                        "|i_t - GT|",
                    ]
                )


    return name


def apply_nl_means_filter(tensor_volume, patch_size=5, patch_distance=6, h_multiplier=0.8):
    """
    Applies Non-Local Means Denoising.
    
    Args:
        patch_size: Size of the patches to compare (5x5 or 7x7 is standard for MRI).
        patch_distance: How far to search for similar patches (radius).
        h_multiplier: Controls smoothing strength. 
                      Higher (1.0+) = Smoother but blurrier. 
                      Lower (0.6-0.8) = Retains more texture.
    """
    vol_np = tensor_volume.cpu().numpy()
    denoised_slices = []
    
    # print(f"Applying NL-Means (Patch={patch_size}, Dist={patch_distance}, h_mult={h_multiplier})...")
    # print(f"tensor vol shape:{tensor_volume.shape}")

    for i in tqdm(range(vol_np.shape[0])):
        slice_img = vol_np[i]
        
        # 1. Estimate the noise standard deviation from the image itself
        # This prevents "blinding" the filter or guessing wrong values.
        sigma_est = np.mean(estimate_sigma(slice_img, channel_axis=None))
        
        # 2. Apply NL-Means
        # h: Cut-off distance (filtering strength). usually 0.6 * sigma_est to 1.0 * sigma_est
        h_val = h_multiplier * sigma_est
        
        denoised = denoise_nl_means(
            slice_img,
            h=h_val, 
            sigma=sigma_est,
            fast_mode=True,  # Speeds it up significantly
            patch_size=patch_size,
            patch_distance=patch_distance,
            channel_axis=None
        )
        denoised_slices.append(denoised)
        
    denoised_vol = np.stack(denoised_slices)
    return torch.from_numpy(denoised_vol)



def rgb_to_grayscale(img):
    # return img[:,0] * 0.2126 + img[:,1] * 0.7152 + img[:,2] * 0.0722
    return (img[:,0] + img[:,1] + img[:,2]) / 3

def encode(tokenizer, img_bhw):
    img = img_bhw.unsqueeze(-1).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)
    enc_tokens = tokenizer.encoder(img)           # [B,256,1024]
    enc_tokens = tokenizer.quant_proj(enc_tokens)       # [B,256,64]
    start_tokens = tokenizer.quantizer.f_to_idx(enc_tokens)  # [B,K_enc,256]
    return start_tokens

def decode_codes(tokenizer, codes_for_decode):
    f = tokenizer.quantizer.idx_to_f(codes_for_decode)          # [B,256,64]
    f = tokenizer.post_quant_proj(f)                            # [B,256,1024]
    dec = tokenizer.decoder(f)                                  # [B,3,H,W] in [-1,1]
    dec = dec.clamp_(-1, 1)
    return dec

def decode_complex_codes(tokenizer, codes_for_decode):
    re_codes = codes_for_decode[:, 0]
    im_codes = codes_for_decode[:, 1]
    re_img = decode_codes(tokenizer, re_codes)
    im_img = decode_codes(tokenizer, im_codes)
    img = torch.stack([re_img, im_img], dim=-1)
    return img

def enc_dec_image(tokenizer, img_bhwc, sample_mask=None):
    print("enc-dec image...")
    tokens_re = encode(tokenizer, img_bhwc[..., 0]) # B,K,L
    tokens_im = encode(tokenizer, img_bhwc[..., 1]) # B,K,L
    tokens = torch.cat((tokens_re.unsqueeze(1), tokens_im.unsqueeze(1)), dim=1) # B,2,K,L

    dec_img = decode_complex_codes(tokenizer, tokens)
    dec_img = rgb_to_grayscale(dec_img) # B,H,W,2

    if sample_mask is not None:
        dec_img = data_consistency(img_bhwc, dec_img, sample_mask)

    return dec_img


def calc_budget(resolution, center_fractions, accelerations):
    total_budget = resolution / accelerations
    cf_budget = int(resolution * center_fractions)  # round down
    acc_budget = total_budget - cf_budget

    return int(np.ceil(acc_budget))  # round up

def norm_0_1(x):
    return (x-x.min()) / (x.max() - x.min())

def load_meditok(config_path, ckpt_path, device):
    with open(config_path, "r") as f:
        enc_cfg = json.load(f)
    vt = MedITok(DotDict(enc_cfg)).to(device).eval()
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    vt.load_state_dict(state, strict=True)
    return vt

def load_gpt(args, device, dtype):
    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=(args.image_size // args.downsample_size) ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        cond_token_num=args.cond_token_num,
        model_type=args.gpt_type,
        num_codebooks=args.num_codebooks,
        n_output_layer=args.num_output_layer,
        vq_embed_path=args.vq_embed_path,
    )
    model = model.to(device, dtype=dtype)
    if args.gpt_ckpt is not None:
        ckpt = torch.load(args.gpt_ckpt, map_location="cpu")
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=True)
        elif "module" in ckpt:
            model.load_state_dict(ckpt["module"], strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)
        print(f"Checkpoint loaded successfully: {args.gpt_ckpt}")
    model.eval()
    if args.compile:
        print("compiling GPT...")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    return model

# def save_pngs(out_path: Path, images_chw: list[torch.Tensor]):
#     """
#     Save a side-by-side grid of the provided images.
#     Expects each item to be CHW floats in [0,1].
#     """
#     panel = torch.stack(images_chw, dim=0)
#     grid = vutils.make_grid(panel, nrow=len(images_chw), padding=2)
#     grid_img = (grid.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
#     Image.fromarray(grid_img).save(out_path)
def save_pngs(out_path: Path, images_chw: list[torch.Tensor], labels: list[str] = None):
    """
    Save a grid of the provided images with optional column labels.

    - If inputs are CHW: one row with len(images_chw) columns.
    - If inputs are BCHW: rows are batch items, columns are entries in images_chw.
    - labels: Optional list of strings to label each column.
    """
    if len(images_chw) == 0:
        raise ValueError("images_chw must be non-empty")

    ndims = {x.ndim for x in images_chw}
    if ndims == {3}:
        panel = torch.stack(images_chw, dim=0)
        nrow = len(images_chw)
    elif ndims == {4}:
        batch_sizes = {x.shape[0] for x in images_chw}
        if len(batch_sizes) != 1:
            raise ValueError(f"All BCHW tensors must have the same batch size, got {sorted(batch_sizes)}")
        bsz = next(iter(batch_sizes))

        panel_list = []
        for b in range(bsz):
            for x in images_chw:
                panel_list.append(x[b])

        panel = torch.stack(panel_list, dim=0)
        nrow = len(images_chw)
    else:
        raise ValueError(f"Expected all tensors to be CHW (3D) or all to be BCHW (4D), got ndims={sorted(ndims)}")

    grid = vutils.make_grid(panel, nrow=nrow, padding=2)
    grid_img = (grid.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()

    if grid_img.ndim == 3 and grid_img.shape[2] == 1:
        grid_img = grid_img[:, :, 0]

    img = Image.fromarray(grid_img)

    if labels is not None and len(labels) > 0:
        from PIL import ImageDraw, ImageFont
        
        label_height = 30
        new_img = Image.new(img.mode, (img.width, img.height + label_height), color=255)
        new_img.paste(img, (0, label_height))
        
        draw = ImageDraw.Draw(new_img)
        font = ImageFont.load_default()
        
        col_width = img.width // nrow
        for i, label in enumerate(labels[:nrow]):
            x_center = (i + 0.5) * col_width
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            x_pos = x_center - text_width / 2
            draw.text((x_pos, 5), label, fill=0, font=font)
        
        img = new_img

    img.save(out_path)


@torch.no_grad()
def validate(
    gpt,
    meditok,
    loader,
    args,
    device=None,
    logger=None,
    out_dir=None,
    n_imgs_to_save=0,
    save_volumes=True,
):
    """
    Run validation/evaluation on a dataset.
    
    Args:
        gpt: GPT model for autoregressive reconstruction
        meditok: MedITok encoder/decoder
        loader: DataLoader for validation dataset
        args: Arguments object containing eval settings
        device: torch.device (if None, uses cuda if available)
        logger: Logger instance (if None, prints to stdout)
        out_dir: Output directory for saving results (if None, uses args.output_dir)
        n_imgs_to_save: how many scans to save for visualization
        save_volumes: Whether to save volume .pt files
    
    Returns:
        metrics: Dictionary containing per-scan and aggregate metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if out_dir is None:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(out_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    def log_info(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    metrics = {}
    scan_buf = {}
    scan_imgs = {}
    scan_gts = {}
    scan_masks = {}
    
    # which func to use
    rec_fn = reconstruct_far
    if hasattr(args, 'active_sampling') and args.active_sampling:
        rec_fn = active_sampling # default, based on gen images
        if hasattr(args, 'active_sampling_fn') and args.active_sampling_fn == "patch_logits":
            rec_fn = active_sampling_patch_logits
        elif hasattr(args, 'active_sampling_fn') and args.active_sampling_fn == "gradient":
            rec_fn = active_sampling_entropy_gradient

    sampling_kwargs = {
        "temperature": getattr(args, 'temperature', 1.0),
        "top_p": getattr(args, 'top_p', 1.0),
        "top_k": getattr(args, 'top_k', 0)
    }

    pbar = tqdm(loader, "recon")
    for idx, batch in enumerate(pbar):
        gt_c_image = batch["gt_c_image"].to(device, non_blocking=True)   # [B,H,W,2],
        masked_c_image = batch["masked_c_image"].to(device, non_blocking=True)   # [B,H,W,2]
        sample_mask = batch["mask"].to(device, non_blocking=True)

        max_val = batch.get("max_val", None)
        fname = batch.get("fname", None)
        slice_idx = batch.get("slice", None)

        # Autoregressive reconstruction: generate new tokens conditioned on masked tokens
        rec_c_img, aux = rec_fn(
            gpt,
            meditok,
            masked_image=masked_c_image,
            gt_image=gt_c_image,
            sample_mask=sample_mask,
            n_steps=getattr(args, 'n_steps', None),
            far_mask_mode=getattr(args, 'far_mask_mode', 'cartesian'),
            complex_mode=getattr(args, 'complex_mode', 'cartesian'),
            viz_gen_steps=getattr(args, 'viz_gen_steps', False),
            budget=getattr(args, 'budget', None),
            rank=getattr(args, 'active_sampling_rank', None),
            n_samples=getattr(args, 'active_sampling_n_samples', 4),
            sampling_kwargs=sampling_kwargs,
            les_zero_mean=getattr(args, 'les_zero_mean', False),
            les_radial_weight_alpha=getattr(args, 'les_radial_weight_alpha', 0.0),
            center_fractions=getattr(args, 'center_fractions', 0.04),
        )  # [B,H,W,2]
        gen_prog = aux.get("gen_progress", None)
        sample_mask = aux.get("sample_mask", sample_mask)
        masked_c_image = aux.get("masked_image", masked_c_image)

        entropy_maps = aux.get("entropy_maps", None)
        freq_maps = aux.get("freq_maps", None)
        sample_masks = aux.get("sample_masks", None)
        sample_progress = aux.get("sample_progress")
        grad_magnitudes = aux.get("grad_magnitudes", None)

        if hasattr(args, 'gt_mode') and args.gt_mode == "autoencoded":
            gt_c_image = enc_dec_image(meditok, gt_c_image, sample_mask)

        if max_val is not None:
            max_val = max_val.reshape(max_val.shape[0], 1, 1, 1).to(rec_c_img.device)
            rec_c_img, gt_c_image, masked_c_image =  rec_c_img * max_val, gt_c_image * max_val, masked_c_image * max_val

        if hasattr(args, 'radial_weighting') and args.radial_weighting:
            rec_c_img = inverse_radial_weighting_transform_image(rec_c_img, weights=loader.dataset.radial_weights)
            gt_c_image = inverse_radial_weighting_transform_image(gt_c_image, weights=loader.dataset.radial_weights)
            masked_c_image = inverse_radial_weighting_transform_image(masked_c_image, weights=loader.dataset.radial_weights)

        rec = transforms.complex_abs(rec_c_img)
        gt = transforms.complex_abs(gt_c_image)
        masked_in = transforms.complex_abs(masked_c_image)
        
        rec_np = rec.detach().cpu().float().numpy()
        if hasattr(args, 'denoise') and args.denoise:
            rec_denoised = apply_nl_means_filter(rec)
            rec_np_denoised = rec_denoised.detach().cpu().numpy()
            mix_alpha = getattr(args, 'mix_alpha', 1.0)
            rec_np = mix_alpha * rec_np_denoised + (1 - mix_alpha) * rec_np
            rec = torch.from_numpy(rec_np).to(rec.device)

        gt_np = gt.detach().cpu().float().numpy()
        # masked_in_np = masked_in.permute(0, 2, 3, 1).detach().cpu().float().numpy()

        # visualization scaling: use max_val if provided, else assume data_range=1
        if max_val is not None:
            # vis_scale = max_val.to(rec.device).view(-1, 1, 1).clamp(min=1e-8)
            vis_scale = gt.max().view(-1,1,1)
        else:
            vis_scale = torch.ones_like(rec[:, :1, :1, :1])
        rec_vis = torch.clamp(rec / vis_scale, 0, 1)
        gt_vis = torch.clamp(gt / vis_scale, 0, 1)
        masked_in_vis = torch.clamp(masked_in / vis_scale, 0, 1)

        # save
        B = rec.shape[0]
        for i in range(B):
            if n_imgs_to_save > 0:
                name = save_slice_viz(
                    args=args,
                    out_dir=out_dir,
                    i=i,
                    fname=fname,
                    slice_idx=slice_idx,
                    rec_vis=rec_vis,
                    gt_vis=gt_vis,
                    masked_in_vis=masked_in_vis,
                    sample_mask=sample_mask,
                    scan_buf=scan_buf,
                    gen_prog=gen_prog,
                    gt_c_image=gt_c_image,
                )
            else:
                name = Path(fname[i]).stem if fname is not None else str(i)

            if n_imgs_to_save > 0:
                save_entropy_maps(
                    args=args,
                    out_dir=out_dir,
                    name=name,
                    i=i,
                    sample_masks=sample_masks,
                    sample_progress=sample_progress,
                    entropy_maps=entropy_maps,
                    freq_maps=freq_maps,
                    reconstructed_image=rec,
                    target_image=gt,
                    scan_buf=scan_buf,
                )
                save_gradient_map(
                    args=args,
                    out_dir=out_dir,
                    name=name,
                    i=i,
                    grad_magnitudes=grad_magnitudes,
                    sample_masks=sample_masks,
                    reconstructed_image=rec,
                    target_image=gt,
                    scan_buf=scan_buf,
                )
                n_imgs_to_save-=B

            # accumulate per-scan metrics
            scan_key = Path(fname[i]).stem if fname is not None else str(i)
            if scan_key not in scan_buf:
                scan_buf[scan_key] = {"rec": [], "gt": [], "slice": []}
                scan_imgs[scan_key] = []
                scan_gts[scan_key] = []
                scan_masks[scan_key] = []
            scan_buf[scan_key]["rec"].append(rec_np[i])
            scan_buf[scan_key]["gt"].append(gt_np[i])
            scan_buf[scan_key]["slice"].append(int(slice_idx[i]) if slice_idx is not None else len(scan_buf[scan_key]["slice"]))
            scan_imgs[scan_key].append((int(slice_idx[i]) if slice_idx is not None else len(scan_imgs[scan_key]), rec_np[i]))
            scan_gts[scan_key].append((int(slice_idx[i]) if slice_idx is not None else len(scan_gts[scan_key]), gt_np[i]))
            scan_masks[scan_key].append((int(slice_idx[i]) if slice_idx is not None else len(scan_masks[scan_key]), sample_mask[i].cpu().numpy()))

    # compute per-scan PSNR/SSIM/LPIPS/DISTS
    metrics = compute_reconstruction_metrics(scan_buf, device)

    # save per-scan volumes as h5 (sorted by slice)
    if save_volumes:
        n_scans_to_save = getattr(args, 'n_scans_to_save', 4)
        recon_dir = out_dir / "recon"
        recon_dir.mkdir(exist_ok=True)
        for i, (k, vals) in enumerate(sorted(scan_imgs.items())):
            rec_sorted = [x[1] for x in sorted(vals, key=lambda p: p[0])]
            # gt_sorted = [x[1] for x in sorted(scan_gts[k], key=lambda p: p[0])]
            rec_vol = np.stack(rec_sorted, axis=0)
            # gt_vol = np.stack(gt_sorted, axis=0)
            if i < n_scans_to_save:
                with h5py.File(recon_dir / f"{k}.h5", 'w') as hf:
                    hf.create_dataset('reconstruction', data=rec_vol, compression='gzip')
                    # hf.create_dataset('ground_truth', data=gt_vol, compression='gzip')
        
        # save masks if active sampling is enabled
        if hasattr(args, 'active_sampling') and args.active_sampling:
            mask_dir = out_dir / "masks"
            mask_dir.mkdir(exist_ok=True)
            for i, (k, vals) in enumerate(sorted(scan_masks.items())):
                mask_sorted = [x[1] for x in sorted(vals, key=lambda p: p[0])]
                mask_vol = torch.from_numpy(np.stack(mask_sorted, axis=0))
                if i < n_scans_to_save:
                    torch.save(mask_vol, mask_dir / f"{k}_mask.pt")

    # save metrics
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log_info("metrics:")
    for k, v in metrics.items():
        log_info(f"{k}: {v}")
    log_info(f"PSNR mean: {metrics['psnr_mean']}, SSIM mean: {metrics['ssim_mean']} LPIPS mean: {metrics['lpips_mean']}, DISTS mean: {metrics['dists_mean']}")
    log_info(f"{out_dir}")
    
    return metrics


@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = create_logger(out_dir)
    logger.info(f"Eval directory created at {out_dir}")
    log_args(args, logger)

    # load models
    meditok = load_meditok(args.encoder_config, args.encoder_ckpt, device)
    gpt = load_gpt(args, device, dtype)

    # dataset
    dataset = build_dataset(args)  # MRICodeWithMaskDataset should return masked_in, gt_tokens, max_val, fname, slice
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Run validation
    metrics = validate(
        gpt=gpt,
        meditok=meditok,
        loader=loader,
        args=args,
        device=device,
        logger=logger,
        out_dir=out_dir,
        n_imgs_to_save=args.n_scans_to_save,
        save_volumes=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i")
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--num-output-layer", type=int, default=1)
    parser.add_argument("--vq-embed-path", type=str, default=None)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--encoder-ckpt", type=str, required=True)
    parser.add_argument("--encoder-config", type=str, default="weights/meditok/config.json")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--downsample-size", type=int, default=16)
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--compile", action='store_true')
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dataset", type=str, default='mri_code')
    parser.add_argument("--code-dir", type=str, default=None)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--acquisition-type", type=str, default=None, choices=[None, 'PD', 'PDFS'], help='Type of acquisition - None, PDFS, PD')
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="recon_samples")
    parser.add_argument("--cls-token-num", type=int, default=0)
    parser.add_argument("--cond-token-num", type=int, default=256)
    parser.add_argument("--n-steps", help="num generation steps", type=int, default=None)
    parser.add_argument("--far-mask-mode", type=str, default='cartesian')
    parser.add_argument("--complex-mode", type=str, default='cartesian', choices=['cartesian', 'polar'])
    parser.add_argument("--center-fractions", type=float, default=0.04)
    parser.add_argument("--accelerations", type=int, default=4)
    parser.add_argument("--viz-gen-steps", action='store_true', help='save result of each generation step')
    parser.add_argument("--n-scans-to-save", type=int, default=4)
    parser.add_argument("--gt-mode", type=str, choices=['raw', 'autoencoded'])
    # active sampling
    parser.add_argument("--active-sampling", action='store_true')
    parser.add_argument("--active-sampling-budget", type=int, default=None)
    parser.add_argument("--active-sampling-steps", type=int, default=None, help="num steps to sample all budget")
    parser.add_argument("--active-sampling-n-samples", type=int, default=4, help="num generation samples per item")
    parser.add_argument("--budget", type=int, default=None, help="num vertical lines to sample, if None is calculated from image size and accelerations")
    parser.add_argument("--active-sampling-rank", type=int, default=None, help="num of sampled line per active sampling iterations")
    parser.add_argument("--active-sampling-fn", type=str, default=None, choices=[None, "patch_logits", "gradient"])
    parser.add_argument("--les-zero-mean", action='store_true',
        help="LES: subtract spatial mean from entropy before FFT to remove DC dominance")
    parser.add_argument("--les-radial-weight-alpha", type=float, default=0.0,
        help="LES: multiply freq_magnitude by r^alpha to boost outer k-space (0=off, 1=linear, 2=quadratic)")
    # Image processing
    parser.add_argument("--denoise", action='store_true', help='apply denoising filter to reconstruction')
    parser.add_argument("--mix-alpha", type=float, default=1.0, help="rec = mix_alpha * rec_denoised + (1-mix_alpha) * rec")
    # Dataset
    parser.add_argument("--radial-weighting", action='store_true')
    parser.add_argument("--data-split", type=str, default=None, help="path to a .txt file containing filename to test on")
    parser.add_argument("--end-slice-skip", type=int, default=0, help="number of end slices to skip (0 for knee, 5 for brain)")


    args = parser.parse_args()
    if args.active_sampling:
        args.budget = calc_budget(args.image_size, args.center_fractions, args.accelerations) if args.budget is None else args.budget
        args.accelerations = float("inf")
        if args.active_sampling_steps == None:
            args.active_sampling_rank = args.budget
        else:
            args.active_sampling_rank = max(1, int(args.budget / args.active_sampling_steps))
            print(f"using AS rank:{args.active_sampling_rank}")
    main(args)