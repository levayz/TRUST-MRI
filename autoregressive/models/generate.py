# Modified from:
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._dynamo.config
import torch._inductor.config
import copy
from data import transforms
from tqdm.auto import tqdm
import torchvision.utils as vutils
from PIL import Image
from pathlib import Path
from autoregressive.utils.mask_utils import radial_circle_radii, radial_circle_mask
from autoregressive.sample.adaptive_sampling import kspace_uncertainty_from_images, kspace_uncertainty_from_patch_logits, image_entropy_to_freq_entropy
from models.meditok import MedITok


# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.triton.unique_kernel_names = True
# torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future


### from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_k_top_p_filtering(
        logits,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample(logits, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, sample_logits=True):
    logits = logits[:, -1, :] / max(temperature, 1e-5) # BL, V/K
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs

def sample_far(logits, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, sample_logits=True):
    # logits BL, 2, 1, V/K or BL, 1, V/K
    n_dims = len(logits.shape)
    if n_dims == 3:
        logits = logits[:, -1, :] / max(temperature, 1e-5)
    elif n_dims == 4:
        logits = logits[:, :, -1, :] / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    else:
        return logits.argmax(dim=-1), None
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        if n_dims == 3:
            idx = torch.multinomial(probs, num_samples=1)
        elif n_dims == 4:
            idx_re = torch.multinomial(probs[:,0], num_samples=1)
            idx_im = torch.multinomial(probs[:,1], num_samples=1)
            idx = torch.cat([idx_re, idx_im], dim=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs


def logits_to_probs(logits, temperature: float = 1.0, top_p: float = 1.0, top_k: int = None, **kwargs):
    logits = logits / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def prefill(model, cond_idx: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, **sampling_kwargs):
    next_token = model(None, cond_idx, input_pos)
    model.setup_head_caches(
        next_token.shape[0],
        model.num_codebooks,
        dtype=next_token.dtype,
        device=next_token.device
    )
    indices = []
    bs = next_token.shape[0]  # [B, 120, 768]
    for i in range(model.num_codebooks):
        start_pos = torch.tensor([i], dtype=torch.int)
        #start_pos = start_pos.repeat_interleave(120, dim=0)
        mask = model.output.head_causal_mask[:bs, None, start_pos]

        if i == 0:
            logits = model.output.forward_test(next_token, input_pos=start_pos, mask=mask)
        else:
            if cfg_scale > 1.0:
                pred_idx = torch.cat([pred_idx, pred_idx])
            logits = model.output.forward_test(next_token, idx=pred_idx, input_pos=start_pos, mask=mask)
            # logits shape: [B, 1, V/K]

        if cfg_scale > 1.0:
            cond_logits, uncond_logits = torch.split(logits, len(logits) // 2, dim=0)
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
            pred_idx = sample(logits, **sampling_kwargs)[0]  # [B, 1]
        else:
            pred_idx = sample(logits, **sampling_kwargs)[0]  # [B, 1]
        indices.append(pred_idx)
    indices = torch.stack(indices, dim=1)
    return indices


def decode_one_token(model, x: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, cfg_flag: bool, **sampling_kwargs):
    assert input_pos.shape[-1] == 1
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        next_token = model(x_combined, cond_idx=None, input_pos=input_pos)
    else:
        next_token = model(x, cond_idx=None, input_pos=input_pos)
    model.setup_head_caches(
        next_token.shape[0],
        model.num_codebooks,
        dtype=next_token.dtype,
        device=next_token.device
    )
    indices = []
    bs = next_token.shape[0]
    for i in range(model.num_codebooks):
        start_pos = torch.tensor([i], dtype=torch.int)
        mask = model.output.head_causal_mask[:bs, None, start_pos]
        if i == 0:
            logits = model.output.forward_test(next_token, input_pos=start_pos, mask=mask)
        else:
            if cfg_scale > 1.0:
                pred_idx = torch.cat([pred_idx, pred_idx]) 
            logits = model.output.forward_test(next_token, idx=pred_idx, input_pos=start_pos, mask=mask)
        if cfg_scale > 1.0:
            cond_logits, uncond_logits = torch.split(logits, len(logits) // 2, dim=0)
            if cfg_flag:
                logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale

            else:
                logits = cond_logits
            pred_idx = sample(logits, **sampling_kwargs)[0]
        else:
            pred_idx = sample(logits, **sampling_kwargs)[0]
        indices.append(pred_idx)
    indices = torch.stack(indices, dim=1)
    return indices


def decode_n_tokens(
        model, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int,
        cfg_scale: float, cfg_interval: int, **sampling_kwargs
    ):
    new_tokens = []
    cfg_flag = True # used if cfg_scale > 1.0
    for i in range(num_new_tokens):
        # Actually better for Inductor to codegen attention here
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            if cfg_interval > -1 and i > cfg_interval:
                cfg_flag = False
            next_token = decode_one_token(model, cur_token, input_pos, cfg_scale, cfg_flag, **sampling_kwargs)
            input_pos += 1
            new_tokens.append(next_token.clone())
            cur_token = next_token
    return new_tokens


@torch.no_grad()
def generate(model, cond, max_new_tokens, num_codebooks, emb_masks=None, cfg_scale=1.0, cfg_interval=-1,
             **sampling_kwargs):
    if model.model_type == 'c2i':
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * model.num_classes
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = 1
    elif model.model_type == 't2i':
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = cond.shape[1]
    else:
        raise Exception("please check model type")

    T_new = T + max_new_tokens
    max_seq_length = T_new
    max_batch_size = cond.shape[0]

    device = cond.device
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size

        if hasattr(model.tok_embeddings, 'codebooks'):
            codebook_dtype = model.tok_embeddings.codebooks[0].weight.dtype
        else:  # unitok embedding
            codebook_dtype = model.tok_embeddings.quantizer.codebooks[0].codebook.weight.dtype

        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=codebook_dtype, device=device)

    if emb_masks is not None:
        assert emb_masks.shape[0] == max_batch_size
        assert emb_masks.shape[-1] == T
        if cfg_scale > 1.0:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
        else:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

        eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
        model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix

    # create an empty tensor of the expected final shape and fill in the current tokens
    seq = torch.empty((max_batch_size, num_codebooks, T_new), dtype=torch.int, device=device)

    input_pos = torch.arange(0, T, device=device)
    #print(f"--------> going to prefill. T={T}, seq={seq.shape}, cond_combined={cond_combined.shape}, input_pos={input_pos.shape}")
    next_token = prefill(model, cond_combined, input_pos, cfg_scale, **sampling_kwargs)
    seq[:, :, T:T+1] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens = decode_n_tokens(model, next_token, input_pos, max_new_tokens - 1, cfg_scale, cfg_interval,
                                          **sampling_kwargs)
    seq[:, :, T+1:] = torch.cat(generated_tokens, dim=-1)

    return seq[:, :, T:]


# -----------------------------------------------------------------------------#
# Masked-to-image reconstruction: conditioning on masked codes (cond_tokens)
# -----------------------------------------------------------------------------#
def prefill_cond_tokens(model, cond_tokens: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, **sampling_kwargs):
    next_token = model(None, cond_idx=None, cond_tokens=cond_tokens, input_pos=input_pos)
    next_token = next_token[:, -1:]
    model.setup_head_caches(
        next_token.shape[0],
        model.num_codebooks,
        dtype=next_token.dtype,
        device=next_token.device
    )
    indices = []
    bs = next_token.shape[0]
    for i in range(model.num_codebooks):
        start_pos = torch.tensor([i], dtype=torch.int)
        mask = model.output.head_causal_mask[:bs, None, start_pos]
        if i == 0:
            logits = model.output.forward_test(next_token, input_pos=start_pos, mask=mask)
        else:
            if cfg_scale > 1.0:
                pred_idx = torch.cat([pred_idx, pred_idx])
            logits = model.output.forward_test(next_token, idx=pred_idx, input_pos=start_pos, mask=mask)
        if cfg_scale > 1.0:
            cond_logits, uncond_logits = torch.split(logits, len(logits) // 2, dim=0)
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
            pred_idx = sample(logits, **sampling_kwargs)[0]
        else:
            pred_idx = sample(logits, **sampling_kwargs)[0]
        indices.append(pred_idx)
    indices = torch.stack(indices, dim=1)
    return indices


# def decode_one_token_cond_tokens(model, x: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, cfg_flag: bool, **sampling_kwargs):
#     assert input_pos.shape[-1] == 1
#     if cfg_scale > 1.0:
#         x_combined = torch.cat([x, x])
#         next_token = model(x_combined, cond_idx=None, input_pos=input_pos)
#     else:
#         next_token = model(x, cond_idx=None, input_pos=input_pos)
#     model.setup_head_caches(
#         next_token.shape[0],
#         model.num_codebooks,
#         dtype=next_token.dtype,
#         device=next_token.device
#     )
#     indices = []
#     bs = next_token.shape[0]
#     for i in range(model.num_codebooks):
#         start_pos = torch.tensor([i], dtype=torch.int)
#         mask = model.output.head_causal_mask[:bs, None, start_pos]
#         if i == 0:
#             logits = model.output.forward_test(next_token, input_pos=start_pos, mask=mask)
#         else:
#             if cfg_scale > 1.0:
#                 pred_idx = torch.cat([pred_idx, pred_idx])
#             logits = model.output.forward_test(next_token, idx=pred_idx, input_pos=start_pos, mask=mask)
#         if cfg_scale > 1.0:
#             cond_logits, uncond_logits = torch.split(logits, len(logits) // 2, dim=0)
#             if cfg_flag:
#                 logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
#             else:
#                 logits = cond_logits
#             pred_idx = sample(logits, **sampling_kwargs)[0]
#         else:
#             pred_idx = sample(logits, **sampling_kwargs)[0]
#         indices.append(pred_idx)
#     indices = torch.stack(indices, dim=1)
#     return indices

@torch.no_grad()
def reconstruct(model, cond_tokens: torch.Tensor, max_new_tokens: int, cfg_scale: float = 1.0, cfg_interval: int = -1,
                **sampling_kwargs):
    """
    Generate codes conditioned on masked-image tokens (cond_tokens).
    cond_tokens: [B, K, L] int codes from masked image.
    Returns: [B, K, max_new_tokens] generated codes.
    """
    device = cond_tokens.device
    B, K, L = cond_tokens.shape
    T = model.cond_token_num + model.cls_token_num # full conditioning sequence length
    T_new = T + max_new_tokens
    max_seq_length = T_new
    max_batch_size = B

    if hasattr(model.tok_embeddings, 'codebooks'):
        codebook_dtype = model.tok_embeddings.codebooks[0].weight.dtype
    else:
        codebook_dtype = model.tok_embeddings.quantizer.codebooks[0].codebook.weight.dtype

    model.setup_caches(max_batch_size=max_batch_size, max_seq_length=max_seq_length, dtype=codebook_dtype, device=device)

    seq = torch.empty((max_batch_size, K, T_new), dtype=torch.int, device=device)

    input_pos = torch.arange(0, T, device=device)
    next_token = prefill_cond_tokens(model, cond_tokens, input_pos, cfg_scale, **sampling_kwargs)
    seq[:, :, T:T+1] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens = decode_n_tokens(model, next_token, input_pos, max_new_tokens - 1, cfg_scale, cfg_interval,
                                       **sampling_kwargs)
    seq[:, :, T+1:] = torch.cat(generated_tokens, dim=-1)

    # drop all conditioning tokens, return only generated segment
    return seq[:, :, T:]


def get_mask_levels(W, center_fraction=0.04, n_steps=None, mode='cartesian'):
    if mode == 'cartesian':
        end_idx =  W // 2 - int(W * center_fraction) // 2
        if n_steps == None:
            step = 1
        else:
            step = int(end_idx / n_steps + 1)
        mask_levels = list(range(0, end_idx, step))
        mask_levels.append(end_idx)
        mask_levels[::-1]
    elif mode == 'radial':
        mask_levels = radial_circle_radii(
        W,
        W,
        num_masks=n_steps,
        center_fraction=center_fraction,
        dtype=torch.float32,
    )
        mask_levels=mask_levels[:-1] # last mask is for the whole image
    else:
        raise ValueError(f"unsupported mask mode: {mode}")

    # print(f"[DEBUG]: using {len(mask_levels)} masks")
    # print(f"[DEBUG]: mask levels:\n{mask_levels}")
    return mask_levels


def get_mask(mask_level, dummy_image, mode='cartesian'):
    if mask_level < 0:
        return torch.zeros_like(dummy_image)
    if mode == 'cartesian':
        sample_mask = torch.ones_like(dummy_image)
        sample_mask[:,:, :mask_level] = 0
        sample_mask[:, :, -mask_level:] = 0

        sample_mask[:,:mask_level] = 0
        sample_mask[:,-mask_level:] = 0
    elif mode =='radial':
        B, H, W, _ = dummy_image.shape
        sample_mask = radial_circle_mask(H, W, mask_level, device=dummy_image.device, dtype=torch.float32).to(dummy_image.dtype)
        sample_mask = sample_mask[None,...,None]


    return sample_mask


def apply_mask(image: torch.Tensor, mask: torch.Tensor):
    kspace = transforms.fft2(image)
    masked_kspace = kspace * mask
    masked_image = transforms.ifft2(masked_kspace)

    return masked_image

def data_consistency(in_c_image, output_c_image, sample_mask):
    """
    output_c_image = IFFT(in_kspace * sample_mask + out_kspace * (1-sample_mask))
    """
    in_kspace = transforms.fft2(in_c_image)
    output_kspace = transforms.fft2(output_c_image) 

    output_kspace = torch.where(sample_mask.bool(), in_kspace, output_kspace)
    
    output_c_image = transforms.ifft2(output_kspace)

    return output_c_image


def save_pngs(out_path: Path, images_hw: list[torch.Tensor]):
    """
    Save a side-by-side grid of the provided images.
    Expects each item to be HW floats in [0,1].
    """
    images_chw = [img.unsqueeze(0) for img in images_hw]
    panel = torch.stack(images_chw, dim=0)
    grid = vutils.make_grid(panel, nrow=len(images_chw), padding=2)
    grid_img = (grid.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(grid_img).save(out_path)

@torch.no_grad()
def reconstruct_far(model, tokenizer:MedITok, masked_image: torch.Tensor, sample_mask: torch.Tensor, sampling_kwargs=None, **kwargs):
    B, H, W, _ = masked_image.shape
    default_sampling_kwargs = {"temperature": 1.0, "top_k": 0, "top_p": 1.0}
    if sampling_kwargs is None:
        sampling_kwargs = default_sampling_kwargs

    center_fraction = kwargs.get("center_fractions", 0.04)
    mask_levels = get_mask_levels(W, center_fraction=center_fraction, n_steps=kwargs.get("n_steps", None), mode=kwargs.get("far_mask_mode", "cartesian"))
    i_hat_prev = masked_image.clone() # [B,H,W,2]
    gen_prog = [(i_hat_prev, sample_mask)]
    for idx, mask_level in enumerate(mask_levels):
        def encode(img_bhw):
            img = img_bhw.unsqueeze(-1).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)
            enc_tokens = tokenizer.encoder(img)           # [B,256,1024]
            enc_tokens = tokenizer.quant_proj(enc_tokens)       # [B,256,64]
            start_tokens = tokenizer.quantizer.f_to_idx(enc_tokens)  # [B,K_enc,256]
            return start_tokens
        
        if kwargs.get("complex_mode", "cartesian") == 'polar':
            i_hat = transforms.cartesian_to_polar(i_hat)

        start_tokens_re = encode(i_hat_prev[..., 0]) # B,K,L
        start_tokens_im = encode(i_hat_prev[..., 1]) # B,K,L
        cond_tokens = torch.cat((start_tokens_re.unsqueeze(1), start_tokens_im.unsqueeze(1)), dim=1) # B,2,K,L

        t = torch.tensor(idx).to(cond_tokens.device).repeat(B)
        h = model(cond_tokens=cond_tokens.long(),
                timestep=t.long()) # (B,L,D)
        if isinstance(h, tuple):
            h, aux = h
        else:
            aux = None
        h_n_dims = len(h.shape)
        if h_n_dims == 3:
            _, L, D = h.shape
        elif h_n_dims == 4:
            _, _, L, D = h.shape
        else:
            raise ValueError()

        # AR head
        pred_idx = []
        logits_list = []
        model.setup_head_caches(
            B * L,
            model.num_codebooks,
            dtype=h.dtype,
            device=h.device
        )
        if h_n_dims == 3:
            h = h.view(B*L, 1, D)
        else:
            h = h.permute(0, 2, 1, 3).contiguous().view(B*L, 2, 1, D)
        for i in range(model.num_codebooks):
            start_pos = torch.tensor([i], dtype=torch.int)
            mask = model.output.head_causal_mask[:B*L, None, start_pos]
            if i == 0:
                logits = model.output.forward_test(h, input_pos=start_pos, mask=mask) # B*L, 2, 1, V/K
            else:
                logits = model.output.forward_test(h, idx=pred_idx[-1], input_pos=start_pos, mask=mask)
            logits_list.append(logits) # BL, 2, 1, V/K
            _pred_idx = sample_far(logits, **sampling_kwargs)[0].unsqueeze(-1)
            pred_idx.append(_pred_idx) # BL, 2, 1
        pred_idx = torch.stack(pred_idx, dim=2) # B*L, 2,  K, 1
        pred_idx = pred_idx.view(B, L, 2, -1).permute(0, 2, 3, 1) # B, L, 2, K -> B, 2, K, L

        logits_list = torch.stack(logits_list, dim=2) # BL, 2, K, V/K
        logits_list = logits_list.view(B, L, 2, model.num_codebooks, -1) # B, L, 2, K, V/K
        logits_list_1d = logits_list.clone()
        H_patch = W_patch = int(L ** 0.5)
        logits_list = logits_list.view(B, H_patch, W_patch, 2, model.num_codebooks, -1)

        def decode_codes(codes_for_decode):
            f = tokenizer.quantizer.idx_to_f(codes_for_decode)          # [B,256,64]
            f = tokenizer.post_quant_proj(f)                            # [B,256,1024]
            dec = tokenizer.decoder(f)                                  # [B,3,H,W] in [-1,1]
            dec = dec.clamp_(-1, 1)
            return dec

        def decode_complex_codes(codes_for_decode):
            re_codes = codes_for_decode[:, 0]
            im_codes = codes_for_decode[:, 1]
            re_img = decode_codes(re_codes)
            im_img = decode_codes(im_codes)
            img = torch.stack([re_img, im_img], dim=-1)
            return img

        i_hat = decode_complex_codes(pred_idx) # B, 3, H, W, 2
        if kwargs.get("complex_mode", "cartesian") == "polar":
            i_hat = transforms.polar_to_cartesian(i_hat)

        def rgb_to_grayscale(img):
            # return img[:,0] * 0.2126 + img[:,1] * 0.7152 + img[:,2] * 0.0722
            return (img[:,0] + img[:,1] + img[:,2]) / 3

        i_hat = rgb_to_grayscale(i_hat) # B, H, W, 2

        # get mask for next level
        if idx == len(mask_levels) - 1: # done, no more masking
            i_hat = data_consistency(masked_image, i_hat, sample_mask)
            if kwargs.get("viz_gen_steps", False):
                gen_prog.append((i_hat, torch.ones_like(sample_mask)))
            break

        mask = get_mask(mask_levels[idx], i_hat, mode=kwargs.get("far_mask_mode", "cartesian"))
        mask_prev = get_mask(mask_levels[idx-1], i_hat, mode=kwargs.get("far_mask_mode", "cartesian"))  
        if kwargs.get("viz_gen_steps", False):
            i_hat = data_consistency(masked_image, i_hat, sample_mask)
            gen_prog.append((i_hat, mask.bool() | sample_mask.bool()))
        i_hat = data_consistency(i_hat_prev, i_hat, mask_prev) # DC step for previous step - keep already generated freqs
        i_hat = apply_mask(i_hat, mask) # zero out high freqs
        i_hat = data_consistency(masked_image, i_hat, sample_mask) # DC to GT samples
        i_hat_prev= i_hat.clone()

    model.rm_head_caches()

    return i_hat, {"gen_progress": gen_prog, "logits": logits_list, "logits_1d": logits_list_1d, "model_aux": aux} # B,H,W,2, [((B,H,W,2), (B,H,W,2)), ...x mask_levels]


@torch.no_grad
def active_sampling(model, tokenizer, gt_image:torch.Tensor, masked_image: torch.Tensor, sample_mask: torch.Tensor, budget: int, rank=1, **kwargs):
    n_samples = kwargs.get("n_samples", 8)
    sampling_kwargs = kwargs["sampling_kwargs"]
    viz_generations = kwargs.get("viz_generations", False) # save generations used in each reconstruction
    del kwargs["sampling_kwargs"]
    B, H, W, _ = masked_image.shape
    n_steps = budget // rank
    # print(f"[DEBUG]: active sampling n_steps:{n_steps}, budget:{budget}, rank:{rank}")
    for i in tqdm(range(n_steps), "active_sampling"):
        samples = masked_image.repeat(n_samples, 1,1,1) # B*N, H, W, 2
        samples, _ = reconstruct_far(
            model,
            tokenizer,
            samples,
            sample_mask.repeat(n_samples,1,1,1),
            sampling_kwargs=sampling_kwargs,
            **kwargs
        )

        samples = samples.view(n_samples, B, H, W, 2).permute(1,0,2,3,4) # B,N,H,W,2
        sample_mask, scores = kspace_uncertainty_from_images( # choose new sample location
            samples.permute(0,1,4,2,3),
            sample_mask.permute(0,3,1,2),
            vertical_mask=(kwargs.get("active_sampling_mask_type", "cartesian") == "cartesian"),
            rank=rank,
        ) 
        sample_mask = sample_mask.permute(0,2,3,1)
        masked_image = data_consistency(gt_image, torch.zeros_like(masked_image), sample_mask) # add sampled location

    rec_image, _ = reconstruct_far(model, tokenizer, masked_image, sample_mask, **kwargs) # reconstruct

    return rec_image, {"sample_mask": sample_mask, "masked_image": masked_image}


@torch.no_grad
def active_sampling_patch_logits(model, tokenizer, gt_image:torch.Tensor, masked_image: torch.Tensor, sample_mask: torch.Tensor, budget: int, rank=1, **kwargs):
    viz_generations = kwargs.get("viz_generations", False) # save generations used in each reconstruction
    B, H, W, _ = masked_image.shape
    n_steps = budget // rank
    entropy_maps = [torch.zeros((B, H, W, 2), device=masked_image.device)]
    freq_maps    = [torch.zeros((B, H, W),    device=masked_image.device)]
    sample_masks = [sample_mask.clone()]
    sample_progress = [masked_image]
    for i in tqdm(range(n_steps), "LES"):
        samples, aux = reconstruct_far(
            model,
            tokenizer,
            masked_image,
            sample_mask,
            **kwargs
        )
        logits = aux["logits"] # B, H_p, W_p, 2, K, V/k

        sample_mask, entropy_complex, freq_magnitude = kspace_uncertainty_from_patch_logits(
            logits,
            sample_mask.permute(0,3,1,2),
            vertical_mask=(kwargs.get("active_sampling_mask_type", "cartesian") == "cartesian"),
            rank=rank,
            les_zero_mean=kwargs.get("les_zero_mean", False),
            les_radial_weight_alpha=kwargs.get("les_radial_weight_alpha", 0.0),
        )
        sample_mask = sample_mask.permute(0,2,3,1)
        sample_progress.append(samples)
        sample_masks.append(sample_mask.clone())
        entropy_maps.append(entropy_complex)
        freq_maps.append(freq_magnitude)
        masked_image = data_consistency(gt_image, torch.zeros_like(masked_image), sample_mask) # add sampled location

    rec_image, _ = reconstruct_far(model, tokenizer, masked_image, sample_mask, **kwargs) # reconstruct

    return rec_image, {"sample_mask": sample_mask, "masked_image": masked_image, "entropy_maps": entropy_maps, "freq_maps": freq_maps, "sample_masks": sample_masks, "sample_progress": sample_progress}


def soft_lookup(z_e, codebooks, tok_embeddings, temperature=1.0):
    """
    Differentiable soft embedding lookup.
    
    Args:
        z_e: continuous pre-quantization features [B, L, 64]
        codebooks: list of K VectorQuantizer objects (from tokenizer.quantizer.codebooks)
        tok_embeddings: TokenEmbedder module with .codebooks attribute
        temperature: softmax temperature
    
    Returns:
        soft_embeddings: [B, L, dim]
    """
    B, L, D = z_e.shape
    K = len(codebooks)
    
    # Split features across codebooks (like VectorQuantizerM.forward does)
    chunk_size = D // K  # 64 / K
    z_e_chunks = z_e.split(chunk_size, dim=-1)  # List of K tensors [B, L, 64/K]
    
    soft_embeds = []
    
    for k in range(K):
        # Normalize features for this chunk
        z_e_k_norm = F.normalize(z_e_chunks[k], dim=-1).float()  # [B, L, 64/K]
        
        # Get normalized codebook weights from VectorQuantizer
        codebook_weights = codebooks[k].codebook.get_norm_weight().float()  # [V/K, 64/K]
        
        # Compute similarities
        similarities = torch.matmul(z_e_k_norm, codebook_weights.T)  # [B, L, V/K]
        
        # Soft assignment weights
        weights = F.softmax(similarities / temperature, dim=-1)  # [B, L, V/K]
        
        # Get transformer embeddings from TokenEmbedder.codebooks[k]
        embed_weights = tok_embeddings.codebooks[k].weight  # [V/K, dim]
        
        # Soft lookup
        soft_embed_k = torch.matmul(weights.to(embed_weights.dtype), embed_weights)  # [B, L, dim]
        soft_embeds.append(soft_embed_k)
    
    return sum(soft_embeds)  # [B, L, dim]

# TODO add timestep, 
# TODO add variable number of sampling steps
# TODO maybe do this after intial reconstruction?
def active_sampling_entropy_gradient( 
    model, 
    tokenizer, 
    gt_image: torch.Tensor,
    masked_image: torch.Tensor,  # [B, H, W, 2] complex image
    sample_mask: torch.Tensor,   # [B, H, W, 2] binary mask
    budget: int, 
    rank=1, 
    sampling_kwargs=None,
    **kwargs
):
    """
    Select k-space locations to acquire by backpropagating from transformer entropy.
    
    Gradient flow: Loss -> Transformer -> Soft Embeddings -> z_e -> Tokenizer -> Image -> K-space
    """
    n_steps = budget // rank
    # print(f"[DEBUG]: active sampling n_steps:{n_steps}, budget:{budget}, rank:{rank}")
    B, H, W = sample_mask.shape[:3]
    device = masked_image.device

    default_sampling_kwargs = {"temperature": 1.0, "top_k": 0, "top_p": 1.0}
    if sampling_kwargs is None:
        sampling_kwargs = default_sampling_kwargs

    grad_magnitude = None
    grad_magnitudes = []
    sample_masks_list = [sample_mask.clone()]
    for _ in tqdm(range(n_steps), "GEO"):
        with torch.enable_grad():
            masked_kspace = transforms.fft2(masked_image)
            masked_kspace.requires_grad_(True)

            # 1. Reconstruct image from k-space
            i_hat = transforms.ifft2(masked_kspace)  # [B, H, W, 2]

            # 2. Encode through tokenizer (get z_e for soft lookup, z_q for hard lookup)
            z_ste_re, z_q_re, z_e_re = tokenizer.encode_with_ste(i_hat[..., 0])  # [B, L, 64]
            z_ste_im, z_q_im, z_e_im = tokenizer.encode_with_ste(i_hat[..., 1])  # [B, L, 64]

            # 3. Soft lookup (differentiable) - uses z_e
            soft_embed_re = soft_lookup(
                z_e_re,
                tokenizer.quantizer.codebooks,
                model.re_tok_embeddings,
                temperature=kwargs.get('temperature', 1.0)
            )  # [B, L, dim]

            soft_embed_im = soft_lookup(
                z_e_im,
                tokenizer.quantizer.codebooks,
                model.im_tok_embeddings,
                temperature=kwargs.get('temperature', 1.0)
            )  # [B, L, dim]

            # 4. Combine into [B, 2, L, dim] format
            soft_cond_embeddings = torch.stack([soft_embed_re, soft_embed_im], dim=1)  # [B, 2, L, dim]

            # 5. Get hard indices for discrete path (what transformer sees in forward)
            indices_re = tokenizer.quantizer.f_to_idx(z_q_re)  # [B, K, L]
            indices_im = tokenizer.quantizer.f_to_idx(z_q_im)  # [B, K, L]
            cond_tokens = torch.stack([indices_re, indices_im], dim=1)  # [B, 2, K, L]

            # 6. Forward through transformer with STE
            timestep = kwargs.get('timestep', torch.zeros(masked_image.shape[0], device=device))
            h = model(
                cond_tokens=cond_tokens.long(),
                soft_cond_embeddings=soft_cond_embeddings,
                timestep=timestep.long()
            )  # [B, L, dim]

            # 7. Autoregressive head forward (differentiable, no KV cache)
            _, L, dim = h.shape
            h_flat = h.view(B * L, 1, dim)  # [B*L, 1, dim]

            all_logits = []
            soft_embeddings = []
            temperature = sampling_kwargs.get('temperature', 1.0)

            for k in range(model.num_codebooks):
                h_k = h_flat if k == 0 else torch.cat([h_flat] + soft_embeddings, dim=1)  # [B*L, 1+k, dim]

                for layer in model.output.layers:
                    h_k = layer(h_k, freqs_cis=None, start_pos=None, mask=None, full_attn=False)
                h_k = model.output.norm(h_k)

                logits_re_k = model.output.linear_head_re(h_k[:, -1:])  # [B*L, 1, V/K]
                logits_im_k = model.output.linear_head_im(h_k[:, -1:])  # [B*L, 1, V/K]
                all_logits.append(torch.stack([logits_re_k, logits_im_k], dim=1))  # [B*L, 2, 1, V/K]

                if k < model.num_codebooks - 1:
                    probs_re = F.softmax(logits_re_k / temperature, dim=-1)
                    probs_im = F.softmax(logits_im_k / temperature, dim=-1)
                    soft_embeddings.append(
                        probs_re @ model.output.codebooks[k].weight +
                        probs_im @ model.output.codebooks[k].weight
                    )  # [B*L, 1, dim]

            all_logits = torch.stack(all_logits, dim=2)  # [B*L, 2, K, 1, V/K]

            # 8. Compute entropy and backpropagate
            probs = F.softmax(all_logits, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
            entropy = entropy.view(B, L, 2, model.num_codebooks)  # [B, L, 2, K]
            loss = -entropy.mean(dim=[1, 2, 3]).sum()

            loss.backward()
            gradients = masked_kspace.grad  # [B, H, W, 2]

        # 9. Select top-rank vertical lines by gradient magnitude
        grad_magnitude = gradients.norm(dim=-1) * (1 - sample_mask[..., 0])  # [B, H, W]
        grad_magnitudes.append(grad_magnitude.detach().clone())
        grad_magnitude_cols = grad_magnitude.mean(dim=1)  # [B, W]
        _, top_indices = torch.topk(grad_magnitude_cols, rank, dim=1)  # [B, rank]

        new_sample_mask = torch.zeros_like(sample_mask)
        batch_indices = torch.arange(B, device=device)[:, None]
        new_sample_mask[batch_indices, :, top_indices, :] = 1
        sample_mask = torch.clamp(sample_mask + new_sample_mask, 0, 1)
        sample_masks_list.append(sample_mask.clone())

        masked_image = data_consistency(gt_image, torch.zeros_like(masked_image), sample_mask)
        model.rm_head_caches()

    rec_image, _ = reconstruct_far(model, tokenizer, masked_image, sample_mask, **kwargs)

    return rec_image, {"sample_mask": sample_mask, "masked_image": masked_image, "grad_magnitude": grad_magnitude, "grad_magnitudes": grad_magnitudes, "sample_masks": sample_masks_list}

