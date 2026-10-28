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
from autoregressive.sample.adaptive_sampling import kspace_uncertainty_from_images, kspace_uncertainty_from_patch_logits, image_entropy_to_freq_entropy, _select_points_poisson_constrained
from models.meditok import MedITok
from models.sense_ops import decode_latent_pair, latent_target


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
        logits = logits[:, -1, :] / max(temperature, 1e-5)   # (BL, V)
    elif n_dims == 4:
        logits = logits[:, :, -1, :] / max(temperature, 1e-5)  # (BL, 2, V)

    do_sample = top_k > 0 or top_p < 1.0
    if not do_sample:
        # greedy argmax — matches original deterministic behavior
        return logits.argmax(dim=-1), None

    # top-k / top-p filtering — top_k_top_p_filtering expects 2D (N, V)
    if n_dims == 3:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    elif n_dims == 4:
        BL, two, V = logits.shape
        logits = top_k_top_p_filtering(logits.reshape(BL * two, V),
                                       top_k=top_k, top_p=top_p).reshape(BL, two, V)

    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        if n_dims == 3:
            idx = torch.multinomial(probs, num_samples=1)
        elif n_dims == 4:
            BL, two, V = probs.shape
            idx = torch.multinomial(probs.reshape(BL * two, V), num_samples=1).reshape(BL, two)
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
    # Data consistency to the measurements: single-image hard k-space replace by default, or a caller-supplied
    # closure (SENSE: expand with the maps -> per-coil replace -> combine; see models.sense_ops.make_sense_dc_fn)
    dc_fn = kwargs.get("dc_fn", None)

    def dc_meas(img, m):
        return dc_fn(img, m) if dc_fn is not None else data_consistency(masked_image, img, m)


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

        mv = getattr(model, 'model_version', 1)
        v2 = mv >= 2.1   # continuous-input models (v2.1)
        if v2:   # v2.1 reads the continuous encoder latents of the input; its input tokens are never used, so they are not computed
            cond_tokens = None
        else:
            start_tokens_re = encode(i_hat_prev[..., 0]) # B,K,L
            start_tokens_im = encode(i_hat_prev[..., 1]) # B,K,L
            cond_tokens = torch.cat((start_tokens_re.unsqueeze(1), start_tokens_im.unsqueeze(1)), dim=1) # B,2,K,L

        t = torch.tensor(idx + int(kwargs.get("timestep_offset", 0))).to(masked_image.device).repeat(B)   # refinement passes use timestep >= 1
        extra_in = {}
        if mv >= 2:   # sampling-mask conditioning
            extra_in["sample_mask_cols"] = sample_mask[:, 0, :, 0].float()
        if v2:        # v2.1: continuous encoder latents instead of input tokens
            extra_in["cond_latent"] = torch.stack([latent_target(tokenizer, i_hat_prev[..., c].float()) for c in range(2)], dim=1)
        if kwargs.get("draft_image") is not None:   # two-view refinement: the previous reconstruction as a second conditioning view
            di = kwargs["draft_image"]
            if v2:
                extra_in["draft_latent"] = torch.stack([latent_target(tokenizer, di[..., c].float()) for c in range(2)], dim=1)
            else:
                extra_in["draft_tokens"] = torch.cat((encode(di[..., 0]).unsqueeze(1), encode(di[..., 1]).unsqueeze(1)), dim=1).long()
        if kwargs.get("draft_latent") is not None:   # progressive commitment: a precomputed, partial second view (committed positions only)
            assert v2, "draft_latent needs a v2.1 model"
            extra_in["draft_latent"] = kwargs["draft_latent"]
            if kwargs.get("draft_keep") is not None:
                extra_in["draft_keep"] = kwargs["draft_keep"]
        h = model(cond_tokens=None if cond_tokens is None else cond_tokens.long(),
                timestep=t.long(), **extra_in) # (B,L,D)
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
        # the fused latent head also reads the head's chain states (mean of the normalised hidden state over the chain steps)
        fused = getattr(model, 'latent_head_type', 'linear') == 'fused' and getattr(model, 'latent_head', None) is not None
        head_states = []

        def head_step(*a, **k):
            out = model.output.forward_test(*a, **k, **({'return_state': True} if fused else {}))
            if fused:
                head_states.append(out[1][:, 0]); return out[0]
            return out
        # KV caches must match the dtype the head produces: under autocast the projections are bf16/fp16 even if h is fp32
        cache_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else h.dtype
        interleaved = bool(getattr(model.output, 'interleaved', False))
        model.setup_head_caches(
            B * L,
            model.num_codebooks * (2 if interleaved else 1),
            dtype=cache_dtype,
            device=h.device
        )
        if h_n_dims == 3:
            h = h.view(B*L, 1, D)
        else:
            h = h.permute(0, 2, 1, 3).contiguous().view(B*L, 2, 1, D)
        if interleaved:   # v2 head: 2K chain steps per position, Re_i at even steps, Im_i (given Re_i) at odd steps
            codes, lgs, prev = [[], []], [[], []], None
            for s in range(2 * model.num_codebooks):
                start_pos = torch.tensor([s], dtype=torch.int)
                mask = model.output.head_causal_mask[:B*L, None, start_pos]
                logits = head_step(h if s == 0 else None, idx=prev, input_pos=start_pos, mask=mask)   # BL, 1, V/K
                prev = sample_far(logits, **sampling_kwargs)[0].reshape(-1)                                           # BL
                codes[s % 2].append(prev); lgs[s % 2].append(logits[:, 0])
            pred_idx = torch.stack([torch.stack(codes[0], 1), torch.stack(codes[1], 1)], 1)       # BL, 2, K
            pred_idx = pred_idx.view(B, L, 2, -1).permute(0, 2, 3, 1)                              # B, 2, K, L
            logits_list = torch.stack([torch.stack(lgs[0], 1), torch.stack(lgs[1], 1)], 1)         # BL, 2, K, V/K
        else:
            for i in range(model.num_codebooks):
                start_pos = torch.tensor([i], dtype=torch.int)
                mask = model.output.head_causal_mask[:B*L, None, start_pos]
                if i == 0:
                    logits = head_step(h, input_pos=start_pos, mask=mask) # B*L, 2, 1, V/K
                else:
                    logits = head_step(h, idx=pred_idx[-1], input_pos=start_pos, mask=mask)
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
        # optional latent-regression head: decode the regressed continuous latent (no quantisation)
        latent_no_dc = None
        if getattr(model, 'latent_head', None) is not None and kwargs.get('latent_decode', True) and h_n_dims == 3:
            if fused:   # conditioned on the codes the token path just committed to (greedy or sampled) and the head's chain state
                head_state = torch.stack(head_states, dim=1).float().mean(1).view(B, L, D)
                z_hat = model.predict_latent(h.view(B, L, D), head_state=head_state, head_codes=pred_idx)
            else:
                z_hat = model.predict_latent(h.view(B, L, D))
            latent_no_dc = decode_latent_pair(tokenizer, z_hat)
            if kwargs.get('rec_mode', 'tokens') == 'latent':
                i_hat, token_no_dc = latent_no_dc, i_hat
        i_hat_no_dc = i_hat.clone()    # decoded image before any data consistency

        # get mask for next level
        if idx == len(mask_levels) - 1: # done, no more masking
            i_hat = dc_meas(i_hat, sample_mask)
            if kwargs.get("viz_gen_steps", False):
                gen_prog.append((i_hat, torch.ones_like(sample_mask)))
            break

        mask = get_mask(mask_levels[idx], i_hat, mode=kwargs.get("far_mask_mode", "cartesian"))
        mask_prev = get_mask(mask_levels[idx-1], i_hat, mode=kwargs.get("far_mask_mode", "cartesian"))  
        if kwargs.get("viz_gen_steps", False):
            i_hat = dc_meas(i_hat, sample_mask)
            gen_prog.append((i_hat, mask.bool() | sample_mask.bool()))
        i_hat = data_consistency(i_hat_prev, i_hat, mask_prev) # DC step for previous step - keep already generated freqs
        i_hat = apply_mask(i_hat, mask) # zero out high freqs
        i_hat = dc_meas(i_hat, sample_mask) # DC to GT samples
        i_hat_prev= i_hat.clone()

    model.rm_head_caches()

    out_aux = {"gen_progress": gen_prog, "logits": logits_list, "logits_1d": logits_list_1d, "model_aux": aux, "decoded_no_dc": i_hat_no_dc}
    out_aux["codes"] = pred_idx                                   # (B, 2, K, L) the codes the token path committed to (greedy or sampled)
    if latent_no_dc is not None:   # the other decoding path as an extra arm (token path when rec_mode=latent, latent path otherwise)
        alt = token_no_dc if kwargs.get('rec_mode', 'tokens') == 'latent' else latent_no_dc
        out_aux["alt_no_dc"] = alt
        out_aux["alt_rec"] = dc_meas(alt, sample_mask)
    return i_hat, out_aux # B,H,W,2, [((B,H,W,2), (B,H,W,2)), ...x mask_levels]


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
            mask_type=kwargs.get("active_sampling_mask_type", "cartesian"),
            poisson_min_dist=kwargs.get("poisson_active_min_dist", 4.0),
        )
        sample_mask = sample_mask.permute(0,2,3,1)
        sample_progress.append(samples)
        sample_masks.append(sample_mask.clone())
        entropy_maps.append(entropy_complex)
        freq_maps.append(freq_magnitude)
        masked_image = data_consistency(gt_image, torch.zeros_like(masked_image), sample_mask) # add sampled location

    rec_image, _ = reconstruct_far(model, tokenizer, masked_image, sample_mask, **kwargs) # reconstruct

    return rec_image, {"sample_mask": sample_mask, "masked_image": masked_image, "entropy_maps": entropy_maps, "freq_maps": freq_maps, "sample_masks": sample_masks, "sample_progress": sample_progress}


@torch.no_grad
def active_sampling_posterior_sampling(model, tokenizer, gt_image: torch.Tensor, masked_image: torch.Tensor, sample_mask: torch.Tensor, budget: int, rank=1, **kwargs):
    """Posterior-sampling active acquisition.

    At each step:
      1. Draw n_samples stochastic reconstructions (top-k / temperature sampling).
      2. Zero-mean the RSS images across samples.
      3. Score each unacquired k-space column by the FFT energy of the centered samples.
      4. Acquire the highest-energy column.

    Sampling kwargs (top_k, temperature, top_p) control the stochasticity of the
    posterior samples.  Use top_k > 0 or top_p < 1 to enable non-greedy sampling.
    """
    from autoregressive.sample.adaptive_sampling import posterior_sampling_scores

    n_samples      = kwargs.get("n_samples", 8)
    sampling_kwargs = kwargs.pop("sampling_kwargs", {"temperature": 1.0, "top_k": 100, "top_p": 1.0})
    B, H, W, _    = masked_image.shape
    n_steps        = budget // rank

    sample_masks    = [sample_mask.clone()]
    sample_progress = [masked_image.clone()]
    score_maps      = []

    for _ in tqdm(range(n_steps), "posterior_sampling"):
        # Batch all n_samples together: one forward pass of size B*S
        masked_rep = masked_image.repeat(n_samples, 1, 1, 1)   # (S*B, H, W, 2)
        mask_rep   = sample_mask.repeat(n_samples, 1, 1, 1)    # (S*B, 1, W, 1)
        rec_batch, _ = reconstruct_far(model, tokenizer, masked_rep, mask_rep,
                                       sampling_kwargs=sampling_kwargs, **kwargs)
        # (S*B, H, W, 2) → (S*B, H, W) → (S, B, H, W)
        samples = transforms.complex_abs(rec_batch).view(n_samples, B, H, W)

        # Score columns per batch item
        new_mask = sample_mask.clone()
        for b in range(B):
            sel_cols, col_scores = posterior_sampling_scores(
                samples[:, b],           # (S, H, W)
                sample_mask[b:b+1],
                rank=rank,
            )
            new_mask[b, 0, sel_cols, 0] = 1.0
            score_maps.append(col_scores.cpu())

        sample_mask = new_mask
        masked_image = data_consistency(gt_image, torch.zeros_like(masked_image), sample_mask)
        sample_masks.append(sample_mask.clone())
        sample_progress.append(masked_image.clone())

    # Final deterministic reconstruction
    rec_image, _ = reconstruct_far(model, tokenizer, masked_image, sample_mask, **kwargs)

    return rec_image, {
        "sample_mask": sample_mask,
        "masked_image": masked_image,
        "sample_masks": sample_masks,
        "sample_progress": sample_progress,
        "score_maps": score_maps,
    }


@torch.no_grad()
def reconstruct_refine(model, tokenizer, masked_image: torch.Tensor, sample_mask: torch.Tensor, sampling_kwargs=None,
                       refine_rounds: int = 1, **kwargs):
    """Iterative refinement for models trained with --refine-rounds (teacher-student scheme):
    pass 1 = greedy reconstruction of the zero-filled image (timestep 0) + DC; then `refine_rounds` passes where the
    DC'd previous reconstruction is re-tokenised and fed back with timestep k (k = 1..rounds), each followed by DC.
    `sampling_kwargs` apply to the refinement passes (pass 1 is always greedy). aux["refine_pass1"] holds the pass-1 image."""
    kwargs.pop("gt_image", None)
    kwargs.pop("timestep_offset", None)
    kwargs.pop("draft_image", None)
    greedy = {"temperature": 1.0, "top_k": 0, "top_p": 1.0}
    x, aux = reconstruct_far(model, tokenizer, masked_image, sample_mask, sampling_kwargs=greedy, timestep_offset=0, **kwargs)
    pass1 = x
    two_view = bool(getattr(model, 'draft_cond', False))
    for k in range(1, int(refine_rounds) + 1):
        if two_view:   # input stays the zero-filled image; the previous reconstruction enters as the second view
            x, aux = reconstruct_far(model, tokenizer, masked_image, sample_mask, sampling_kwargs=sampling_kwargs, timestep_offset=k, draft_image=x, **kwargs)
        else:
            x, aux = reconstruct_far(model, tokenizer, x, sample_mask, sampling_kwargs=sampling_kwargs, timestep_offset=k, **kwargs)
    aux["refine_pass1"] = pass1
    return x, aux


@torch.no_grad()
def reconstruct_commit(model, tokenizer, masked_image: torch.Tensor, sample_mask: torch.Tensor, sampling_kwargs=None,
                       commit_schedule=(0.55, 0.8), commit_mode: str = "conf", **kwargs):
    """Progressive commitment for two-view models trained with --commit. Pass 1 (greedy, timestep 0) gives per-position chunk
    distributions; for each fraction f of the schedule the f most confident positions (by the expected-embedding concentration,
    previously kept ones first) are committed: their renormalised expected embeddings form the second view, zero elsewhere, and the
    model re-predicts with the timestep of f. The last pass uses `sampling_kwargs` (greedy by default); the returned image is that
    pass's reconstruction (+ DC, latent path when rec_mode='latent'). aux["commit_passes"] holds every pass after pass 1 as dicts
    main / main_no_dc / alt / alt_no_dc (main = token path, alt = latent path, unless rec_mode='latent'); aux["refine_pass1"] = pass 1."""
    from autoregressive.models.commit import expected_embedding_view, commit_keep_mask, commit_timestep
    from autoregressive.models.gpt_kar import codebook_from_quantizer
    for k in ("gt_image", "timestep_offset", "draft_image", "draft_latent", "draft_keep", "refine_rounds"):
        kwargs.pop(k, None)
    greedy = {"temperature": 1.0, "top_k": 0, "top_p": 1.0}
    codebook = model.latent_codebook if getattr(model, "latent_codebook_ready", False) else codebook_from_quantizer(tokenizer.quantizer).to(masked_image.device)
    x, aux = reconstruct_far(model, tokenizer, masked_image, sample_mask, sampling_kwargs=greedy, timestep_offset=0, **kwargs)
    pass1 = x
    view, conf = expected_embedding_view(aux["logits_1d"], codebook)
    B, L = conf.shape
    keep = torch.zeros(B, L, device=conf.device)
    committed = torch.zeros_like(view)
    passes = []
    schedule = list(commit_schedule)
    for k, f in enumerate(schedule, 1):
        new_keep = commit_keep_mask(conf, f, commit_mode, already=keep)
        newly = (new_keep > keep)[:, None, :, None]                                       # (B, 1, L, 1)
        committed = torch.where(newly, view, committed)
        keep = new_keep
        t = int(commit_timestep(f)[0])
        kw = sampling_kwargs if (k == len(schedule) and sampling_kwargs is not None) else greedy
        x, aux = reconstruct_far(model, tokenizer, masked_image, sample_mask, sampling_kwargs=kw, timestep_offset=t,
                                 draft_latent=committed, draft_keep=keep, **kwargs)
        passes.append({"main": x, "main_no_dc": aux.get("decoded_no_dc"), "alt": aux.get("alt_rec"), "alt_no_dc": aux.get("alt_no_dc"),
                       "keep": keep.clone(), "timestep": t})
        view_k, conf_k = expected_embedding_view(aux["logits_1d"], codebook)
        view = torch.where(keep.bool()[:, None, :, None], committed, view_k)
        conf = torch.where(keep.bool(), torch.full_like(conf_k, float("inf")), conf_k)
    aux["commit_passes"] = passes
    aux["refine_pass1"] = pass1
    return x, aux


@torch.no_grad()
def posterior_sampling_reconstruct(model, tokenizer, masked_image: torch.Tensor, sample_mask: torch.Tensor,
                                   sampling_kwargs=None, n_samples: int = 8, ps_average: str = "magnitude",
                                   ps_greedy_baseline: bool = False, **kwargs):
    """Posterior-sampling reconstruction ("ps_recon").

    Plain single-step reconstruction, repeated: draw ``n_samples`` stochastic reconstructions of
    the *same* masked measurements (``sampling_kwargs`` must enable sampling, e.g. top_k > 0),
    each with its own data-consistency step inside ``reconstruct_far``, and average them:
      * ps_average="magnitude": mean of |x_s|   (the quantity PSNR/SSIM are computed on)
      * ps_average="complex":   mean of x_s     (|.| is taken downstream)

    Returns ``(rec, aux)`` where ``rec`` is (B,H,W,2) such that ``transforms.complex_abs(rec)``
    is the averaged magnitude image. ``aux["ps_sample0"]`` is the first stochastic draw and, if
    ``ps_greedy_baseline``, ``aux["greedy_rec"]`` is the argmax reconstruction on the same mask.
    """
    B, H, W, _ = masked_image.shape
    kwargs.pop("gt_image", None)   # only the active-sampling variants need the ground truth

    if sampling_kwargs is None or (sampling_kwargs.get("top_k", 0) <= 0 and sampling_kwargs.get("top_p", 1.0) >= 1.0):
        print("[ps_recon] WARNING: sampling_kwargs are greedy (top_k=0, top_p=1); all samples will be identical")

    masked_rep = masked_image.repeat(n_samples, 1, 1, 1)   # (S*B, H, W, 2): identical measurements
    mask_rep   = sample_mask.repeat(n_samples, 1, 1, 1)    # (S*B, H, W, 2): identical mask
    refine_rounds = int(kwargs.pop("refine_rounds", 0) or 0)
    if refine_rounds > 0:   # refinement-trained model: greedy pass 1, sampled refinement passes
        rec_batch, rec_aux = reconstruct_refine(model, tokenizer, masked_rep, mask_rep, sampling_kwargs=sampling_kwargs,
                                                refine_rounds=refine_rounds, **kwargs)
    else:
        rec_batch, rec_aux = reconstruct_far(model, tokenizer, masked_rep, mask_rep,
                                             sampling_kwargs=sampling_kwargs, **kwargs)
    rec = rec_batch.reshape(n_samples, B, H, W, 2)         # every sample is already data-consistent

    if ps_average == "complex":
        rec_mean = rec.mean(dim=0)                                        # (B, H, W, 2)
    elif ps_average == "magnitude":
        mag_mean = transforms.complex_abs(rec).mean(dim=0)                # (B, H, W)
        rec_mean = torch.stack([mag_mean, torch.zeros_like(mag_mean)], dim=-1)
    else:
        raise ValueError(f"unknown ps_average: {ps_average}")

    aux = {"ps_sample0": rec[0]}
    # the same average over the decoded samples BEFORE data consistency ("PS (re+im)" without DC)
    no_dc = rec_aux.get("decoded_no_dc", None)
    if no_dc is not None:
        no_dc = no_dc.reshape(n_samples, B, H, W, 2)
        if ps_average == "complex":
            aux["ps_mean_no_dc"] = no_dc.mean(dim=0)
        else:
            m_no_dc = transforms.complex_abs(no_dc).mean(dim=0)
            aux["ps_mean_no_dc"] = torch.stack([m_no_dc, torch.zeros_like(m_no_dc)], dim=-1)
    if ps_greedy_baseline:
        greedy_kwargs = {"temperature": 1.0, "top_k": 0, "top_p": 1.0}
        if refine_rounds > 0:
            aux["greedy_rec"], _ = reconstruct_refine(model, tokenizer, masked_image, sample_mask, sampling_kwargs=greedy_kwargs,
                                                      refine_rounds=refine_rounds, **kwargs)
        else:
            aux["greedy_rec"], _ = reconstruct_far(model, tokenizer, masked_image, sample_mask,
                                                   sampling_kwargs=greedy_kwargs, **kwargs)
    return rec_mean, aux


@torch.no_grad()
def active_sampling_tds(
    model,
    tokenizer,
    gt_image: torch.Tensor,
    masked_image: torch.Tensor,
    sample_mask: torch.Tensor,
    budget: int,
    rank: int = 1,
    **kwargs,
):
    """Token Distribution Shift (TDS) active acquisition loop.

    At each step:
      1. reconstruct_far → logits + decoded x_hat  (decode happens here, once)
      2. tds_score_step  → simulate each unacquired line, pick best
      3. Update mask and apply data-consistency with ground-truth

    All TDS hyperparameters are passed through **kwargs with ``tds_`` prefix:
        tds_distance, tds_foreground_mode, tds_foreground_percentile,
        tds_foreground_threshold, tds_top_m, tds_alpha, tds_beta,
        tds_candidate_mode, tds_candidate_k, tds_entropy_normalization,
        tds_line_aggregation, tds_candidate_batch_size
    """
    import time as _time
    from autoregressive.sample.tds import tds_score_step

    B, H, W, _ = masked_image.shape
    device = masked_image.device
    n_steps = budget // rank

    sampling_kwargs = kwargs.get("sampling_kwargs", {"temperature": 1.0, "top_k": 0, "top_p": 1.0})

    tds_kwargs = {
        k: kwargs[k]
        for k in kwargs
        if k.startswith("tds_")
    }

    sample_masks    = [sample_mask.clone()]
    sample_progress = [masked_image.clone()]
    diagnostics_all = []

    for _step in tqdm(range(n_steps), "TDS"):
        t_step = _time.perf_counter()

        # Reconstruction + decode-once
        t_decode = _time.perf_counter()
        rec_img, aux = reconstruct_far(model, tokenizer, masked_image, sample_mask, **kwargs)
        t_decode_end = _time.perf_counter()

        logits_old = aux["logits"]         # (B, H_p, W_p, 2, K, V_K)
        x_hat      = aux["decoded_no_dc"]  # (B, H, W, 2) — before data-consistency

        y_omega = transforms.fft2(masked_image)  # (B, H, W, 2)

        result = tds_score_step(
            model, tokenizer,
            y_omega, sample_mask, x_hat, logits_old,
            device, sampling_kwargs,
            **tds_kwargs,
        )

        best_col  = result["selected_col"]
        diag      = result["diagnostics"]
        diag["runtime_decode_s"] = t_decode_end - t_decode
        diag["runtime_total_s"]  = _time.perf_counter() - t_step
        diagnostics_all.append(diag)

        # Update Cartesian mask (full column)
        new_mask = sample_mask.clone()
        new_mask[:, :, best_col, :] = 1.0
        sample_mask = new_mask

        masked_image = data_consistency(
            gt_image, torch.zeros_like(masked_image), sample_mask
        )
        sample_masks.append(sample_mask.clone())
        sample_progress.append(masked_image.clone())

    # Final reconstruction
    rec_image, _ = reconstruct_far(model, tokenizer, masked_image, sample_mask, **kwargs)

    return rec_image, {
        "sample_mask": sample_mask,
        "masked_image": masked_image,
        "sample_masks": sample_masks,
        "sample_progress": sample_progress,
        "diagnostics": diagnostics_all,
    }


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

        # 9. Select new k-space points by gradient magnitude
        grad_magnitude = gradients.norm(dim=-1) * (1 - sample_mask[..., 0])  # [B, H, W]
        grad_magnitudes.append(grad_magnitude.detach().clone())

        mask_type = kwargs.get("active_sampling_mask_type", "cartesian")

        if mask_type in ("cartesian", "1d"):
            grad_magnitude_cols = grad_magnitude.mean(dim=1)  # [B, W]
            _, top_indices = torch.topk(grad_magnitude_cols, rank, dim=1)  # [B, rank]
            new_sample_mask = torch.zeros_like(sample_mask)
            batch_indices = torch.arange(B, device=device)[:, None]
            new_sample_mask[batch_indices, :, top_indices, :] = 1
            sample_mask = torch.clamp(sample_mask + new_sample_mask, 0, 1)

        elif mask_type == "poisson":
            measured = sample_mask[..., 0].unsqueeze(1).bool()  # [B, 1, H, W]
            poisson_min_dist = kwargs.get("poisson_active_min_dist", 4.0)
            measured = _select_points_poisson_constrained(grad_magnitude, measured, rank=rank, min_dist=poisson_min_dist)
            sample_mask = measured.squeeze(1).float().unsqueeze(-1).expand(B, H, W, 2).clone()

        else:  # "2d" — top-k individual 2D point selection
            flat_scores = grad_magnitude.reshape(B, -1)
            k_sel = min(rank, H * W)
            vals, flat_idx = torch.topk(flat_scores, k=k_sel, dim=1)
            new_sample_mask = torch.zeros_like(sample_mask)
            valid = torch.isfinite(vals)
            if valid.any():
                rows = (flat_idx // W).long()
                cols = (flat_idx % W).long()
                bb = torch.arange(B, device=device)[:, None].expand(B, k_sel).reshape(-1)
                rr = rows.reshape(-1)
                cc = cols.reshape(-1)
                vv = valid.reshape(-1)
                new_sample_mask[bb[vv], rr[vv], cc[vv], :] = 1
            sample_mask = torch.clamp(sample_mask + new_sample_mask, 0, 1)
        sample_masks_list.append(sample_mask.clone())

        masked_image = data_consistency(gt_image, torch.zeros_like(masked_image), sample_mask)
        model.rm_head_caches()

    rec_image, _ = reconstruct_far(model, tokenizer, masked_image, sample_mask, **kwargs)

    return rec_image, {"sample_mask": sample_mask, "masked_image": masked_image, "grad_magnitude": grad_magnitude, "grad_magnitudes": grad_magnitudes, "sample_masks": sample_masks_list}


@torch.no_grad
def active_sampling_probe_attention(
    model, 
    tokenizer, 
    gt_image: torch.Tensor, 
    masked_image: torch.Tensor, 
    sample_mask: torch.Tensor, 
    budget: int, 
    rank=1, 
    **kwargs
):
    """
    Active Sampling loop using KspaceProbeTransformer Attention + Entropy.
    """
    B, H, W, _ = masked_image.shape
    device = masked_image.device
    
    # Setup logging similar to your old function
    n_steps = budget // rank
    sample_masks = [sample_mask.clone()]
    sample_progress = [masked_image.clone()]

    model.eval()

    for i in tqdm(range(n_steps), desc="Active Sampling (Probe)"):
        # Encode current partial image to tokens
        samples, aux = reconstruct_far(
            model,
            tokenizer,
            masked_image,
            sample_mask,
            **kwargs
        )
        # Extract critical signals
        # Shape: [B, Heads, W_probes, Image_Tokens + 1]
        probe_attn = aux["model_aux"]["probe_attention"] 
        # Shape: [B, Image_Tokens, Codebook_Size]
        image_logits = aux["logits_1d"] # B, L, 2, K, V/K  

        # 2. Select New Lines (The New Logic)
        current_mask_1d = sample_mask[:, 0, :, 0] # Extract the 1D mask from [B, H, W, 2] - >[B,W]
        
        new_mask_1d, selected_indices = get_attention_based_selection_mask(
            probe_attention=probe_attn,
            image_logits=image_logits,
            current_mask=current_mask_1d,
            num_new_lines=rank
        )
        
        # Expand 1D mask back to [B, H, W, 2]
        # (Assuming cartesian mask where columns are sampled)
        new_mask_4d = new_mask_1d.view(B, 1, W, 1).expand(B, H, W, 2)
        
        # Physics update: Add the REAL ground truth data at the new locations
        masked_image = data_consistency(gt_image, masked_image, new_mask_4d)
        sample_mask = new_mask_4d

        # 4. Logging
        sample_progress.append(masked_image.clone())
        sample_masks.append(sample_mask.clone())
        # We can log the aggregate scores as the "entropy map" visualization
        # entropy_maps.append(scores) 

    # Final Reconstruction after all budget is spent
    # (Optional: do one final high-quality pass or just return last output)
    rec_image, aux = reconstruct_far(
        model,
        tokenizer,
        masked_image,
        sample_mask,
        **kwargs
    )

    return rec_image, {
        "sample_mask": sample_mask, 
        "masked_image": masked_image, 
        "sample_masks": sample_masks, 
        "sample_progress": sample_progress
    }

def get_attention_based_selection_mask(probe_attention, image_logits, current_mask, num_new_lines):
    """
    Selects k-space lines based on Entropy-Weighted Attention.
    Handles multi-codebook logits: [B, L, 2, K, V_k]
    """
    B, Heads, Probes, Tokens = probe_attention.shape
    
    # 1. Compute Uncertainty (Total Entropy per Image Token)
    # Input: [B, Tokens, 2 (Re/Im), K (Codebooks), V_k (Vocab)]
    
    # Calculate probabilities for the vocab dimension (dim=-1)
    probs = torch.softmax(image_logits, dim=-1)
    log_probs = torch.log_softmax(image_logits, dim=-1)
    
    # Entropy of each small distribution: -Sum(p * log(p))
    # Reduces [B, L, 2, K, V_k] -> [B, L, 2, K]
    component_entropy = -torch.sum(probs * log_probs, dim=-1)
    
    # Sum over Real/Imag (dim 2) and Codebooks (dim 3) to get total token uncertainty
    # Reduces [B, L, 2, K] -> [B, L]
    token_entropy = component_entropy.sum(dim=(2, 3))

    # 2. Aggregation (Max over heads)
    # We take the maximum attention any head pays to a token
    # [B, Heads, Probes, Tokens] -> [B, Probes, Tokens]
    attn_max, _ = probe_attention.max(dim=1) 
    
    # 3. Weighted Score
    # Multiply attention by the uncertainty of the target token
    # [B, Probes, Tokens] * [B, 1, Tokens] -> Sum over Tokens -> [B, Probes]
    scores = torch.sum(attn_max * token_entropy.unsqueeze(1), dim=2)
    
    # 4. Filter & Select (Same as before)
    candidate_scores = scores.clone()
    
    # Ensure current_mask is boolean for indexing
    # If current_mask is floats (0.0/1.0), convert to bool
    is_acquired = current_mask > 0.5 
    candidate_scores[is_acquired] = -float('inf') 
    
    # Pick Top-K
    top_scores, indices = torch.topk(candidate_scores, k=num_new_lines, dim=1)
    
    # 5. Update Mask
    new_mask = current_mask.clone()
    
    # Scatter expects dim index to match src, so we usually just pass 1.0
    # src needs to be broadcastable. We can just loop or use scatter with a value.
    # scatter_(dim, index, src)
    new_mask.scatter_(dim=1, index=indices, value=1.0)
    
    return new_mask, indices