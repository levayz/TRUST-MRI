"""
autoresearch sampler (EDITABLE): how a trained model turns the zero-filled SENSE input into the scored reconstruction.

prepare.evaluate calls reconstruct(model, tokenizer, x_in, mask_cols, dc_fn) per validation batch:
    x_in      (B, 512, 512, 2) zero-filled SENSE image on the token grid (Re / Im, max-normalised)
    mask_cols (B, 256) the sampled k-space columns
    dc_fn     per-coil data consistency on token-grid images: dc_fn(img (B, 512, 512, 2)) -> (B, 512, 512, 2)
and expects (rec, arms): rec is scored as val_psnr, every image in arms is scored too and reported as arm_<name>_psnr.

Baseline = the single-pass inference of run19: one body pass at timestep 0, greedy roll-out of the token head, decode the
codes and the regressed latent with the frozen tokenizer decoder, per-coil DC on both; the latent path is scored (it is
0.7 dB ahead of the token path at full scale), the token path is reported as an arm.
"""
import torch

from models.sense_ops import latent_target, decode_latent_pair

REC_MODE = "latent"                                    # "latent" | "tokens": which decoding path is scored
SAMPLING = dict(temperature=1.0, top_k=0, top_p=1.0)   # token head sampling; top_k 0 and top_p 1 = greedy argmax


@torch.no_grad()
def reconstruct(model, tokenizer, x_in, mask_cols, dc_fn):
    B = x_in.shape[0]
    cond_latent = torch.stack([latent_target(tokenizer, x_in[..., c].float()) for c in range(2)], dim=1)   # (B, 2, L, 64)
    t = torch.zeros(B, dtype=torch.long, device=x_in.device)
    h = model(cond_latent=cond_latent, sample_mask_cols=mask_cols, timestep=t)                            # (B, L, dim)
    codes = rollout_token_head(model, h, **SAMPLING)                                                        # (B, 2, K, L)
    arms = {"tokens": dc_fn(decode_codes(tokenizer, codes))}
    if model.latent_head is not None:
        arms["latent"] = dc_fn(decode_latent_pair(tokenizer, model.predict_latent(h)))
    mode = REC_MODE if REC_MODE in arms else "tokens"
    rec = arms.pop(mode)
    return rec, arms


def rollout_token_head(model, h, temperature=1.0, top_k=0, top_p=1.0):
    """KV-cached roll-out of the token head's chain at every position. h (B, L, D) -> codes (B, 2, K, L) long."""
    B, L, D = h.shape
    K = model.num_codebooks
    interleaved = bool(getattr(model.output, "interleaved", False))
    model.setup_head_caches(B * L, K * (2 if interleaved else 1), dtype=h.dtype, device=h.device)
    hb = h.reshape(B * L, 1, D)
    if interleaved:   # 2K steps: Re_i at even steps, Im_i (given Re_i) at odd steps
        codes, prev = [[], []], None
        for s in range(2 * K):
            pos = torch.tensor([s], dtype=torch.int)
            mask = model.output.head_causal_mask[:B * L, None, pos]
            logits = model.output.forward_test(hb if s == 0 else None, idx=prev, input_pos=pos, mask=mask)   # (N, 1, V)
            prev = sample_codes(logits[:, -1], temperature, top_k, top_p)                                     # (N,)
            codes[s % 2].append(prev)
        out = torch.stack([torch.stack(codes[0], 1), torch.stack(codes[1], 1)], 1)                            # (N, 2, K)
    else:             # K steps: Re_i and Im_i jointly at step i
        pred = []
        for i in range(K):
            pos = torch.tensor([i], dtype=torch.int)
            mask = model.output.head_causal_mask[:B * L, None, pos]
            logits = model.output.forward_test(hb if i == 0 else None, idx=None if i == 0 else pred[-1], input_pos=pos, mask=mask)
            pred.append(sample_codes(logits[:, :, -1], temperature, top_k, top_p).unsqueeze(-1))             # (N, 2, 1)
        out = torch.cat(pred, dim=2)                                                                            # (N, 2, K)
    model.rm_head_caches()
    return out.view(B, L, 2, K).permute(0, 2, 3, 1).contiguous()


def sample_codes(logits, temperature=1.0, top_k=0, top_p=1.0):
    """logits (..., V) -> codes (...): greedy argmax unless top_k > 0 or top_p < 1 (then multinomial after filtering)."""
    if top_k <= 0 and top_p >= 1.0:
        return logits.argmax(dim=-1)
    shape = logits.shape[:-1]
    lg = logits.reshape(-1, logits.shape[-1]).float() / max(temperature, 1e-5)
    if top_k > 0:
        kth = torch.topk(lg, min(top_k, lg.shape[-1]))[0][..., -1, None]
        lg = lg.masked_fill(lg < kth, float("-inf"))
    if top_p < 1.0:
        sorted_lg, idx = torch.sort(lg, descending=True)
        cum = torch.cumsum(torch.softmax(sorted_lg, dim=-1), dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        lg = lg.masked_fill(remove.scatter(1, idx, remove), float("-inf"))
    return torch.multinomial(torch.softmax(lg, dim=-1), 1).reshape(shape)


def decode_codes(tokenizer, codes):
    """codes (B, 2, K, L) -> (B, 512, 512, 2): the frozen tokenizer decoder on each stream (RGB output averaged to one channel)."""
    imgs = []
    for s in range(2):
        f = tokenizer.post_quant_proj(tokenizer.quantizer.idx_to_f(codes[:, s]))
        imgs.append(tokenizer.decoder(f).clamp(-1, 1).float().mean(1))
    return torch.stack(imgs, dim=-1)
