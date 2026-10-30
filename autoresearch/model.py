"""
autoresearch model (EDITABLE).

The SENSE GPT of TRUST-MRI run19 (model version 2.1, asymmetric Re/Im head, linear latent head) reduced to the one configuration
in use and sized down for the proxy budget. Parameter names match autoregressive/models/gpt_kar.py, so the full-scale
checkpoints load into build_model(depth=24, dim=1024, n_head=16) with strict=True (check_harness.py relies on this).

Input per slice: the tokenizer's continuous Re / Im latents of the zero-filled SENSE image, (B, 2, L, 64) with L = 1024
positions of the 32 x 32 token grid of the 512 tokenizer grid; the sampling-mask columns (B, 256); a timestep (B,) (0 = the
plain zero-filled input at inference).
Body: full-attention transformer over the L positions plus one timestep prefix token, 2-D rotary embeddings, the mask columns
projected and added to every position (gated), the timestep embedding added to every position (gated, zero at timestep 0).
Outputs per position: the token head predicts the K = 8 codebook codes of the Re and Im streams as a chain of K steps (Re_i and
Im_i jointly at step i, conditioned on the previous codes through per-stream tables); the latent head regresses the tokenizer
latent (2 x 64, unit-norm chunks of 8) that the decoder can consume without quantisation.
"""
from dataclasses import dataclass
from typing import Optional
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: Optional[int] = None
    n_output_layer: int = 1              # transformer blocks of the token head's chain
    multiple_of: int = 256               # SwiGLU hidden width rounded up to a multiple of this
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    num_codebooks: int = 8
    vocab_size: int = 32768              # product codebook: 8 x 4096
    num_classes: int = 8                 # timestep vocabulary
    class_dropout_prob: float = 0.1      # timestep dropout (keeps a null row in the table like the full-scale model)
    cls_token_num: int = 1               # the timestep prefix token
    block_size: int = 1024               # token positions (32 x 32)
    latent_dim: int = 64                 # width of the tokenizer latent per stream (0 = no latent head)
    mask_width: int = 256                # acquisition grid width = length of the sampling-mask column vector
    head_interleaved: bool = False       # False = asymmetric head (K steps), True = interleaved Re/Im chain (2K steps)
    cont_in_dim: int = 64                # width of the tokenizer latent per stream on the input side


def find_multiple(n: int, k: int):
    return n if n % k == 0 else n + k - (n % k)


# ----------------------------------------------------------------------------------------------------------------------
# building blocks
# ----------------------------------------------------------------------------------------------------------------------

class LabelEmbedder(nn.Module):
    """Timestep embedding table with label dropout (the extra row is the null timestep)."""

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels, train):
        if train and self.dropout_prob > 0:
            drop = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            labels = torch.where(drop, self.num_classes, labels)
        return self.embedding_table(labels).unsqueeze(1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + self.eps)).type_as(x) * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = int(2 * (4 * config.dim) / 3)
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)
        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        assert input_pos.shape[0] == k_val.shape[2]
        self.k_cache[:, :, input_pos] = k_val
        self.v_cache[:, :, input_pos] = v_val
        return self.k_cache, self.v_cache


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(self, x, freqs_cis=None, input_pos=None, mask=None, full_attn=False):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)
        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        if freqs_cis is not None:
            xq = apply_rotary_emb(xq, freqs_cis)
            xk = apply_rotary_emb(xk, freqs_cis)
        xq, xk, xv = map(lambda t: t.transpose(1, 2), (xq, xk, xv))
        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        out = F.scaled_dot_product_attention(xq, keys, values, attn_mask=mask,
                                             is_causal=(mask is None and not full_attn),
                                             dropout_p=self.attn_dropout_p if self.training else 0)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        return self.resid_dropout(self.wo(out))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x, freqs_cis=None, start_pos=None, mask=None, full_attn=False):
        h = x + self.attention(self.attention_norm(x), freqs_cis, start_pos, mask, full_attn)
        return h + self.feed_forward(self.ffn_norm(h))


# ----------------------------------------------------------------------------------------------------------------------
# token heads: per position, an autoregressive chain over the K codebooks (KV-cached at inference)
# ----------------------------------------------------------------------------------------------------------------------

class AutoRegressiveHeadAsym(nn.Module):
    """K chain steps per position; step i predicts Re_i and Im_i jointly from the body state and the previous codes of both
    streams (per-stream tables, concatenated and projected by `merge`), plus a learned chain-position embedding."""
    interleaved = False

    def __init__(self, config):
        super().__init__()
        self.num_codebooks = K = config.num_codebooks
        self.sub_vocab_size = V = config.vocab_size // K
        self.codebooks_re = nn.ModuleList([nn.Embedding(V, config.dim) for _ in range(K - 1)])
        self.codebooks_im = nn.ModuleList([nn.Embedding(V, config.dim) for _ in range(K - 1)])
        self.merge = nn.Linear(2 * config.dim, config.dim, bias=False)
        with torch.no_grad():   # == E_re + E_im (overwritten by Transformer.initialize_weights, like the full-scale model)
            self.merge.weight.copy_(torch.cat([torch.eye(config.dim), torch.eye(config.dim)], dim=1))
        self.pos_emb = nn.Parameter(torch.zeros(K, config.dim))
        self.layers = nn.ModuleList([TransformerBlock(copy.deepcopy(config)) for _ in range(config.n_output_layer)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.linear_head_re = nn.Linear(config.dim, V)
        self.linear_head_im = nn.Linear(config.dim, V)
        self.head_causal_mask = None

    def _merge(self, i, re_idx, im_idx):
        return self.merge(torch.cat([self.codebooks_re[i](re_idx), self.codebooks_im[i](im_idx)], dim=-1))

    def forward_train(self, base_tokens, targets):
        """base_tokens (B, L, C), targets (B, 2, K, L) -> logits (B, 2, K, L, V) (teacher forcing)."""
        K = self.num_codebooks
        B, L, C = base_tokens.shape
        t_re = targets[:, 0].permute(0, 2, 1).reshape(B * L, K)[:, :-1]
        t_im = targets[:, 1].permute(0, 2, 1).reshape(B * L, K)[:, :-1]
        prev = torch.stack([self._merge(i, t_re[:, i], t_im[:, i]) for i in range(K - 1)], dim=1)   # (N, K-1, C)
        h = torch.cat((base_tokens.reshape(B * L, 1, C), prev), dim=1) + self.pos_emb[None, :, :].to(base_tokens.dtype)
        for layer in self.layers:
            h = layer(h)                                                                               # causal over the chain
        h = self.norm(h)
        logits_re = self.linear_head_re(h).reshape(B, L, K, -1).permute(0, 2, 1, 3)
        logits_im = self.linear_head_im(h).reshape(B, L, K, -1).permute(0, 2, 1, 3)
        return torch.stack((logits_re, logits_im), dim=1)

    def forward_test(self, base_tokens, idx=None, input_pos=None, mask=None):
        """One KV-cached chain step. s = int(input_pos): s == 0 takes the base state (N, 1, C), s > 0 takes idx (N, 2, 1) =
        the codes of step s-1. Returns logits (N, 2, 1, V)."""
        s = int(input_pos)
        h = base_tokens if idx is None else self._merge(s - 1, idx[:, 0], idx[:, 1])
        h = h + self.pos_emb[s][None, None, :].to(h.dtype)
        for layer in self.layers:
            h = layer(h, start_pos=input_pos, mask=mask)
        h = self.norm(h)
        return torch.stack((self.linear_head_re(h), self.linear_head_im(h)), dim=1)


class AutoRegressiveHeadInterleaved(nn.Module):
    """2K chain steps per position: step 2i predicts Re_i, step 2i+1 predicts Im_i given Re_i (the variant run13 trained; 0.1 dB
    behind the asymmetric head at full scale)."""
    interleaved = True

    def __init__(self, config):
        super().__init__()
        self.num_codebooks = K = config.num_codebooks
        self.sub_vocab_size = V = config.vocab_size // K
        self.codebooks_re = nn.ModuleList([nn.Embedding(V, config.dim) for _ in range(K)])
        self.codebooks_im = nn.ModuleList([nn.Embedding(V, config.dim) for _ in range(K - 1)])
        self.pos_emb = nn.Parameter(torch.zeros(2 * K, config.dim))
        self.layers = nn.ModuleList([TransformerBlock(copy.deepcopy(config)) for _ in range(config.n_output_layer)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.linear_head_re = nn.Linear(config.dim, V)
        self.linear_head_im = nn.Linear(config.dim, V)
        self.head_causal_mask = None

    def forward_train(self, base_tokens, targets):
        K = self.num_codebooks
        B, L, C = base_tokens.shape
        t_re = targets[:, 0].permute(0, 2, 1).reshape(B * L, K)
        t_im = targets[:, 1].permute(0, 2, 1).reshape(B * L, K)
        xs = [base_tokens.reshape(B * L, C)]
        for i in range(K):
            xs.append(self.codebooks_re[i](t_re[:, i]))
            if i < K - 1:
                xs.append(self.codebooks_im[i](t_im[:, i]))
        h = torch.stack(xs, dim=1) + self.pos_emb[None, :, :].to(base_tokens.dtype)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        logits = torch.stack((self.linear_head_re(h[:, 0::2]), self.linear_head_im(h[:, 1::2])), dim=1)   # (N, 2, K, V)
        return logits.reshape(B, L, 2, K, -1).permute(0, 2, 3, 1, 4).contiguous()

    def forward_test(self, base_tokens, idx=None, input_pos=None, mask=None):
        """s == 0: base state (N, 1, C); s = 2i+1: idx = Re_i (N,); s = 2i+2: idx = Im_i. Returns logits (N, 1, V) of the stream
        predicted at step s (even = Re, odd = Im)."""
        s = int(input_pos)
        if s == 0:
            h = base_tokens
        elif s % 2 == 1:
            h = self.codebooks_re[(s - 1) // 2](idx.reshape(-1))[:, None, :]
        else:
            h = self.codebooks_im[(s - 2) // 2](idx.reshape(-1))[:, None, :]
        h = h + self.pos_emb[s][None, None, :].to(h.dtype)
        for layer in self.layers:
            h = layer(h, start_pos=input_pos, mask=mask)
        h = self.norm(h)
        return (self.linear_head_re if s % 2 == 0 else self.linear_head_im)(h)


# ----------------------------------------------------------------------------------------------------------------------
# the model
# ----------------------------------------------------------------------------------------------------------------------

class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.num_codebooks = config.num_codebooks
        self.cls_token_num = config.cls_token_num
        self.block_size = config.block_size
        self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        # input: [Re latent | Im latent] -> MLP (the quantiser is skipped on the input side)
        self.cont_in_dim = config.cont_in_dim
        self.in_proj = nn.Sequential(nn.Linear(2 * config.cont_in_dim, config.dim), nn.GELU(), nn.Linear(config.dim, config.dim))
        self.tok_dropout = nn.Dropout(config.token_dropout_p)
        self.token_ln = nn.LayerNorm(config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = AutoRegressiveHeadInterleaved(config) if config.head_interleaved else AutoRegressiveHeadAsym(config)
        self.latent_dim = int(config.latent_dim)
        self.latent_head = (nn.Sequential(RMSNorm(config.dim, eps=config.norm_eps), nn.Linear(config.dim, 2 * self.latent_dim))
                            if self.latent_dim > 0 else None)
        # timestep conditioning (zero at timestep 0) and sampling-mask conditioning, both gated
        self.t_gate = nn.Parameter(torch.full((1,), 0.5))
        self.mask_width = int(config.mask_width)
        self.mask_embed = nn.Linear(self.mask_width, config.dim)
        self.m_gate = nn.Parameter(torch.full((1,), 0.5))
        grid = int(config.block_size ** 0.5)
        assert grid * grid == config.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid, config.dim // config.n_head, config.rope_base, config.cls_token_num)
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)
        nn.init.constant_(self.output.linear_head_re.weight, 0)
        nn.init.constant_(self.output.linear_head_im.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(self, cond_latent, sample_mask_cols, timestep, targets=None, return_latent=False):
        """cond_latent (B, 2, L, Dz), sample_mask_cols (B, W) in {0, 1}, timestep (B,) long.
        targets None -> the body's hidden states (B, L, dim).
        targets (B, 2, K, L) -> (logits (B, 2, K, L, V), token loss, latent (B, 2, L, latent_dim) or None)."""
        w_dtype = self.in_proj[0].weight.dtype
        z = torch.cat((cond_latent[:, 0], cond_latent[:, 1]), dim=-1).to(w_dtype)                  # (B, L, 2 Dz)
        x = self.token_ln(self.in_proj(z))                                                           # (B, L, dim)
        scale = x.shape[-1] ** 0.5
        assert sample_mask_cols.shape[-1] == self.mask_width, f"mask width {sample_mask_cols.shape[-1]} != {self.mask_width}"
        m_vec = self.mask_embed((2.0 * sample_mask_cols.float() - 1.0).to(self.mask_embed.weight.dtype)).float()
        m_vec = F.normalize(m_vec, dim=-1) * scale                                                   # same RMS as the tokens
        x = x + (self.m_gate.float() * m_vec)[:, None, :].to(x.dtype)
        t_prefix = self.cls_embedding(timestep, train=self.training)[:, :self.cls_token_num]        # (B, 1, dim)
        e_t = self.cls_embedding.embedding_table(timestep).float()
        t_vec = F.normalize(e_t, dim=-1) * scale * (timestep > 0).float()[:, None]
        x = x + (self.t_gate.float() * t_vec)[:, None, :].to(x.dtype)
        h = self.tok_dropout(torch.cat([t_prefix, x], dim=1))                                        # (B, 1 + L, dim)
        self.freqs_cis = self.freqs_cis.to(h.device)
        for layer in self.layers:
            h = layer(h, self.freqs_cis, None, None, full_attn=True)
        h = self.norm(h)[:, self.cls_token_num:].contiguous()                                        # (B, L, dim)
        if targets is None:
            return h
        logits = self.output.forward_train(h, targets)                                               # (B, 2, K, L, V)
        V = logits.size(-1)
        loss_re = F.cross_entropy(logits[:, 0].reshape(-1, V), targets[:, 0].reshape(-1))
        loss_im = F.cross_entropy(logits[:, 1].reshape(-1, V), targets[:, 1].reshape(-1))
        loss = (loss_re + loss_im) / 2
        latent = self.predict_latent(h) if (return_latent and self.latent_head is not None) else None
        return logits, loss, latent

    def predict_latent(self, h):
        """h (B, L, dim) -> (B, 2, L, latent_dim): the regressed tokenizer latent of the Re / Im streams, each chunk of 8
        L2-normalised (the decoder consumes unit-norm chunks; see models.sense_ops.latent_to_image)."""
        z = self.latent_head(h).float()
        B, L, _ = z.shape
        z = z.view(B, L, 2, self.latent_dim).permute(0, 2, 1, 3)
        nchunk = 8 if self.latent_dim % 8 == 0 else 1
        return F.normalize(z.reshape(B, 2, L, nchunk, -1), dim=-1).reshape(B, 2, L, self.latent_dim)

    # KV caches of the token head's chain (inference)
    def setup_head_caches(self, max_batch_size, max_seq_length, dtype, device):
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length, self.max_batch_size = max_seq_length, max_batch_size
        for b in self.output.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype).to(device)
        causal = torch.tril(torch.ones(max_seq_length, max_seq_length, dtype=torch.bool, device=device))
        self.output.head_causal_mask = causal.unsqueeze(0).repeat(max_batch_size, 1, 1)

    def rm_head_caches(self):
        for b in self.output.layers:
            b.attention.kv_cache = None
        self.output.head_causal_mask = None
        self.max_seq_length = self.max_batch_size = -1


# ----------------------------------------------------------------------------------------------------------------------
# 2-D rotary embeddings (gpt-fast)
# ----------------------------------------------------------------------------------------------------------------------

def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, zero_prefix_len: int = 0):
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_grid = torch.cat([freqs[:, None, :].expand(-1, grid_size, -1), freqs[None, :, :].expand(grid_size, -1, -1)], dim=-1)
    cache = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1).flatten(0, 1)   # (grid^2, n_elem/2, 2)
    return torch.cat([torch.zeros(zero_prefix_len, n_elem // 2, 2), cache])


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack([xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
                          xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1]], dim=-1)
    return x_out2.flatten(3).type_as(x)


def build_model(depth: int, dim: int, n_head: int, **kwargs) -> Transformer:
    """depth / dim / n_head of the body; the full-scale line is GPT-L = (24, 1024, 16), GPT-B = (12, 768, 12)."""
    return Transformer(ModelArgs(n_layer=depth, dim=dim, n_head=n_head, **kwargs))


def num_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
