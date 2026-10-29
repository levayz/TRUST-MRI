
# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py  
#   nanoKAR_GPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple
import sys
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from .quant import VectorQuantizerM, AttnProjection
import copy
from data import transforms



def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(torch.nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)


def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    n_output_layer: int = 1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02

    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_codebooks: int = 8
    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = 'c2i'

    vocab_size: int = 16384
    cls_token_num: int = 1
    cond_token_num: int = 0  # number of conditioning prefix tokens
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048
    vq_embed_path: str = None
    latent_dim: int = 0  # >0: regression head predicting the tokenizer's continuous latent (Re+Im per position) next to the token head
    draft_cond: bool = False  # second conditioning view for refinement: tokens of the previous (DC'd) reconstruction, added to the zero-filled view
    model_version: float = 1  # 1: legacy GPT (prefix timestep token only; norm ~1 vs ~32 for the tokens -> invisible in practice)
                            # 2: v1 + timestep conditioning (RMS-scaled embedding added to every conditioning token, gated, zero at timestep 0)
                            #        + sampling-mask conditioning (mask column vector -> linear -> RMS-scaled -> gated -> added to every token)
                            # 2.1: v2 + continuous input (tokenizer latents of Re|Im -> MLP, no input tokens) + interleaved Re/Im output head
    mask_width: int = 256   # v2: width of the acquisition grid = length of the sampling-mask column vector
    head_interleaved: bool = True   # v2.1 head: True = interleaved Re/Im chain (2K steps), False = parallel asymmetric head (K steps)
    cont_in_dim: int = 64   # v2.1: width of the tokenizer's continuous latent per stream (the input is [Re latent, Im latent] -> MLP, no quantisation)
    latent_head_type: str = 'linear'   # latent head: 'linear' = RMSNorm -> Linear; 'mlp' = + residual MLP on the same hidden state;
                                       # 'fused' = + residual MLP on [hidden state, token head's chain state, codebook latents of the
                                       # token head's greedy codes] (the latent head sees what the token head committed to)
    latent_mlp_width: int = 0          # hidden width of the latent MLP (0 = 2 * dim)

    block_type: str = None # default self attn block, supports also complex

#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


#################################################################################
#                      Embedding Layers for Text Feature                        #
#################################################################################
class CaptionEmbedder(nn.Module):
    """
    Embeds text caption into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()
        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        embeddings = self.cap_proj(caption)
        return embeddings


def load_unitok_embeddings(p):
    ckpt = torch.load(p, map_location='cpu')['trainer']['unitok']
    keys = list(ckpt.keys())
    useful_keys = [k for k in keys if 'quantizer' in k or 'post_quant_proj' in k]
    useful_params = {k:ckpt[k] for k in useful_keys}
    del ckpt
    return useful_params


class CustomTokenEmbedder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # hard coding for unitok, need to be fixed
        self.num_codebooks = 8
        self.quantizer = VectorQuantizerM(32768, 64, 0.25, False, 0.01, 8)
        self.post_quant_proj = AttnProjection(64, 1024, 16)
        self.mm_projector = nn.Sequential(
            nn.LayerNorm(1024, eps=1e-6),
            nn.Linear(1024, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, indices):  # input [bz,num-codebook,256]
        assert indices.shape[1] == self.num_codebooks
        features = self.quantizer.idx_to_f(indices)  # [bz,256,C]
        features = self.post_quant_proj(features)  # [bz,256,1024]
        latent_features = self.mm_projector(features)  # [bz,256,hidden_size]
        return latent_features  # [bz,256,hidden_size]


class TokenEmbedder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_codebooks):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebooks = nn.ModuleList()
        for _ in range(num_codebooks):
            codebook = nn.Embedding(vocab_size // num_codebooks, hidden_size)
            self.codebooks.append(codebook)

    def forward(self, indices):
        assert indices.shape[1] == self.num_codebooks
        latent_features = []
        for i in range(self.num_codebooks):
            latent_feature = self.codebooks[i](indices[:, i])
            latent_features.append(latent_feature)
        latent_features = torch.stack(latent_features).sum(dim=0)
        return latent_features


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                                  KAR_GPT Model                                    #
#################################################################################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
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
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor = None,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        full_attn: Optional[bool] = False,
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)

        if freqs_cis is not None:
            xq = apply_rotary_emb(xq, freqs_cis)
            xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        output = F.scaled_dot_product_attention(
            xq, keys, values,
            attn_mask=mask,
            is_causal=True if (mask is None and full_attn == False) else False,  # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output

class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.n_head = config.n_head
        self.head_dim = config.dim // config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head

        kv_size = self.n_kv_head * self.head_dim

        self.wq = nn.Linear(config.dim, config.dim, bias=False)
        self.wkv = nn.Linear(config.dim, 2 * kv_size, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)

        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
        self,
        x1: torch.Tensor,  # (B, Q, D) query source
        x2: torch.Tensor,  # (B, K, D) key/value source
        freqs_cis_q: Optional[torch.Tensor] = None,
        freqs_cis_k: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,  # broadcastable to (B, n_head, Q, K)
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        bsz, qlen, d1 = x1.shape
        _, klen, d2 = x2.shape
        assert d1 == self.dim and d2 == self.dim

        kv_size = self.n_kv_head * self.head_dim

        xq = self.wq(x1).view(bsz, qlen, self.n_head, self.head_dim)
        xk, xv = self.wkv(x2).split([kv_size, kv_size], dim=-1)
        xk = xk.view(bsz, klen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, klen, self.n_kv_head, self.head_dim)

        if freqs_cis_q is not None:
            xq = apply_rotary_emb(xq, freqs_cis_q)
        if freqs_cis_k is not None:
            xk = apply_rotary_emb(xk, freqs_cis_k)

        xq = xq.transpose(1, 2)  # (B, n_head, Q, hd)
        xk = xk.transpose(1, 2)  # (B, n_kv_head, K, hd)
        xv = xv.transpose(1, 2)  # (B, n_kv_head, K, hd)

        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            xk = xk.repeat_interleave(repeat, dim=1)
            xv = xv.repeat_interleave(repeat, dim=1)

        if return_attention_weights:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights_dropped = F.dropout(attn_weights, p=self.attn_dropout_p, training=self.training)
            out = attn_weights_dropped @ xv
            out = out.transpose(1, 2).contiguous().view(bsz, qlen, self.dim)
            return self.resid_dropout(self.wo(out)), attn_weights
        else:
            out = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=mask,
                is_causal=False,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
            )
            out = out.transpose(1, 2).contiguous().view(bsz, qlen, self.dim)
            return self.resid_dropout(self.wo(out))



class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None,
                full_attn=False):
        h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis, start_pos, mask, full_attn))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class TransformerCrossAttentionBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()
        self.attention = CrossAttention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, freqs_cis_q: torch.Tensor, freqs_cis_k: torch.Tensor, return_attn_weights=False):
        if return_attn_weights:
            attn_out, attn_weights = self.attention(
                self.attention_norm(x1), 
                self.attention_norm(x2), 
                freqs_cis_q, 
                freqs_cis_k,
                return_attention_weights=True
            )
            h = x1 + self.drop_path(attn_out)
            out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
            return out, attn_weights
        else:
            h = x1 + self.drop_path(self.attention(self.attention_norm(x1), self.attention_norm(x2), freqs_cis_q, freqs_cis_k))
            out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
            return out


class TransformerComplexBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()
        self.re_self_attn_block = TransformerBlock(config, drop_path) 
        self.re_cross_attn_block = TransformerCrossAttentionBlock(config, drop_path)

        self.im_self_attn_block = TransformerBlock(config, drop_path) 
        self.im_cross_attn_block = TransformerCrossAttentionBlock(config, drop_path)


    def forward(self, x: torch.Tensor, freqs_cis_q: torch.Tensor, freqs_cis_k: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None,
                full_attn=True, return_attn_weights=False):
        """
        x: (B, 2, L, D) - x[:,0] Re(IMG) or |IMG|, x[:,1] Im(IMG) or angle(IMG)
        """
        re = x[:,0]
        im = x[:, 1]

        h_re = self.re_self_attn_block(re, freqs_cis_q, start_pos, mask, full_attn)
        h_im = self.im_self_attn_block(im, freqs_cis_q, start_pos, mask, full_attn)

        if return_attn_weights:
            h_re_x, attn_weights_re = self.re_cross_attn_block(h_re, h_im, freqs_cis_q, freqs_cis_k, return_attn_weights=True)
            h_im_x, attn_weights_im = self.im_cross_attn_block(h_im, h_re, freqs_cis_q, freqs_cis_k, return_attn_weights=True)
            h = torch.stack([h_re_x, h_im_x], dim=1)
            return h, (attn_weights_re, attn_weights_im)
        else:
            h_re_x = self.re_cross_attn_block(h_re, h_im, freqs_cis_q, freqs_cis_k)
            h_im_x = self.im_cross_attn_block(h_im, h_re, freqs_cis_q, freqs_cis_k)
            h = torch.stack([h_re_x, h_im_x], dim=1)
            return h


class AutoRegressiveHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_codebooks = config.num_codebooks
        self.sub_vocab_size = config.vocab_size // config.num_codebooks

        self.codebooks = nn.ModuleList()
        for _ in range(self.num_codebooks - 1):
            codebook = nn.Embedding(self.sub_vocab_size, config.dim)
            self.codebooks.append(codebook)

        self.layers = torch.nn.ModuleList()

        config_clone = copy.deepcopy(config)
        # config_clone.dim = config_clone.dim * 2
        for _ in range(config.n_output_layer):
            self.layers.append(TransformerBlock(config_clone, drop_path=0.))

        self.norm = RMSNorm(config_clone.dim, eps=config.norm_eps)
        self.linear_head_re = nn.Linear(config_clone.dim, self.sub_vocab_size)
        self.linear_head_im = nn.Linear(config_clone.dim, self.sub_vocab_size)

    def forward_train(self, base_tokens, targets):
        # base_tokens[B, 1+(Li-1), 768], targets[B, 2 K, Li]

        K = targets.shape[2]
        B, L, C = base_tokens.shape
        base_tokens = base_tokens.reshape(B * L, 1, C)  # [B*(1+(Li-1)), 1, 768]

        targets_re = targets[:, 0, :, :]
        targets_re = targets_re.permute(0, 2, 1).reshape(B * L, K)[:, :-1]  # [B*Li, K-1]

        targets_im = targets[:, 1, :, :]
        targets_im = targets_im.permute(0, 2, 1).reshape(B * L, K)[:, :-1]  # [B*Li, K-1]
        
        index_embeddings_re = []
        index_embeddings_im = []
        for i in range(self.num_codebooks - 1):
            index_embed_re = self.codebooks[i](targets_re[:, i])
            index_embed_im = self.codebooks[i](targets_im[:, i])
            index_embeddings_re.append(index_embed_re)
            index_embeddings_im.append(index_embed_im)
        index_embeddings_re = torch.stack(index_embeddings_re, dim=1)  # [B*Li, K-1, 768]
        index_embeddings_im = torch.stack(index_embeddings_im, dim=1)  # [B*Li, K-1, 768]

        # concat along channel dimension TODO if concat then cant concat the base tokens
        # index_embeddings = torch.cat((index_embeddings_re, index_embeddings_im), dim=-1)  # [B*Li, K-1, 1536]
        index_embeddings = index_embeddings_re + index_embeddings_im # TODO maybe something better, add NORM followed by projection? conct and project back to 768?
        h = torch.cat((base_tokens, index_embeddings), dim=1)  # [B*(1+(Li-1)), 1+(K-1), 768]
        for layer in self.layers:
            h = layer(h, freqs_cis=None, start_pos=None, mask=None, full_attn=False)
        h = self.norm(h)  # [B*(1+(Li-1)), 1+(K-1), 768]
        logits_re = self.linear_head_re(h)  # [B*(1+(Li-1)), 1+(K-1), V/K]. Next codebook prediction??
        logits_re = logits_re.reshape(B, L, K, -1).permute(0, 2, 1, 3)  # [B, 1+(K-1), 1+(Li-1), V/K]

        logits_im = self.linear_head_im(h)  # [B*(1+(Li-1)), 1+(K-1), V/K]. Next codebook prediction??
        logits_im = logits_im.reshape(B, L, K, -1).permute(0, 2, 1, 3)  # [B, 1+(K-1), 1+(Li-1), V/K]
        logits = torch.stack((logits_re, logits_im), dim=1)  # [B, 2, 1+(K-1), 1+(Li-1), V/K]
        return logits

    def forward_test(self, base_tokens, idx=None, input_pos=None, mask=None, return_state=False):
        if idx is not None:
            re_idx = idx[:,0]
            im_idx = idx[:,1]
            h_re = self.codebooks[input_pos - 1](re_idx)
            h_im = self.codebooks[input_pos - 1](im_idx)
            h = h_re + h_im # TODO add norm?
        else:
            h = base_tokens
        
        for layer in self.layers:
            h = layer(h, freqs_cis=None, start_pos=input_pos, mask=mask)
        h = self.norm(h)

        logits_re = self.linear_head_re(h)
        logits_im = self.linear_head_im(h)  # [B*(1+(Li-1)), 1, V/K]. Next codebook prediction

        logits = torch.stack((logits_re, logits_im), dim=1)  # [B*L, 2, 1, V/K] here K=1 because of KV cache
        return (logits, h) if return_state else logits

    @torch.no_grad()
    def rollout(self, base):
        """Greedy roll-out without KV caches (training-time use). base (N, C) -> codes (N, 2, K) long and the chain state (N, C):
        the mean over the K chain steps of the normalised hidden state (the quantity forward_test returns with return_state=True)."""
        K = self.num_codebooks
        xs, re, im, states = [base], [], [], []
        for s in range(K):
            h = torch.stack(xs, dim=1)
            for layer in self.layers:
                h = layer(h, freqs_cis=None, start_pos=None, mask=None, full_attn=False)
            h = self.norm(h)[:, -1]
            states.append(h)
            re.append(self.linear_head_re(h).argmax(-1)); im.append(self.linear_head_im(h).argmax(-1))
            if s < K - 1:
                xs.append(self.codebooks[s](re[-1]) + self.codebooks[s](im[-1]))
        return torch.stack([torch.stack(re, 1), torch.stack(im, 1)], dim=1), torch.stack(states, 1).float().mean(1)


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes # this acts as a number of timesteps (mask index)
        self.model_type = config.model_type
        self.num_codebooks = config.num_codebooks
        self.cls_token_num = config.cls_token_num # this is for the timestep embedding tokens
        self.cond_token_num = config.cond_token_num
        self.model_version = float(getattr(config, 'model_version', 1) or 1)
        if self.model_type == 'c2i':
            self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        elif self.model_type == 't2i':
            self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob)
        else:
            raise Exception("please check model type")

        vq_embed_path = getattr(config, 'vq_embed_path', False)
        if vq_embed_path:
            self.re_tok_embeddings = CustomTokenEmbedder(config.dim)
            self.im_tok_embeddings = CustomTokenEmbedder(config.dim)
            print(f"Loading embeddings from: {vq_embed_path}")
            missing_keys, unexpected_keys = self.tok_embeddings.load_state_dict(load_unitok_embeddings(vq_embed_path), strict=False)
            print(f"---missing keys: {missing_keys}\n---unexpected keys: {unexpected_keys}")
        elif self.model_version >= 2.1:
            # v2.1 input: the tokenizer's continuous Re / Im latents (unit-norm codebook chunks), concatenated along the channel
            # dimension and projected by an MLP; the quantiser is skipped on the input side
            self.cont_in_dim = int(getattr(config, 'cont_in_dim', 64))
            self.in_proj = nn.Sequential(nn.Linear(2 * self.cont_in_dim, config.dim), nn.GELU(), nn.Linear(config.dim, config.dim))
        else:
            # self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
            self.re_tok_embeddings = TokenEmbedder(config.vocab_size, config.dim, config.num_codebooks) 
            self.im_tok_embeddings = TokenEmbedder(config.vocab_size, config.dim, config.num_codebooks) 
        if self.cond_token_num == 0:
            self.cond_proj = nn.Sequential(
                nn.Linear(config.dim, config.dim),
                nn.GELU(),
                nn.Linear(config.dim, config.dim),
            )
        self.tok_dropout = nn.Dropout(config.token_dropout_p)
        self.token_ln = nn.LayerNorm(config.dim)

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            block_class = TransformerComplexBlock if self.config.block_type == "complex" else TransformerBlock
            self.layers.append(block_class(config, dpr[layer_id]))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        if self.model_version >= 2.1:
            self.output = AutoRegressiveHeadInterleaved(config) if getattr(config, 'head_interleaved', True) else AutoRegressiveHeadAsym(config)
        else:
            self.output = AutoRegressiveHead(config)
        self.latent_dim = int(getattr(config, 'latent_dim', 0) or 0)
        self.latent_head = (nn.Sequential(RMSNorm(config.dim, eps=config.norm_eps), nn.Linear(config.dim, 2 * self.latent_dim))
                            if self.latent_dim > 0 else None)
        self.latent_head_type = str(getattr(config, 'latent_head_type', 'linear') or 'linear')
        assert self.latent_head_type in ('linear', 'mlp', 'fused'), self.latent_head_type
        self.latent_mlp = None
        if self.latent_dim > 0 and self.latent_head_type != 'linear':
            width = int(getattr(config, 'latent_mlp_width', 0) or 0) or 2 * config.dim
            fused = self.latent_head_type == 'fused'
            in_dim = config.dim + (config.dim + 2 * self.latent_dim if fused else 0)
            # residual branch on the linear readout; its last layer is zero-initialised (initialize_weights) so a warm start from a
            # linear-head checkpoint reproduces that model exactly at step 0
            self.latent_mlp = nn.Sequential(nn.Linear(in_dim, width), nn.GELU(), nn.Linear(width, 2 * self.latent_dim))
            if fused:
                self.latent_fuse_ln = RMSNorm(config.dim, eps=config.norm_eps)
                K = config.num_codebooks
                # the tokenizer's codebook rows (K, V/K, d), set by set_latent_codebook; not part of the checkpoint
                self.register_buffer('latent_codebook', torch.zeros(K, config.vocab_size // K, self.latent_dim // K), persistent=False)
                self.latent_codebook_ready = False
        if self.model_version >= 2:
            # timestep conditioning (the additive vector is zero at timestep 0, so a warm-started pass 1 is unchanged)
            self.t_gate = nn.Parameter(torch.full((1,), 0.5))
            # sampling-mask conditioning
            self.mask_width = int(config.mask_width)
            self.mask_embed = nn.Linear(self.mask_width, config.dim)
            self.m_gate = nn.Parameter(torch.full((1,), 0.5))
        self.draft_cond = bool(getattr(config, 'draft_cond', False))
        if self.draft_cond:   # two-view conditioning: zero-filled tokens + (gated) tokens of the previous reconstruction, same embedders
            self.draft_ln = RMSNorm(config.dim, eps=config.norm_eps)
            self.draft_gate = nn.Parameter(torch.full((1,), 0.5))

        # 2d rotary pos embedding
        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        prefix_len = self.cls_token_num + self.cond_token_num
        self.freqs_cis = precompute_freqs_cis_2d(
            grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num
        )

        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)
        # exit()

        # Zero-out output layers:
        nn.init.constant_(self.output.linear_head_re.weight, 0)
        nn.init.constant_(self.output.linear_head_im.weight, 0)
        if getattr(self, 'latent_mlp', None) is not None:   # residual latent MLP starts as a no-op
            nn.init.zeros_(self.latent_mlp[-1].weight)
            nn.init.zeros_(self.latent_mlp[-1].bias)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def setup_caches(self, max_batch_size, max_seq_length, dtype, device=None):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
        #     return
        device = device if device is not None else None
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            kv = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)
            b.attention.kv_cache = kv.to(device) if device is not None else kv

        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        prefix_len = self.cls_token_num + self.cond_token_num
        self.freqs_cis = precompute_freqs_cis_2d(
            grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num, self.cond_token_num,
        )
        self.freqs_cis = self.freqs_cis.to(device) if device is not None else self.freqs_cis 

    def setup_head_caches(self, max_batch_size, max_seq_length, dtype, device):
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length # TODO probably should be self.output.max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.output.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype).to(
                device)
        causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)).to(device)
        self.output.head_causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)

    def rm_head_caches(self):
        for b in self.output.layers:
            b.attention.kv_cache = None
        self.output.head_causal_mask = None
        self.max_seq_length = -1
        self.max_batch_size = -1

    def forward(
            self,
            cond_tokens: torch.Tensor = None,  # [B, 2, K, L] (model v1 input)
            soft_cond_embeddings: torch.Tensor = None, # used for STE [B,2,L,dim]
            timestep: torch.Tensor = None, 
            input_pos: Optional[torch.Tensor] = None,
            targets: Optional[torch.Tensor] = None, # [B, 2, K, L]
            mask: Optional[torch.Tensor] = None,
            valid: Optional[torch.Tensor] = None,
            return_latent: bool = False,
            draft_tokens: torch.Tensor = None,  # [B, 2, K, L] tokens of the previous reconstruction (refinement passes, --draft-cond)
            sample_mask_cols: torch.Tensor = None,  # [B, W] sampling-mask columns of the acquisition grid (model v2)
            cond_latent: torch.Tensor = None,  # [B, 2, L, Dz] continuous encoder latents of the input (model v2.1)
            draft_latent: torch.Tensor = None,  # [B, 2, L, Dz] latents of the previous reconstruction (model v2.1 two-view refinement)
            draft_keep: torch.Tensor = None,  # [B, L] progressive commitment: 1 = the draft view is committed at this position, 0 = no draft there
            loss_mask: torch.Tensor = None,  # [B, L] per-position weights of the token loss (e.g. the positions not revealed by the draft view)
            **kwargs,
    ):
        assert self.cond_token_num > 0, "cond_token_num must be greater than 0"
        if getattr(self, 'model_version', 1) >= 2.1:
            assert cond_latent is not None, "model v2.1 takes continuous encoder latents: cond_latent (B, 2, L, D)"
            w_dtype = self.in_proj[0].weight.dtype
            z = torch.cat((cond_latent[:, 0], cond_latent[:, 1]), dim=-1).to(w_dtype)              # (B, L, 2D): Re | Im along channels
            cond_embeddings = self.token_ln(self.in_proj(z))                                         # [B, Lc, dim]
            if draft_latent is not None:
                assert getattr(self, 'draft_cond', False), "draft_latent given but the model was built without draft_cond"
                zd = torch.cat((draft_latent[:, 0], draft_latent[:, 1]), dim=-1).to(w_dtype)
                d = self.draft_ln(self.in_proj(zd))
                if draft_keep is not None:   # partial draft: dropped positions see the zero-filled view only
                    d = d * draft_keep.to(d.dtype)[..., None]
                cond_embeddings = cond_embeddings + (self.draft_gate.float() * d.float()).to(cond_embeddings.dtype)
            cond_embed_re = cond_embed_im = None
        else:
            cond_tokens_re = cond_tokens[:, 0, :, :]
            cond_tokens_im = cond_tokens[:, 1, :, :]
            cond_embed_re = self.re_tok_embeddings(cond_tokens_re)  # [B, Lc, dim]
            cond_embed_im = self.im_tok_embeddings(cond_tokens_im)  # [B, Lc, dim]

        if soft_cond_embeddings is not None:
            soft_cond_embed_re = soft_cond_embeddings[:, 0]
            soft_cond_embed_im = soft_cond_embeddings[:, 1]

            cond_embed_re = soft_cond_embed_re + (cond_embed_re - soft_cond_embed_re).detach()
            cond_embed_im = soft_cond_embed_im + (cond_embed_im - soft_cond_embed_im).detach()
        
        if cond_embed_re is not None:   # v1 token path
            cond_embeddings = self.token_ln(cond_embed_re + cond_embed_im) # [B, Lc, dim]
        if draft_tokens is not None:
            assert getattr(self, 'draft_cond', False), "draft_tokens given but the model was built without draft_cond"
            d = self.draft_ln(self.re_tok_embeddings(draft_tokens[:, 0]) + self.im_tok_embeddings(draft_tokens[:, 1]))   # [B, Lc, dim]
            if draft_keep is not None:
                d = d * draft_keep.to(d.dtype)[..., None]
            cond_embeddings = cond_embeddings + (self.draft_gate.float() * d.float()).to(cond_embeddings.dtype)
        if getattr(self, 'model_version', 1) >= 2:
            assert sample_mask_cols is not None, "model v2 needs sample_mask_cols (B, W)"
            assert sample_mask_cols.shape[-1] == self.mask_width, f"mask width {sample_mask_cols.shape[-1]} != model mask_width {self.mask_width}"
            m_vec = self.mask_embed((2.0 * sample_mask_cols.float() - 1.0).to(self.mask_embed.weight.dtype)).float()   # (B, dim)
            m_vec = F.normalize(m_vec, dim=-1) * (cond_embeddings.shape[-1] ** 0.5)                               # same RMS as the tokens
            cond_embeddings = cond_embeddings + (self.m_gate.float() * m_vec)[:, None, :].to(cond_embeddings.dtype)

        if timestep is not None and self.cls_token_num > 0:
            t_embeddings = self.cls_embedding(timestep, train=self.training)[:, :self.cls_token_num]
            if getattr(self, 'model_version', 1) >= 2:
                # the prefix token alone is invisible (norm ~1 among 1024 tokens of norm ~32): also add the timestep embedding,
                # rescaled to the tokens' RMS, to every conditioning position (gated, gate init 0)
                # (no label dropout here; zero for timestep 0 so pass 1 of a warm-started model is bit-identical to before)
                e_t = self.cls_embedding.embedding_table(timestep).float()                                   # (B, dim)
                t_vec = F.normalize(e_t, dim=-1) * (cond_embeddings.shape[-1] ** 0.5) * (timestep > 0).float()[:, None]
                cond_embeddings = cond_embeddings + (self.t_gate.float() * t_vec)[:, None, :].to(cond_embeddings.dtype)
            cond_embeddings = torch.cat([t_embeddings, cond_embeddings], dim=1) # [B, 1+Lm, dim]

        h = self.tok_dropout(cond_embeddings)
        self.freqs_cis = self.freqs_cis.to(h.device)

        freqs_cis = self.freqs_cis

        # transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, mask, full_attn=True)

        # output layers
        h = self.norm(h)
        # if self.training:
        h = h[:, self.cls_token_num:].contiguous()

        if targets is None:
            return h
        else:
            logits = self.output.forward_train(h, targets=targets)  # [B,2, 1+(K-1), 1+(Li-1), V/K]
        latent = None
        if return_latent and self.latent_head is not None:
            if self.latent_head_type == 'fused':   # greedy roll-out of the token head (no grad): the codes the token path commits to
                codes, state = self.rollout_head(h)
                latent = self.predict_latent(h, head_state=state, head_codes=codes)
            else:
                latent = self.predict_latent(h)

        # if we are given some desired targets also calculate the loss
        loss = None
        if valid is not None:
            # loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            # valid_all = valid[:, None].repeat(1, targets.shape[1]).view(-1)
            # loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
            pass
        elif targets is not None and loss_mask is not None:
            # weighted token loss: rows of logits[:, s].reshape(-1, V) are ordered (B, K, L), like targets[:, s]
            V = logits.size(-1)
            w = loss_mask.float()[:, None, :].expand(-1, targets.shape[2], -1).reshape(-1)
            l_re = F.cross_entropy(logits[:, 0, :, :, :].reshape(-1, V), targets[:, 0].reshape(-1), reduction='none')
            l_im = F.cross_entropy(logits[:, 1, :, :, :].reshape(-1, V), targets[:, 1].reshape(-1), reduction='none')
            loss = ((l_re + l_im) * w).sum() / (2 * w.sum().clamp_min(1.0))
        elif targets is not None:
            # targets shape: [B, K, L]
            # logits shape (after view): [B*K*L, V/K]
            loss_re = F.cross_entropy(logits[:, 0, :, :, :].reshape(-1, logits.size(-1)), targets[:, 0].reshape(-1))
            loss_im = F.cross_entropy(logits[:, 1, :, :, :].reshape(-1, logits.size(-1)), targets[:, 1].reshape(-1))
            loss = (loss_re + loss_im) / 2

        if return_latent:
            return logits, loss, latent
        return logits, loss

    def predict_latent(self, h: torch.Tensor, head_state: torch.Tensor = None, head_codes: torch.Tensor = None) -> torch.Tensor:
        """h (B, L, dim) -> (B, 2, L, latent_dim): regressed tokenizer latent for the Re / Im streams, each 8-dim codebook
        chunk L2-normalised (the decoder consumes unit-norm chunks; see models.sense_ops.latent_to_image).
        'linear': RMSNorm -> Linear. 'mlp': + residual MLP on the same normalised hidden state. 'fused': + residual MLP on
        [normalised hidden state, token head's chain state, codebook latents of the token head's codes]; head_state (B, L, dim)
        and head_codes (B, 2, K, L) come from rollout_head (training) or from the sampler's cached head loop (inference)."""
        hn = self.latent_head[0](h)
        z = self.latent_head[1](hn).float()                              # (B, L, 2*D)
        if self.latent_mlp is not None:
            if self.latent_head_type == 'fused':
                assert head_state is not None and head_codes is not None, "the fused latent head needs the token head's state and codes"
                q = self.codes_to_latent(head_codes).to(hn.dtype)        # (B, 2, L, D): unit-norm codebook chunks of the committed codes
                x = torch.cat([hn, self.latent_fuse_ln(head_state.to(hn.dtype)), q[:, 0], q[:, 1]], dim=-1)
            else:
                x = hn
            z = z + self.latent_mlp(x).float()
        B, L, _ = z.shape
        z = z.view(B, L, 2, self.latent_dim).permute(0, 2, 1, 3)       # (B, 2, L, D)
        nchunk = 8 if self.latent_dim % 8 == 0 else 1
        z = F.normalize(z.reshape(B, 2, L, nchunk, -1), dim=-1).reshape(B, 2, L, self.latent_dim)
        return z

    @torch.no_grad()
    def rollout_head(self, h: torch.Tensor):
        """Greedy roll-out of the token head from the body's hidden states h (B, L, dim), without KV caches (training-time use;
        the sampler's cached loop yields the same codes and states). Returns the codes (B, 2, K, L) the token path commits to and
        the head's chain state (B, L, dim): the mean over the chain steps of its normalised hidden state."""
        B, L, D = h.shape
        codes, state = self.output.rollout(h.reshape(B * L, D))                         # (N, 2, K), (N, D)
        return codes.view(B, L, 2, -1).permute(0, 2, 3, 1).contiguous(), state.view(B, L, D)

    def set_latent_codebook(self, codebook):
        """codebook: the tokenizer's product quantiser (models.quant.VectorQuantizerM) or a (K, V/K, d) tensor of unit-norm rows.
        Stored as a non-persistent buffer (it belongs to the tokenizer, not to the checkpoint); needed by the fused latent head."""
        cb = codebook if torch.is_tensor(codebook) else codebook_from_quantizer(codebook)
        assert tuple(cb.shape) == tuple(self.latent_codebook.shape), f"codebook {tuple(cb.shape)} != {tuple(self.latent_codebook.shape)}"
        self.latent_codebook.copy_(cb.to(self.latent_codebook.dtype))
        self.latent_codebook_ready = True

    def codes_to_latent(self, codes: torch.Tensor) -> torch.Tensor:
        """codes (B, 2, K, L) -> (B, 2, L, K*d): the tokenizer's codebook rows of the codes (what quantizer.idx_to_f returns)."""
        assert getattr(self, 'latent_codebook_ready', False), "call set_latent_codebook(tokenizer.quantizer) first"
        return torch.cat([self.latent_codebook[k][codes[:, :, k]] for k in range(codes.shape[2])], dim=-1)

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)



class AutoRegressiveHeadInterleaved(nn.Module):
    """Model-v2.1 output head. Per position one causal chain of 2K steps: step 2i predicts the Re code of codebook i, step 2i+1
    the Im code of codebook i (conditioned on Re_i and on all earlier codes). Each (stream, codebook) has its own embedding
    table and every chain step has a learned position embedding (even = Re, odd = Im), so the stream identity is explicit."""
    interleaved = True

    def __init__(self, config):
        super().__init__()
        self.num_codebooks = K = config.num_codebooks
        self.sub_vocab_size = V = config.vocab_size // K
        self.codebooks_re = nn.ModuleList([nn.Embedding(V, config.dim) for _ in range(K)])        # re_i is the input of step 2i+1
        self.codebooks_im = nn.ModuleList([nn.Embedding(V, config.dim) for _ in range(K - 1)])    # im_i is the input of step 2i+2
        self.pos_emb = nn.Parameter(torch.zeros(2 * K, config.dim))
        config_clone = copy.deepcopy(config)
        self.layers = torch.nn.ModuleList([TransformerBlock(config_clone, drop_path=0.) for _ in range(config.n_output_layer)])
        self.norm = RMSNorm(config_clone.dim, eps=config.norm_eps)
        self.linear_head_re = nn.Linear(config_clone.dim, V)
        self.linear_head_im = nn.Linear(config_clone.dim, V)
        self.head_causal_mask = None

    def chain_inputs(self, base, t_re, t_im):
        """base (N, C), t_re / t_im (N, K) target codes -> chain inputs (N, 2K, C) (teacher forcing)."""
        xs = [base]
        for i in range(self.num_codebooks):
            xs.append(self.codebooks_re[i](t_re[:, i]))
            if i < self.num_codebooks - 1:
                xs.append(self.codebooks_im[i](t_im[:, i]))
        x = torch.stack(xs, dim=1)
        return x + self.pos_emb[None, :, :].to(x.dtype)

    def forward_train(self, base_tokens, targets):
        K = self.num_codebooks
        B, L, C = base_tokens.shape
        t_re = targets[:, 0].permute(0, 2, 1).reshape(B * L, K)
        t_im = targets[:, 1].permute(0, 2, 1).reshape(B * L, K)
        h = self.chain_inputs(base_tokens.reshape(B * L, C), t_re, t_im)
        for layer in self.layers:
            h = layer(h, freqs_cis=None, start_pos=None, mask=None, full_attn=False)   # causal over the chain
        h = self.norm(h)
        logits_re = self.linear_head_re(h[:, 0::2])                                    # (N, K, V): step 2i -> re_i
        logits_im = self.linear_head_im(h[:, 1::2])                                    # (N, K, V): step 2i+1 -> im_i
        logits = torch.stack((logits_re, logits_im), dim=1)                            # (N, 2, K, V)
        return logits.reshape(B, L, 2, K, -1).permute(0, 2, 3, 1, 4).contiguous()      # (B, 2, K, L, V)

    def forward_test(self, base_tokens, idx=None, input_pos=None, mask=None, return_state=False):
        """One chain step with the KV cache. s = int(input_pos): s == 0 takes the base state (N, 1, C); s = 2i+1 takes idx = re_i;
        s = 2i+2 takes idx = im_i (idx: (N,) long). Returns the logits (N, 1, V) of the stream predicted at step s (even Re, odd Im),
        plus the normalised hidden state (N, 1, C) with return_state."""
        s = int(input_pos)
        if s == 0:
            h = base_tokens
        elif s % 2 == 1:
            h = self.codebooks_re[(s - 1) // 2](idx.reshape(-1))[:, None, :]
        else:
            h = self.codebooks_im[(s - 2) // 2](idx.reshape(-1))[:, None, :]
        h = h + self.pos_emb[s][None, None, :].to(h.dtype)
        for layer in self.layers:
            h = layer(h, freqs_cis=None, start_pos=input_pos, mask=mask)
        h = self.norm(h)
        logits = (self.linear_head_re if s % 2 == 0 else self.linear_head_im)(h)
        return (logits, h) if return_state else logits

    @torch.no_grad()
    def rollout(self, base):
        """Greedy roll-out without KV caches. base (N, C) -> codes (N, 2, K) long, chain state (N, C) = mean over the 2K steps of the
        normalised hidden state."""
        K = self.num_codebooks
        xs, re, im, states = [base], [], [], []
        for s in range(2 * K):
            h = torch.stack(xs, dim=1) + self.pos_emb[None, :s + 1, :].to(base.dtype)
            for layer in self.layers:
                h = layer(h, freqs_cis=None, start_pos=None, mask=None, full_attn=False)
            h = self.norm(h)[:, -1]
            states.append(h)
            if s % 2 == 0:
                c = self.linear_head_re(h).argmax(-1); re.append(c); xs.append(self.codebooks_re[s // 2](c))
            else:
                c = self.linear_head_im(h).argmax(-1); im.append(c)
                if s < 2 * K - 1:
                    xs.append(self.codebooks_im[s // 2](c))
        return torch.stack([torch.stack(re, 1), torch.stack(im, 1)], dim=1), torch.stack(states, 1).float().mean(1)


def codebook_from_quantizer(quantizer) -> torch.Tensor:
    """(K, V/K, d) unit-norm rows of the tokenizer's product quantiser (models.quant.VectorQuantizerM), in chunk order."""
    return torch.stack([cb.codebook.get_norm_weight().detach().float() for cb in quantizer.codebooks])


def infer_model_args_from_state_dict(sd, default_mask_width: int = 256) -> dict:
    """Architecture flags a checkpoint implies (to rebuild a model without its training args): model_version, mask_width,
    draft_cond, head_interleaved, latent_dim, latent_head_type, latent_mlp_width."""
    dim = int(sd["norm.weight"].shape[0])
    out = dict(model_version=(2.1 if "in_proj.0.weight" in sd else (2 if "mask_embed.weight" in sd else 1)),
               mask_width=(int(sd["mask_embed.weight"].shape[1]) if "mask_embed.weight" in sd else default_mask_width),
               draft_cond="draft_ln.weight" in sd,
               head_interleaved="output.merge.weight" not in sd,
               latent_dim=(int(sd["latent_head.1.weight"].shape[0] // 2) if "latent_head.1.weight" in sd else 0),
               latent_head_type="linear", latent_mlp_width=0)
    if "latent_mlp.0.weight" in sd:
        w = sd["latent_mlp.0.weight"]
        out["latent_mlp_width"] = int(w.shape[0])
        out["latent_head_type"] = "fused" if int(w.shape[1]) > dim else "mlp"
    return out


def adapt_mask_width(sd, new_width: int):
    """Warm-start a v2+ model on another acquisition grid: resample the sampling-mask embedding (Linear(mask_width -> dim), one
    weight column per k-space column) to the new width by linear interpolation along the column axis, so the learned mask
    conditioning carries over (its output is RMS-normalised in the model, so the scale change does not matter)."""
    k = "mask_embed.weight"
    if k not in sd or int(sd[k].shape[1]) == int(new_width):
        return sd
    out = dict(sd)
    w = sd[k].float()                                                                          # (dim, old_width)
    out[k] = F.interpolate(w[None], size=int(new_width), mode="linear", align_corners=True)[0].to(sd[k].dtype)
    return out


def remap_v1_to_v2(sd):
    """Warm-start a v2.1 model from a v1 / v2 checkpoint: drop the v1 input token tables (v2 projects continuous encoder latents) and
    copy the v1 head tables into both per-stream tables of the interleaved head (Re_7 and the chain positions start fresh)."""
    out = {}
    for k, v in sd.items():
        if k.startswith("re_tok_embeddings.") or k.startswith("im_tok_embeddings."):
            continue
        if k.startswith("output.codebooks."):
            i = k.split(".")[2]
            out[f"output.codebooks_re.{i}.weight"] = v
            out[f"output.codebooks_im.{i}.weight"] = v
            continue
        out[k] = v
    return out


class AutoRegressiveHeadAsym(nn.Module):
    """Parallel Re/Im head (K chain steps per position, Re_i and Im_i predicted jointly at step i like the v1 head) with the
    Re/Im symmetry of the conditioning removed: per-stream embedding tables for the previous codes, concatenated along the
    channel dimension and projected (the projection is initialised to the v1 sum, so a warm start is unchanged at step 0),
    plus a learned chain-position embedding."""
    interleaved = False

    def __init__(self, config):
        super().__init__()
        self.num_codebooks = K = config.num_codebooks
        self.sub_vocab_size = V = config.vocab_size // K
        self.codebooks_re = nn.ModuleList([nn.Embedding(V, config.dim) for _ in range(K - 1)])
        self.codebooks_im = nn.ModuleList([nn.Embedding(V, config.dim) for _ in range(K - 1)])
        self.merge = nn.Linear(2 * config.dim, config.dim, bias=False)
        with torch.no_grad():
            self.merge.weight.copy_(torch.cat([torch.eye(config.dim), torch.eye(config.dim)], dim=1))   # == E_re + E_im at init
        self.pos_emb = nn.Parameter(torch.zeros(K, config.dim))
        config_clone = copy.deepcopy(config)
        self.layers = torch.nn.ModuleList([TransformerBlock(config_clone, drop_path=0.) for _ in range(config.n_output_layer)])
        self.norm = RMSNorm(config_clone.dim, eps=config.norm_eps)
        self.linear_head_re = nn.Linear(config_clone.dim, V)
        self.linear_head_im = nn.Linear(config_clone.dim, V)
        self.head_causal_mask = None

    def _merge(self, i, re_idx, im_idx):
        return self.merge(torch.cat([self.codebooks_re[i](re_idx), self.codebooks_im[i](im_idx)], dim=-1))

    def forward_train(self, base_tokens, targets):
        K = self.num_codebooks
        B, L, C = base_tokens.shape
        t_re = targets[:, 0].permute(0, 2, 1).reshape(B * L, K)[:, :-1]
        t_im = targets[:, 1].permute(0, 2, 1).reshape(B * L, K)[:, :-1]
        prev = torch.stack([self._merge(i, t_re[:, i], t_im[:, i]) for i in range(K - 1)], dim=1)   # (N, K-1, C)
        h = torch.cat((base_tokens.reshape(B * L, 1, C), prev), dim=1) + self.pos_emb[None, :, :].to(base_tokens.dtype)
        for layer in self.layers:
            h = layer(h, freqs_cis=None, start_pos=None, mask=None, full_attn=False)
        h = self.norm(h)
        logits_re = self.linear_head_re(h).reshape(B, L, K, -1).permute(0, 2, 1, 3)
        logits_im = self.linear_head_im(h).reshape(B, L, K, -1).permute(0, 2, 1, 3)
        return torch.stack((logits_re, logits_im), dim=1)                                       # (B, 2, K, L, V)

    def forward_test(self, base_tokens, idx=None, input_pos=None, mask=None, return_state=False):
        s = int(input_pos)
        h = base_tokens if idx is None else self._merge(s - 1, idx[:, 0], idx[:, 1])          # idx (N, 2, 1) -> (N, 1, C)
        h = h + self.pos_emb[s][None, None, :].to(h.dtype)
        for layer in self.layers:
            h = layer(h, freqs_cis=None, start_pos=input_pos, mask=mask)
        h = self.norm(h)
        logits = torch.stack((self.linear_head_re(h), self.linear_head_im(h)), dim=1)           # (N, 2, 1, V)
        return (logits, h) if return_state else logits

    @torch.no_grad()
    def rollout(self, base):
        """Greedy roll-out without KV caches. base (N, C) -> codes (N, 2, K) long, chain state (N, C) = mean over the K steps of the
        normalised hidden state."""
        K = self.num_codebooks
        xs, re, im, states = [base], [], [], []
        for s in range(K):
            h = torch.stack(xs, dim=1) + self.pos_emb[None, :s + 1, :].to(base.dtype)
            for layer in self.layers:
                h = layer(h, freqs_cis=None, start_pos=None, mask=None, full_attn=False)
            h = self.norm(h)[:, -1]
            states.append(h)
            re.append(self.linear_head_re(h).argmax(-1)); im.append(self.linear_head_im(h).argmax(-1))
            if s < K - 1:
                xs.append(self._merge(s, re[-1], im[-1]))
        return torch.stack([torch.stack(re, 1), torch.stack(im, 1)], dim=1), torch.stack(states, 1).float().mean(1)


class AutoRegressiveHeadSingleStream(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_codebooks = config.num_codebooks
        self.sub_vocab_size = config.vocab_size // config.num_codebooks

        self.codebooks = nn.ModuleList()
        for _ in range(self.num_codebooks - 1):
            codebook = nn.Embedding(self.sub_vocab_size, config.dim)
            self.codebooks.append(codebook)

        self.layers = torch.nn.ModuleList()

        config_clone = copy.deepcopy(config)
        # config_clone.dim = config_clone.dim * 2
        for _ in range(config.n_output_layer):
            self.layers.append(TransformerBlock(config_clone, drop_path=0.))

        self.norm = RMSNorm(config_clone.dim, eps=config.norm_eps)
        self.linear_head = nn.Linear(config_clone.dim, self.sub_vocab_size)

    def forward_train(self, base_tokens, targets):
        # base_tokens[B, 1+(Li-1), 768], targets[B, K, Li]
        K = targets.shape[1]
        B, L, C = base_tokens.shape
        base_tokens = base_tokens.reshape(B * L, 1, C)  # [B*(1+(Li-1)), 1, 768]

        targets = targets.permute(0, 2, 1).reshape(B * L, K)[:, :-1]  # [B*Li, K-1]
        
        index_embeddings = []
        for i in range(self.num_codebooks - 1):
            index_embed = self.codebooks[i](targets[:, i])
            index_embeddings.append(index_embed)
        index_embeddings = torch.stack(index_embeddings, dim=1)  # [B*Li, K-1, 768]

        h = torch.cat((base_tokens, index_embeddings), dim=1)  # [B*(1+(Li-1)), 1+(K-1), 768]
        for layer in self.layers:
            h = layer(h, freqs_cis=None, start_pos=None, mask=None, full_attn=False)
        h = self.norm(h)  # [B*(1+(Li-1)), 1+(K-1), 768]
        logits = self.linear_head(h)  # [B*(1+(Li-1)), 1+(K-1), V/K]. Next codebook prediction??
        logits = logits.reshape(B, L, K, -1).permute(0, 2, 1, 3)  # [B, 1+(K-1), 1+(Li-1), V/K]

        return logits

    def forward_test(self, base_tokens, idx=None, input_pos=None, mask=None):
        if idx is not None:
            h = self.codebooks[input_pos - 1](idx)
        else:
            h = base_tokens
        
        for layer in self.layers:
            h = layer(h, freqs_cis=None, start_pos=input_pos, mask=mask)
        h = self.norm(h)
        logits = self.linear_head(h)

        return logits


class AutoRegressiveHeadDoubleStream(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.re_head = AutoRegressiveHeadSingleStream(config)
        self.im_head = AutoRegressiveHeadSingleStream(config)

    def rm_head_caches(self):
        for b in self.re_head.layers:
            b.attention.kv_cache = None
        self.re_head.head_causal_mask = None

        for b in self.im_head.layers:
            b.attention.kv_cache = None
        self.im_head.head_causal_mask = None

    def forward_train(self, base_tokens, targets):
        # base_tokens[B, 2, 1+(Li-1), 768], targets[B, 2 K, Li]
        base_tokens_re = base_tokens[:,0]
        base_tokens_im = base_tokens[:,1]
        targets_re = targets[:,0]
        targets_im = targets[:,1]

        logits_re = self.re_head.forward_train(base_tokens_re, targets_re)
        logits_im = self.im_head.forward_train(base_tokens_im, targets_im)

        logits = torch.stack([logits_re, logits_im], dim=1)

        return logits

    def forward_test(self, base_tokens, idx=None, input_pos=None, mask=None):
        if idx is not None:
            re_idx = idx[:,0]
            im_idx = idx[:,1]
        else:
            re_idx = im_idx = None

        logits_re = self.re_head.forward_test(base_tokens[:,0], re_idx, input_pos, mask)
        logits_im = self.im_head.forward_test(base_tokens[:,1], im_idx, input_pos, mask)

        logits = torch.stack([logits_re, logits_im], dim=1)
        return logits


class ComplexTransformer(Transformer):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        del self.output
        self.output = AutoRegressiveHeadDoubleStream(config)
        self.initialize_weights_output()
    
    def initialize_weights_output(self):
        self.apply(self._init_weights)
        # zero output layers
        nn.init.constant_(self.output.re_head.linear_head.weight, 0)
        nn.init.constant_(self.output.im_head.linear_head.weight, 0)

    
    def rm_head_caches(self):
        self.output.rm_head_caches()
        self.output.head_causal_mask = None
        self.max_seq_length = -1
        self.max_batch_size = -1

    def forward(
            self,
            cond_tokens: torch.Tensor,  # [B, 2, K, L]
            timestep: torch.Tensor = None, 
            input_pos: Optional[torch.Tensor] = None,
            targets: Optional[torch.Tensor] = None, # [B, 2, K, L]
            mask: Optional[torch.Tensor] = None,
            valid: Optional[torch.Tensor] = None,
    ):
        assert self.cond_token_num > 0, "cond_token_num must be greater than 0"
        cond_tokens_re = cond_tokens[:, 0, :, :]
        cond_tokens_im = cond_tokens[:, 1, :, :]
        cond_embed_re = self.re_tok_embeddings(cond_tokens_re)  # [B, Lc, dim]
        cond_embed_im = self.im_tok_embeddings(cond_tokens_im)  # [B, Lc, dim]

        cond_embeddings = torch.stack([cond_embed_re,cond_embed_im], dim=1) # [B, 2, Lc, dim]

        if timestep is not None and self.cls_token_num > 0:
            t_embeddings = self.cls_embedding(timestep, train=self.training)[:, :self.cls_token_num]
            t2 = t_embeddings.unsqueeze(1).expand(-1, 2, -1, -1) 
            cond_embeddings = torch.cat([t2, cond_embeddings], dim=2) # [B, 2, 1+Lm, dim]

        h = self.tok_dropout(cond_embeddings)
        self.freqs_cis = self.freqs_cis.to(h.device)

        freqs_cis = self.freqs_cis

        # transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis_q=freqs_cis, freqs_cis_k=freqs_cis, start_pos=input_pos, mask=mask, full_attn=True)

        # output layers
        h = self.norm(h)
        # if self.training:
        h = h[:, :, self.cls_token_num:].contiguous() # H,2,L,dim

        if targets is None:
            return h
        else:
            logits = self.output.forward_train(h, targets=targets) 

        # if we are given some desired targets also calculate the loss
        loss = None
        if valid is not None:
            # loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            # valid_all = valid[:, None].repeat(1, targets.shape[1]).view(-1)
            # loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
            pass
        elif targets is not None:
            # targets shape: [B, K, L]
            # logits shape (after view): [B*K*L, V/K]
            loss_re = F.cross_entropy(logits[:, 0, :, :, :].reshape(-1, logits.size(-1)), targets[:, 0].reshape(-1))
            loss_im = F.cross_entropy(logits[:, 1, :, :, :].reshape(-1, logits.size(-1)), targets[:, 1].reshape(-1))
            loss = (loss_re + loss_im) / 2

        return logits, loss

    def setup_head_caches(self, max_batch_size, max_seq_length, dtype, device):
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length # TODO probably should be self.output.max_seq_length
        self.max_batch_size = max_batch_size

        for b in self.output.re_head.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype).to(
                device)

        for b in self.output.im_head.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype).to(
                device)

        causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)).to(device)
        self.output.head_causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        self.output.re_head.head_causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        self.output.im_head.head_causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)


class KspaceProbeBlock(nn.Module):
    def __init__(self, config, drop_path=0.0):
        super().__init__()
        self.cross_attn_block = TransformerCrossAttentionBlock(config, drop_path)
        self.self_attn_block = TransformerBlock(config, drop_path)

    def forward(self, q, kv, freqs_cis_k, return_attn_weights=False):
        q = self.cross_attn_block(q, kv, freqs_cis_q=None, freqs_cis_k=freqs_cis_k, return_attn_weights=return_attn_weights)
        if return_attn_weights:
            q, attn_weights = q
        else:
            attn_weights = None
        q = self.self_attn_block(q, freqs_cis=None, start_pos=None, mask=None, full_attn=True)

        return q, attn_weights


class KspaceProbeTransformer(nn.Module):
    def __init__(self, transformer, num_kspace_lines=320, vertical_line_height=320):
        """
        Args:
            transformer: The pre-trained backbone model.
            num_kspace_lines: Number of learnable probe tokens (candidates).
            kspace_readout_width: The dimension of the readout line (for Aux loss).
        """
        super().__init__()
        self.backbone = transformer
        config = transformer.config
        self.dim = config.dim
        self.H = vertical_line_height
        self.W = num_kspace_lines
        
        # 1. Learnable Probe Tokens (The "Sensors")
        # Shape: [1, N_probes, Dim]
        self.probe_tokens = nn.Parameter(torch.randn(1, num_kspace_lines, self.dim))
        nn.init.trunc_normal_(self.probe_tokens, std=0.02)
        
        # 2. Sidecar Layers
        # We create one ProbeBlock for every layer in the backbone
        # We use the backbone's dropout settings for consistency
        drop_path_rates = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        
        self.probe_layers = nn.ModuleList([
            KspaceProbeBlock(config, drop_path=drop_path_rates[i])
            for i in range(config.n_layer)
        ])
        
        # 3. Auxiliary Physics Head 
        # Predicts Real and Imaginary parts of the k-space line
        # Output: [B, num_kspace_lines, 2 * vertical_line_width]
        # self.kspace_head = nn.Linear(self.dim, vertical_line_height * 2)
        self.kspace_head = nn.Sequential(
            nn.Linear(self.dim, self.dim * 4),    
            nn.LayerNorm(self.dim * 4),           
            nn.GELU(),                          
            nn.Linear(self.dim * 4, self.H * 2)   
        )
        
        nn.init.zeros_(self.kspace_head[-1].weight)
        nn.init.zeros_(self.kspace_head[-1].bias)
    
    @property
    def output(self):
        return self.backbone.output
    
    @property
    def num_codebooks(self):
        return self.backbone.num_codebooks

    def configure_backbone_grad(self, freeze: bool = True):
            """
            Helper to freeze/unfreeze the backbone parameters.
            Useful for two-stage training (Probe Warmup -> Full Finetuning).
            """
            for param in self.backbone.parameters():
                param.requires_grad = not freeze
            
            status = "Frozen" if freeze else "Unfrozen"
            print(f"Backbone parameters set to: {status}")
    
    def setup_caches(self, max_batch_size, max_seq_length, dtype, device=None):
        return self.backbone.setup_caches(max_batch_size, max_seq_length, dtype, device)

    def setup_head_caches(self, max_batch_size, max_seq_length, dtype, device):
        return self.backbone.setup_head_caches(max_batch_size, max_seq_length, dtype, device)

    def rm_head_caches(self):
        self.backbone.rm_head_caches()

    def forward(
        self, 
        cond_tokens: torch.Tensor, 
        soft_cond_embeddings: torch.Tensor = None,
        timestep: torch.Tensor = None,
        input_pos: torch.Tensor = None,
        targets: torch.Tensor = None,
        target_image: torch.Tensor=None,
        mask: torch.Tensor = None,
        valid: torch.Tensor = None
    ):
        # =================================================================
        # PART 1: EMBEDDING LOGIC (Replicated from Backbone)
        # =================================================================
        # prepare the 'h' (image stream) exactly as the backbone does.
        
        cond_tokens_re = cond_tokens[:, 0, :, :]
        cond_tokens_im = cond_tokens[:, 1, :, :]
        
        # Access embeddings from the backbone
        cond_embed_re = self.backbone.re_tok_embeddings(cond_tokens_re)
        cond_embed_im = self.backbone.im_tok_embeddings(cond_tokens_im)

        if soft_cond_embeddings is not None:
            soft_cond_embed_re = soft_cond_embeddings[:, 0]
            soft_cond_embed_im = soft_cond_embeddings[:, 1]
            cond_embed_re = soft_cond_embed_re + (cond_embed_re - soft_cond_embed_re).detach()
            cond_embed_im = soft_cond_embed_im + (cond_embed_im - soft_cond_embed_im).detach()
        
        cond_embeddings = self.backbone.token_ln(cond_embed_re + cond_embed_im)

        if timestep is not None and self.backbone.cls_token_num > 0:
            t_embeddings = self.backbone.cls_embedding(timestep, train=self.training)[:, :self.backbone.cls_token_num]
            cond_embeddings = torch.cat([t_embeddings, cond_embeddings], dim=1)

        h = self.backbone.tok_dropout(cond_embeddings)
        
        # Setup Rotary Embeddings for the Image Stream
        self.backbone.freqs_cis = self.backbone.freqs_cis.to(h.device)
        freqs_cis = self.backbone.freqs_cis

        # =================================================================
        # PART 2: THE probe LOOP
        # =================================================================
        
        # Initialize Probe Stream
        B = h.shape[0]
        probe_stream = self.probe_tokens.expand(B, -1, -1)
        
        final_attn_weights = None

        # Iterate over both streams simultaneously
        for i, (image_layer, probe_layer) in enumerate(zip(self.backbone.layers, self.probe_layers)):
            
            # A. Image Stream Update (Backbone)
            # Note: We hardcode start_pos=0 as this is usually for parallel training/encoding
            h = image_layer(h, freqs_cis, start_pos=0, mask=mask, full_attn=True)
            
            # B. Probe Stream Update (Sidecar)
            # Check if this is the last layer to save compute on attention weights
            is_last_layer = (i == len(self.backbone.layers) - 1)
            
            probe_stream, attn_weights = probe_layer(
                q=probe_stream,
                kv=h,               # The updated image features serve as Keys/Values
                freqs_cis_k=freqs_cis, # Image RoPE is required for the Key/Value
                return_attn_weights=is_last_layer
            )
            
            if is_last_layer:
                final_attn_weights = attn_weights[:,:,:, self.backbone.cls_token_num:]

        # =================================================================
        # PART 3: HEADS & OUTPUTS
        # =================================================================
        
        # --- A. Image Reconstruction Output (Standard) ---
        h_norm = self.backbone.norm(h)
        h_out = h_norm[:, self.backbone.cls_token_num:].contiguous()

        if targets is None:
            image_logits = h_out
            ce_loss = None
        else:
            image_logits = self.backbone.output.forward_train(h_out, targets=targets)
            
            # # Calculate Standard Reconstruction Loss
            loss_re = F.cross_entropy(image_logits[:, 0].reshape(-1, image_logits.size(-1)), targets[:, 0].reshape(-1))
            loss_im = F.cross_entropy(image_logits[:, 1].reshape(-1, image_logits.size(-1)), targets[:, 1].reshape(-1))
            ce_loss = (loss_re + loss_im) / 2
        
        if target_image is None:
            return h_out, {"probe_attention": final_attn_weights}

        # --- B. K-Space Aux Output (Physics Probe) ---
        kspace_pred = self.kspace_head(probe_stream) # B, W, H*2
        kspace_pred = kspace_pred.view(B, self.W, self.H, 2).permute(0, 2, 1, 3).contiguous()
        image_pred = transforms.ifft2(kspace_pred)

        if target_image is None:
            kspace_loss = None
        else:
            kspace_loss = F.l1_loss(image_pred, target_image)

        loss = kspace_loss
        # if ce_loss is not None:
        #     loss +=ce_loss
        aux = {
            "image_logits": image_logits,
            "ce_loss": ce_loss,
            "kspace_loss": kspace_loss,
            "kspace_pred": kspace_pred,
            "image_pred": image_pred,
            "probe_attention": final_attn_weights # [B, N_Heads, N_Probes, N_Img]
        }
            
        return image_logits, loss, aux
    


#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, prefix_len: int = 0):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    prefix_cache = torch.cat([torch.zeros(prefix_len, n_elem // 2, 2), cache])
    return prefix_cache


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, zero_prefix_len: int = 0):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)],
                             dim=-1)  # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    prefix_cache = torch.cat([torch.zeros(zero_prefix_len, n_elem // 2, 2), cache])
    return prefix_cache


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)  # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)  # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
        xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
        xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


#################################################################################
#                                KAR_GPT Configs                                    #
#################################################################################
### text-conditional
def KAR_GPT_7B(**kwargs):
    return Transformer(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs))  # 6.6B


def KAR_GPT_3B(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs))  # 3.1B


def KAR_GPT_1B(**kwargs):
    return Transformer(ModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs))  # 1.2B


### class-conditional
def KAR_GPT_XXXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs))  # 3.9B


def KAR_GPT_XXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs))  # 1.4B


def KAR_GPT_XL(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs))  # 775M


def KAR_GPT_L(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs))  # 343M


def KAR_GPT_B(**kwargs):
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs))  # 111M

### Complex

def KAR_COMPLEX_GPT_7B(**kwargs):
    return ComplexTransformer(ModelArgs(n_layer=32, n_head=32, dim=4096, block_type="complex", **kwargs))  # 6.6B


def KAR_COMPLEX_GPT_3B(**kwargs):
    return ComplexTransformer(ModelArgs(n_layer=24, n_head=32, dim=3200, block_type="complex", **kwargs))  # 3.1B


def KAR_COMPLEX_GPT_1B(**kwargs):
    return ComplexTransformer(ModelArgs(n_layer=22, n_head=32, dim=2048, block_type="complex", **kwargs))  # 1.2B


def KAR_COMPLEX_GPT_XXXL(**kwargs):
    return ComplexTransformer(ModelArgs(n_layer=48, n_head=40, dim=2560, block_type="complex", **kwargs))  # 3.9B


def KAR_COMPLEX_GPT_XXL(**kwargs):
    return ComplexTransformer(ModelArgs(n_layer=48, n_head=24, dim=1536, block_type="complex", **kwargs))  # 1.4B


def KAR_COMPLEX_GPT_XL(**kwargs):
    return ComplexTransformer(ModelArgs(n_layer=36, n_head=20, dim=1280, block_type="complex", **kwargs))  # 775M


def KAR_COMPLEX_GPT_L(**kwargs):
    return ComplexTransformer(ModelArgs(n_layer=24, n_head=16, dim=1024, block_type="complex", **kwargs))  # 343M


def KAR_COMPLEX_GPT_B(**kwargs):
    return ComplexTransformer(ModelArgs(n_layer=12, n_head=12, dim=768, block_type="complex", **kwargs))  # 111M


# Kspace Probe
def PROBE_GPT_B(resolution, backbone, backbone_checkpoint, freeze_backbone=True):
    if backbone_checkpoint is not None:
        checkpoint = torch.load(backbone_checkpoint, map_location="cpu")
        backbone.load_state_dict(checkpoint["model"])
    
    model = KspaceProbeTransformer(backbone, num_kspace_lines=resolution, vertical_line_height=resolution)

    if freeze_backbone:
        model.configure_backbone_grad(freeze=True)

    return model

KAR_GPT_models = {
    'GPT-B': KAR_GPT_B, 'GPT-L': KAR_GPT_L, 'GPT-XL': KAR_GPT_XL, 'GPT-XXL': KAR_GPT_XXL, 'GPT-XXXL': KAR_GPT_XXXL,
    'GPT-1B': KAR_GPT_1B, 'GPT-3B': KAR_GPT_3B, 'GPT-7B': KAR_GPT_7B,
    'COMPLEX-GPT-B': KAR_COMPLEX_GPT_B, 'COMPLEX-GPT-L': KAR_COMPLEX_GPT_L, 'COMPLEX-GPT-XL': KAR_COMPLEX_GPT_XL, 
    'COMPLEX-GPT-XXL': KAR_COMPLEX_GPT_XXL, 'COMPLEX-GPT-XXXL': KAR_COMPLEX_GPT_XXXL,
    'COMPLEX-GPT-1B': KAR_COMPLEX_GPT_1B, 'COMPLEX-GPT-3B': KAR_COMPLEX_GPT_3B, 'COMPLEX-GPT-7B': KAR_COMPLEX_GPT_7B,
}