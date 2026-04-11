"""
Multicoil MRI fusion modules.

ComplexFusion:  per-coil real/imag feature fusion  (B,L,Dh) × 2 → (B,L,D)
CrossCoilFusion: cross-coil attention + learned pooling  (B,C,L,D) → (B,L,D)
"""

import torch
import torch.nn as nn
from torch import Tensor


class ComplexFusion(nn.Module):
    """Fuse real and imaginary encoder features for one coil.

    Uses a complex-multiplication-inspired linear structure so that the
    network can learn phase-aware combinations:

        u_re = W_rr(h_re) - W_ri(h_im)
        u_im = W_ri(h_re) + W_rr(h_im)
        h_c  = LN( Linear([u_re, u_im]) ) + spatial_emb + coil_emb[c]

    Args:
        Dh:        encoder pre-quantization feature dim  (64 for MedITok)
        D:         output fusion dimension
        L:         spatial sequence length               (256 for MedITok)
        max_coils: upper bound on number of coils (sets Embedding table size)
    """

    def __init__(self, Dh: int, D: int, L: int, max_coils: int):
        super().__init__()
        # W_rr and W_ri are each a single Linear called twice in forward —
        # PyTorch shares the weights automatically since it is the same object.
        self.W_rr = nn.Linear(Dh, Dh, bias=False)
        self.W_ri = nn.Linear(Dh, Dh, bias=False)

        self.fuse = nn.Linear(2 * Dh, D)
        self.ln   = nn.LayerNorm(D)

        # Spatial embedding p_l: (1, L, D), shared across all coils
        self.spatial_emb = nn.Parameter(torch.zeros(1, L, D))

        # Coil embedding e_c: looked up by integer index c
        self.coil_emb = nn.Embedding(max_coils, D)

    def forward(self, h_re: Tensor, h_im: Tensor, c: int) -> Tensor:
        """
        Args:
            h_re: (B, L, Dh)  real-part encoder features
            h_im: (B, L, Dh)  imag-part encoder features
            c:    int          coil index in [0, max_coils)

        Returns:
            h_c: (B, L, D)
        """
        # Complex-multiplication branches — W_rr and W_ri each called twice
        u_re = self.W_rr(h_re) - self.W_ri(h_im)          # (B, L, Dh)
        u_im = self.W_ri(h_re) + self.W_rr(h_im)          # (B, L, Dh)

        # Project concatenated complex output and normalise
        h_c = self.ln(self.fuse(torch.cat([u_re, u_im], dim=-1)))  # (B, L, D)

        # Coil embedding: (1, D) → (1, 1, D) to broadcast over (B, L, D)
        c_idx = torch.tensor([c], device=h_re.device)
        e_c   = self.coil_emb(c_idx).view(1, 1, -1)        # (1, 1, D)

        return h_c + self.spatial_emb + e_c                 # (B, L, D)


class CrossCoilFusion(nn.Module):
    """Merge features from C coils into a single sequence via attention pooling.

    Pipeline:
        (B, C, L, D)  →  reshape to (B*L, C, D)
        → bidirectional MHSA block (Pre-LN)
        → MLP block (Pre-LN)
        → learned attention pooling over coil dim
        → (B, L, D)

    C is inferred at runtime from the input — no constructor parameter needed.

    Args:
        D:          feature dimension (must match ComplexFusion output D)
        n_heads:    number of attention heads; D must be divisible by n_heads
        mlp_ratio:  MLP hidden dim multiplier (default 4 → hidden = 4*D)
    """

    def __init__(self, D: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        # Pre-LN before MHSA (spec writes MHSA(H)+H without explicit LN, but
        # Pre-LN is added here for training stability, consistent with ViT style)
        self.ln_attn  = nn.LayerNorm(D)
        self.attn     = nn.MultiheadAttention(D, n_heads, batch_first=True)

        # Pre-LN MLP block
        self.ln_mlp   = nn.LayerNorm(D)
        self.mlp      = nn.Sequential(
            nn.Linear(D, mlp_ratio * D),
            nn.GELU(),
            nn.Linear(mlp_ratio * D, D),
        )

        # Learned attention pooling: score each coil with a scalar
        self.pool_proj = nn.Linear(D, 1)

    def forward(self, H: Tensor) -> Tensor:
        """
        Args:
            H: (B, C, L, D)  stacked per-coil features

        Returns:
            h: (B, L, D)     coil-fused features
        """
        B, C, L, D = H.shape

        # Merge batch and spatial dims so attention runs over the coil axis
        x = H.permute(0, 2, 1, 3).reshape(B * L, C, D)     # (B*L, C, D)

        # Bidirectional MHSA sublayer (Pre-LN, no causal mask)
        x_ln = self.ln_attn(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln)           # (B*L, C, D)
        U = attn_out + x                                      # residual

        # MLP sublayer (Pre-LN)
        V = self.mlp(self.ln_mlp(U)) + U                     # (B*L, C, D)

        # Learned attention pooling over coil dimension
        scores = self.pool_proj(V)                            # (B*L, C, 1)
        alpha  = torch.softmax(scores, dim=1)                 # (B*L, C, 1)
        h      = (alpha * V).sum(dim=1)                       # (B*L, D)

        return h.reshape(B, L, D)                             # (B, L, D)
