"""Unit tests for ComplexFusion and CrossCoilFusion."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import pytest

from models.multicoil_fusion import ComplexFusion, CrossCoilFusion

# ---------------------------------------------------------------------------
# Shared test constants (match MedITok defaults)
# ---------------------------------------------------------------------------
B         = 2    # batch size
C         = 4    # number of coils
L         = 256  # spatial sequence length (16×16 MedITok grid)
Dh        = 64   # encoder pre-quantization feature dim (vocab_width)
D         = 128  # fusion output dim
MAX_COILS = 8
N_HEADS   = 4


# ===========================================================================
# ComplexFusion tests
# ===========================================================================

class TestComplexFusion:
    @pytest.fixture
    def model(self):
        return ComplexFusion(Dh=Dh, D=D, L=L, max_coils=MAX_COILS)

    @pytest.fixture
    def inputs(self):
        h_re = torch.randn(B, L, Dh)
        h_im = torch.randn(B, L, Dh)
        return h_re, h_im

    def test_output_shape(self, model, inputs):
        h_re, h_im = inputs
        out = model(h_re, h_im, c=0)
        assert out.shape == (B, L, D), f"Expected ({B}, {L}, {D}), got {out.shape}"

    def test_different_coil_indices_differ(self, model, inputs):
        h_re, h_im = inputs
        out0 = model(h_re, h_im, c=0)
        out1 = model(h_re, h_im, c=1)
        assert out0.shape == out1.shape
        assert not torch.allclose(out0, out1), \
            "Different coil indices should produce different outputs"

    def test_spatial_emb_shape_and_type(self, model):
        assert isinstance(model.spatial_emb, nn.Parameter), \
            "spatial_emb must be an nn.Parameter"
        assert model.spatial_emb.shape == (1, L, D), \
            f"Expected (1, {L}, {D}), got {model.spatial_emb.shape}"

    def test_coil_emb_shape(self, model):
        assert model.coil_emb.weight.shape == (MAX_COILS, D), \
            f"Expected ({MAX_COILS}, {D}), got {model.coil_emb.weight.shape}"

    def test_W_rr_W_ri_are_single_instances(self, model):
        # W_rr and W_ri must be single Linear objects, not duplicated
        assert isinstance(model.W_rr, nn.Linear)
        assert isinstance(model.W_ri, nn.Linear)
        # They must be distinct from each other (different weights)
        assert model.W_rr is not model.W_ri

    def test_gradient_flow(self, model, inputs):
        h_re, h_im = inputs
        out = model(h_re, h_im, c=0)
        out.sum().backward()
        assert model.W_rr.weight.grad is not None, "W_rr must receive gradients"
        assert model.W_ri.weight.grad is not None, "W_ri must receive gradients"
        assert model.spatial_emb.grad is not None, "spatial_emb must receive gradients"
        assert model.coil_emb.weight.grad is not None, "coil_emb must receive gradients"

    def test_fuse_layer_shape(self, model):
        assert model.fuse.in_features == 2 * Dh
        assert model.fuse.out_features == D

    def test_all_coil_indices_valid(self, model, inputs):
        h_re, h_im = inputs
        for c in range(MAX_COILS):
            out = model(h_re, h_im, c=c)
            assert out.shape == (B, L, D), f"Failed for coil index {c}"


# ===========================================================================
# CrossCoilFusion tests
# ===========================================================================

class TestCrossCoilFusion:
    @pytest.fixture
    def model(self):
        return CrossCoilFusion(D=D, n_heads=N_HEADS)

    @pytest.fixture
    def H(self):
        return torch.randn(B, C, L, D)

    def test_output_shape(self, model, H):
        out = model(H)
        assert out.shape == (B, L, D), f"Expected ({B}, {L}, {D}), got {out.shape}"

    def test_intermediate_shapes(self, model, H):
        """Verify shapes at each stage by running a manual forward pass."""
        B_, C_, L_, D_ = H.shape
        assert (B_, C_, L_, D_) == (B, C, L, D)

        # Step 1: reshape
        x = H.permute(0, 2, 1, 3).reshape(B_ * L_, C_, D_)
        assert x.shape == (B * L, C, D), f"reshape: {x.shape}"

        # Step 2: MHSA (use the actual module layers)
        x_ln = model.ln_attn(x)
        attn_out, _ = model.attn(x_ln, x_ln, x_ln)
        assert attn_out.shape == (B * L, C, D), f"attn_out: {attn_out.shape}"
        U = attn_out + x
        assert U.shape == (B * L, C, D), f"U: {U.shape}"

        # Step 3: MLP
        V = model.mlp(model.ln_mlp(U)) + U
        assert V.shape == (B * L, C, D), f"V: {V.shape}"

        # Step 4: pooling scores
        scores = model.pool_proj(V)
        assert scores.shape == (B * L, C, 1), f"scores: {scores.shape}"

        # Step 5: softmax and weighted sum
        alpha = torch.softmax(scores, dim=1)
        assert alpha.shape == (B * L, C, 1), f"alpha: {alpha.shape}"
        h = (alpha * V).sum(dim=1)
        assert h.shape == (B * L, D), f"h before reshape: {h.shape}"

        # Step 6: final reshape
        h = h.reshape(B_, L_, D_)
        assert h.shape == (B, L, D), f"final: {h.shape}"

    def test_variable_coil_count(self, model):
        """CrossCoilFusion must handle any number of coils without reinstantiation."""
        for c in (3, 6, 8):
            H_var = torch.randn(B, c, L, D)
            out = model(H_var)
            assert out.shape == (B, L, D), \
                f"Failed for C={c}: got {out.shape}"

    def test_softmax_weights_sum_to_one(self, model, H):
        """Attention weights over coil dim must sum to 1."""
        # Use a forward hook to capture pool_proj output
        captured = {}

        def hook(module, inp, out):
            captured["scores"] = out.detach()

        handle = model.pool_proj.register_forward_hook(hook)
        try:
            model(H)
        finally:
            handle.remove()

        scores = captured["scores"]                         # (B*L, C, 1)
        alpha  = torch.softmax(scores, dim=1)               # (B*L, C, 1)
        sums   = alpha.sum(dim=1)                           # (B*L, 1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            "Attention weights must sum to 1 along coil dim"

    def test_gradient_flow(self, model, H):
        out = model(H)
        out.sum().backward()
        assert model.pool_proj.weight.grad is not None, \
            "pool_proj must receive gradients"
        assert model.mlp[0].weight.grad is not None, \
            "mlp fc1 must receive gradients"
        assert model.attn.in_proj_weight.grad is not None, \
            "attn in_proj_weight must receive gradients"


# ===========================================================================
# Integration test: ComplexFusion → CrossCoilFusion
# ===========================================================================

class TestIntegration:
    def test_complex_then_cross_coil(self):
        fusion = ComplexFusion(Dh=Dh, D=D, L=L, max_coils=MAX_COILS)
        cross  = CrossCoilFusion(D=D, n_heads=N_HEADS)

        # Simulate C per-coil encoder outputs
        per_coil = []
        for c in range(C):
            h_re = torch.randn(B, L, Dh)
            h_im = torch.randn(B, L, Dh)
            h_c  = fusion(h_re, h_im, c=c)             # (B, L, D)
            assert h_c.shape == (B, L, D)
            per_coil.append(h_c)

        # Stack along coil dimension
        H = torch.stack(per_coil, dim=1)                # (B, C, L, D)
        assert H.shape == (B, C, L, D)

        # Fuse across coils
        out = cross(H)                                  # (B, L, D)
        assert out.shape == (B, L, D), \
            f"Integration output shape mismatch: {out.shape}"

    def test_backward_through_full_pipeline(self):
        fusion = ComplexFusion(Dh=Dh, D=D, L=L, max_coils=MAX_COILS)
        cross  = CrossCoilFusion(D=D, n_heads=N_HEADS)

        per_coil = []
        for c in range(C):
            h_re = torch.randn(B, L, Dh)
            h_im = torch.randn(B, L, Dh)
            per_coil.append(fusion(h_re, h_im, c=c))

        H   = torch.stack(per_coil, dim=1)
        out = cross(H)
        out.sum().backward()

        # Both modules should have received gradients
        assert fusion.W_rr.weight.grad is not None
        assert cross.pool_proj.weight.grad is not None
