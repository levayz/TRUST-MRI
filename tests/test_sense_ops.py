"""Unit tests for models/sense_ops.py and the padded-coil path of SensitivityModel."""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from models.sense_ops import (acs_col_range, acs_num_low_freqs, c2r, crop_native_to_grid, estimate_maps, fft2c,
                              ifft2c, make_sense_dc_fn, mask_bhw, mask_cols, mask_for_sens_net, phase_ref_coil0,
                              r2c, sense_combine, sense_dc, sense_expand, support_from_maps)
from models.varnet_sens import SensitivityModel

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "fastmri/brain/multicoil_val"
MAPS = ROOT / "fastmri/brain/multicoil_val_sens_maps_gpu"
torch.manual_seed(0)


def synth_maps(B=2, C=4, H=32, W=32, r=12.0):
    """Smooth complex maps with a disc support, unit RSS inside, 0 outside (ESPIRiT-like, coil-0 gauge)."""
    yy, xx = torch.meshgrid(torch.arange(H).float() - H / 2, torch.arange(W).float() - W / 2, indexing="ij")
    support = (yy ** 2 + xx ** 2).sqrt() < r
    maps = []
    for c in range(C):
        ang = 2 * torch.pi * c / C
        mag = torch.exp(-((yy - r * torch.cos(torch.tensor(ang))) ** 2 + (xx - r * torch.sin(torch.tensor(ang))) ** 2) / (2 * r ** 2))
        ph = 0.3 * c * (xx / W) + 0.2 * (yy / H) * c
        maps.append(mag * torch.exp(1j * ph))
    maps = torch.stack(maps)
    maps = maps * phase_ref_coil0(maps[None])[0].sgn().conj()[0:1].conj() if False else maps
    maps = maps / (maps.abs() ** 2).sum(0).sqrt().clamp_min(1e-12) * support
    maps = phase_ref_coil0(maps[None])[0]
    return maps[None].repeat(B, 1, 1, 1).to(torch.complex64), support[None].repeat(B, 1, 1)


def synth_image(B=2, H=32, W=32):
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    x = (torch.sin(yy / 3) * torch.cos(xx / 5) + 1.5) * torch.exp(1j * 0.4 * xx / W)
    return x[None].repeat(B, 1, 1).to(torch.complex64)


def col_mask(B, H, W, cols):
    m = torch.zeros(W, dtype=torch.bool); m[cols] = True
    return m[None, None, :, None].expand(B, H, W, 1).float()


class TestLayouts:
    def test_mask_cols_layouts(self):
        B, H, W = 2, 8, 16
        cols = torch.tensor([0, 3, 7, 8, 15])
        m4 = col_mask(B, H, W, cols)
        ref = mask_cols(m4)
        assert ref.shape == (B, W) and ref.sum() == 2 * len(cols)
        assert torch.equal(mask_cols(m4[..., :1].expand(B, H, W, 2)), ref)
        assert torch.equal(mask_cols(m4[:, :1]), ref)                       # (B,1,W,1)
        assert torch.equal(mask_cols(mask_for_sens_net(m4)), ref)           # (B,1,1,W,1)
        assert torch.equal(mask_cols(m4[0], batch=B), ref)                  # (H,W,1)
        assert torch.equal(mask_cols(m4[0, 0, :, 0], batch=B), ref)         # (W,)
        assert mask_for_sens_net(m4).shape == (B, 1, 1, W, 1)
        assert mask_bhw(m4, H).shape == (B, H, W)

    def test_acs(self):
        assert acs_num_low_freqs(256, 0.04) == 10
        assert acs_col_range(256, 0.04) == (123, 133)
        from eval_demod_ceiling import acs_col_range as ref
        assert acs_col_range(256, 0.04) == ref(256, 0.04)
        assert acs_col_range(320, 0.04) == ref(320, 0.04)
        from dataset.data.subsample import create_mask_for_mask_type
        mf = create_mask_for_mask_type("random", [0.04], [8])
        for seed in range(5):
            m, nlf = mf((1, 256, 256, 2), seed=(seed,))
            cols = mask_cols(m).squeeze(0)
            assert nlf == 10 and cols[123:133].all()
        from models.varnet_sens import SensitivityModel as SM
        pad, n = SM.get_pad_and_num_low_freqs(mask_for_sens_net(m), 10)
        assert int(pad) == 123 and int(n) == 10

    def test_crop(self):
        x = torch.arange(640 * 320).float().reshape(1, 640, 320)
        y = crop_native_to_grid(x, (320, 320), 256)
        assert y.shape == (1, 256, 256)
        assert torch.equal(y, x[:, 160 + 32:160 + 32 + 256, 32:32 + 256])


class TestSense:
    def test_combine_expand(self):
        maps, support = synth_maps()
        x = synth_image() * support
        assert torch.equal(support_from_maps(maps), support)
        xr = sense_combine(sense_expand(x, maps), maps)
        assert torch.allclose(xr[support], x[support], atol=1e-5)
        assert torch.all(xr[~support] == 0)

    def test_projection(self):
        maps, support = synth_maps()
        coils = torch.randn(2, 4, 32, 32, dtype=torch.complex64)
        P = lambda c: sense_expand(sense_combine(c, maps), maps)
        assert torch.allclose(P(P(coils)), P(coils), atol=1e-5)

    def test_dc_identity(self):
        maps, support = synth_maps()
        x0 = synth_image() * support
        mask = col_mask(2, 32, 32, torch.tensor([0, 5, 14, 15, 16, 17, 30]))
        k_meas = fft2c(sense_expand(x0, maps)) * mask_bhw(mask, 32)[:, None]
        assert torch.allclose(sense_dc(x0, maps, k_meas, mask), x0, atol=1e-5)
        # a wrong estimate is pulled back onto the measured lines
        xh = sense_dc(torch.zeros_like(x0), maps, k_meas, mask)
        kk = fft2c(sense_expand(xh, maps))
        m = mask_bhw(mask, 32)[:, None].expand_as(kk)
        assert (kk[m] - k_meas[m]).abs().max() < (kk[~m] - fft2c(sense_expand(x0, maps))[~m]).abs().max()

    def test_dc_fn_tiling(self):
        maps, support = synth_maps()
        x0 = synth_image() * support
        mask = col_mask(2, 32, 32, torch.tensor([1, 2, 15, 16, 17, 29]))
        k_meas = fft2c(sense_expand(x0, maps)) * mask_bhw(mask, 32)[:, None]
        dc_fn = make_sense_dc_fn(k_meas, maps, mask)
        i_hat = c2r(x0 * 0.5 + 0.1)
        single = dc_fn(i_hat, mask)
        assert single.shape == i_hat.shape
        assert torch.allclose(r2c(single), sense_dc(x0 * 0.5 + 0.1, maps, k_meas, mask), atol=1e-6)
        n = 3
        tiled = dc_fn(i_hat.repeat(n, 1, 1, 1), mask.repeat(n, 1, 1, 1))
        assert torch.allclose(tiled, single.repeat(n, 1, 1, 1), atol=1e-6)
        assert torch.allclose(dc_fn(i_hat.repeat(n, 1, 1, 1)), single.repeat(n, 1, 1, 1), atol=1e-6)
        with pytest.raises(AssertionError):
            dc_fn(i_hat[:1])

    def test_phase_ref(self):
        maps, support = synth_maps()
        rot = maps * torch.exp(1j * torch.linspace(0, 3, 32))[None, None, :, None]
        ref = phase_ref_coil0(rot)
        assert torch.allclose(ref.abs(), maps.abs(), atol=1e-6)
        assert ref[:, 0].imag.abs().max() < 1e-5
        assert torch.allclose(ref[support[:, None].expand_as(ref)], maps[support[:, None].expand_as(maps)], atol=1e-4)
        x = synth_image()
        assert torch.allclose(sense_combine(sense_expand(x, maps), ref).abs(), sense_combine(sense_expand(x, maps), maps).abs(), atol=1e-5)
        # gradient is finite where coil 0 is 0
        z = torch.zeros(1, 2, 4, 4, dtype=torch.complex64, requires_grad=True)
        phase_ref_coil0(z).abs().sum().backward()
        assert torch.isfinite(z.grad).all()


class TestSensNetPadding:
    def test_padded_coils(self):
        torch.manual_seed(0)
        net = SensitivityModel(chans=4, num_pools=2).eval()
        B, C, Cp, H, W = 2, 4, 7, 32, 32
        k = torch.randn(B, C, H, W, dtype=torch.complex64)
        kp = torch.zeros(B, Cp, H, W, dtype=torch.complex64); kp[:, :C] = k
        mask = col_mask(B, H, W, torch.tensor([13, 14, 15, 16, 17, 18, 3, 27]))
        m5 = mask_for_sens_net(mask)
        with torch.no_grad():
            ref = net(c2r(k), m5, num_low_frequencies=6)
            full = net(c2r(kp), m5, num_low_frequencies=6)
            out = net(c2r(kp), m5, num_low_frequencies=6, num_coils=torch.tensor([C, C]))
        assert torch.isnan(full).any(), "zero coils through NormUnet should produce NaN (this is the bug we guard)"
        assert torch.isfinite(out).all()
        assert torch.equal(out[:, C:], torch.zeros_like(out[:, C:]))
        assert torch.allclose(out[:, :C], ref, atol=1e-6)
        maps = estimate_maps(net, kp, mask, num_coils=torch.tensor([C, C]), num_low_freqs=6)
        assert maps.shape == (B, Cp, H, W) and torch.isfinite(maps).all()
        assert maps[:, 0].imag.abs().max() < 1e-5
        rss = (maps.abs() ** 2).sum(1).sqrt()
        assert torch.allclose(rss, torch.ones_like(rss), atol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for bf16 autocast")
    def test_padded_coils_bf16(self):
        net = SensitivityModel(chans=4, num_pools=2).eval().cuda()
        B, C, Cp, H, W = 2, 3, 5, 32, 32
        kp = torch.zeros(B, Cp, H, W, dtype=torch.complex64, device="cuda")
        kp[:, :C] = torch.randn(B, C, H, W, dtype=torch.complex64, device="cuda")
        mask = col_mask(B, H, W, torch.tensor([13, 14, 15, 16, 17, 18])).cuda()
        nc = torch.tensor([C, C], device="cuda")
        with torch.no_grad():
            ref = net(c2r(kp), mask_for_sens_net(mask), num_low_frequencies=6, num_coils=nc)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(c2r(kp), mask_for_sens_net(mask), num_low_frequencies=6, num_coils=nc)
        assert out.dtype == torch.float32 and torch.isfinite(out).all()
        assert torch.equal(out[:, C:], torch.zeros_like(out[:, C:]))
        err = (out[:, :C] - ref[:, :C]).abs()
        assert err.mean() < 1e-2 and err.max() < 0.3   # bf16 convs on random noise inputs


@pytest.mark.skipif(not (DATA.exists() and MAPS.exists()), reason="fastMRI brain multicoil val data not available")
class TestRealSlice:
    def test_dc_matches_eval_sense_block_formula(self):
        import h5py
        from dataset.data.subsample import create_mask_for_mask_type
        name = [l.strip() for l in open(ROOT / "fastmri/splits/suno_brain_test.txt") if l.strip()][0]
        with h5py.File(DATA / name) as f:
            k = torch.from_numpy(f["kspace"][5])
        with h5py.File(MAPS / name) as f:
            sm = torch.from_numpy(f["sens_maps"][5])
        coils = crop_native_to_grid(ifft2c(k), sm.shape[-2:], 256)[None]
        maps = crop_native_to_grid(sm, sm.shape[-2:], 256)[None]
        x0 = sense_combine(coils, maps)
        mv = x0.abs().max()
        x0, coils = x0 / mv, coils / mv
        m, _ = create_mask_for_mask_type("random", [0.04], [8])((1, 256, 256, 2), seed=tuple(map(ord, name)))
        mask_hw = mask_cols(m).squeeze(0)[None, :].expand(256, 256)
        k_meas = fft2c(coils) * mask_hw
        x_zf = sense_combine(ifft2c(k_meas), maps)
        # eval_sense_block.py formula
        ss = (maps.abs() ** 2).sum(1); support = ss > 1e-6
        def combine(c):
            num = (maps.conj() * c).sum(1)
            return torch.where(support, num / ss.clamp_min(1e-6), torch.zeros_like(num))
        def dc(xhat):
            kk = fft2c(maps * xhat[:, None])
            return combine(ifft2c(torch.where(mask_hw, k_meas, kk)))
        assert torch.allclose(x_zf, combine(ifft2c(k_meas)), atol=1e-6)
        assert torch.allclose(sense_dc(x_zf * 0.7, maps, k_meas, m), dc(x_zf * 0.7), atol=1e-6)
        # the DC identity needs measurements in the span of the maps (real coils are not exactly): synthesize them
        k_syn = fft2c(sense_expand(x0, maps)) * mask_hw
        assert torch.allclose(sense_dc(x0, maps, k_syn, m), x0, atol=1e-5)
        # oracle maps are coil-0 gauged already
        assert torch.allclose(phase_ref_coil0(maps), maps, atol=1e-5)


class TestTokenGridResize:
    def test_shapes_and_roundtrip(self):
        from models.sense_ops import tok_upsample, tok_downsample, wrap_dc_fn_resized
        maps, support = synth_maps(B=2, C=4, H=64, W=64, r=24.0)
        x0 = synth_image(B=2, H=64, W=64) * support
        x0 = x0 / x0.abs().amax(dim=(1, 2), keepdim=True)
        x = c2r(x0)
        up = tok_upsample(x, 128)
        assert up.shape == (2, 128, 128, 2) and up.abs().max() <= 1.0
        assert torch.equal(tok_upsample(x, 64), x) and torch.equal(tok_downsample(x, 64), x)
        back = tok_downsample(up, 64)
        assert back.shape == x.shape
        err = (back - x).abs().mean() / x.abs().mean()
        assert err < 0.05, float(err)
        # a DC closure on the 64 grid applied to 128-grid images
        mask = col_mask(2, 64, 64, torch.tensor([0, 5, 30, 31, 32, 33, 60]))
        k_meas = fft2c(sense_expand(x0, maps)) * mask_bhw(mask, 64)[:, None]
        dc_tok = wrap_dc_fn_resized(make_sense_dc_fn(k_meas, maps, mask), 64, 128)
        out = dc_tok(up, mask)
        assert out.shape == up.shape
        assert (tok_downsample(out, 64) - x).abs().mean() / x.abs().mean() < 0.05
        assert wrap_dc_fn_resized(make_sense_dc_fn(k_meas, maps, mask), 64, 64)(x, mask).shape == x.shape


class TestExpectedDecoding:
    def test_expected_latent_one_hot_equals_lookup(self):
        import json
        from models.quant import VectorQuantizerM
        from models.sense_ops import expected_latent
        q = VectorQuantizerM(vocab_size=64, vocab_width=16, num_codebooks=4)   # 4 codebooks x 16 entries x 4 dims
        B, L, K, V = 2, 5, 4, 16
        idx = torch.randint(0, V, (B, K, L))
        f_ref = q.idx_to_f(idx)                                                # (B, L, 16)
        logits = torch.full((B, K, L, V), -30.0)
        logits.scatter_(-1, idx[..., None], 30.0)                              # near one-hot
        f = expected_latent(q, logits, 1.0)
        assert f.shape == f_ref.shape and torch.allclose(f, f_ref, atol=1e-4)
        # uniform logits -> codebook mean
        f_u = expected_latent(q, torch.zeros(B, K, L, V), 1.0)
        means = torch.cat([cb.codebook.get_norm_weight().mean(0) for cb in q.codebooks])
        assert torch.allclose(f_u, means.expand(B, L, -1), atol=1e-5)
        # differentiable w.r.t. logits
        lg = torch.zeros(B, K, L, V, requires_grad=True)
        expected_latent(q, lg).sum().backward()
        assert lg.grad is not None and torch.isfinite(lg.grad).all()
