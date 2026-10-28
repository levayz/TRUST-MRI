"""Latent-head variants: linear / mlp / fused (the fused head reads the body state, the token head's chain state and the codebook
latents of the token head's greedy codes). Checks: the cache-free greedy roll-out equals the sampler's cached loop (codes and
states) for the v1, 8-step and 16-step heads; the roll-out is self-consistent under teacher forcing; a fused / mlp model
warm-started from a linear-head checkpoint reproduces it exactly; the checkpoint-to-args inference.

Run: PYTHONPATH="$(ls results/_pipcache/*.whl | tr '\n' ':')$(pwd)" python -m pytest tests/test_latent_head.py -q
"""
import pytest
import torch

from autoregressive.models.gpt_kar import ModelArgs, Transformer, infer_model_args_from_state_dict

K, V, DIM, GRID, DZ = 8, 16 * 8, 64, 2, 16     # 8 codebooks of 16 entries, tiny model, 2x2 grid, 16-d latent (8 chunks of 2)
L = GRID * GRID
HEADS = {"v1": dict(version=1), "8-step": dict(version=2.1, head_interleaved=False), "16-step": dict(version=2.1, head_interleaved=True)}


def make(version, seed=0, **kw):
    torch.manual_seed(seed)
    cfg = ModelArgs(n_layer=1, n_head=4, dim=DIM, vocab_size=V, num_codebooks=K, block_size=L, cond_token_num=L, cls_token_num=1,
                    num_classes=8, model_version=version, mask_width=16, cont_in_dim=8, class_dropout_prob=0.0, **kw)
    m = Transformer(cfg).eval()
    for p in m.parameters():
        p.data.normal_(0, 0.1)
    return m


def fake_codebook(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(K, V // K, DZ // K, generator=g), dim=-1)


def cached_greedy_loop(m, base):
    """The sampler's head loop (generate.py) with greedy codes, returning codes (B, 2, K, L) and the mean chain state (B, L, DIM)."""
    B = base.shape[0]
    N = B * L
    interleaved = bool(getattr(m.output, "interleaved", False))
    m.setup_head_caches(N, 2 * K if interleaved else K, dtype=torch.float32, device=torch.device("cpu"))
    h = base.reshape(N, 1, DIM)
    states = []
    if interleaved:
        codes, prev = [[], []], None
        for s in range(2 * K):
            pos = torch.tensor([s], dtype=torch.int)
            mask = m.output.head_causal_mask[:N, None, pos]
            lg, st = m.output.forward_test(h if s == 0 else None, idx=prev, input_pos=pos, mask=mask, return_state=True)
            prev = lg[:, 0].argmax(-1); codes[s % 2].append(prev); states.append(st[:, 0])
        codes = torch.stack([torch.stack(codes[0], 1), torch.stack(codes[1], 1)], 1)             # (N, 2, K)
    else:
        codes = []
        for i in range(K):
            pos = torch.tensor([i], dtype=torch.int)
            mask = m.output.head_causal_mask[:N, None, pos]
            lg, st = m.output.forward_test(h, idx=None if i == 0 else codes[-1], input_pos=pos, mask=mask, return_state=True)
            codes.append(lg.argmax(-1)); states.append(st[:, 0])                                   # (N, 2, 1)
        codes = torch.stack(codes, 2)[..., 0]                                                      # (N, 2, K)
    m.rm_head_caches()
    return codes.view(B, L, 2, K).permute(0, 2, 3, 1), torch.stack(states, 1).mean(1).view(B, L, DIM)


@pytest.mark.parametrize("head", list(HEADS))
def test_rollout_matches_cached_loop(head):
    kw = dict(HEADS[head]); m = make(kw.pop("version"), latent_dim=DZ, latent_head_type="fused", **kw)
    base = torch.randn(2, L, DIM)
    with torch.no_grad():
        codes, state = m.rollout_head(base)
        codes_c, state_c = cached_greedy_loop(m, base)
    assert codes.shape == (2, 2, K, L) and state.shape == (2, L, DIM)
    assert torch.equal(codes, codes_c)
    assert torch.allclose(state, state_c, atol=1e-4), float((state - state_c).abs().max())


@pytest.mark.parametrize("head", list(HEADS))
def test_rollout_self_consistent_under_teacher_forcing(head):
    """feeding the rolled-out codes back as targets, the teacher-forced argmax must reproduce them step by step."""
    kw = dict(HEADS[head]); m = make(kw.pop("version"), seed=1, latent_dim=DZ, latent_head_type="fused", **kw)
    base = torch.randn(2, L, DIM)
    with torch.no_grad():
        codes, _ = m.rollout_head(base)
        logits = m.output.forward_train(base, codes)                      # (B, 2, K, L, V/K)
    assert torch.equal(logits.argmax(-1), codes)


@pytest.mark.parametrize("kind", ["mlp", "fused"])
def test_warm_start_from_linear_head_is_identity(kind):
    lin = make(2.1, seed=2, head_interleaved=False, latent_dim=DZ, latent_head_type="linear")
    new = make(2.1, seed=3, head_interleaved=False, latent_dim=DZ, latent_head_type=kind)
    res = new.load_state_dict(lin.state_dict(), strict=False)
    assert not res.unexpected_keys
    assert all(k.startswith("latent_mlp.") or k.startswith("latent_fuse_ln.") for k in res.missing_keys), res.missing_keys
    assert "latent_codebook" not in lin.state_dict() and "latent_codebook" not in new.state_dict()   # the codebook is the tokenizer's
    with torch.no_grad():                                                  # fresh residual branch = zero last layer (initialize_weights)
        new.latent_mlp[-1].weight.zero_(); new.latent_mlp[-1].bias.zero_()
    if kind == "fused":
        new.set_latent_codebook(fake_codebook())
    B = 2
    z_in = torch.nn.functional.normalize(torch.randn(B, 2, L, 8), dim=-1)
    cols = (torch.rand(B, 16) > 0.5).float()
    t = torch.zeros(B, dtype=torch.long)
    targets = torch.randint(0, V // K, (B, 2, K, L))
    with torch.no_grad():
        _, _, z_lin = lin(cond_latent=z_in, sample_mask_cols=cols, timestep=t, targets=targets, return_latent=True)
        _, _, z_new = new(cond_latent=z_in, sample_mask_cols=cols, timestep=t, targets=targets, return_latent=True)
    assert z_lin.shape == (B, 2, L, DZ)
    assert torch.allclose(z_lin, z_new, atol=1e-5)
    chunks = z_new.reshape(B, 2, L, K, -1).norm(dim=-1)
    assert torch.allclose(chunks, torch.ones_like(chunks), atol=1e-4)      # unit-norm chunks
    with torch.no_grad():                                                  # a non-zero branch changes the output
        new.latent_mlp[-1].weight.normal_(0, 0.1)
        _, _, z_new2 = new(cond_latent=z_in, sample_mask_cols=cols, timestep=t, targets=targets, return_latent=True)
    assert not torch.allclose(z_lin, z_new2, atol=1e-3)


def test_fused_head_sees_the_token_codes():
    """with the codebook set, changing the codes the fused head is given changes its output (the codes are an input)."""
    m = make(2.1, seed=4, head_interleaved=False, latent_dim=DZ, latent_head_type="fused")
    m.set_latent_codebook(fake_codebook())
    h = torch.randn(2, L, DIM)
    with torch.no_grad():
        codes, state = m.rollout_head(h)
        a = m.predict_latent(h, head_state=state, head_codes=codes)
        b = m.predict_latent(h, head_state=state, head_codes=(codes + 1) % (V // K))
        q = m.codes_to_latent(codes)
    assert q.shape == (2, 2, L, DZ) and not torch.allclose(a, b)
    with pytest.raises(AssertionError):
        make(2.1, seed=4, head_interleaved=False, latent_dim=DZ, latent_head_type="fused").codes_to_latent(codes)   # codebook not set


def test_infer_model_args_from_state_dict():
    for kind, width in [("linear", 0), ("mlp", 2 * DIM), ("fused", 2 * DIM)]:
        m = make(2.1, head_interleaved=False, latent_dim=DZ, latent_head_type=kind)
        got = infer_model_args_from_state_dict(m.state_dict(), default_mask_width=16)
        assert got == dict(model_version=2.1, mask_width=16, draft_cond=False, head_interleaved=False, latent_dim=DZ,
                           latent_head_type=kind, latent_mlp_width=width), (kind, got)
    v1 = make(1)
    got = infer_model_args_from_state_dict(v1.state_dict(), default_mask_width=16)
    assert got["model_version"] == 1 and got["latent_dim"] == 0 and got["latent_head_type"] == "linear"


def test_adapt_mask_width_interpolates_the_mask_embedding():
    from autoregressive.models.gpt_kar import adapt_mask_width
    m16 = make(2.1, seed=5, head_interleaved=False)
    sd = m16.state_dict()
    sd20 = adapt_mask_width(sd, 20)
    assert tuple(sd20["mask_embed.weight"].shape) == (DIM, 20) and tuple(sd["mask_embed.weight"].shape) == (DIM, 16)
    assert torch.allclose(sd20["mask_embed.weight"][:, 0], sd["mask_embed.weight"][:, 0])       # end columns preserved
    assert torch.allclose(sd20["mask_embed.weight"][:, -1], sd["mask_embed.weight"][:, -1])
    assert adapt_mask_width(sd, 16) is sd                                                        # no-op at the same width
    cfg = ModelArgs(n_layer=1, n_head=4, dim=DIM, vocab_size=V, num_codebooks=K, block_size=L, cond_token_num=L, cls_token_num=1,
                    num_classes=8, model_version=2.1, mask_width=20, cont_in_dim=8, class_dropout_prob=0.0, head_interleaved=False)
    m20 = Transformer(cfg)
    res = m20.load_state_dict(sd20, strict=False)
    assert not res.unexpected_keys and not res.missing_keys
