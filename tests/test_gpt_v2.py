"""Model v2 unit tests: interleaved Re/Im head (teacher-forced vs step-wise consistency), continuous input, v1->v2 remap.

Run: PYTHONPATH="$(ls results/_pipcache/*.whl | tr '\n' ':')$(pwd)" python -m pytest tests/test_gpt_v2.py -q
"""
import torch

from autoregressive.models.gpt_kar import ModelArgs, Transformer, remap_v1_to_v2

K, V, DIM, GRID = 8, 16 * 8, 64, 2          # 8 codebooks of 16 entries, tiny model, 2x2 token grid
L = GRID * GRID


def make(version, **kw):
    cfg = ModelArgs(n_layer=1, n_head=4, dim=DIM, vocab_size=V, num_codebooks=K, block_size=L, cond_token_num=L, cls_token_num=1,
                    num_classes=8, model_version=version, mask_width=16, cont_in_dim=8, class_dropout_prob=0.0, **kw)
    m = Transformer(cfg).eval()
    for p in m.parameters():                 # non-degenerate weights (the output heads are zero-initialised by design)
        p.data.normal_(0, 0.1)
    return m


def test_interleaved_head_train_matches_stepwise():
    torch.manual_seed(0)
    m = make(2.1)
    B = 2
    base = torch.randn(B, L, DIM)
    targets = torch.randint(0, V // K, (B, 2, K, L))
    with torch.no_grad():
        logits_tf = m.output.forward_train(base, targets)                       # (B, 2, K, L, V/K)
        # step-wise with the KV cache, feeding the same target codes
        m.setup_head_caches(B * L, 2 * K, dtype=torch.float32, device=torch.device("cpu"))
        h = base.reshape(B * L, 1, DIM)
        t_re = targets[:, 0].permute(0, 2, 1).reshape(B * L, K)
        t_im = targets[:, 1].permute(0, 2, 1).reshape(B * L, K)
        got = {0: [], 1: []}
        prev = None
        for s in range(2 * K):
            pos = torch.tensor([s], dtype=torch.int)
            mask = m.output.head_causal_mask[:B * L, None, pos]
            lg = m.output.forward_test(h if s == 0 else None, idx=prev, input_pos=pos, mask=mask)   # (N, 1, V/K)
            got[s % 2].append(lg[:, 0])
            prev = t_re[:, s // 2] if s % 2 == 0 else t_im[:, s // 2]
        m.rm_head_caches()
    sw = torch.stack([torch.stack(got[0], 1), torch.stack(got[1], 1)], 1)       # (N, 2, K, V/K)
    sw = sw.reshape(B, L, 2, K, -1).permute(0, 2, 3, 1, 4)
    assert torch.allclose(logits_tf, sw, atol=1e-4), float((logits_tf - sw).abs().max())


def test_im_conditioned_on_re_same_codebook():
    """changing Re code i must change the Im-code-i logits (interleaving), but not the Re-code-i logits (causality)."""
    torch.manual_seed(1)
    m = make(2.1)
    base = torch.randn(1, L, DIM)
    t = torch.randint(0, V // K, (1, 2, K, L))
    t2 = t.clone(); t2[0, 0, 3] = (t2[0, 0, 3] + 1) % (V // K)      # flip Re code of codebook 3 everywhere
    with torch.no_grad():
        a, b = m.output.forward_train(base, t), m.output.forward_train(base, t2)
    assert torch.allclose(a[:, 0, :4], b[:, 0, :4])                  # Re logits of codebooks 0..3 unchanged
    assert not torch.allclose(a[:, 1, 3], b[:, 1, 3])                 # Im logits of codebook 3 changed
    assert torch.allclose(a[:, 1, :3], b[:, 1, :3])                   # Im logits of codebooks 0..2 unchanged


def test_v21_forward_continuous_input_and_mask():
    torch.manual_seed(2)
    m = make(2.1)
    B = 2
    z = torch.nn.functional.normalize(torch.randn(B, 2, L, 8), dim=-1)
    cols = (torch.rand(B, 16) > 0.5).float()
    t = torch.zeros(B, dtype=torch.long)
    targets = torch.randint(0, V // K, (B, 2, K, L))
    with torch.no_grad():
        h = m(cond_latent=z, sample_mask_cols=cols, timestep=t)
        assert h.shape == (B, L, DIM)
        logits, loss = m(cond_latent=z, sample_mask_cols=cols, timestep=t, targets=targets)[:2]
        assert logits.shape == (B, 2, K, L, V // K) and torch.isfinite(loss)
        h2 = m(cond_latent=z.flip(0), sample_mask_cols=cols, timestep=t)
        assert not torch.allclose(h, h2)                                         # the input matters
    try:
        m(cond_latent=z, timestep=t); raise AssertionError("v2.1 must refuse to run without the mask")
    except AssertionError as e:
        assert "sample_mask_cols" in str(e) or "mask" in str(e)
    assert not hasattr(m, "re_tok_embeddings")                                   # no token tables in v2.1


def test_remap_v1_to_v2_loads():
    v1, v2 = make(1), make(2.1)
    sd = remap_v1_to_v2(v1.state_dict())
    res = v2.load_state_dict(sd, strict=False)
    assert not res.unexpected_keys, res.unexpected_keys
    missing = set(res.missing_keys)
    assert missing == {"t_gate", "m_gate", "mask_embed.weight", "mask_embed.bias", "in_proj.0.weight", "in_proj.0.bias", "in_proj.2.weight",
                       "in_proj.2.bias", "output.pos_emb", f"output.codebooks_re.{K - 1}.weight"}, missing
    assert torch.equal(v2.output.codebooks_re[0].weight, v1.output.codebooks[0].weight)
    assert torch.equal(v2.output.codebooks_im[0].weight, v1.output.codebooks[0].weight)


def test_v2_token_input_with_mask():
    """v2 = v1 token input + mask / timestep conditioning (what run9 trains); the old head is kept."""
    torch.manual_seed(3)
    m = make(2)
    assert hasattr(m, "re_tok_embeddings") and not getattr(m.output, "interleaved", False) and hasattr(m, "mask_embed")
    B = 2
    cond = torch.randint(0, V // K, (B, 2, K, L))
    cols = (torch.rand(B, 16) > 0.5).float()
    t = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        h = m(cond_tokens=cond, sample_mask_cols=cols, timestep=t)
        assert h.shape == (B, L, DIM)
        h2 = m(cond_tokens=cond, sample_mask_cols=cols.flip(0), timestep=t)
        assert not torch.allclose(h, h2)
    v1 = make(1)
    res = m.load_state_dict(v1.state_dict(), strict=False)
    assert not res.unexpected_keys and set(res.missing_keys) == {"t_gate", "m_gate", "mask_embed.weight", "mask_embed.bias"}


def test_asym_head_train_matches_stepwise_and_warm_start():
    torch.manual_seed(4)
    m = make(2.1, head_interleaved=False)
    assert not getattr(m.output, "interleaved", True)
    B = 2
    base = torch.randn(B, L, DIM)
    targets = torch.randint(0, V // K, (B, 2, K, L))
    with torch.no_grad():
        tf = m.output.forward_train(base, targets)
        m.setup_head_caches(B * L, K, dtype=torch.float32, device=torch.device("cpu"))
        h = base.reshape(B * L, 1, DIM)
        t = targets.permute(0, 3, 1, 2).reshape(B * L, 2, K)                     # (N, 2, K)
        outs = []
        for i in range(K):
            pos = torch.tensor([i], dtype=torch.int)
            mask = m.output.head_causal_mask[:B * L, None, pos]
            lg = m.output.forward_test(h if i == 0 else None, idx=None if i == 0 else t[:, :, i - 1:i], input_pos=pos, mask=mask)  # (N, 2, 1, V)
            outs.append(lg[:, :, 0])
        m.rm_head_caches()
    sw = torch.stack(outs, 2).reshape(B, L, 2, K, -1).permute(0, 2, 3, 1, 4)
    assert torch.allclose(tf, sw, atol=1e-4), float((tf - sw).abs().max())
    # warm start: with v1 weights remapped, the asym head equals the v1 head at init (merge == sum, pos_emb == 0)
    v1 = make(1)
    m2 = make(2.1, head_interleaved=False)
    res = m2.load_state_dict(remap_v1_to_v2(v1.state_dict()), strict=False)
    assert not res.unexpected_keys
    with torch.no_grad():
        m2.output.merge.weight.copy_(torch.cat([torch.eye(DIM), torch.eye(DIM)], 1)); m2.output.pos_emb.zero_()
        a = v1.output.forward_train(base, targets); b = m2.output.forward_train(base, targets)
    assert torch.allclose(a, b, atol=1e-5)
