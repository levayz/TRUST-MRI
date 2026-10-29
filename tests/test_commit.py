"""Progressive commitment: the partial draft view (draft_keep), the expected-embedding view and the keep-mask / timestep helpers.
Run: /opt/miniconda3/envs/meditok/bin/python -m pytest tests/test_commit.py -q
"""
import torch
import torch.nn.functional as F

from autoregressive.models.gpt_kar import ModelArgs, Transformer
from autoregressive.models.commit import expected_embedding_view, commit_keep_mask, commit_timestep, codes_to_view

K, V, DIM, GRID, DZ = 8, 16 * 8, 64, 2, 16
L = GRID * GRID


def make(seed=0):
    torch.manual_seed(seed)
    cfg = ModelArgs(n_layer=1, n_head=4, dim=DIM, vocab_size=V, num_codebooks=K, block_size=L, cond_token_num=L, cls_token_num=1,
                    num_classes=8, model_version=2.1, head_interleaved=False, mask_width=16, cont_in_dim=DZ, class_dropout_prob=0.0,
                    draft_cond=True, latent_dim=DZ)
    m = Transformer(cfg).eval()
    for p in m.parameters():
        p.data.normal_(0, 0.1)
    return m


def inputs(B=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    cond = F.normalize(torch.randn(B, 2, L, DZ, generator=g).reshape(B, 2, L, K, DZ // K), dim=-1).reshape(B, 2, L, DZ)
    draft = F.normalize(torch.randn(B, 2, L, DZ, generator=g).reshape(B, 2, L, K, DZ // K), dim=-1).reshape(B, 2, L, DZ)
    targets = torch.randint(0, V // K, (B, 2, K, L), generator=g)
    cols = (torch.rand(B, 16, generator=g) > 0.5).float()
    return cond, draft, targets, cols


def run(m, cond, targets, cols, t=1, **kw):
    with torch.no_grad():
        out = m(targets=targets, cond_latent=cond, timestep=torch.full((cond.shape[0],), t, dtype=torch.long), sample_mask_cols=cols, **kw)
    return out[0] if isinstance(out, tuple) else out


def test_keep_zero_equals_no_draft_and_keep_one_equals_full_draft():
    m = make(); cond, draft, targets, cols = inputs()
    base = run(m, cond, targets, cols)
    full = run(m, cond, targets, cols, draft_latent=draft)
    none = run(m, cond, targets, cols, draft_latent=draft, draft_keep=torch.zeros(2, L))
    ones = run(m, cond, targets, cols, draft_latent=draft, draft_keep=torch.ones(2, L))
    assert torch.allclose(none, base, atol=1e-5) and torch.allclose(ones, full, atol=1e-5)
    assert not torch.allclose(full, base, atol=1e-3), "the draft view must change the logits"


def test_partial_keep_only_changes_through_kept_positions():
    m = make(); cond, draft, targets, cols = inputs()
    keep = torch.zeros(2, L); keep[:, 0] = 1
    draft2 = draft.clone(); draft2[:, :, 1:] = torch.randn_like(draft2[:, :, 1:])        # dropped positions: the draft content is irrelevant
    a = run(m, cond, targets, cols, draft_latent=draft, draft_keep=keep)
    b = run(m, cond, targets, cols, draft_latent=draft2, draft_keep=keep)
    assert torch.allclose(a, b, atol=1e-5)


def test_expected_embedding_view_one_hot_and_uniform():
    g = torch.Generator().manual_seed(0)
    cb = F.normalize(torch.randn(K, V // K, DZ // K, generator=g), dim=-1)
    codes = torch.randint(0, V // K, (2, L, 2, K), generator=g)
    logits = torch.full((2, L, 2, K, V // K), -1e4); logits.scatter_(-1, codes[..., None], 1e4)   # one-hot
    view, conf = expected_embedding_view(logits, cb)
    assert view.shape == (2, 2, L, DZ) and conf.shape == (2, L)
    assert torch.allclose(conf, torch.ones(2, L), atol=1e-5)
    rows = torch.stack([cb[k][codes[..., k]] for k in range(K)], dim=-2)                     # (2, L, 2, K, d)
    assert torch.allclose(view, rows.permute(0, 2, 1, 3, 4).reshape(2, 2, L, DZ), atol=1e-5)
    _, conf_u = expected_embedding_view(torch.zeros(2, L, 2, K, V // K), cb)                  # uniform: |mean of the rows|^2 << 1
    assert conf_u.max() < 0.5


def test_keep_mask_fraction_monotone_and_random():
    conf = torch.rand(3, L)
    k1 = commit_keep_mask(conf, 0.5, "conf")
    assert torch.equal(k1.sum(1), torch.full((3,), round(0.5 * L), dtype=torch.float))
    top = conf.argsort(1, descending=True)[:, : round(0.5 * L)]
    assert all(k1[i, top[i]].all() for i in range(3))
    k2 = commit_keep_mask(torch.rand(3, L), 0.75, "conf", already=k1)
    assert (k2 >= k1).all() and torch.equal(k2.sum(1), torch.full((3,), round(0.75 * L), dtype=torch.float))
    kr = commit_keep_mask(conf, torch.tensor([0.25, 0.5, 1.0]), "random")
    assert kr.sum(1).tolist() == [round(0.25 * L), round(0.5 * L), L]


def test_timestep_from_fraction():
    assert commit_timestep(torch.tensor([0.0, 0.3, 0.55, 0.8, 0.9, 1.0])).tolist() == [1, 2, 3, 4, 5, 5]


def test_loss_mask_all_ones_equals_plain_loss_and_single_position_matches_manual():
    m = make(); cond, draft, targets, cols = inputs()
    with torch.no_grad():
        _, loss = m(targets=targets, cond_latent=cond, timestep=torch.ones(2, dtype=torch.long), sample_mask_cols=cols)
        _, loss_ones = m(targets=targets, cond_latent=cond, timestep=torch.ones(2, dtype=torch.long), sample_mask_cols=cols, loss_mask=torch.ones(2, L))
        assert torch.allclose(loss, loss_ones, atol=1e-6)
        mask = torch.zeros(2, L); mask[:, 1] = 1
        logits, loss_one = m(targets=targets, cond_latent=cond, timestep=torch.ones(2, dtype=torch.long), sample_mask_cols=cols, loss_mask=mask)
        V = logits.shape[-1]
        manual = sum(F.cross_entropy(logits[:, s, :, 1, :].reshape(-1, V), targets[:, s, :, 1].reshape(-1)) for s in range(2)) / 2
        assert torch.allclose(loss_one, manual, atol=1e-5), (float(loss_one), float(manual))


def test_codes_to_view_matches_the_fused_head_lookup():
    g = torch.Generator().manual_seed(0)
    cb = F.normalize(torch.randn(K, V // K, DZ // K, generator=g), dim=-1)
    codes = torch.randint(0, V // K, (2, 2, K, L), generator=g)
    m = make(); m.set_latent_codebook_tensor(cb) if hasattr(m, "set_latent_codebook_tensor") else None
    view = codes_to_view(codes, cb)
    assert view.shape == (2, 2, L, DZ)
    rows = torch.cat([cb[k][codes[:, :, k]] for k in range(K)], dim=-1)
    assert torch.equal(view, rows)
    one_hot = torch.full((2, L, 2, K, V // K), -1e4); one_hot.scatter_(-1, codes.permute(0, 3, 1, 2)[..., None], 1e4)
    ev, _ = expected_embedding_view(one_hot, cb)
    assert torch.allclose(ev, view, atol=1e-5), "hard rows must equal the expected embedding of a one-hot belief"
