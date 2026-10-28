"""Privileged-teacher distillation helpers (autoregressive/train/train.py): the enriched teacher mask and the chunked KL.

Run: PYTHONPATH="$(ls results/_pipcache/*.whl | tr '\n' ':')$(pwd)" python -m pytest tests/test_distill.py -q
"""
import torch

from autoregressive.train.train import enrich_mask_cols, kl_to_teacher


def test_enrich_mask_cols_adds_random_columns_to_the_target_budget():
    torch.manual_seed(0)
    B, W = 64, 256
    cols = torch.rand(B, W) < 0.12                                   # ~31 sampled columns, like the x8 random mask
    out = enrich_mask_cols(cols, 2.0, 4.0)
    assert out.shape == cols.shape and out.dtype == torch.bool
    assert bool((out | cols).eq(out).all())                          # superset of the student's columns
    n = out.sum(1)
    assert int(n.min()) >= 60 and int(n.max()) <= 130                # W/4 = 64 ... W/2 = 128 (rounding)
    assert int(n.min()) >= int(cols.sum(1).min())
    full = torch.ones(2, W, dtype=torch.bool)
    assert bool(enrich_mask_cols(full, 2.0, 4.0).all())              # nothing to add when everything is sampled
    assert bool(enrich_mask_cols(cols, 1.0, 1.0).all())              # R = 1 fills every column


def test_kl_to_teacher_zero_for_identical_logits_and_weighted_per_sample():
    torch.manual_seed(1)
    B, K, L, V = 3, 8, 5, 16
    ls = torch.randn(B, 2, K, L, V)
    w = torch.ones(B)
    assert float(kl_to_teacher(ls, ls, 1.0, w)) < 1e-6
    lt = ls + torch.randn_like(ls)
    kl_all = kl_to_teacher(ls, lt, 1.0, w)
    assert float(kl_all) > 0
    per = [float(kl_to_teacher(ls[i:i + 1], lt[i:i + 1], 1.0, torch.ones(1))) for i in range(B)]
    assert abs(float(kl_all) - sum(per) / B) < 1e-5                # mean over samples with unit weights
    w2 = torch.tensor([1.0, 0.0, 0.0])
    assert abs(float(kl_to_teacher(ls, lt, 1.0, w2)) - per[0]) < 1e-5   # zero-weighted samples drop out
    assert float(kl_to_teacher(ls, lt, 1.0, torch.zeros(B))) == 0.0
    # reference: torch's kl_div on the full tensor
    ref = torch.nn.functional.kl_div(torch.log_softmax(ls, -1), torch.log_softmax(lt, -1), log_target=True, reduction="none").sum(-1).mean()
    assert abs(float(kl_all) - float(ref)) < 1e-5
