"""Progressive commitment (MaskGIT-style) helpers for the two-view SENSE GPT.

After pass 1 the token head gives, per position, 16 chunk distributions (2 streams x 8 codebooks). The expected codebook embedding
of each chunk, mu = sum_v p(v) e_v with unit-norm rows e_v, has |mu|^2 = 1 - spread: it is 1 when the distribution sits on one row
(or on near-duplicate rows) and small when it is spread out. The confidence of a position is the mean of |mu|^2 over its 16 chunks
(diagnostic of Oct 28: rank correlation -0.88 with the latent error inside the head; top-1 probability, entropy and draw
agreement have none). The committed second view at a kept position is the renormalised expected embedding, not the argmax code's row.
"""
import torch
import torch.nn.functional as F


def expected_embedding_view(logits: torch.Tensor, codebook: torch.Tensor, temperature: float = 1.0):
    """logits (B, L, 2, K, V) of pass 1, codebook (K, V, d) unit-norm rows ->
    view (B, 2, L, K*d): per-chunk renormalised expected embeddings in the draft_latent layout; conf (B, L): mean |mu|^2 over the chunks."""
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    mu = torch.einsum("blskv,kvd->blskd", probs, codebook.float())                     # (B, L, 2, K, d)
    conf = (mu ** 2).sum(-1).mean((2, 3))                                                # (B, L)
    B, L, S, K, d = mu.shape
    view = F.normalize(mu, dim=-1).permute(0, 2, 1, 3, 4).reshape(B, S, L, K * d)        # (B, 2, L, K*d)
    return view, conf


def codes_to_view(codes: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """codes (B, 2, K, L) -> (B, 2, L, K*d): the codebook rows of the codes in the draft_latent layout (the quantised latent;
    the ground truth's rows in MaskGIT-style training, the sampled codes' rows at inference)."""
    return torch.cat([codebook[k][codes[:, :, k]] for k in range(codes.shape[2])], dim=-1)


def commit_keep_mask(conf: torch.Tensor, frac, mode: str = "conf", already: torch.Tensor = None) -> torch.Tensor:
    """(B, L) float mask with round(frac * L) ones per sample: the most confident positions (mode 'conf') or random ones ('random').
    Positions already kept (`already`, (B, L)) stay kept and count toward the fraction."""
    B, L = conf.shape
    frac = torch.as_tensor(frac, device=conf.device, dtype=torch.float32).reshape(-1).expand(B) if torch.as_tensor(frac).numel() == 1 \
        else torch.as_tensor(frac, device=conf.device, dtype=torch.float32)
    n_keep = (frac * L).round().long().clamp(0, L)                                       # (B,)
    score = conf.float().clone() if mode == "conf" else torch.rand(B, L, device=conf.device)
    if already is not None:
        score = torch.where(already.bool(), torch.full_like(score, float("inf")), score)
    order = score.argsort(dim=1, descending=True)
    rank = torch.empty_like(order); rank.scatter_(1, order, torch.arange(L, device=conf.device).expand(B, L))
    keep = (rank < n_keep[:, None]).float()
    if already is not None:
        keep = torch.maximum(keep, already.float())
    return keep


def commit_timestep(frac) -> torch.Tensor:
    """Committed fraction -> timestep in 1..5 (pass 1 is timestep 0): 1 + round(4 * frac)."""
    f = torch.as_tensor(frac, dtype=torch.float32).reshape(-1)
    return (1 + (4 * f).round().long()).clamp(1, 5)
