"""Token Distribution Shift (TDS) active k-space acquisition policy.

Core idea:
  At each acquisition step, decode once to get x_hat, compute the predicted
  full k-space y_hat = FFT(x_hat), then for every unacquired phase-encoding
  line j simulate its acquisition and re-run the encoder+Transformer (no
  decode) to measure how much the token distributions shift.  The line that
  produces the largest foreground-weighted distribution shift is acquired.

References in the codebase:
  - generate.py: reconstruct_far, active_sampling_patch_logits, data_consistency
  - adaptive_sampling.py: kspace_uncertainty_from_patch_logits (LES baseline)
"""

from __future__ import annotations

import math
import time
import warnings

import torch
import torch.nn.functional as F

from data import transforms

EPS = 1e-8


# ---------------------------------------------------------------------------
# Probability / entropy utilities
# ---------------------------------------------------------------------------

def logits_to_probs_tds(logits: torch.Tensor) -> torch.Tensor:
    """Convert patch logits to clamped probability distributions.

    Args:
        logits: (B, H_p, W_p, 2, K, V_K)  — from reconstruct_far aux["logits"]

    Returns:
        probs: (B, L, C, V_K)  where L = H_p * W_p, C = 2 * K
    """
    B, H_p, W_p, ch, K, V_K = logits.shape
    L = H_p * W_p
    logits_flat = logits.reshape(B, L, ch * K, V_K).float()
    probs = F.softmax(logits_flat, dim=-1).clamp(min=EPS)
    return probs


def compute_patch_entropy(probs: torch.Tensor):
    """Per-patch entropy and total entropy (natural log, nats).

    Args:
        probs: (B, L, C, V_K) — already clamped

    Returns:
        h:       (B, L) per-patch entropy summed over codebooks
        H_total: (B,)   sum of all patch entropies
    """
    log_p = probs.log()
    h_per_cb = -(probs * log_p).sum(dim=-1)  # (B, L, C)
    h = h_per_cb.sum(dim=-1)                  # (B, L)
    H_total = h.sum(dim=-1)                   # (B,)
    return h, H_total


# ---------------------------------------------------------------------------
# Distribution distance functions
# ---------------------------------------------------------------------------

def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence per patch, summed over codebooks.

    JS(p||q) = 0.5*KL(p||m) + 0.5*KL(q||m),  m = 0.5*(p+q)

    Args:
        p, q: (B, L, C, V_K)

    Returns:
        (B, L)
    """
    m = (0.5 * (p + q)).clamp(min=EPS)
    p_s = p.clamp(min=EPS)
    q_s = q.clamp(min=EPS)
    kl_pm = (p_s * (p_s.log() - m.log())).sum(dim=-1)   # (B, L, C)
    kl_qm = (q_s * (q_s.log() - m.log())).sum(dim=-1)
    return (0.5 * kl_pm + 0.5 * kl_qm).sum(dim=-1)      # (B, L)


def symmetric_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Symmetric KL divergence per patch, summed over codebooks.

    Args:
        p, q: (B, L, C, V_K)

    Returns:
        (B, L)
    """
    p_s = p.clamp(min=EPS)
    q_s = q.clamp(min=EPS)
    kl_pq = (p_s * (p_s.log() - q_s.log())).sum(dim=-1)  # (B, L, C)
    kl_qp = (q_s * (q_s.log() - p_s.log())).sum(dim=-1)
    return (0.5 * (kl_pq + kl_qp)).sum(dim=-1)            # (B, L)


def topk_l1(p: torch.Tensor, q: torch.Tensor, top_k: int = 10) -> torch.Tensor:
    """L1 distance restricted to the top-K entries of p per codebook.

    Args:
        p, q: (B, L, C, V_K)
        top_k: number of top entries per codebook to include

    Returns:
        (B, L)
    """
    k = min(top_k, p.shape[-1])
    vals, _ = p.topk(k, dim=-1)                      # (B, L, C, k)
    threshold = vals[..., -1:].clamp(min=0.0)         # (B, L, C, 1)
    mask = (p >= threshold).float()
    l1_per_cb = ((p - q).abs() * mask).sum(dim=-1)   # (B, L, C)
    return l1_per_cb.sum(dim=-1)                      # (B, L)


# ---------------------------------------------------------------------------
# Foreground weighting
# ---------------------------------------------------------------------------

def compute_foreground_weights(
    x_hat: torch.Tensor,
    h_old: torch.Tensor,
    H_p: int,
    W_p: int,
    *,
    mode: str,
    threshold: float = 0.0,
    percentile: float = 20.0,
    top_m: int = 64,
    entropy_normalization: bool = True,
    C: int | None = None,
    V_K: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Foreground-aware uncertainty weights  w_l = fg_l * h_norm_l.

    Args:
        x_hat:  (B, H, W, 2) decoded reconstruction (no DC)
        h_old:  (B, L) per-patch entropy
        H_p, W_p: patch grid dimensions
        mode:   "none" | "threshold" | "percentile" | "top_m_entropy_foreground"
        threshold: absolute magnitude threshold (mode="threshold")
        percentile: percentile cutoff (mode="percentile")
        top_m:  keep this many top-entropy foreground patches
        entropy_normalization: divide h by C*log(V_K)
        C, V_K: codebook count and vocab size (needed for normalization)

    Returns:
        w:     (B, L) weight per patch
        n_fg:  number of foreground patches (first batch item)
    """
    B, H_img, W_img, _ = x_hat.shape
    L = H_p * W_p
    device = x_hat.device

    # Normalized entropy
    if entropy_normalization and C is not None and V_K is not None:
        max_H = C * math.log(max(V_K, 2))
        h_norm = h_old / max(max_H, EPS)
    else:
        h_norm = h_old.clone()

    # Per-patch mean magnitude
    mag = transforms.complex_abs(x_hat)          # (B, H_img, W_img)
    ph = H_img // H_p
    pw = W_img // W_p
    patch_mag = (
        mag.reshape(B, H_p, ph, W_p, pw)
        .mean(dim=(2, 4))                          # (B, H_p, W_p)
        .reshape(B, L)                             # (B, L)
    )

    if mode == "none":
        fg = torch.ones(B, L, device=device, dtype=h_old.dtype)

    elif mode == "threshold":
        fg = (patch_mag > threshold).to(h_old.dtype)

    elif mode == "percentile":
        pct_val = torch.quantile(
            patch_mag.float(), percentile / 100.0, dim=-1, keepdim=True
        )
        fg = (patch_mag >= pct_val).to(h_old.dtype)

    elif mode == "top_m_entropy_foreground":
        pct_val = torch.quantile(patch_mag.float(), 0.20, dim=-1, keepdim=True)
        fg_base = (patch_mag >= pct_val).to(h_old.dtype)
        h_fg = h_old * fg_base
        m = min(top_m, L)
        _, top_idx = h_fg.topk(m, dim=-1)
        fg = torch.zeros(B, L, device=device, dtype=h_old.dtype)
        fg.scatter_(-1, top_idx, 1.0)

    else:
        raise ValueError(f"Unknown tds_foreground_mode: {mode!r}")

    n_fg = int((fg[0] > 0).sum().item())
    if n_fg == 0:
        warnings.warn(
            f"TDS: all foreground patches are zero (mode={mode!r}). "
            "Falling back to uniform weighting."
        )
        fg = torch.ones_like(fg)
        n_fg = L

    w = fg * h_norm  # (B, L)
    return w, n_fg


# ---------------------------------------------------------------------------
# Lightweight encode-and-infer (no decode)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode_and_get_logits(
    model,
    tokenizer,
    image: torch.Tensor,
    device: torch.device,
    sampling_kwargs: dict,
) -> torch.Tensor:
    """Encode `image` through encoder + Transformer + AR head — **no decode**.

    Replicates the encode/infer code path from reconstruct_far (timestep=0,
    single FAR step) but skips the tokenizer decoder.

    Args:
        model:           the reconstruction Transformer model
        tokenizer:       MedITok tokenizer (frozen)
        image:           (B, H, W, 2) — zero-filled simulated image
        device:          torch device
        sampling_kwargs: passed to sample_far (controls AR head conditioning)

    Returns:
        logits: (B, H_p, W_p, 2, K, V_K) — same format as aux["logits"]
    """
    from autoregressive.models.generate import sample_far  # lazy — avoid circular import

    B, H, W, _ = image.shape

    def img_to_tokens(img_bhw: torch.Tensor) -> torch.Tensor:
        """(B, H, W) → (B, K, L) discrete indices — mirrors local encode() in reconstruct_far."""
        img_3ch = img_bhw.unsqueeze(-1).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)  # (B,3,H,W)
        enc = tokenizer.encoder(img_3ch)       # (B, L, D_enc)
        enc = tokenizer.quant_proj(enc)         # (B, L, D_vq)
        return tokenizer.quantizer.f_to_idx(enc)  # (B, K, L)

    tok_re = img_to_tokens(image[..., 0])  # (B, K, L)
    tok_im = img_to_tokens(image[..., 1])
    cond_tokens = torch.cat(
        [tok_re.unsqueeze(1), tok_im.unsqueeze(1)], dim=1
    )  # (B, 2, K, L)

    t = torch.zeros(B, device=device, dtype=torch.long)
    h = model(cond_tokens=cond_tokens.long(), timestep=t)
    if isinstance(h, tuple):
        h = h[0]

    h_ndim = h.ndim
    if h_ndim == 3:
        _, L, D = h.shape
    elif h_ndim == 4:
        _, _, L, D = h.shape
    else:
        raise ValueError(f"Unexpected transformer output ndim: {h_ndim}, shape={tuple(h.shape)}")

    model.setup_head_caches(B * L, model.num_codebooks, dtype=h.dtype, device=device)

    if h_ndim == 3:
        h_flat = h.view(B * L, 1, D)
    else:
        h_flat = h.permute(0, 2, 1, 3).contiguous().view(B * L, 2, 1, D)

    pred_idx_list: list[torch.Tensor] = []
    logits_list:   list[torch.Tensor] = []

    for i in range(model.num_codebooks):
        start_pos = torch.tensor([i], dtype=torch.int)
        mask = model.output.head_causal_mask[:B * L, None, start_pos]
        if i == 0:
            lg = model.output.forward_test(h_flat, input_pos=start_pos, mask=mask)
        else:
            lg = model.output.forward_test(
                h_flat, idx=pred_idx_list[-1], input_pos=start_pos, mask=mask
            )
        logits_list.append(lg)
        _pred = sample_far(lg, **sampling_kwargs)[0].unsqueeze(-1)
        pred_idx_list.append(_pred)

    model.rm_head_caches()

    # Stack and reshape to match reconstruct_far's logits format
    # logits_list: K × (B*L, 2, 1, V_K) → stack → (B*L, 2, K, 1, V_K)
    logits = torch.stack(logits_list, dim=2)                          # (B*L, 2, K, 1, V_K)
    logits = logits.view(B, L, 2, model.num_codebooks, -1)            # (B, L, 2, K, V_K)
    H_p = W_p = int(L ** 0.5)
    logits = logits.view(B, H_p, W_p, 2, model.num_codebooks, -1)    # (B, H_p, W_p, 2, K, V_K)
    return logits


# ---------------------------------------------------------------------------
# Main TDS scoring step
# ---------------------------------------------------------------------------

@torch.no_grad()
def tds_score_step(
    model,
    tokenizer,
    y_omega: torch.Tensor,
    sample_mask: torch.Tensor,
    x_hat: torch.Tensor,
    logits_old: torch.Tensor,
    device: torch.device,
    sampling_kwargs: dict | None = None,
    *,
    tds_distance: str = "js",
    tds_foreground_mode: str = "percentile",
    tds_foreground_threshold: float = 0.0,
    tds_foreground_percentile: float = 20.0,
    tds_top_m: int = 64,
    tds_alpha: float = 1.0,
    tds_beta: float = 0.0,
    tds_candidate_mode: str = "all",
    tds_candidate_k: int = 16,
    tds_entropy_normalization: bool = True,
    tds_line_aggregation: str = "sum",
    tds_candidate_batch_size: int = 16,
) -> dict:
    """Select the next phase-encoding line using Token Distribution Shift.

    Decode-once protocol:
      - x_hat and logits_old come from a single prior reconstruct_far call.
      - For each candidate, only encode → Transformer → AR head (no decode).

    Args:
        model, tokenizer: frozen models
        y_omega:      (B, H, W, 2) current acquired k-space (zeros at unacquired)
        sample_mask:  (B, H, W, 2) binary acquisition mask
        x_hat:        (B, H, W, 2) decoded prediction, no data-consistency applied
                      (= aux["decoded_no_dc"] from reconstruct_far)
        logits_old:   (B, H_p, W_p, 2, K, V_K) from aux["logits"]
        device:       torch device
        sampling_kwargs: kwargs passed to sample_far in the AR head
        tds_distance: "js" | "symmetric_kl" | "topk_l1"
        tds_foreground_mode: "none" | "threshold" | "percentile" | "top_m_entropy_foreground"
        tds_foreground_threshold: magnitude threshold (threshold mode)
        tds_foreground_percentile: percentile cutoff (percentile mode)
        tds_top_m:    number of top-entropy patches (top_m_entropy_foreground mode)
        tds_alpha:    weight on TDS score
        tds_beta:     weight on entropy-reduction term (0 → plain TDS)
        tds_candidate_mode: "all" | "geo_topk"
        tds_candidate_k: shortlist size for geo_topk mode
        tds_entropy_normalization: divide h by C*log(V_K) before weighting
        tds_line_aggregation: "sum" | "mean" over patches
        tds_candidate_batch_size: number of candidates processed per model call

    Returns:
        dict with keys:
            selected_col (int)
            scores_1d    (B, W) float tensor, -inf at acquired columns
            diagnostics  dict
    """
    if sampling_kwargs is None:
        sampling_kwargs = {"temperature": 1.0, "top_k": 0, "top_p": 1.0}

    B, H, W, _ = y_omega.shape
    if B > 1:
        raise NotImplementedError(
            "TDS currently supports only B=1 (single-slice evaluation). "
            f"Got B={B}."
        )

    B_logits, H_p, W_p, ch, K, V_K = logits_old.shape
    C_total = ch * K   # e.g. 2*8 = 16 combined codebooks (re+im × K)

    # ---- 1. Old distributions and entropies ----
    p_old = logits_to_probs_tds(logits_old)       # (B, L, C, V_K)
    h_old, H_old = compute_patch_entropy(p_old)   # (B, L), (B,)

    # ---- 2. Foreground weights ----
    w, n_fg = compute_foreground_weights(
        x_hat, h_old, H_p, W_p,
        mode=tds_foreground_mode,
        threshold=tds_foreground_threshold,
        percentile=tds_foreground_percentile,
        top_m=tds_top_m,
        entropy_normalization=tds_entropy_normalization,
        C=C_total,
        V_K=V_K,
    )  # (B, L)

    # ---- 3. Predicted full k-space ----
    y_hat = transforms.fft2(x_hat)  # (B, H, W, 2)

    # ---- 4. Candidate columns ----
    acquired = sample_mask[:, 0, :, 0].bool()   # (B, W)
    unacquired = (~acquired[0]).nonzero(as_tuple=True)[0]  # (N_cand,)

    if tds_candidate_mode == "geo_topk":
        # One cheap GEO gradient pass scores all columns; TDS only evaluates top-K.
        # This gives ~(N_cand / K) × speedup (typically 15–20×).
        from autoregressive.sample.adaptive_sampling import geo_step as _geo_step
        masked_image_cur = transforms.ifft2(y_omega)   # (B, H, W, 2)
        _, geo_scores = _geo_step(
            model, tokenizer, masked_image_cur, sample_mask, rank=1, device=device
        )  # geo_scores: (W,) — already 0 at acquired cols
        k_short = min(tds_candidate_k, len(unacquired))
        unacquired_scores = geo_scores[unacquired]      # (N_cand,) scores for unacquired only
        _, local_idx = unacquired_scores.topk(k_short)
        unacquired = unacquired[local_idx]              # (k_short,) top-K candidates

    N_cand = len(unacquired)
    scores_1d = torch.full((B, W), float("-inf"), device=device, dtype=torch.float32)
    H_j_per_col: dict[int, float] = {}

    # ---- 5. Candidate evaluation loop ----
    t_loop_start = time.perf_counter()

    for batch_start in range(0, N_cand, tds_candidate_batch_size):
        batch_cols = unacquired[batch_start: batch_start + tds_candidate_batch_size]
        M = len(batch_cols)

        # Build M simulated k-spaces: copy y_omega, insert one predicted line per item
        y_sim = y_omega.expand(M, -1, -1, -1).clone()   # (M, H, W, 2)
        for m_idx in range(M):
            col_j = batch_cols[m_idx].item()
            y_sim[m_idx, :, col_j, :] = y_hat[0, :, col_j, :]

        x_sim = transforms.ifft2(y_sim)  # (M, H, W, 2)

        # Encode + infer — no decode
        logits_j = _encode_and_get_logits(
            model, tokenizer, x_sim, device, sampling_kwargs
        )  # (M, H_p, W_p, 2, K, V_K)

        p_j = logits_to_probs_tds(logits_j)        # (M, L, C, V_K)
        h_j, H_j = compute_patch_entropy(p_j)      # (M, L), (M,)

        # Broadcast p_old and w from (1, L, C, V_K) / (1, L) to (M, ...)
        p_old_exp = p_old.expand(M, -1, -1, -1)    # (M, L, C, V_K)
        w_exp = w.expand(M, -1)                     # (M, L)

        # Distribution distance per patch
        if tds_distance == "js":
            dist = js_divergence(p_old_exp, p_j)    # (M, L)
        elif tds_distance == "symmetric_kl":
            dist = symmetric_kl(p_old_exp, p_j)
        elif tds_distance == "topk_l1":
            dist = topk_l1(p_old_exp, p_j)
        else:
            raise ValueError(f"Unknown tds_distance: {tds_distance!r}")

        # Weighted TDS score
        if tds_line_aggregation == "sum":
            score_tds = (w_exp * dist).sum(dim=-1)  # (M,)
        else:  # "mean"
            score_tds = (w_exp * dist).mean(dim=-1)

        # Optional entropy-reduction term
        H_old_exp = H_old.expand(M)
        entropy_drop = (H_old_exp - H_j).clamp(min=0.0)
        score_final = tds_alpha * score_tds + tds_beta * entropy_drop  # (M,)

        # Store per-column scores
        for m_idx in range(M):
            col_int = batch_cols[m_idx].item()
            scores_1d[0, col_int] = score_final[m_idx].item()
            H_j_per_col[col_int] = H_j[m_idx].item()

    t_loop_end = time.perf_counter()

    # ---- 6. Select best unacquired column ----
    scores_1d.masked_fill_(acquired, float("-inf"))
    best_col = int(scores_1d[0].argmax().item())
    selected_score = float(scores_1d[0, best_col].item())
    H_j_selected = H_j_per_col.get(best_col, float("nan"))

    if not math.isfinite(selected_score):
        warnings.warn(f"TDS: selected score is not finite: {selected_score}")

    # Top-10 scores for diagnostics
    finite_mask = scores_1d[0].isfinite()
    top10_scores: list[tuple[int, float]] = []
    if finite_mask.any():
        k_top = min(10, int(finite_mask.sum().item()))
        top10_vals, top10_cols = scores_1d[0].topk(k_top)
        top10_scores = list(zip(top10_cols.cpu().tolist(), top10_vals.cpu().tolist()))

    diagnostics: dict = {
        "selected_col": best_col,
        "selected_tds_score": selected_score,
        "H_old": float(H_old[0].item()),
        "H_j_selected": float(H_j_selected),
        "entropy_drop_selected": float(max(H_old[0].item() - H_j_selected, 0.0)),
        "n_fg_patches": n_fg,
        "candidate_count": N_cand,
        "candidate_mode": tds_candidate_mode,
        "geo_topk_used": tds_candidate_mode == "geo_topk",
        "runtime_loop_s": t_loop_end - t_loop_start,
        "top10_scores": top10_scores,
    }

    return {
        "selected_col": best_col,
        "scores_1d": scores_1d,
        "diagnostics": diagnostics,
    }
