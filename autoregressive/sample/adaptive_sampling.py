import math
import time

import torch
import torch.nn.functional as F
from dataset.data.transforms import fft2
from data import transforms

@torch.no_grad()
def kspace_uncertainty_from_images(
    x: torch.Tensor,                  # (B, N, C, H, W), C==2
    already_measured: torch.Tensor,    # (B, 1, H, W) bool/int
    *,
    vertical_mask: bool,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 5:
        raise ValueError(f"x must be (B,N,C,H,W), got {tuple(x.shape)}")
    b, n, c, h, w = x.shape
    if c != 2:
        raise ValueError(f"Expected C==2 (real, imag), got C=={c}")

    device = x.device
    measured = already_measured.to(device=device, dtype=torch.bool).clone()

    # FFT per sample; center over N only (do not mix across batch)
    k = fft2(x.permute(0, 1, 3, 4, 2))              # (B, N, H, W, 2)
    k_centered = k - k.mean(dim=1, keepdim=True)         # (B, N, H, W, 2)
    p2 = k_centered.pow(2)

    if vertical_mask:
        # scores: (B, W) = mean over N, H, complex
        scores = p2.mean(dim=1).mean(dim=1).mean(dim=-1)  # (B, W)

        # measured_cols: (B, W) = any over C and H
        measured_cols = measured.any(dim=1).any(dim=1)     # (B, W)

        scores_masked = scores.masked_fill(measured_cols, float("-inf"))
        k_sel = min(rank, w)
        vals, cols = torch.topk(scores_masked, k=k_sel, dim=1)  # (B, k_sel)

        valid = torch.isfinite(vals)  # (B, k_sel)
        if valid.any():
            bb = torch.arange(b, device=device)[:, None].expand(b, k_sel).reshape(-1)
            cc = cols.reshape(-1)
            vv = valid.reshape(-1)

            bb = bb[vv]
            cc = cc[vv]
            measured[bb, :, :, cc] = True  # sets whole vertical lines for each selected column

        return measured, scores

    # scores_hw: (B, H, W) = mean over N and complex
    scores_hw = p2.mean(dim=1).mean(dim=-1)               # (B, H, W)

    measured_points = measured.any(dim=1)                 # (B, H, W)
    scores_hw_masked = scores_hw.masked_fill(measured_points, float("-inf"))
    flat_scores = scores_hw_masked.reshape(b, -1)         # (B, H*W)

    k_sel = min(rank, h * w)
    vals, flat_idx = torch.topk(flat_scores, k=k_sel, dim=1)  # (B, k_sel)

    valid = torch.isfinite(vals)  # (B, k_sel)
    if valid.any():
        rows = (flat_idx // w).long()
        cols = (flat_idx % w).long()

        bb = torch.arange(b, device=device)[:, None].expand(b, k_sel).reshape(-1)
        rr = rows.reshape(-1)
        cc = cols.reshape(-1)
        vv = valid.reshape(-1)

        bb = bb[vv]
        rr = rr[vv]
        cc = cc[vv]
        measured[bb, :, rr, cc] = True  # sets both complex channels at selected (row,col)

    return measured, scores_hw.reshape(b, -1)


def image_entropy_to_freq_entropy(entropy_complex, zero_mean=False,
                                   radial_weight_alpha=0.0, already_measured=None):
    if zero_mean:
        # Subtract spatial mean to eliminate DC dominance from brain/background bias.
        # The FFT then captures spatial *variation* in entropy (scan-specific).
        entropy_complex = entropy_complex - entropy_complex.mean(dim=(-3, -2), keepdim=True)

    freq_data = fft2(entropy_complex)

    # Compute Magnitude (Energy of the uncertainty at each frequency)
    freq_magnitude = torch.norm(freq_data, dim=-1)  # (B, H_img, W_img)

    if radial_weight_alpha > 0.0:
        # Multiply by r^alpha to suppress DC (center) and boost outer k-space.
        # alpha=1 → linear, alpha=2 → quadratic; r normalized to [0, 1].
        from autoregressive.utils.mask_utils import make_centered_radius_grid
        H, W = freq_magnitude.shape[-2:]
        r = make_centered_radius_grid(H, W, device=freq_magnitude.device,
                                      dtype=freq_magnitude.dtype)
        r = r / r.max().clamp(min=1e-8)
        freq_magnitude = freq_magnitude * r.unsqueeze(0).pow(radial_weight_alpha)

    return freq_magnitude


def _select_points_poisson_constrained(
    freq_magnitude: torch.Tensor,  # (B, H, W) scores, already zeroed for measured
    measured: torch.Tensor,         # (B, 1, H, W) bool
    rank: int,
    min_dist: float,
) -> torch.Tensor:
    """Select up to `rank` 2D k-space points per batch item by descending score,
    subject to a minimum inter-point distance constraint (Poisson disk).

    Returns updated `measured` (B, 1, H, W).
    """
    import math
    B, _, H, W = measured.shape
    cell_size = max(1, math.ceil(min_dist))
    min_dist_sq = min_dist ** 2

    for b in range(B):
        scores = freq_magnitude[b]  # (H, W)
        flat_scores = scores.reshape(-1)

        # Sort all candidates by descending score; ignore already-measured
        sorted_idx = torch.argsort(flat_scores, descending=True)
        meas_b = measured[b, 0]  # (H, W) bool

        # Build spatial hash from already-measured points
        grid = {}
        meas_coords = meas_b.nonzero(as_tuple=False)  # (N_meas, 2)
        for rc in meas_coords:
            row, col = rc[0].item(), rc[1].item()
            cell_key = (row // cell_size, col // cell_size)
            grid.setdefault(cell_key, []).append((row, col))

        n_accepted = 0
        for idx in sorted_idx:
            if n_accepted >= rank:
                break
            idx_item = idx.item()
            row, col = idx_item // W, idx_item % W
            if meas_b[row, col]:
                continue  # already measured
            # Check 5x5 cell neighbourhood
            ci, cj = row // cell_size, col // cell_size
            ok = True
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    for pr, pc in grid.get((ci + di, cj + dj), []):
                        if (row - pr) ** 2 + (col - pc) ** 2 < min_dist_sq:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                measured[b, :, row, col] = True
                cell_key = (ci, cj)
                grid.setdefault(cell_key, []).append((row, col))
                n_accepted += 1

    return measured


@torch.no_grad()
def kspace_uncertainty_from_patch_logits(
    logits: torch.Tensor,           # (B, H_patch, W_patch, 2, K, V_per_K)
    already_measured: torch.Tensor, # (B, 1, H_img, W_img)
    *,
    vertical_mask: bool,
    rank: int,
    les_zero_mean: bool = False,
    les_radial_weight_alpha: float = 0.0,
    mask_type: str = "cartesian",   # "cartesian" | "2d" | "poisson"
    poisson_min_dist: float = 4.0,  # pixels, only used when mask_type=="poisson"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates acquisition scores using Logit-Based Spectral Uncertainty.

    Returns: (updated_measured, entropy_complex_upsampled, freq_magnitude_upsampled)
      - entropy_complex_upsampled: (B, H_img, W_img, 2) spatial entropy for display
      - freq_magnitude_upsampled:  (B, H_img, W_img)   freq entropy for display (pre-mask)
    """
    # 1. Validate Input Shapes
    if logits.ndim != 6:
        raise ValueError(f"logits must be (B, H, W, 2, K, V), got {tuple(logits.shape)}")

    b, h_patch, w_patch, c_complex, k_books, v_sub = logits.shape
    b_mask, c_mask, h_img, w_img = already_measured.shape

    device = logits.device
    measured = already_measured.to(device=device, dtype=torch.bool).clone()

    # -------------------------------------------------------------------
    # 2. Compute Spatial Uncertainty (Entropy) in patch space
    # -------------------------------------------------------------------
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    # Entropy shape: (B, H_patch, W_patch, 2, K)
    entropy_per_codebook = -torch.sum(probs * log_probs, dim=-1)

    # Sum over codebooks → (B, H_patch, W_patch, 2): [Entropy_Real, Entropy_Imag]
    entropy_complex = entropy_per_codebook.sum(dim=-1)

    # -------------------------------------------------------------------
    # 3. Spectral Projection in PATCH space (critical: do FFT BEFORE upsampling)
    #
    # If we upsample the 16×16 entropy to 256×256 first, bilinear interpolation
    # creates a smooth surface whose highest spatial frequency fits within the
    # already-measured center fraction (~8px radius vs ~10px measured at 4% CF).
    # After masking, nothing is left. FFT in patch space keeps the 16 frequency
    # bands meaningful relative to the measurement budget.
    # -------------------------------------------------------------------
    freq_magnitude_patch = image_entropy_to_freq_entropy(
        entropy_complex,       # (B, H_patch, W_patch, 2)
        zero_mean=les_zero_mean,
        radial_weight_alpha=les_radial_weight_alpha,
    )  # (B, H_patch, W_patch)

    # Upsample freq_magnitude to image resolution for masking and scoring
    if (h_patch != h_img) or (w_patch != w_img):
        freq_magnitude = F.interpolate(
            freq_magnitude_patch.unsqueeze(1),  # (B, 1, H_patch, W_patch)
            size=(h_img, w_img),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)  # (B, H_img, W_img)
    else:
        freq_magnitude = freq_magnitude_patch

    # Keep a copy for display (before masking) so save_entropy_maps can
    # apply its own mask and show the full freq entropy pattern.
    freq_magnitude_for_display = freq_magnitude.clone()

    # Upsample entropy_complex to image resolution (for display only)
    if (h_patch != h_img) or (w_patch != w_img):
        entropy_complex = F.interpolate(
            entropy_complex.permute(0, 3, 1, 2),  # (B, 2, H_patch, W_patch)
            size=(h_img, w_img),
            mode='bilinear',
            align_corners=False,
        ).permute(0, 2, 3, 1)  # (B, H_img, W_img, 2)

    # -------------------------------------------------------------------
    # 4. Mask out already measured positions before scoring
    # -------------------------------------------------------------------
    measured_points = measured.any(dim=1)  # (B, H_img, W_img)
    freq_magnitude = freq_magnitude.masked_fill(measured_points, 0.0)

    # -------------------------------------------------------------------
    # 5. Score and Select Measurements
    # -------------------------------------------------------------------
    if vertical_mask or mask_type == "cartesian":
        scores = freq_magnitude.mean(dim=-2)  # (B, W_img)

        measured_cols = measured.any(dim=1).any(dim=1)
        scores_masked = scores.masked_fill(measured_cols, float("-inf"))

        k_sel = min(rank, w_img)
        vals, cols = torch.topk(scores_masked, k=k_sel, dim=1)

        valid = torch.isfinite(vals)
        if valid.any():
            bb = torch.arange(b, device=device)[:, None].expand(b, k_sel).reshape(-1)
            cc = cols.reshape(-1)
            vv = valid.reshape(-1)

            bb = bb[vv]
            cc = cc[vv]
            measured[bb, :, :, cc] = True

        return measured, entropy_complex, freq_magnitude_for_display

    elif mask_type == "poisson":
        measured = _select_points_poisson_constrained(
            freq_magnitude,  # (B, H_img, W_img), already zeroed at measured positions
            measured,        # (B, 1, H_img, W_img)
            rank=rank,
            min_dist=poisson_min_dist,
        )
        return measured, entropy_complex, freq_magnitude_for_display

    else:  # "2d" — top-k 2D point selection
        scores_hw = freq_magnitude

        measured_points = measured.any(dim=1)
        scores_hw_masked = scores_hw.masked_fill(measured_points, float("-inf"))

        flat_scores = scores_hw_masked.reshape(b, -1)
        k_sel = min(rank, h_img * w_img)

        vals, flat_idx = torch.topk(flat_scores, k=k_sel, dim=1)

        valid = torch.isfinite(vals)
        if valid.any():
            rows = (flat_idx // w_img).long()
            cols = (flat_idx % w_img).long()

            bb = torch.arange(b, device=device)[:, None].expand(b, k_sel).reshape(-1)
            rr = rows.reshape(-1)
            cc = cols.reshape(-1)
            vv = valid.reshape(-1)

            bb = bb[vv]
            rr = rr[vv]
            cc = cc[vv]
            measured[bb, :, rr, cc] = True

        return measured, entropy_complex, freq_magnitude_for_display


@torch.no_grad()
def posterior_sampling_scores(
    samples: torch.Tensor,          # (S, H, W) real RSS images
    already_measured: torch.Tensor, # (1, 1, W, 1) or (B, 1, W, 1) cartesian mask
    rank: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select k-space columns by posterior variance energy (magnitude-based).

    For each unacquired column c:
        score(c) = sum_s sum_h |FFT(x_s - mean)[h, c]|^2

    NOTE: operates on RSS magnitude images; DC-biased due to 1/f^2 image spectrum.
    See posterior_sampling_scores_kspace for the AdaSense-compatible variant.

    Args:
        samples:          (S, H, W) — S stochastic reconstructions, real-valued RSS
        already_measured: (1, 1, W, 1) cartesian mask (1 = acquired)
        rank:             number of columns to select

    Returns:
        selected_cols: (rank,) column indices
        col_scores:    (W,) energy score per column (already-measured = 0)
    """
    S, H, W = samples.shape
    device = samples.device

    # Zero-mean across samples: centered[s, h, w]
    centered = samples - samples.mean(dim=0, keepdim=True)   # (S, H, W)

    # FFT of each centered sample (treat as real: imag = 0)
    centered_c = torch.stack([centered, torch.zeros_like(centered)], dim=-1)  # (S, H, W, 2)
    K = fft2(centered_c)   # (S, H, W, 2) — real/imag parts of 2D DFT

    # Energy at each column: sum over samples and rows of |K[s,h,c]|^2
    K_energy = K[..., 0] ** 2 + K[..., 1] ** 2   # (S, H, W)
    col_scores = K_energy.sum(dim=(0, 1))          # (W,)

    # Zero out already-measured columns
    measured = already_measured[0, 0, :, 0].bool()   # (W,)
    col_scores = col_scores.masked_fill(measured, 0.0)

    selected_cols = col_scores.topk(rank).indices  # (rank,)
    return selected_cols, col_scores


def posterior_sampling_scores_kspace(
    samples: torch.Tensor,          # (S, H, W, 2) complex reconstructions
    already_measured: torch.Tensor, # (1, 1, W, 1) or (B, 1, W, 1) cartesian mask
    rank: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select k-space columns by posterior variance in k-space (AdaSense-compatible).

    For each unacquired column c:
        score(c) = mean_{s,h,ri} (K[s,h,c,ri] - mean_s K[s,h,c,ri])^2

    Variance is computed directly on the complex k-space of each sample, so
    unacquired columns — which are unconstrained across samples — score high,
    while acquired columns (fixed by data consistency) score near zero.

    Args:
        samples:          (S, H, W, 2) — S complex reconstructions (real/imag)
        already_measured: (1, 1, W, 1) cartesian mask (1 = acquired)
        rank:             number of columns to select

    Returns:
        selected_cols: (rank,) column indices
        col_scores:    (W,) variance score per column (already-measured = 0)
    """
    S, H, W, _ = samples.shape

    K = fft2(samples)                              # (S, H, W, 2)
    K_centered = K - K.mean(dim=0, keepdim=True)  # (S, H, W, 2)
    col_scores = K_centered.pow(2).mean(dim=(0, 1, 3))  # (W,)

    measured = already_measured[0, 0, :, 0].bool()
    col_scores = col_scores.masked_fill(measured, 0.0)

    selected_cols = col_scores.topk(rank).indices
    return selected_cols, col_scores


# ---------------------------------------------------------------------------
# GEO — Gradient Entropy Optimization
# ---------------------------------------------------------------------------

def geo_step(
    model,
    tokenizer,
    masked_image: torch.Tensor,
    sample_mask: torch.Tensor,
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One GEO step: compute k-space gradient entropy, select highest-scoring column.

    Returns:
        new_mask:        (B, H, W, 2) updated mask with `rank` new columns acquired
        geo_line_scores: (W,) float32 gradient magnitude per column (0 at acquired cols)
    """
    from autoregressive.models.generate import soft_lookup  # lazy — avoid circular import

    B, H, W, _ = masked_image.shape

    with torch.enable_grad():
        masked_kspace = transforms.fft2(masked_image)
        masked_kspace.requires_grad_(True)
        i_hat = transforms.ifft2(masked_kspace)

        z_ste_re, z_q_re, z_e_re = tokenizer.encode_with_ste(i_hat[..., 0])
        z_ste_im, z_q_im, z_e_im = tokenizer.encode_with_ste(i_hat[..., 1])

        soft_embed_re = soft_lookup(
            z_e_re, tokenizer.quantizer.codebooks, model.re_tok_embeddings,
        )
        soft_embed_im = soft_lookup(
            z_e_im, tokenizer.quantizer.codebooks, model.im_tok_embeddings,
        )
        soft_cond_embeddings = torch.stack([soft_embed_re, soft_embed_im], dim=1)

        indices_re = tokenizer.quantizer.f_to_idx(z_q_re)
        indices_im = tokenizer.quantizer.f_to_idx(z_q_im)
        cond_tokens = torch.stack([indices_re, indices_im], dim=1)

        timestep = torch.zeros(B, device=device, dtype=torch.long)
        h = model(
            cond_tokens=cond_tokens.long(),
            soft_cond_embeddings=soft_cond_embeddings,
            timestep=timestep,
        )
        if isinstance(h, tuple):
            h = h[0]
        _, L, dim = h.shape
        h_flat = h.view(B * L, 1, dim)

        all_logits = []
        soft_embeddings = []
        for k in range(model.num_codebooks):
            h_k = h_flat if k == 0 else torch.cat([h_flat] + soft_embeddings, dim=1)
            for layer in model.output.layers:
                h_k = layer(h_k, freqs_cis=None, start_pos=None, mask=None, full_attn=False)
            h_k = model.output.norm(h_k)
            logits_re_k = model.output.linear_head_re(h_k[:, -1:])
            logits_im_k = model.output.linear_head_im(h_k[:, -1:])
            all_logits.append(torch.stack([logits_re_k, logits_im_k], dim=1))
            if k < model.num_codebooks - 1:
                probs_re = F.softmax(logits_re_k, dim=-1)
                probs_im = F.softmax(logits_im_k, dim=-1)
                soft_embeddings.append(
                    probs_re @ model.output.codebooks[k].weight
                    + probs_im @ model.output.codebooks[k].weight
                )

        all_logits_t = torch.stack(all_logits, dim=2)
        probs = F.softmax(all_logits_t, dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
        loss = -entropy.mean(dim=[1, 2, 3]).sum()
        loss.backward()
        gradients = masked_kspace.grad  # (B, H, W, 2)

    grad_mag = gradients.norm(dim=-1) * (1 - sample_mask[..., 0])  # (B, H, W)
    geo_line_scores = grad_mag.mean(dim=1)[0]  # (W,) — column scores for batch[0]

    grad_mag_cols = grad_mag.mean(dim=1)  # (B, W)
    measured_cols = sample_mask[..., 0].any(dim=1)  # (B, W)
    grad_mag_cols = grad_mag_cols.masked_fill(measured_cols, float("-inf"))
    _, top_cols = torch.topk(grad_mag_cols, k=rank, dim=1)

    new_mask = sample_mask.clone()
    for b in range(B):
        for c in top_cols[b]:
            new_mask[b, :, c.item(), :] = 1.0

    model.rm_head_caches()
    return new_mask.detach(), geo_line_scores.detach()


# ---------------------------------------------------------------------------
# GLES — Gradient-weighted Latent Entropy Selection
# ---------------------------------------------------------------------------

def _patch_entropy(all_logits_t: torch.Tensor, B: int, L: int) -> torch.Tensor:
    """Reduce stacked AR-head logits to per-patch entropy.

    Args:
        all_logits_t: (B*L, 2, K, 1, V_K)
        B: batch size, L: number of latent patches

    Returns:
        h_patch: (B, L) — entropy averaged over re/im branches, codebooks, and seq dim
    """
    probs = F.softmax(all_logits_t, dim=-1)                         # (B*L, 2, K, 1, V_K)
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)           # (B*L, 2, K, 1)
    return entropy.mean(dim=[1, 2, 3]).view(B, L)                   # (B, L)


def gles_step(
    model,
    tokenizer,
    masked_image: torch.Tensor,
    sample_mask: torch.Tensor,
    rank: int,
    device: torch.device,
    *,
    M: int | None = 8,
    use_foreground: bool = False,
    foreground_percentile: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One GLES step: entropy-weighted patchwise gradient selection.

    Computes s_GLES(j) = Σ_l h_l · mean_i |∂h_l/∂y_{j,i}| for each column j,
    where h_l is per-patch entropy and y is k-space.

    Unlike GEO (|Σ_l ∂h_l|), GLES takes absolute values per-patch before
    summing, preventing gradient cancellation across patches.

    Args:
        M:                   Top-M high-entropy patches to use; None = all.
        use_foreground:      Restrict patch selection to foreground patches.
        foreground_percentile: Exclude bottom-p% of patches by magnitude.

    Returns:
        new_mask:         (B, H, W, 2) updated mask with `rank` new columns acquired
        gles_line_scores: (W,) float32 GLES score per column (0 at acquired cols)
    """
    from autoregressive.models.generate import soft_lookup  # lazy — avoid circular import

    B, H, W, _ = masked_image.shape

    with torch.enable_grad():
        masked_kspace = transforms.fft2(masked_image)
        masked_kspace.requires_grad_(True)
        i_hat = transforms.ifft2(masked_kspace)

        z_ste_re, z_q_re, z_e_re = tokenizer.encode_with_ste(i_hat[..., 0])
        z_ste_im, z_q_im, z_e_im = tokenizer.encode_with_ste(i_hat[..., 1])

        soft_embed_re = soft_lookup(
            z_e_re, tokenizer.quantizer.codebooks, model.re_tok_embeddings,
        )
        soft_embed_im = soft_lookup(
            z_e_im, tokenizer.quantizer.codebooks, model.im_tok_embeddings,
        )
        soft_cond_embeddings = torch.stack([soft_embed_re, soft_embed_im], dim=1)

        indices_re = tokenizer.quantizer.f_to_idx(z_q_re)
        indices_im = tokenizer.quantizer.f_to_idx(z_q_im)
        cond_tokens = torch.stack([indices_re, indices_im], dim=1)

        timestep = torch.zeros(B, device=device, dtype=torch.long)
        h = model(
            cond_tokens=cond_tokens.long(),
            soft_cond_embeddings=soft_cond_embeddings,
            timestep=timestep,
        )
        if isinstance(h, tuple):
            h = h[0]
        _, L, dim = h.shape
        h_flat = h.view(B * L, 1, dim)

        all_logits = []
        soft_embeddings = []
        for k in range(model.num_codebooks):
            h_k = h_flat if k == 0 else torch.cat([h_flat] + soft_embeddings, dim=1)
            for layer in model.output.layers:
                h_k = layer(h_k, freqs_cis=None, start_pos=None, mask=None, full_attn=False)
            h_k = model.output.norm(h_k)
            logits_re_k = model.output.linear_head_re(h_k[:, -1:])
            logits_im_k = model.output.linear_head_im(h_k[:, -1:])
            all_logits.append(torch.stack([logits_re_k, logits_im_k], dim=1))
            if k < model.num_codebooks - 1:
                probs_re = F.softmax(logits_re_k, dim=-1)
                probs_im = F.softmax(logits_im_k, dim=-1)
                soft_embeddings.append(
                    probs_re @ model.output.codebooks[k].weight
                    + probs_im @ model.output.codebooks[k].weight
                )

        all_logits_t = torch.stack(all_logits, dim=2)       # (B*L, 2, K, 1, V_K)
        h_patch = _patch_entropy(all_logits_t, B, L)        # (B, L)

        # Select patches by mean entropy across batch
        h_select = h_patch.mean(dim=0)  # (L,)

        if use_foreground:
            H_p = W_p = int(L ** 0.5)
            mag = (i_hat[..., 0] ** 2 + i_hat[..., 1] ** 2).sqrt().detach()  # (B, H, W)
            ph, pw = H // H_p, W // W_p
            patch_mag = (
                mag.reshape(B, H_p, ph, W_p, pw)
                .mean(dim=(2, 4))
                .reshape(B, L)
                .mean(dim=0)                                 # (L,)
            )
            fg_threshold = torch.quantile(patch_mag, foreground_percentile / 100.0)
            h_select = h_select.clone()
            h_select[patch_mag < fg_threshold] = float("-inf")

        if M is not None:
            n_sel = min(M, int(torch.isfinite(h_select).sum().item()))
            patch_ids = h_select.topk(n_sel).indices
        else:
            patch_ids = torch.where(torch.isfinite(h_select))[0]

        # Accumulate entropy-weighted per-patch gradient magnitudes
        n_patches = len(patch_ids)
        line_score = torch.zeros(B, W, device=device)

        for idx_pos, l in enumerate(patch_ids):
            l_item = l.item()
            h_l = h_patch[:, l_item].sum()

            # Use .backward() instead of autograd.grad() — the latter is
            # incompatible with gradient checkpointing used inside the model.
            if masked_kspace.grad is not None:
                masked_kspace.grad.zero_()
            h_l.backward(retain_graph=(idx_pos < n_patches - 1))
            grad_l = masked_kspace.grad.detach().clone()    # (B, H, W, 2)

            grad_mag_l = grad_l.norm(dim=-1)                # (B, H, W)
            patch_line_score = grad_mag_l.mean(dim=1)       # (B, W)
            weight = h_patch[:, l_item].detach().view(B, 1)
            line_score = line_score + weight * patch_line_score

    # Diagnostics-friendly scores (0 at acquired cols)
    measured_mask_1d = sample_mask[0, :, :, 0].any(dim=0)  # (W,)
    gles_line_scores = line_score[0].masked_fill(measured_mask_1d, 0.0)

    measured_cols = sample_mask[..., 0].any(dim=1)          # (B, W)
    line_score = line_score.masked_fill(measured_cols, float("-inf"))
    _, top_cols = torch.topk(line_score, k=rank, dim=1)

    new_mask = sample_mask.clone()
    for b in range(B):
        for c in top_cols[b]:
            new_mask[b, :, c.item(), :] = 1.0

    model.rm_head_caches()
    return new_mask.detach(), gles_line_scores.detach()


# ---------------------------------------------------------------------------
# ERSE — Entropy-weighted Residual Spectral Energy
# ---------------------------------------------------------------------------

@torch.no_grad()
def erse_step(
    model,
    tokenizer,
    masked_image: torch.Tensor,   # (B, H, W, 2)
    sample_mask: torch.Tensor,    # (B, H, W, 2)
    rank: int,
    device: torch.device,
    *,
    residual_mode: str = "pred_minus_current",  # "pred_minus_current" | "pred" | "current"
    weight_mode: str = "soft_entropy",          # "soft_entropy" | "top_m_entropy" | "foreground_soft_entropy"
    top_m: int | None = 32,
    smoothing: bool = True,
    smoothing_kernel: int = 5,
    entropy_normalization: bool = True,
    line_aggregation: str = "mean",             # "mean" | "sum"
    use_complex_residual: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One ERSE step: Entropy-weighted Residual Spectral Energy acquisition.

    Cheap no-gradient, no-candidate-simulation policy.
    Cost: one forward pass + one token decode + one FFT.

    Returns:
        new_mask:         (B, H, W, 2) updated mask with `rank` new columns acquired
        erse_line_scores: (W,) float32 ERSE score per column (0 at acquired cols, batch[0])
    """
    from autoregressive.models.generate import sample_far  # lazy — avoid circular import

    t0 = time.perf_counter()
    B, H, W, _ = masked_image.shape

    # ------------------------------------------------------------------
    # 1. Current zero-filled reconstruction
    # ------------------------------------------------------------------
    masked_kspace = transforms.fft2(masked_image)   # (B, H, W, 2)
    x_omega = transforms.ifft2(masked_kspace)        # (B, H, W, 2)

    # ------------------------------------------------------------------
    # 2. Encode through tokenizer (hard quantisation — no STE needed)
    # ------------------------------------------------------------------
    def _img_to_tokens(img_bhw: torch.Tensor) -> torch.Tensor:
        """(B, H, W) -> (B, K, L) discrete token indices."""
        img_3ch = img_bhw.unsqueeze(-1).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)  # (B,3,H,W)
        enc = tokenizer.encoder(img_3ch)
        enc = tokenizer.quant_proj(enc)
        return tokenizer.quantizer.f_to_idx(enc)  # (B, K, L)

    tok_re = _img_to_tokens(masked_image[..., 0])  # (B, K, L)
    tok_im = _img_to_tokens(masked_image[..., 1])
    cond_tokens = torch.cat(
        [tok_re.unsqueeze(1), tok_im.unsqueeze(1)], dim=1
    )  # (B, 2, K, L)

    # ------------------------------------------------------------------
    # 3. Transformer forward pass
    # ------------------------------------------------------------------
    t = torch.zeros(B, device=device, dtype=torch.long)
    h = model(cond_tokens=cond_tokens.long(), timestep=t)
    if isinstance(h, tuple):
        h = h[0]

    h_ndim = h.ndim
    if h_ndim == 3:
        _, L, D = h.shape
        h_flat = h.view(B * L, 1, D)
    elif h_ndim == 4:
        _, _, L, D = h.shape
        h_flat = h.permute(0, 2, 1, 3).contiguous().view(B * L, 2, 1, D)
    else:
        raise ValueError(f"Unexpected transformer output ndim: {h_ndim}")

    # ------------------------------------------------------------------
    # 4. AR head — collect logits and greedy predicted token indices
    # ------------------------------------------------------------------
    model.setup_head_caches(B * L, model.num_codebooks, dtype=h.dtype, device=device)

    logits_list: list[torch.Tensor] = []
    pred_idx_list: list[torch.Tensor] = []

    for i in range(model.num_codebooks):
        start_pos = torch.tensor([i], dtype=torch.int)
        mask_ar = model.output.head_causal_mask[:B * L, None, start_pos]
        if i == 0:
            lg = model.output.forward_test(h_flat, input_pos=start_pos, mask=mask_ar)
        else:
            lg = model.output.forward_test(
                h_flat, idx=pred_idx_list[-1], input_pos=start_pos, mask=mask_ar
            )
        logits_list.append(lg)                               # (B*L, 2, 1, V_K)
        _pred = sample_far(lg)[0].unsqueeze(-1)              # greedy argmax → (B*L, 2, 1)
        pred_idx_list.append(_pred)

    # all_logits_t: (B*L, 2, K, 1, V_K)
    all_logits_t = torch.stack(logits_list, dim=2)

    # ------------------------------------------------------------------
    # 5. Per-patch entropy and normalization
    # ------------------------------------------------------------------
    h_patch = _patch_entropy(all_logits_t, B, L)  # (B, L)

    if entropy_normalization:
        V_K = all_logits_t.shape[-1]
        h_norm = (h_patch / math.log(V_K)).clamp(0.0, 1.0)
    else:
        h_norm = h_patch

    # ------------------------------------------------------------------
    # 6. Entropy weight map   w_h: (B, H, W)
    # ------------------------------------------------------------------
    H_p = W_p = int(L ** 0.5)

    def _patches_to_image(h_n: torch.Tensor) -> torch.Tensor:
        """(B, L) -> (B, H, W) via bilinear upsample."""
        grid = h_n.view(B, 1, H_p, W_p).float()
        return F.interpolate(
            grid, size=(H, W), mode='bilinear', align_corners=False
        ).squeeze(1)  # (B, H, W)

    if weight_mode == "soft_entropy":
        w_h = _patches_to_image(h_norm)

    elif weight_mode == "top_m_entropy":
        h_hard = h_norm.clone()
        if top_m is not None:
            k_sel = min(top_m, L)
            kth_val = h_hard.topk(k_sel, dim=1).values[:, -1:]  # (B, 1)
            h_hard = h_hard * (h_hard >= kth_val).float()
        w_h = _patches_to_image(h_hard)

    elif weight_mode == "foreground_soft_entropy":
        mag = (x_omega[..., 0] ** 2 + x_omega[..., 1] ** 2).sqrt()  # (B, H, W)
        ph, pw = H // H_p, W // W_p
        patch_mag = (
            mag.reshape(B, H_p, ph, W_p, pw)
            .mean(dim=(2, 4))
            .reshape(B, L)
        )
        patch_mag_norm = patch_mag / (patch_mag.amax(dim=1, keepdim=True) + 1e-8)
        w_h = _patches_to_image(h_norm * patch_mag_norm)

    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode!r}")

    if smoothing:
        k = smoothing_kernel
        pad = k // 2
        box = torch.ones(1, 1, k, k, device=device, dtype=w_h.dtype) / (k * k)
        w_h = F.conv2d(w_h.unsqueeze(1), box, padding=pad).squeeze(1)  # (B, H, W)

    # ------------------------------------------------------------------
    # 7. Decode greedy predicted reconstruction   x_pred: (B, H, W, 2)
    # ------------------------------------------------------------------
    pred_idx = torch.stack(pred_idx_list, dim=2)           # (B*L, 2, K, 1)
    pred_idx = pred_idx.view(B, L, 2, -1).permute(0, 2, 3, 1)  # (B, 2, K, L)

    def _decode_channel(codes: torch.Tensor) -> torch.Tensor:
        """(B, K, L) -> (B, H, W) grayscale float in [-1, 1]."""
        f = tokenizer.quantizer.idx_to_f(codes)   # (B, L, D_vq)
        f = tokenizer.post_quant_proj(f)           # (B, L, D_enc)
        dec = tokenizer.decoder(f).clamp(-1, 1)   # (B, 3, H, W)
        return (dec[:, 0] + dec[:, 1] + dec[:, 2]) / 3  # (B, H, W)

    x_pred = torch.stack(
        [_decode_channel(pred_idx[:, 0]), _decode_channel(pred_idx[:, 1])],
        dim=-1,
    )  # (B, H, W, 2)

    # ------------------------------------------------------------------
    # 8. Residual
    # ------------------------------------------------------------------
    if residual_mode == "pred_minus_current":
        r = x_pred - x_omega
    elif residual_mode == "pred":
        r = x_pred
    elif residual_mode == "current":
        r = x_omega
    else:
        raise ValueError(f"Unknown residual_mode: {residual_mode!r}")

    if not use_complex_residual:
        # magnitude-only residual: keep real channel, zero imag
        r_mag = (r[..., 0] ** 2 + r[..., 1] ** 2).sqrt()
        r = torch.stack([r_mag, torch.zeros_like(r_mag)], dim=-1)

    # ------------------------------------------------------------------
    # 9. Entropy-weighted spectral energy per column
    # ------------------------------------------------------------------
    x_erse = r * w_h.unsqueeze(-1)                   # (B, H, W, 2)
    y_erse = transforms.fft2(x_erse)                 # (B, H, W, 2)
    energy = (y_erse ** 2).sum(dim=-1)               # (B, H, W)

    if line_aggregation == "mean":
        line_score = energy.mean(dim=1)              # (B, W)
    else:
        line_score = energy.sum(dim=1)               # (B, W)

    # ------------------------------------------------------------------
    # 10. Mask acquired columns, select top-rank, update mask
    # ------------------------------------------------------------------
    measured_cols = sample_mask[..., 0].any(dim=1)   # (B, W)

    # Diagnostic scores: 0 at acquired columns (batch[0])
    erse_line_scores = line_score[0].masked_fill(measured_cols[0], 0.0)  # (W,)

    line_score = line_score.masked_fill(measured_cols, float("-inf"))
    _, top_cols = torch.topk(line_score, k=rank, dim=1)

    new_mask = sample_mask.clone()
    for b in range(B):
        for c in top_cols[b]:
            new_mask[b, :, c.item(), :] = 1.0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t0
    n_nonzero = (h_norm > 0.1).sum().item()
    print(
        f"[ERSE] {elapsed:.3f}s | "
        f"entropy min={h_patch.min():.3f} max={h_patch.max():.3f} mean={h_patch.mean():.3f} | "
        f"h_norm min={h_norm.min():.3f} max={h_norm.max():.3f} | "
        f"w_h min={w_h.min():.3f} max={w_h.max():.3f} | "
        f"r_norm={r.norm():.3f} | "
        f"energy min={energy.min():.3e} max={energy.max():.3e} | "
        f"score min={erse_line_scores.min():.3e} max={erse_line_scores.max():.3e} | "
        f"n_high_entropy_patches={n_nonzero} | "
        f"smoothing={smoothing} | "
        f"cols={top_cols[0].tolist()}"
    )

    model.rm_head_caches()
    return new_mask.detach(), erse_line_scores.detach()