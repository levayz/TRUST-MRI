import torch
import torch.nn.functional as F
from data.transforms import fft2

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


@torch.no_grad()
def kspace_uncertainty_from_patch_logits(
    logits: torch.Tensor,           # (B, H_patch, W_patch, 2, K, V_per_K)
    already_measured: torch.Tensor, # (B, 1, H_img, W_img)
    *,
    vertical_mask: bool,
    rank: int,
    les_zero_mean: bool = False,
    les_radial_weight_alpha: float = 0.0,
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
    if vertical_mask:
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

    else:
        scores_hw = freq_magnitude

        measured_points = measured.any(dim=1)
        scores_hw_masked = scores_hw.masked_fill(measured_points, float("-inf"))
        B, H, W = scores_hw_masked.shape
        scores_hw_masked_normed = F.softmax(scores_hw_masked.view(B, -1), dim=1).view(B, H, W)

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