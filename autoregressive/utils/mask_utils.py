"""
Utility functions for generating radial and cartesian k-space masks.
These are shared between training and sampling/reconstruction.
"""
import torch


def make_centered_radius_grid(H, W, device=None, dtype=torch.float32):
    """Create a grid of distances from center."""
    device = device or "cpu"
    yy = torch.arange(H, device=device, dtype=dtype) - (H // 2)
    xx = torch.arange(W, device=device, dtype=dtype) - (W // 2)
    return torch.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)


def radial_circle_mask(H, W, radius, device=None, dtype=torch.float32):
    """Create a circular mask of given radius."""
    rr = make_centered_radius_grid(H, W, device=device, dtype=dtype)
    return rr <= float(radius)


def radial_circle_radii(H, W, num_masks=32, center_fraction=0.08, device=None, dtype=torch.float32):
    """Generate radii for progressive radial masks."""
    num_masks = 1 if num_masks is None else int(num_masks)

    min_dim = min(H, W)
    r_min = max(1, int(min_dim * center_fraction) // 2)

    rr_max = make_centered_radius_grid(H, W, device=device, dtype=dtype).max().to(torch.float32)

    radii = torch.linspace(float(r_min), float(rr_max.item()), steps=num_masks, device=device, dtype=torch.float32)
    radii = torch.cat([radii, torch.tensor([rr_max]).to(radii.device)])
    return radii


def sample_circle_radius_idx(curr_iter, total_iters, num_masks, exponent=2.0, device=None):
    """Sample a radius index with curriculum scheduling."""
    p = float(curr_iter) / float(max(1, total_iters))
    p = max(0.0, min(1.0, p))
    max_keep_idx = int((p**exponent) * (num_masks - 1))
    max_keep_idx = max(0, min(max_keep_idx, num_masks - 1))
    return torch.randint(low=0, high=max_keep_idx + 1, size=(1,), device=device or "cpu").item()


def sample_circle_radius_idx_weighted(
    radii,
    *,
    alpha: float = 0.7,
    eps: float = 1e-6,
    device=None,
) -> int:
    """Sample a radius index with inverse-radius weighting."""
    r = radii
    if not torch.is_tensor(r):
        r = torch.as_tensor(r, dtype=torch.float32, device=device or "cpu")
    else:
        r = r.to(device=device or r.device, dtype=torch.float32)

    weights = 1.0 / (r.clamp_min(0.0) + eps).pow(alpha)
    probs = weights / weights.sum()
    return torch.multinomial(probs, num_samples=1).item()
