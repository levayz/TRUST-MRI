"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import numpy as np
import torch


def create_mask_for_mask_type(mask_type_str, center_fractions, accelerations, **kwargs):
    print(f"creating mask with cf={center_fractions}, acc={accelerations}")
    if mask_type_str in ('cart_1d', 'random'):
        return RandomMaskFunc(center_fractions, accelerations)
    elif mask_type_str == 'equispaced':
        return EquispacedMaskFunc(center_fractions, accelerations)
    elif mask_type_str == 'vds2d':
        return VariableDensity2DMaskFunc(accelerations, **kwargs)
    elif mask_type_str == 'poisson':
        return PoissonDiskMaskFunc(accelerations, **kwargs)
    else:
        raise Exception(f"{mask_type_str} not supported")


class MaskFunc():
    def __init__(self, center_fractions, accelerations):
        """
        Args:
            center_fractions (List[float]): Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is chosen uniformly
                each time.

            accelerations (List[int]): Amount of under-sampling. This should have the same length
                as center_fractions. If multiple values are provided, then one of these is chosen
                uniformly each time.
        """
        if len(center_fractions) != len(accelerations):
            raise ValueError('Number of center fractions should match number of accelerations')

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def choose_acceleration(self):
        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]
        return center_fraction, acceleration


class RandomMaskFunc(MaskFunc):
    """
    RandomMaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the k-space data has N
    columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center corresponding to
           low-frequencies
        2. The other columns are selected uniformly at random with a probability equal to:
           prob = (N / acceleration - N_low_freqs) / (N - N_low_freqs).
    This ensures that the expected number of columns selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which case one possible
    (center_fraction, acceleration) is chosen uniformly at random each time the RandomMaskFunc object is
    called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04], then there
    is a 50% probability that 4-fold acceleration with 8% center fraction is selected and a 50%
    probability that 8-fold acceleration with 4% center fraction is selected.
    """

    def __init__(self, center_fractions, accelerations):
        """
        Args:
            center_fractions (List[float]): Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is chosen uniformly
                each time.

            accelerations (List[int]): Amount of under-sampling. This should have the same length
                as center_fractions. If multiple values are provided, then one of these is chosen
                uniformly each time. An acceleration of 4 retains 25% of the columns, but they may
                not be spaced evenly.
        """
        if len(center_fractions) != len(accelerations):
            raise ValueError('Number of center fractions should match number of accelerations')

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def __call__(self, shape, seed=None):
        """
        Args:
            shape (iterable[int]): The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last dimension.
            seed (int, optional): Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same shape.
        Returns:
            torch.Tensor: A mask of the specified shape.
        """
        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')

        if seed is not None:
            print(f"setting random mask seed")
            self.rng.seed(seed)
        num_cols = shape[-2]
        center_fraction, acceleration = self.choose_acceleration()

        # Create the mask
        num_low_freqs = int(round(num_cols * center_fraction))
        prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
        mask = self.rng.uniform(size=num_cols) < prob
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = True

        # Reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask

class EquispacedMaskFunc(MaskFunc):
    """
    EquispacedMaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the k-space data has N
    columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center corresponding to
           low-frequencies
        2. The other columns are selected with equal spacing at a proportion that reaches the
           desired acceleration rate taking into consideration the number of low frequencies. This
           ensures that the expected number of columns selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which case one possible
    (center_fraction, acceleration) is chosen uniformly at random each time the EquispacedMaskFunc
    object is called.
    """
    def __call__(self, shape, seed):
        """
        Args:
            shape (iterable[int]): The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last dimension.
            seed (int, optional): Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same shape.
        Returns:
            torch.Tensor: A mask of the specified shape.
        """
        if len(shape) < 3:
           raise ValueError('Shape should have 3 or more dimensions')

        self.rng.seed(seed)
        center_fraction, acceleration = self.choose_acceleration()
        num_cols = shape[-2]
        num_low_freqs = int(round(num_cols * center_fraction))

        # Create the mask
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = True

        # Determine acceleration rate by adjusting for the number of low frequencies
        adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (num_low_freqs * acceleration - num_cols)
        offset = self.rng.randint(0, round(adjusted_accel))

        accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
        accel_samples = np.around(accel_samples).astype(np.uint)
        mask[accel_samples] = True

        # Reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask


class VariableDensity2DMaskFunc:
    """
    Samples individual k-space points on a 2D Cartesian grid with probability
    biased toward low frequencies (center).

    If acs_fraction > 0, the center square of side round(min(H,W)*acs_fraction)
    is always fully sampled first (hard ACS zone). The remaining budget is then
    filled by density-weighted random sampling outside that square.
    If acs_fraction == 0 (default), purely probabilistic sampling is used with
    no guaranteed center region.

    Returns a [1, H, W, 1] float32 mask (2D point mask, not column mask).
    center_fractions is intentionally not used.
    """

    def __init__(self, accelerations, profile="power_law", decay=2.0, sigma=0.25, acs_fraction=0.0):
        """
        Args:
            accelerations (List[int]): Acceleration factors to choose from uniformly.
            profile (str): "power_law" or "gaussian".
            decay (float): Exponent for power_law profile; p(r) ∝ (1 + r)^{-decay}.
            sigma (float): Relative sigma for gaussian profile (fraction of r_max).
            acs_fraction (float): Side length of the hard ACS center square as a
                fraction of min(H, W). E.g. 0.025 → 8×8 on a 320×320 image.
                0.0 = no hard ACS zone.
        """
        self.accelerations = np.asarray(accelerations).ravel()
        self.profile = profile
        self.decay = decay
        self.sigma = sigma
        self.acs_fraction = acs_fraction
        self.rng = np.random.RandomState()

    def __call__(self, shape, seed=None):
        """
        Args:
            shape: (..., H, W, 2) — at least 3 dims; H=shape[-3], W=shape[-2].
            seed: optional int or tuple for reproducibility.
        Returns:
            torch.Tensor: float32 mask of shape [1, H, W, 1].
        """
        if seed is not None:
            if isinstance(seed, (tuple, list)):
                seed = int(sum(seed) % (2 ** 31))
            self.rng.seed(seed)

        H, W = shape[-3], shape[-2]
        acceleration = int(self.rng.choice(self.accelerations))
        n_total = max(1, round(H * W / acceleration))

        # Build 2D density profile centered at (H//2, W//2)
        yy = np.arange(H, dtype=np.float32) - H // 2
        xx = np.arange(W, dtype=np.float32) - W // 2
        r = np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)  # (H, W)
        r_max = r.max()

        if self.profile == "power_law":
            density = 1.0 / (1.0 + r / max(r_max, 1e-8)) ** self.decay
        else:  # gaussian
            sigma_abs = self.sigma * max(r_max, 1e-8)
            density = np.exp(-(r ** 2) / (2.0 * sigma_abs ** 2))

        density_flat = density.ravel()
        density_flat = density_flat / density_flat.sum()

        if self.acs_fraction > 0.0:
            acs_half = round(min(H, W) * self.acs_fraction / 2)
            cy, cx = H // 2, W // 2
            mask_flat = np.zeros(H * W, dtype=np.float32)
            for yi in range(cy - acs_half, cy + acs_half):
                for xi in range(cx - acs_half, cx + acs_half):
                    mask_flat[yi * W + xi] = 1.0
            n_acs = int(mask_flat.sum())
            n_remaining = max(0, n_total - n_acs)
            non_acs_idx = np.where(mask_flat == 0)[0]
            p_non_acs = density_flat[non_acs_idx]
            p_non_acs = p_non_acs / p_non_acs.sum()
            chosen = self.rng.choice(non_acs_idx, size=min(n_remaining, len(non_acs_idx)), replace=False, p=p_non_acs)
            mask_flat[chosen] = 1.0
        else:
            chosen = self.rng.choice(H * W, size=min(n_total, H * W), replace=False, p=density_flat)
            mask_flat = np.zeros(H * W, dtype=np.float32)
            mask_flat[chosen] = 1.0

        mask = torch.from_numpy(mask_flat.reshape(1, H, W, 1))
        return mask


class PoissonDiskMaskFunc:
    """
    Samples individual k-space points with a minimum inter-point distance
    constraint (Poisson disk), optionally biased toward low frequencies via a
    VDS density profile. No hard ACS zone.

    Returns a [1, H, W, 1] float32 mask.
    """

    def __init__(self, accelerations, min_dist_frac=0.03, vds_weighting=True,
                 profile="power_law", decay=2.0):
        """
        Args:
            accelerations (List[int]): Acceleration factors to choose from.
            min_dist_frac (float): Minimum distance between samples as a
                fraction of min(H, W).
            vds_weighting (bool): If True, weight candidate selection by VDS
                density (more likely to pick center candidates first).
            profile (str): "power_law" or "gaussian" for VDS weighting.
            decay (float): Power-law decay exponent.
        """
        self.accelerations = accelerations
        self.min_dist_frac = min_dist_frac
        self.vds_weighting = vds_weighting
        self.profile = profile
        self.decay = decay
        self.rng = np.random.RandomState()

    def __call__(self, shape, seed=None):
        """
        Args:
            shape: (..., H, W, 2); H=shape[-3], W=shape[-2].
            seed: optional int or tuple.
        Returns:
            torch.Tensor: float32 mask of shape [1, H, W, 1].
        """
        if seed is not None:
            if isinstance(seed, (tuple, list)):
                seed = int(sum(seed) % (2 ** 31))
            self.rng.seed(seed)

        H, W = shape[-3], shape[-2]
        acceleration = self.rng.choice(self.accelerations)
        n_total = max(1, round(H * W / acceleration))
        min_dist = self.min_dist_frac * min(H, W)
        cell_size = max(1, math.ceil(min_dist))

        # Build VDS density for candidate ordering
        yy = np.arange(H, dtype=np.float32) - H // 2
        xx = np.arange(W, dtype=np.float32) - W // 2
        r = np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)  # (H, W)
        r_max = r.max()

        if self.vds_weighting:
            density = 1.0 / (1.0 + r / max(r_max, 1e-8)) ** self.decay
            density_flat = density.ravel()
            density_flat = density_flat / density_flat.sum()
            # Draw all candidates in density-weighted order (without replacement)
            candidate_order = self.rng.choice(
                H * W, size=H * W, replace=False, p=density_flat
            )
        else:
            candidate_order = self.rng.permutation(H * W)

        # Spatial hash: maps cell (ci, cj) -> list of (row, col) accepted points
        grid = {}

        def _neighbors_ok(row, col):
            ci, cj = row // cell_size, col // cell_size
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    for pr, pc in grid.get((ci + di, cj + dj), []):
                        if (row - pr) ** 2 + (col - pc) ** 2 < min_dist ** 2:
                            return False
            return True

        mask_flat = np.zeros(H * W, dtype=np.float32)
        n_accepted = 0
        for idx in candidate_order:
            if n_accepted >= n_total:
                break
            row, col = int(idx // W), int(idx % W)
            if _neighbors_ok(row, col):
                mask_flat[idx] = 1.0
                cell_key = (row // cell_size, col // cell_size)
                grid.setdefault(cell_key, []).append((row, col))
                n_accepted += 1

        if n_accepted < n_total:
            import warnings
            warnings.warn(
                f"PoissonDiskMaskFunc: could only place {n_accepted}/{n_total} points "
                f"(min_dist={min_dist:.1f}px, acc={acceleration}). "
                "Consider reducing min_dist_frac or acceleration."
            )

        mask = torch.from_numpy(mask_flat.reshape(1, H, W, 1))
        return mask
