"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import contextlib
import math
import warnings
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch


@contextlib.contextmanager
def temp_seed(rng: np.random.RandomState, seed: Optional[Union[int, Tuple[int, ...]]]):
    """A context manager for temporarily adjusting the random seed."""
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class MaskFunc:
    """
    An object for GRAPPA-style sampling masks.

    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.

    When called, ``MaskFunc`` uses internal functions create mask by 1)
    creating a mask for the k-space center, 2) create a mask outside of the
    k-space center, and 3) combining them into a total mask. The internals are
    handled by ``sample_mask``, which calls ``calculate_center_mask`` for (1)
    and ``calculate_acceleration_mask`` for (2). The combination is executed
    in the ``MaskFunc`` ``__call__`` function.

    If you would like to implement a new mask, simply subclass ``MaskFunc``
    and overwrite the ``sample_mask`` logic. See examples in ``RandomMaskFunc``
    and ``EquispacedMaskFunc``.
    """

    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[int],
        allow_any_combination: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
            allow_any_combination: Whether to allow cross combinations of
                elements from ``center_fractions`` and ``accelerations``.
            seed: Seed for starting the internal random number generator of the
                ``MaskFunc``.
        """
        if len(center_fractions) != len(accelerations) and not allow_any_combination:
            raise ValueError(
                "Number of center fractions should match number of accelerations "
                "if allow_any_combination is False."
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.allow_any_combination = allow_any_combination
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Sample and return a k-space mask.

        Args:
            shape: Shape of k-space.
            offset: Offset from 0 to begin mask (for equispaced masks). If no
                offset is given, then one is selected randomly.
            seed: Seed for random number generator for reproducibility.

        Returns:
            A 2-tuple containing 1) the k-space mask and 2) the number of
            center frequency lines.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_mask, accel_mask, num_low_frequencies = self.sample_mask(
                shape, offset
            )

        # combine masks together
        return torch.max(center_mask, accel_mask), num_low_frequencies

    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Sample a new k-space mask.

        This function samples and returns two components of a k-space mask: 1)
        the center mask (e.g., for sensitivity map calculation) and 2) the
        acceleration mask (for the edge of k-space). Both of these masks, as
        well as the integer of low frequency samples, are returned.

        Args:
            shape: Shape of the k-space to subsample.
            offset: Offset from 0 to begin mask (for equispaced masks).

        Returns:
            A 3-tuple contaiing 1) the mask for the center of k-space, 2) the
            mask for the high frequencies of k-space, and 3) the integer count
            of low frequency samples.
        """
        num_cols = shape[-2]
        center_fraction, acceleration = self.choose_acceleration()
        num_low_frequencies = round(num_cols * center_fraction)
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, acceleration, offset, num_low_frequencies
            ),
            shape,
        )

        return center_mask, acceleration_mask, num_low_frequencies

    def reshape_mask(self, mask: np.ndarray, shape: Sequence[int]) -> torch.Tensor:
        """Reshape mask to desired output shape."""
        num_cols = shape[-2]
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols

        return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking (for equispaced masks).
            num_low_frequencies: Integer count of low-frequency lines sampled.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        raise NotImplementedError

    def calculate_center_mask(
        self, shape: Sequence[int], num_low_freqs: int
    ) -> np.ndarray:
        """
        Build center mask based on number of low frequencies.

        Args:
            shape: Shape of k-space to mask.
            num_low_freqs: Number of low-frequency lines to sample.

        Returns:
            A mask for hte low spatial frequencies of k-space.
        """
        num_cols = shape[-2]
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = 1
        assert mask.sum() == num_low_freqs

        return mask

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        if self.allow_any_combination:
            return self.rng.choice(self.center_fractions), self.rng.choice(
                self.accelerations
            )
        else:
            choice = self.rng.randint(len(self.center_fractions))
            return self.center_fractions[choice], self.accelerations[choice]


class RandomMaskFunc(MaskFunc):
    """
    Creates a random sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected uniformly at random with a
        probability equal to: prob = (N / acceleration - N_low_freqs) /
        (N - N_low_freqs). This ensures that the expected number of columns
        selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the ``RandomMaskFunc`` object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        prob = (num_cols / acceleration - num_low_frequencies) / (
            num_cols - num_low_frequencies
        )

        return self.rng.uniform(size=num_cols) < prob


class EquiSpacedMaskFunc(MaskFunc):
    """
    Sample data with equally-spaced k-space lines.

    The lines are spaced exactly evenly, as is done in standard GRAPPA-style
    acquisitions. This means that with a densely-sampled center,
    ``acceleration`` will be greater than the true acceleration rate.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Not used.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        if offset is None:
            offset = self.rng.randint(0, high=round(acceleration))

        mask = np.zeros(num_cols, dtype=np.float32)
        mask[offset::acceleration] = 1

        return mask


class EquispacedMaskFractionFunc(MaskFunc):
    """
    Equispaced mask with approximate acceleration matching.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected with equal spacing at a proportion
           that reaches the desired acceleration rate taking into consideration
           the number of low frequencies. This ensures that the expected number
           of columns selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the EquispacedMaskFunc object is called.

    Note that this function may not give equispaced samples (documented in
    https://github.com/facebookresearch/fastMRI/issues/54), which will require
    modifications to standard GRAPPA approaches. Nonetheless, this aspect of
    the function has been preserved to match the public multicoil data.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Number of low frequencies. Used to adjust mask
                to exactly match the target acceleration.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        # determine acceleration rate by adjusting for the number of low frequencies
        adjusted_accel = (acceleration * (num_low_frequencies - num_cols)) / (
            num_low_frequencies * acceleration - num_cols
        )
        if offset is None:
            offset = self.rng.randint(0, high=round(adjusted_accel))

        mask = np.zeros(num_cols)
        accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
        accel_samples = np.around(accel_samples).astype(np.uint)
        mask[accel_samples] = 1.0

        return mask


class MagicMaskFunc(MaskFunc):
    """
    Masking function for exploiting conjugate symmetry via offset-sampling.

    This function applies the mask described in the following paper:

    Defazio, A. (2019). Offset Sampling Improves Deep Learning based
    Accelerated MRI Reconstructions by Exploiting Symmetry. arXiv preprint,
    arXiv:1912.01101.

    It is essentially an equispaced mask with an offset for the opposite site
    of k-space. Since MRI images often exhibit approximate conjugate k-space
    symmetry, this mask is generally more efficient than a standard equispaced
    mask.

    Similarly to ``EquispacedMaskFunc``, this mask will usually undereshoot the
    target acceleration rate.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Not used.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        if offset is None:
            offset = self.rng.randint(0, high=acceleration)

        if offset % 2 == 0:
            offset_pos = offset + 1
            offset_neg = offset + 2
        else:
            offset_pos = offset - 1 + 3
            offset_neg = offset - 1 + 0

        poslen = (num_cols + 1) // 2
        neglen = num_cols - (num_cols + 1) // 2
        mask_positive = np.zeros(poslen, dtype=np.float32)
        mask_negative = np.zeros(neglen, dtype=np.float32)

        mask_positive[offset_pos::acceleration] = 1
        mask_negative[offset_neg::acceleration] = 1
        mask_negative = np.flip(mask_negative)

        mask = np.concatenate((mask_positive, mask_negative))

        return np.fft.fftshift(mask)  # shift mask and return


class MagicMaskFractionFunc(MagicMaskFunc):
    """
    Masking function for exploiting conjugate symmetry via offset-sampling.

    This function applies the mask described in the following paper:

    Defazio, A. (2019). Offset Sampling Improves Deep Learning based
    Accelerated MRI Reconstructions by Exploiting Symmetry. arXiv preprint,
    arXiv:1912.01101.

    It is essentially an equispaced mask with an offset for the opposite site
    of k-space. Since MRI images often exhibit approximate conjugate k-space
    symmetry, this mask is generally more efficient than a standard equispaced
    mask.

    Similarly to ``EquispacedMaskFractionFunc``, this method exactly matches
    the target acceleration by adjusting the offsets.
    """

    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Sample a new k-space mask.

        This function samples and returns two components of a k-space mask: 1)
        the center mask (e.g., for sensitivity map calculation) and 2) the
        acceleration mask (for the edge of k-space). Both of these masks, as
        well as the integer of low frequency samples, are returned.

        Args:
            shape: Shape of the k-space to subsample.
            offset: Offset from 0 to begin mask (for equispaced masks).

        Returns:
            A 3-tuple contaiing 1) the mask for the center of k-space, 2) the
            mask for the high frequencies of k-space, and 3) the integer count
            of low frequency samples.
        """
        num_cols = shape[-2]
        fraction_low_freqs, acceleration = self.choose_acceleration()
        num_cols = shape[-2]
        num_low_frequencies = round(num_cols * fraction_low_freqs)

        # bound the number of low frequencies between 1 and target columns
        target_columns_to_sample = round(num_cols / acceleration)
        num_low_frequencies = max(min(num_low_frequencies, target_columns_to_sample), 1)

        # adjust acceleration rate based on target acceleration.
        adjusted_target_columns_to_sample = (
            target_columns_to_sample - num_low_frequencies
        )
        adjusted_acceleration = 0
        if adjusted_target_columns_to_sample > 0:
            adjusted_acceleration = round(num_cols / adjusted_target_columns_to_sample)

        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        accel_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, adjusted_acceleration, offset, num_low_frequencies
            ),
            shape,
        )

        return center_mask, accel_mask, num_low_frequencies


class VariableDensity2DMaskFunc:
    """
    Samples individual k-space points on a 2D Cartesian grid with probability
    biased toward low frequencies (center).

    If acs_fraction > 0, the center square of side round(min(H,W)*acs_fraction)
    is always fully sampled first (hard ACS zone). The remaining budget is then
    filled by density-weighted random sampling outside that square.
    If acs_fraction == 0 (default), purely probabilistic sampling is used.

    Returns a [1, H, W, 1] float32 mask (2D point mask, not column mask).

    Note: center_fractions is not used by this class; pass it only to satisfy
    the create_mask_for_mask_type interface.
    """

    def __init__(
        self,
        accelerations: Sequence[int],
        profile: str = "power_law",
        decay: float = 2.0,
        sigma: float = 0.25,
        acs_fraction: float = 0.0,
    ):
        """
        Args:
            accelerations: Acceleration factors to choose from uniformly.
            profile: "power_law" or "gaussian".
            decay: Exponent for power_law profile; p(r) ∝ (1 + r)^{-decay}.
            sigma: Relative sigma for gaussian profile (fraction of r_max).
            acs_fraction: Side length of the hard ACS center square as a
                fraction of min(H, W). E.g. 0.025 → 8×8 on a 320×320 image.
                0.0 = no hard ACS zone.
        """
        self.accelerations = np.asarray(accelerations).ravel()
        self.profile = profile
        self.decay = decay
        self.sigma = sigma
        self.acs_fraction = acs_fraction
        self.rng = np.random.RandomState()

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Args:
            shape: (..., H, W, 2) — at least 3 dims; H=shape[-3], W=shape[-2].
            offset: Unused; kept for interface compatibility.
            seed: Optional int or tuple for reproducibility.

        Returns:
            Tuple of (float32 mask of shape [1, H, W, 1], num_low_frequencies=0).
        """
        if seed is not None:
            if isinstance(seed, (tuple, list)):
                seed = int(sum(seed) % (2 ** 31))
            self.rng.seed(seed)

        H, W = shape[-3], shape[-2]
        acceleration = int(self.rng.choice(self.accelerations))
        n_total = max(1, round(H * W / acceleration))

        yy = np.arange(H, dtype=np.float32) - H // 2
        xx = np.arange(W, dtype=np.float32) - W // 2
        r = np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
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
            chosen = self.rng.choice(
                non_acs_idx,
                size=min(n_remaining, len(non_acs_idx)),
                replace=False,
                p=p_non_acs,
            )
            mask_flat[chosen] = 1.0
        else:
            chosen = self.rng.choice(
                H * W, size=min(n_total, H * W), replace=False, p=density_flat
            )
            mask_flat = np.zeros(H * W, dtype=np.float32)
            mask_flat[chosen] = 1.0

        mask = torch.from_numpy(mask_flat.reshape(1, H, W, 1))
        return mask, 0


class PoissonDiskMaskFunc:
    """
    Samples individual k-space points with a minimum inter-point distance
    constraint (Poisson disk), optionally biased toward low frequencies via a
    VDS density profile. No hard ACS zone.

    Returns a [1, H, W, 1] float32 mask.
    """

    def __init__(
        self,
        accelerations: Sequence[int],
        min_dist_frac: float = 0.03,
        vds_weighting: bool = True,
        profile: str = "power_law",
        decay: float = 2.0,
    ):
        """
        Args:
            accelerations: Acceleration factors to choose from.
            min_dist_frac: Minimum distance between samples as a fraction of
                min(H, W).
            vds_weighting: If True, weight candidate selection by VDS density
                (more likely to pick center candidates first).
            profile: "power_law" or "gaussian" for VDS weighting.
            decay: Power-law decay exponent.
        """
        self.accelerations = accelerations
        self.min_dist_frac = min_dist_frac
        self.vds_weighting = vds_weighting
        self.profile = profile
        self.decay = decay
        self.rng = np.random.RandomState()

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Args:
            shape: (..., H, W, 2); H=shape[-3], W=shape[-2].
            offset: Unused; kept for interface compatibility.
            seed: Optional int or tuple.

        Returns:
            Tuple of (float32 mask of shape [1, H, W, 1], num_low_frequencies=0).
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

        yy = np.arange(H, dtype=np.float32) - H // 2
        xx = np.arange(W, dtype=np.float32) - W // 2
        r = np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
        r_max = r.max()

        if self.vds_weighting:
            density = 1.0 / (1.0 + r / max(r_max, 1e-8)) ** self.decay
            density_flat = density.ravel()
            density_flat = density_flat / density_flat.sum()
            candidate_order = self.rng.choice(
                H * W, size=H * W, replace=False, p=density_flat
            )
        else:
            candidate_order = self.rng.permutation(H * W)

        grid: dict = {}

        def _neighbors_ok(row: int, col: int) -> bool:
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
            warnings.warn(
                f"PoissonDiskMaskFunc: could only place {n_accepted}/{n_total} points "
                f"(min_dist={min_dist:.1f}px, acc={acceleration}). "
                "Consider reducing min_dist_frac or acceleration."
            )

        mask = torch.from_numpy(mask_flat.reshape(1, H, W, 1))
        return mask, 0


def create_mask_for_mask_type(
    mask_type_str: str,
    center_fractions: Sequence[float],
    accelerations: Sequence[int],
    **kwargs,
) -> object:
    """
    Creates a mask of the specified type.

    Args:
        mask_type_str: One of "random"/"cart_1d", "equispaced",
            "equispaced_fraction", "magic", "magic_fraction", "vds2d",
            "poisson".
        center_fractions: Fraction of low-frequency columns to retain
            (not used by vds2d/poisson).
        accelerations: Acceleration factors to apply.
        **kwargs: Extra keyword arguments forwarded to 2D mask constructors.

    Returns:
        A mask function for the target mask type.
    """
    if mask_type_str in ("random", "cart_1d"):
        return RandomMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced":
        return EquiSpacedMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced_fraction":
        return EquispacedMaskFractionFunc(center_fractions, accelerations)
    elif mask_type_str == "magic":
        return MagicMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "magic_fraction":
        return MagicMaskFractionFunc(center_fractions, accelerations)
    elif mask_type_str == "vds2d":
        return VariableDensity2DMaskFunc(accelerations, **kwargs)
    elif mask_type_str == "poisson":
        return PoissonDiskMaskFunc(accelerations, **kwargs)
    else:
        raise ValueError(f"{mask_type_str} not supported")
