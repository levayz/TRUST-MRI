"""
Multicoil MRI dataset for training/evaluating MulticoilRecon.

Usage:
    from dataset.data.subsample import RandomMaskFunc
    from dataset.multicoil_data import build_multicoil_dataset

    mask_func = RandomMaskFunc(center_fractions=[0.08], accelerations=[4])
    dataset   = build_multicoil_dataset(
        root      = "fastmri/brain/multicoil_train",
        mask_func = mask_func,
        crop_size = (256, 256),
    )
    sample = dataset[0]
    # sample["x_us"]  : Tensor(C, 256, 256, 2)  -- undersampled coil images
    # sample["x_gt"]  : Tensor(C, 256, 256, 2)  -- GT coil images
    # sample["mask"]  : Tensor(1, 1, W, 1)       -- 1-D Cartesian mask
    # sample["target_rss"] : Tensor(256, 256)    -- GT RSS image
    # sample["max_val"]    : float
    # sample["fname"], sample["slice_num"]
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

import warnings
# Suppress TypedStorage deprecation warnings that appear in spawned DataLoader
# workers when loading pre-PyTorch-2.0 .pt files (VCC matrices).
# Must live here so it fires on re-import in each spawned worker process.
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")

import numpy as np
import torch

# local FFT / transform utilities
from dataset.data.fftc import ifft2c_new as ifft2c, fft2c_new as fft2c
from dataset.data.transforms import (
    apply_mask,
    complex_abs,
    complex_center_crop,
    complex_pad,
    root_sum_of_squares,
    to_tensor,
)

# fastMRI SliceDataset (fixed import in dataset/data/transforms.py)
from dataset.data import SliceDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rss_from_coil_images(x: torch.Tensor) -> torch.Tensor:
    """Compute root-sum-of-squares across coil dimension.

    Args:
        x: (C, H, W, 2) complex coil images (real/imag stacked on last dim)

    Returns:
        (H, W) RSS magnitude
    """
    # magnitude per coil: (C, H, W)
    mag = complex_abs(x)            # complex_abs handles last-dim=2
    # RSS across coil dim (0)
    return root_sum_of_squares(mag, dim=0)   # (H, W)


# ---------------------------------------------------------------------------
# Transform callable
# ---------------------------------------------------------------------------

class MulticoilTransform:
    """Transform for multicoil fastMRI slices.

    Called by SliceDataset for each slice:
        (kspace, mask_from_file, target, attrs, fname, slice_num) → dict

    Args:
        mask_func:  callable from data.subsample (e.g. RandomMaskFunc)
        crop_size:  spatial size after center-crop; must match MedITok input (256×256)
        use_seed:   if True, seed mask RNG from filename for reproducibility
    """

    def __init__(
        self,
        mask_func: Callable,
        crop_size: Tuple[int, int] = (256, 256),
        use_seed: bool = True,
        max_coils: int = 20,
        vcc_matrix_dir: Optional[str] = None,
        n_virtual_coils: int = 8,
    ):
        self.mask_func       = mask_func
        self.crop_size       = crop_size
        self.use_seed        = use_seed
        self.max_coils       = max_coils
        self.vcc_dir         = Path(vcc_matrix_dir) if vcc_matrix_dir else None
        self.n_virtual_coils = n_virtual_coils
        self._W_cache: dict  = {}   # stem → (C', C) complex tensor or None

    def __call__(
        self,
        kspace: np.ndarray,         # (C, H, W) complex
        mask_from_file: np.ndarray, # ignored — we generate our own mask
        target: np.ndarray,         # (H, W) precomputed RSS (may be None)
        attrs: dict,
        fname: str,
        slice_num: int,
    ) -> dict:
        # ------------------------------------------------------------------ #
        # 1. Convert to torch tensor  (C, H, W, 2)
        # ------------------------------------------------------------------ #
        kspace_gt: torch.Tensor = to_tensor(kspace).float()  # (C, H, W, 2)

        # ------------------------------------------------------------------ #
        # 2. IFFT full k-space → image space, then pad+crop to target size
        #    Cropping in IMAGE space preserves correct FOV.
        # ------------------------------------------------------------------ #
        ch, cw = self.crop_size
        x_gt = ifft2c(kspace_gt)                                         # (C, H, W, 2)
        x_gt = complex_center_crop(complex_pad(x_gt, ch, cw), self.crop_size)  # (C, ch, cw, 2)

        # ------------------------------------------------------------------ #
        # 3. FFT the cropped GT image → k-space of the cropped FOV,
        #    then apply undersampling mask at the target resolution.
        # ------------------------------------------------------------------ #
        kspace_gt = fft2c(x_gt)                                          # (C, ch, cw, 2)

        # ------------------------------------------------------------------ #
        # 3b. RSS from physical coils — always computed before VCC so the
        #     target image is never distorted by coil compression.
        # ------------------------------------------------------------------ #
        rss_gt = _rss_from_coil_images(x_gt)  # (H', W') physical RSS

        # ------------------------------------------------------------------ #
        # 3c. Optional VCC: project physical k-space → virtual coil domain.
        #     W is loaded lazily per scan (cached by stem in self._W_cache).
        # ------------------------------------------------------------------ #
        if self.vcc_dir is not None:
            stem = Path(fname).stem
            if stem not in self._W_cache:
                wpath = self.vcc_dir / f"{stem}.pt"
                if wpath.exists():
                    d = torch.load(wpath, map_location="cpu", weights_only=True)
                    self._W_cache[stem] = d.get(f"W_{self.n_virtual_coils}")
                else:
                    self._W_cache[stem] = None  # missing file → skip VCC
            W = self._W_cache[stem]
            if W is not None:
                kspace_c  = torch.view_as_complex(kspace_gt.contiguous())      # (C, H, W)
                C_phys, H_, W_ = kspace_c.shape
                kspace_v  = (W @ kspace_c.reshape(C_phys, -1)).reshape(W.shape[0], H_, W_)
                kspace_gt = torch.view_as_real(kspace_v)                        # (C', H, W, 2)
                x_gt      = ifft2c(kspace_gt)                                   # (C', H, W, 2)

        seed = tuple(map(ord, fname)) if self.use_seed else None
        kspace_us, mask, _ = apply_mask(kspace_gt, self.mask_func, seed=seed)
        # kspace_us: (C or C', ch, cw, 2), mask: (1, 1, cw, 1)

        # ------------------------------------------------------------------ #
        # 4. IFFT undersampled k-space → per-coil undersampled images
        # ------------------------------------------------------------------ #
        x_us = ifft2c(kspace_us)    # (C or C', ch, cw, 2)

        # ------------------------------------------------------------------ #
        # 5. RSS GT magnitude + normalisation
        #    rss_gt was computed from physical coils above (before VCC).
        #    All coil images are divided by RSS max so values lie in [-1, 1].
        # ------------------------------------------------------------------ #
        max_val = float(rss_gt.max()) + 1e-8
        target_rss = (rss_gt / max_val).clamp(-1.0, 1.0)  # (H', W')

        x_gt = (x_gt / max_val).clamp(-1.0, 1.0)          # (C, H', W', 2)
        x_us = (x_us / max_val).clamp(-1.0, 1.0)          # (C, H', W', 2)
        # kspace_us was raw (unnormalized); recompute from normalized x_us so that
        # kspace_us / max_val = fft2c(x_us) — consistent scale with decoded predictions
        kspace_us = fft2c(x_us)                            # (C, H', W', 2)

        # ------------------------------------------------------------------ #
        # 6. Pad coil dimension so batches can be stacked.
        #    When VCC is active all scans produce exactly n_virtual_coils coils,
        #    so pad to that instead of max_coils to avoid unnecessary zero coils.
        # ------------------------------------------------------------------ #
        C = x_gt.shape[0]
        target_c = self.n_virtual_coils if self.vcc_dir is not None else self.max_coils
        target_c = min(target_c, self.max_coils)   # never exceed model table size
        if C < target_c:
            pad_c = target_c - C
            H2, W2 = self.crop_size
            zeros = x_gt.new_zeros(pad_c, H2, W2, 2)
            x_gt      = torch.cat([x_gt,      zeros], dim=0)
            x_us      = torch.cat([x_us,      zeros], dim=0)
            kspace_gt = torch.cat([kspace_gt, zeros], dim=0)
            kspace_us = torch.cat([kspace_us, zeros], dim=0)
        elif C > target_c:
            x_gt      = x_gt[:target_c]
            x_us      = x_us[:target_c]
            kspace_gt = kspace_gt[:target_c]
            kspace_us = kspace_us[:target_c]

        return {
            "kspace_gt":  kspace_gt,   # (target_c, H', W', 2)
            "kspace_us":  kspace_us,   # (target_c, H', W', 2)
            "mask":       mask,        # (1, 1, W, 1) or (1, H, W, 1)
            "x_gt":       x_gt,        # (target_c, H', W', 2)
            "x_us":       x_us,        # (target_c, H', W', 2)
            "target_rss": target_rss,  # (H', W')
            "num_coils":  torch.tensor(min(C, target_c), dtype=torch.long),
            "max_val":    max_val,
            "fname":      fname,
            "slice_num":  slice_num,
        }


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def _cf_tag(cf: float) -> str:
    """0.04 → 'cf040'"""
    return f"cf{int(round(cf * 1000)):03d}"


class TokenCachedDataset(torch.utils.data.Dataset):
    """
    Wraps a SliceDataset and attaches pre-computed MedITok tokens from disk.

    Expects the cache layout produced by precompute_tokens.py:
        {cache_dir}/acc{acceleration}_{cf_tag}/{fname_stem}.pt

    Each volume file contains N_MASKS undersampling variations. A random mask
    is selected on each __getitem__ call, providing mask augmentation at zero
    encoding cost.

    Keys added to each sample:
        tok_us_re  : (max_coils, K, L)  long  — US coil real tokens (random mask)
        tok_us_im  : (max_coils, K, L)  long  — US coil imag tokens
        tok_gt_rss : (K, L)             long  — GT RSS tokens (mask-independent)
        mask       : (1, 1, W, 1)       bool  — the mask used for tok_us_*
        num_coils  : scalar             long

    Falls back to raw images (no token keys) on cache miss.
    """

    def __init__(
        self,
        base_dataset: SliceDataset,
        cache_dir: str,
        acceleration: int,
        center_fraction: float,
        max_coils: int = 20,
        include_images: bool = True,
    ):
        self.base            = base_dataset
        self.acc_dir         = Path(cache_dir) / f"acc{acceleration}_{_cf_tag(center_fraction)}"
        self.max_coils       = max_coils
        self.include_images  = include_images

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i: int):
        import random
        sample    = self.base[i]
        fname     = sample["fname"]
        slice_num = int(sample["slice_num"])
        stem      = Path(fname).stem
        tok_path  = self.acc_dir / f"{stem}.pt"

        if not tok_path.exists():
            return sample   # cache miss — fall back to on-the-fly encoding

        vol = torch.load(tok_path, map_location="cpu", weights_only=True)

        # Locate this slice in the volume
        slice_idx = (vol["slice_nums"] == slice_num).nonzero(as_tuple=True)[0].item()

        C       = int(vol["num_coils"][slice_idx])
        n_masks = vol["tok_us_re"].shape[1]
        K       = vol["tok_us_re"].shape[3]
        L       = vol["tok_us_re"].shape[4]

        # Pick a random mask variation
        mask_idx = random.randrange(n_masks)

        def _pad(t):  # (C, K, L) → (max_coils, K, L)
            if C < self.max_coils:
                pad = t.new_zeros(self.max_coils - C, K, L)
                return torch.cat([t, pad], dim=0)
            return t[:self.max_coils]

        # int16 → long for embedding lookups
        sample["tok_us_re"]  = _pad(vol["tok_us_re"] [slice_idx, mask_idx].long())
        sample["tok_us_im"]  = _pad(vol["tok_us_im"] [slice_idx, mask_idx].long())
        sample["tok_gt_rss"] =      vol["tok_gt_rss"][slice_idx].long()           # (K, L)
        sample["num_coils"]  = torch.tensor(C, dtype=torch.long)

        # Restore mask shape (1, 1, W, 1) expected by DC layer
        raw_mask = vol["masks"][slice_idx, mask_idx]   # (W,) bool
        sample["mask"] = raw_mask.view(1, 1, -1, 1)

        if not self.include_images:
            for key in ("x_us", "x_gt", "kspace_gt", "kspace_us"):
                sample.pop(key, None)

        return sample


def build_multicoil_dataset(
    root: str,
    mask_func: Callable,
    crop_size: Tuple[int, int] = (256, 256),
    challenge: str = "multicoil",
    use_seed: bool = True,
    max_coils: int = 20,
    token_cache_dir: Optional[str] = None,
    acceleration: int = 4,
    center_fraction: float = 0.04,
    include_images: bool = True,
    vcc_matrix_dir: Optional[str] = None,
    n_virtual_coils: int = 8,
    min_coils: int = 0,
) -> torch.utils.data.Dataset:
    """Build a multicoil dataset with MulticoilTransform applied.

    Args:
        root:             path to directory of H5 files
        mask_func:        callable from data.subsample (used when not using cache)
        crop_size:        spatial crop; default (256, 256)
        challenge:        "multicoil" or "singlecoil"
        use_seed:         seed mask RNG from filename for reproducible masks
        token_cache_dir:  if set, wrap with TokenCachedDataset loading pre-computed tokens
        acceleration:     undersampling factor — selects the correct cache subdir
        center_fraction:  ACS fraction — selects the correct cache subdir
        include_images:   when using cache, set False to drop raw image tensors
        vcc_matrix_dir:   if set, apply VCC projection using pre-computed W matrices
        n_virtual_coils:  C' virtual coils to use (key into the .pt files)
        min_coils:        skip scans with fewer than this many physical coils (0 = no filter)
    """
    import h5py as _h5py
    transform = MulticoilTransform(mask_func=mask_func, crop_size=crop_size,
                                   use_seed=use_seed, max_coils=max_coils,
                                   vcc_matrix_dir=vcc_matrix_dir,
                                   n_virtual_coils=n_virtual_coils)
    base = SliceDataset(root=Path(root), transform=transform, challenge=challenge)

    if min_coils > 0:
        # Read coil count once per file, filter raw_samples list in-place.
        _coil_cache: dict = {}
        filtered = []
        for sample in base.raw_samples:
            fname = sample.fname
            if fname not in _coil_cache:
                try:
                    with _h5py.File(fname, "r") as hf:
                        _coil_cache[fname] = int(hf["kspace"].shape[1])
                except Exception:
                    _coil_cache[fname] = 0
            if _coil_cache[fname] >= min_coils:
                filtered.append(sample)
        n_before = len(base.raw_samples)
        base.raw_samples = filtered
        logging.getLogger(__name__).info(
            f"min_coils={min_coils}: kept {len(filtered)}/{n_before} slices "
            f"({n_before - len(filtered)} dropped)"
        )

    if token_cache_dir is not None:
        return TokenCachedDataset(
            base,
            cache_dir=token_cache_dir,
            acceleration=acceleration,
            center_fraction=center_fraction,
            max_coils=max_coils,
            include_images=include_images,
        )
    return base
