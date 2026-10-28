"""
SENSE multicoil slice dataset for the TRUST-MRI GPT and the sensitivity-net trainer.

Per slice (conventions of eval_sense_block.py / MediTok_official datasets/fastmri_dataset.py):
    coils   x_c = crop(crop(ifft2c(k_c native), ESPIRiT-map grid), resolution)
    maps    oracle ESPIRiT maps (sum|S|^2 = 1 on the support, 0 outside, coil-0 gauge) cropped alike
    GT      x0 = sum_c conj(S_c) x_c / sum_c |S_c|^2 on the support
    scale   everything divided by max|x0| of the slice (the single-coil dataset's max_val convention)
    mask    1D Cartesian (cf, R) via mask_func, seeded by the filename when use_seed -> (H, W, 1) like mri_code
    ZF      x_zf = combine(ifft2c(fft2c(x_c) * mask), oracle maps)   (placeholder; the trainers recompute
            it on the GPU with the estimated maps)
Returned keys (coil tensors are zero-padded to max_coils):
    coils (max_coils,H,W,2) float32, num_coils int, support (H,W) bool, gt_c_image (H,W,2), masked_c_image (H,W,2),
    mask (H,W,1) float32, max_val float, fname str, slice int, gt_tokens zeros [2,8,256] (API compat),
    maps_oracle (max_coils,H,W,2) only when return_oracle_maps=True.
"""
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")   # DDP ranks x workers open the same files

import pathlib
import random
from collections import Counter
from typing import Optional, Sequence, Union

import h5py
import torch
from torch.utils.data import Dataset

from dataset.data.subsample import create_mask_for_mask_type
from models.sense_ops import c2r, center_crop_or_pad, crop_native_to_grid, fft2c, ifft2c, sense_combine, support_from_maps

SENSE_MASK_TYPES = ("random", "cart_1d", "equispaced")


class SenseSliceDataset(Dataset):
    def __init__(self, root, sens_dirs: Union[str, Sequence[str]], mask_func, resolution: int = 256,
                 max_coils: int = 20, use_seed: bool = True, custom_split: Optional[str] = None,
                 num_samples: Optional[int] = None, seed: int = 42, end_slice_skip: int = 0,
                 return_oracle_maps: bool = False, min_coils: int = 0, gt_cache_dir: Optional[str] = None,
                 gt_cache_token_size: Optional[int] = None):
        self.root = pathlib.Path(root)
        if isinstance(sens_dirs, str):
            sens_dirs = [d for d in sens_dirs.split(",") if d]
        self.sens_dirs = [pathlib.Path(d) for d in sens_dirs]
        self.mask_func = mask_func
        self.resolution = resolution
        self.max_coils = max_coils
        self.use_seed = use_seed
        self.end_slice_skip = end_slice_skip
        self.return_oracle_maps = return_oracle_maps
        self.gt_cache_dir = pathlib.Path(gt_cache_dir) if gt_cache_dir else None   # precompute_gt_cache.py: <dir>/<volume>.h5
        self.gt_cache_token_size = gt_cache_token_size
        random.seed(seed)

        if custom_split is not None:
            with open(custom_split, "r") as f:
                split_filenames = {line.strip() for line in f if line.strip()}
            files = [self.root / fname for fname in split_filenames if (self.root / fname).exists()]
            print(f"[SenseSliceDataset] custom split {custom_split}: {len(files)} files found under {self.root}")
        else:
            files = [p for p in self.root.iterdir() if p.suffix == ".h5"]
            if num_samples is not None:
                random.shuffle(files)
                files = files[:num_samples]

        self.examples, self.map_paths, self.num_coils = [], {}, {}
        skipped = Counter()
        for fname in sorted(files):
            map_path = next((d / fname.name for d in self.sens_dirs if (d / fname.name).exists()), None)
            if map_path is None:
                skipped["no_sens_maps"] += 1
                continue
            with h5py.File(fname, "r") as data:
                num_slices, C = data["kspace"].shape[:2]
            if C > max_coils:
                skipped[f"coils>{max_coils}"] += 1
                continue
            if min_coils > 0 and C < min_coils:
                skipped[f"coils<{min_coils}"] += 1
                continue
            self.map_paths[fname] = map_path
            self.num_coils[fname] = int(C)
            for s in range(num_slices - self.end_slice_skip):
                self.examples.append((fname, s))
        print(f"[SenseSliceDataset] {self.root}: {len(self.map_paths)} volumes, {len(self.examples)} slices, "
              f"max_coils={max_coils}, use_seed={use_seed}, oracle_maps={return_oracle_maps}; skipped={dict(skipped)}")
        if self.gt_cache_dir is not None:   # every volume must be cached (one batch must not mix cached and uncached items)
            missing = [f.name for f in self.map_paths if not (self.gt_cache_dir / f.name).exists()]
            assert not missing, f"[SenseSliceDataset] {len(missing)} of {len(self.map_paths)} volumes have no ground-truth cache under {self.gt_cache_dir} (e.g. {missing[:3]}): run precompute_gt_cache.py"
            first = next(iter(self.map_paths))
            with h5py.File(self.gt_cache_dir / first.name, "r") as data:
                a = dict(data.attrs)
                assert int(a["resolution"]) == resolution, f"ground-truth cache was built at resolution {a['resolution']}, dataset uses {resolution}"
                assert gt_cache_token_size is None or int(a["token_size"]) == gt_cache_token_size, f"ground-truth cache token size {a['token_size']} != {gt_cache_token_size}"
            print(f"[SenseSliceDataset] ground-truth tokens and latents from {self.gt_cache_dir} (token size {a['token_size']}, {a.get('encoder_ckpt', '?')})")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, s = self.examples[i]
        res = self.resolution
        with h5py.File(fname, "r") as data:
            k = torch.from_numpy(data["kspace"][s])                       # (C, H, W) complex64 native grid
        with h5py.File(self.map_paths[fname], "r") as data:
            maps = torch.from_numpy(data["sens_maps"][s])                 # (C, Hr, Wr) complex64 map grid
        coils = crop_native_to_grid(ifft2c(k), maps.shape[-2:], res)       # (C, res, res)
        maps = center_crop_or_pad(maps, (res, res))
        x0 = sense_combine(coils[None], maps[None])[0]                     # (res, res) physical units
        support = support_from_maps(maps[None])[0]
        max_val = x0.abs().max().clamp_min(1e-12)
        coils, x0 = coils / max_val, x0 / max_val

        seed = None if not self.use_seed else tuple(map(ord, pathlib.Path(fname).name))
        mask, _ = self.mask_func((res, res, 2), seed=seed)                 # 1D column mask (1, W, 1)
        assert mask.dim() == 3, f"SENSE datasets need a 1D Cartesian mask, got shape {tuple(mask.shape)}"
        mask = mask.repeat(res, 1, 1).float()                              # (H, W, 1)
        k_us = fft2c(coils) * mask[..., 0][None]
        x_zf = sense_combine(ifft2c(k_us)[None], maps[None])[0]

        C = coils.shape[0]
        coils_pad = torch.zeros(self.max_coils, res, res, dtype=torch.complex64)
        coils_pad[:C] = coils
        out = {
            "coils": c2r(coils_pad),                 # (max_coils, H, W, 2)
            "num_coils": C,
            "support": support,                      # (H, W) bool
            "gt_c_image": c2r(x0),                   # (H, W, 2) SENSE GT, max-normalised
            "masked_c_image": c2r(x_zf),             # (H, W, 2) zero-filled SENSE image (oracle maps)
            "mask": mask,                            # (H, W, 1)
            "max_val": float(max_val),
            "fname": pathlib.Path(fname).name,
            "slice": s,
            "gt_tokens": torch.zeros([2, 8, 256]),
        }
        if self.gt_cache_dir is not None:
            with h5py.File(self.gt_cache_dir / pathlib.Path(fname).name, "r") as data:
                out["gt_tokens"] = torch.from_numpy(data["tokens"][s].astype("int64"))        # (2, K, L) codes of the GT on the token grid
                out["gt_latent"] = torch.from_numpy(data["latent"][s]).float()               # (2, L, D) unit-norm chunks
            out["gt_cached"] = True
        if self.return_oracle_maps:
            maps_pad = torch.zeros(self.max_coils, res, res, dtype=torch.complex64)
            maps_pad[:C] = maps
            out["maps_oracle"] = c2r(maps_pad)
        return out


def _none_if_empty(s):
    return None if s in (None, "none", "") else s


def build_sense(args, **kwargs):
    mask_type = getattr(args, "mask_type", "cart_1d")
    assert mask_type in SENSE_MASK_TYPES, f"--dataset sense needs a 1D Cartesian mask, got {mask_type}"
    mask = create_mask_for_mask_type(mask_type, [getattr(args, "center_fractions", 0.04)],
                                     [getattr(args, "accelerations", 8)])
    sens_dirs = getattr(args, "sens_dirs", None)
    assert sens_dirs, "--sens-dirs is required for --dataset sense"
    return SenseSliceDataset(
        root=args.data_root,
        sens_dirs=sens_dirs,
        mask_func=mask,
        resolution=getattr(args, "sense_crop_size", None) or args.image_size,
        max_coils=getattr(args, "max_coils", 20),
        use_seed=getattr(args, "use_seed", False),
        custom_split=_none_if_empty(getattr(args, "data_split", None)),
        num_samples=getattr(args, "num_samples", None),
        seed=getattr(args, "global_seed", 42),
        end_slice_skip=getattr(args, "end_slice_skip", 0),
        return_oracle_maps=getattr(args, "return_oracle_maps", False),
        min_coils=getattr(args, "min_coils", 0),
        gt_cache_dir=_none_if_empty(getattr(args, "gt_cache_dir", None)),
        gt_cache_token_size=getattr(args, "image_size", None),
    )
