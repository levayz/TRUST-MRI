#!/usr/bin/env python
"""Generate patch-level foreground masks for all slices in a fastMRI H5 directory.

Two methods available (--method):

  edges (default)
    Scans the column-summed magnitude profile for the first and last large
    horizontal edge to locate the left/right brain boundary.  All patch
    columns between those bounds are marked foreground — giving a single
    connected strip with no holes, regardless of brain shape.

  percentile
    Marks patches whose mean magnitude exceeds the foreground_percentile
    threshold.  Simpler but can produce disconnected regions.

Output — one .npz per scan under <output_dir>/:
    fg_masks   (n_slices, H_p, H_p)  float32   1=foreground, 0=background
    patch_mags (n_slices, H_p, H_p)  float32   mean |pixel| per patch

Usage:
    cd /fast/lev/TRUST-MRI
    python generate_foreground_masks.py                          # edge method, all scans
    python generate_foreground_masks.py --force                  # overwrite existing
    python generate_foreground_masks.py --method percentile      # old percentile method
    python generate_foreground_masks.py --edge_fraction 0.10 --smooth_width 5
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Pure-numpy image utilities (no torch — safe for multiprocessing workers)
# ---------------------------------------------------------------------------

def _ifft2(kspace: np.ndarray) -> np.ndarray:
    """Centered ortho-norm 2D IFFT matching data.transforms.ifft2."""
    shifted = np.fft.ifftshift(kspace, axes=(-2, -1))
    img     = np.fft.ifftn(shifted, axes=(-2, -1), norm="ortho")
    img     = np.fft.fftshift(img, axes=(-2, -1))
    return np.stack([img.real, img.imag], axis=-1).astype(np.float32)


def _pad(img: np.ndarray, H_out: int, W_out: int) -> np.ndarray:
    H, W, C = img.shape
    ph = max(0, H_out - H)
    pw = max(0, W_out - W)
    return np.pad(img, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)))


def _center_crop(img: np.ndarray, size: int) -> np.ndarray:
    H, W, _ = img.shape
    r0, c0 = (H - size) // 2, (W - size) // 2
    return img[r0:r0 + size, c0:c0 + size, :]


def load_slice_numpy(h5_path: pathlib.Path, slice_idx: int, resolution: int) -> np.ndarray:
    """Returns (resolution, resolution, 2) float32 complex image, max-normalised."""
    with h5py.File(h5_path, "r") as f:
        kspace_np = f["kspace"][slice_idx]

    img = _ifft2(kspace_np)
    H, W, _ = img.shape
    if H < resolution or W < resolution:
        img = _pad(img, resolution, resolution)
    img = _center_crop(img, resolution)

    max_val = float(np.sqrt(img[..., 0] ** 2 + img[..., 1] ** 2).max())
    if max_val > 0:
        img = img / max_val
    return img


# ---------------------------------------------------------------------------
# Foreground methods
# ---------------------------------------------------------------------------

def compute_fg_mask_edges(
    gt_image: np.ndarray,           # (H, W, 2)
    H_p: int = 16,
    edge_fraction: float = 0.10,
    smooth_width: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Foreground by left/right brain boundary detection.

    Algorithm:
      1. Sum pixel magnitudes over rows → column energy profile (W,)
      2. Smooth with a box filter to suppress noise
      3. Take absolute first-difference → edge strength profile (W-1,)
      4. Threshold at edge_fraction × max(edge_strength)
      5. Left boundary  = pixel column of first strong edge
         Right boundary = pixel column of last  strong edge
      6. All patch columns in [left_patch, right_patch] → foreground

    The result is always one connected horizontal strip with no holes.
    """
    H, W, _ = gt_image.shape
    ph = H // H_p
    pw = W // H_p

    mag = np.sqrt(gt_image[..., 0] ** 2 + gt_image[..., 1] ** 2)  # (H, W)
    col_energy = mag.sum(axis=0).astype(np.float64)                  # (W,)

    # Box-filter smoothing
    if smooth_width > 1:
        kernel = np.ones(smooth_width) / smooth_width
        col_energy = np.convolve(col_energy, kernel, mode="same")

    col_diff = np.abs(np.diff(col_energy))                           # (W-1,)
    threshold = col_diff.max() * edge_fraction

    strong_edges = np.where(col_diff > threshold)[0]
    if len(strong_edges) >= 2:
        left_px  = int(strong_edges[0])
        right_px = int(strong_edges[-1]) + 1
    elif len(strong_edges) == 1:
        left_px, right_px = int(strong_edges[0]), W - 1
    else:
        left_px, right_px = 0, W - 1

    # Map pixel boundaries → patch indices (clamp to valid range)
    left_p  = max(0, left_px // pw)
    right_p = min(H_p - 1, right_px // pw)

    fg_mask = np.zeros((H_p, H_p), dtype=np.float32)
    fg_mask[:, left_p:right_p + 1] = 1.0

    patch_mag = mag.reshape(H_p, ph, H_p, pw).mean(axis=(1, 3)).astype(np.float32)
    return fg_mask, patch_mag


def compute_fg_mask_percentile(
    gt_image: np.ndarray,           # (H, W, 2)
    H_p: int = 16,
    foreground_percentile: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Foreground by patch-magnitude percentile threshold (original method)."""
    H, W, _ = gt_image.shape
    ph = H // H_p
    pw = W // H_p

    mag = np.sqrt(gt_image[..., 0] ** 2 + gt_image[..., 1] ** 2)
    patch_mag = mag.reshape(H_p, ph, H_p, pw).mean(axis=(1, 3)).astype(np.float32)

    threshold = np.percentile(patch_mag, foreground_percentile)
    fg_mask = (patch_mag >= threshold).astype(np.float32)
    return fg_mask, patch_mag


# ---------------------------------------------------------------------------
# Per-scan worker
# ---------------------------------------------------------------------------

def process_scan(
    h5_path_str: str,
    output_dir_str: str,
    resolution: int,
    H_p: int,
    force: bool,
    method: str,
    # edges params
    edge_fraction: float,
    smooth_width: int,
    # percentile params
    percentile: float,
) -> str:
    h5_path    = pathlib.Path(h5_path_str)
    output_dir = pathlib.Path(output_dir_str)
    out_path   = output_dir / (h5_path.stem + ".npz")

    if out_path.exists() and not force:
        return h5_path.stem

    with h5py.File(h5_path, "r") as f:
        n_slices = f["kspace"].shape[0]

    fg_masks   = np.zeros((n_slices, H_p, H_p), dtype=np.float32)
    patch_mags = np.zeros((n_slices, H_p, H_p), dtype=np.float32)

    for s in range(n_slices):
        img = load_slice_numpy(h5_path, s, resolution)
        if method == "edges":
            fg, pm = compute_fg_mask_edges(
                img, H_p=H_p, edge_fraction=edge_fraction, smooth_width=smooth_width,
            )
        else:
            fg, pm = compute_fg_mask_percentile(img, H_p=H_p, foreground_percentile=percentile)
        fg_masks[s]   = fg
        patch_mags[s] = pm

    np.savez_compressed(out_path, fg_masks=fg_masks, patch_mags=patch_mags)
    return h5_path.stem


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Generate patch-level foreground masks for fastMRI H5 slices"
    )
    p.add_argument("--input_dir",  default="fastmri/brain/singlecoil_emulated_val")
    p.add_argument("--output_dir", default="fastmri/brain/singlecoil_emulated_val_fg_masks")
    p.add_argument("--resolution", default=256, type=int)
    p.add_argument("--H_p", default=16, type=int,
                   help="Patch grid size H_p×H_p (default 16)")
    p.add_argument("--method", default="edges", choices=["edges", "percentile"],
                   help="Foreground detection method (default: edges)")
    # edges params
    p.add_argument("--edge_fraction", default=0.10, type=float,
                   help="Edge threshold as fraction of max column-diff (default 0.10)")
    p.add_argument("--smooth_width", default=5, type=int,
                   help="Box-filter width for column-profile smoothing (default 5)")
    # percentile params
    p.add_argument("--percentile", default=20.0, type=float,
                   help="Bottom-p%% of patches excluded (percentile method, default 20)")
    p.add_argument("--workers", default=4, type=int)
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing .npz files")
    args = p.parse_args()

    root = pathlib.Path(__file__).resolve().parent

    def _resolve(path_str):
        p = pathlib.Path(path_str)
        return p if p.is_absolute() else root / p

    input_dir  = _resolve(args.input_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(input_dir.glob("*.h5"))
    if not h5_files:
        print(f"No .h5 files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    pending = [h for h in h5_files if args.force or not (output_dir / (h.stem + ".npz")).exists()]
    already_done = len(h5_files) - len(pending)

    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")
    print(f"Method : {args.method}", end="")
    if args.method == "edges":
        print(f"  (edge_fraction={args.edge_fraction}, smooth_width={args.smooth_width})")
    else:
        print(f"  (percentile={args.percentile})")
    print(f"Scans  : {len(h5_files)} total  |  {already_done} already done  |  {len(pending)} pending")

    if not pending:
        print("Nothing to do. Use --force to overwrite.")
        return

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_scan,
                str(h), str(output_dir),
                args.resolution, args.H_p, args.force,
                args.method, args.edge_fraction, args.smooth_width, args.percentile,
            ): h
            for h in pending
        }
        for fut in tqdm(as_completed(futures), total=len(pending), desc="Scans"):
            exc = fut.exception()
            if exc:
                print(f"\n  ERROR {futures[fut].name}: {exc}")

    print(f"\nDone. {len(list(output_dir.glob('*.npz')))} NPZ files in {output_dir}")


if __name__ == "__main__":
    main()
