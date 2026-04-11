"""
Pre-compute VCC (Virtual Coil Compression) matrices per scan.

Computes W ∈ ℂ^{C'×C} from the ACS (auto-calibration signal) region of
the 256×256 center-cropped k-space, stacked across all slices of a scan.
W is fixed across all slices (one matrix per scan file).

ACS for 1-D Cartesian cf=0.04, W=256:
    n_acs = int(round(256 * 0.04)) = 10 columns
    pad   = (256 - 10 + 1) // 2   = 123
    ACS columns: kspace_c[:, :, 123:133]

Output layout:
    {vcc_dir}/{stem}.pt
        W_4:              (4,  C) complex64   — compression matrix for C'=4
        W_8:              (8,  C) complex64   — compression matrix for C'=8
        singular_values:  (C,)    float32     — full spectrum (for elbow plots)
        n_coils_original: int
        sampling:         "1d_cartesian_cf0040"
        center_fraction:  0.04

Usage:
    python precompute_vcc.py --data_root fastmri/brain/multicoil_train --vcc_dir vcc_matrices/brain/train
    python precompute_vcc.py --data_root fastmri/brain/multicoil_val   --vcc_dir vcc_matrices/brain/val
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from dataset.data.fftc import ifft2c_new as ifft2c, fft2c_new as fft2c
from dataset.data.transforms import to_tensor, complex_center_crop, complex_pad

CROP = 256


def _acs_col_range(W: int, center_fraction: float):
    n_acs = int(round(W * center_fraction))
    pad   = (W - n_acs + 1) // 2
    return pad, pad + n_acs


def load_cropped_kspace(h5path: Path) -> torch.Tensor:
    """Load all slices, IFFT → center-crop 256×256 → FFT.

    Returns:
        (S, C, 256, 256) complex64
    """
    with h5py.File(h5path, "r") as f:
        raw = f["kspace"][()]          # (S, C, H, W) complex64
    kspace_np = raw.astype(np.complex64)
    slices = []
    for s in range(kspace_np.shape[0]):
        ks = to_tensor(kspace_np[s]).float()                            # (C, H, W, 2)
        x  = ifft2c(ks)                                                 # (C, H, W, 2)
        x  = complex_center_crop(complex_pad(x, CROP, CROP), (CROP, CROP))
        ks_crop = fft2c(x)                                              # (C, 256, 256, 2)
        slices.append(torch.view_as_complex(ks_crop.contiguous()))      # (C, 256, 256)
    return torch.stack(slices)   # (S, C, 256, 256)


def compute_vcc(kspace: torch.Tensor, center_fraction: float, n_coils_list: list):
    """SVD on stacked ACS columns → compression matrices.

    Returns:
        matrices: dict[C'] -> (C', C) complex64
        svals:    (C,) float32
    """
    S, C, H, W = kspace.shape
    col_start, col_end = _acs_col_range(W, center_fraction)
    acs = kspace[:, :, :, col_start:col_end]           # (S, C, H, n_acs)
    A   = acs.permute(1, 0, 2, 3).reshape(C, -1)       # (C, S*H*n_acs)
    U, s, _ = torch.linalg.svd(A, full_matrices=False)  # U: (C, min(C,P))
    matrices = {cp: U[:, :min(cp, C)].conj().T for cp in n_coils_list}
    return matrices, s.float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",            required=True,
                        help="Directory of multicoil .h5 files")
    parser.add_argument("--vcc_dir",              required=True,
                        help="Output directory for per-scan .pt files")
    parser.add_argument("--n_virtual_coils_list", type=int, nargs="+", default=[4, 8],
                        help="C' values to store (default: 4 8)")
    parser.add_argument("--center_fraction",      type=float, default=0.04)
    parser.add_argument("--overwrite",            action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    vcc_dir   = Path(args.vcc_dir)
    vcc_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(data_root.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {data_root}")

    col_start, col_end = _acs_col_range(CROP, args.center_fraction)
    cf_tag = f"cf{int(round(args.center_fraction * 1000)):03d}"
    print(f"Scans       : {len(h5_files)}")
    print(f"C' values   : {args.n_virtual_coils_list}")
    print(f"ACS columns : [{col_start}:{col_end}]  ({col_end - col_start} lines)")
    print(f"Output dir  : {vcc_dir}\n")

    skipped = 0
    errors  = 0
    saved   = 0
    for h5path in tqdm(h5_files, desc="Computing VCC"):
        out_path = vcc_dir / f"{h5path.stem}.pt"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            kspace = load_cropped_kspace(h5path)
        except Exception as e:
            tqdm.write(f"WARNING: skipping {h5path.name} — {e}")
            errors += 1
            continue

        _, C, _, _ = kspace.shape
        matrices, svals = compute_vcc(kspace, args.center_fraction, args.n_virtual_coils_list)

        payload = {
            "singular_values":  svals,
            "n_coils_original": C,
            "sampling":         f"1d_cartesian_{cf_tag}",
            "center_fraction":  args.center_fraction,
        }
        for cp, W in matrices.items():
            payload[f"W_{cp}"] = W.to(torch.complex64)

        torch.save(payload, out_path)
        saved += 1

    print(f"\nDone. Saved {saved}, skipped {skipped}, errors {errors}.")
    print(f"Output: {vcc_dir}/")


if __name__ == "__main__":
    main()
