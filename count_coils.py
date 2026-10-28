"""
Print the number of coils for every scan in a fastMRI multicoil directory.

Usage:
    python count_coils.py fastmri/brain/multicoil_val
    python count_coils.py fastmri/brain/multicoil_train --sort coils
"""

import argparse
from collections import Counter
from pathlib import Path

import h5py


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir", help="directory of .h5 files")
    p.add_argument("--sort", choices=["name", "coils"], default="name",
                   help="sort output by filename (default) or coil count")
    args = p.parse_args()

    files = sorted(Path(args.data_dir).glob("*.h5"))
    if not files:
        print(f"No .h5 files found in {args.data_dir}")
        return

    rows = []
    for f in files:
        try:
            with h5py.File(f, "r") as h:
                ksp = h["kspace"]
                n_coils  = ksp.shape[1]   # (slices, coils, H, W)
                n_slices = ksp.shape[0]
            rows.append((f.name, n_coils, n_slices))
        except Exception as e:
            rows.append((f.name, -1, -1))
            print(f"  ERROR {f.name}: {e}")

    if args.sort == "coils":
        rows.sort(key=lambda r: r[1])

    # Header
    print(f"{'Filename':<50}  {'Coils':>5}  {'Slices':>6}")
    print("-" * 65)
    for name, nc, ns in rows:
        if nc == -1:
            print(f"{name:<50}  {'ERR':>5}")
        else:
            print(f"{name:<50}  {nc:>5}  {ns:>6}")

    # Summary
    coil_counts = [nc for _, nc, _ in rows if nc > 0]
    dist = Counter(coil_counts)
    print()
    print(f"Total scans : {len(rows)}")
    print(f"Coil count distribution:")
    for nc in sorted(dist):
        print(f"  {nc:>2} coils : {dist[nc]:>4} scans")


if __name__ == "__main__":
    main()
