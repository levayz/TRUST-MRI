"""
Scan the FastMRI singlecoil knee splits and write a CSV mapping each file to
its acquisition type (PD or PDFS).

Output: fastmri/knee/scan_types.csv
Columns: filename, split, acquisition_raw, scan_type
"""

import csv
import os
import h5py

SPLITS = {
    "train": "/fast/lev/TRUST-MRI/fastmri/knee/singlecoil_train",
    "val": "/fast/lev/TRUST-MRI/fastmri/knee/singlecoil_val",
}

ACQ_MAP = {
    "CORPD_FBK": "PD",
    "CORPDFS_FBK": "PDFS",
}

OUT_PATH = "/fast/lev/TRUST-MRI/fastmri/knee/scan_types.csv"


def main():
    rows = []
    for split, directory in SPLITS.items():
        for fname in sorted(os.listdir(directory)):
            if not fname.endswith(".h5"):
                continue
            fpath = os.path.join(directory, fname)
            with h5py.File(fpath, "r") as hf:
                acq_raw = hf.attrs.get("acquisition", "UNKNOWN")
            scan_type = ACQ_MAP.get(acq_raw, "UNKNOWN")
            rows.append(
                {
                    "filename": fname,
                    "split": split,
                    "acquisition_raw": acq_raw,
                    "scan_type": scan_type,
                }
            )

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "split", "acquisition_raw", "scan_type"]
        )
        writer.writeheader()
        writer.writerows(rows)

    pd_count = sum(1 for r in rows if r["scan_type"] == "PD")
    pdfs_count = sum(1 for r in rows if r["scan_type"] == "PDFS")
    unknown = sum(1 for r in rows if r["scan_type"] == "UNKNOWN")
    print(f"Wrote {len(rows)} entries to {OUT_PATH}")
    print(f"  PD:      {pd_count}")
    print(f"  PDFS:    {pdfs_count}")
    if unknown:
        print(f"  UNKNOWN: {unknown}")


if __name__ == "__main__":
    main()
