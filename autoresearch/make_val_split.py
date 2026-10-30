"""Carve the autoresearch validation split out of fastMRI brain multicoil_val, disjoint from the suno test split (which stays
reserved for the human's confirmation runs). Stratified like suno (11 AXT2 + 1 each of AXFLAIR / AXT1POST / AXT1PRE / AXT1),
seeded, volumes with <= 20 coils only. Writes splits/autoresearch_val.txt. Run once; the split is committed."""
import os, random, sys, pathlib
import h5py

ROOT = pathlib.Path(__file__).resolve().parents[1]
VAL = ROOT / "fastmri/brain/multicoil_val"
SUNO = ROOT / "fastmri/splits/suno_brain_test.txt"
OUT = pathlib.Path(__file__).resolve().parent / "splits/autoresearch_val.txt"
QUOTA = {"AXT2": 11, "AXFLAIR": 1, "AXT1POST": 1, "AXT1PRE": 1, "AXT1": 1}
SEED = 0

suno = {l.strip() for l in open(SUNO) if l.strip()}
by_acq = {k: [] for k in QUOTA}
for name in sorted(os.listdir(VAL)):
    if not name.endswith(".h5") or name in suno:
        continue
    with h5py.File(VAL / name, "r") as f:
        coils = f["kspace"].shape[1]
        acq = f.attrs.get("acquisition", "")
    if coils <= 20 and acq in by_acq:
        by_acq[acq].append(name)
rng = random.Random(SEED)
chosen = []
for acq, n in QUOTA.items():
    chosen += rng.sample(by_acq[acq], n)
chosen.sort()
OUT.write_text("\n".join(chosen) + "\n")
n_slices = 0
for name in chosen:
    with h5py.File(VAL / name, "r") as f:
        n_slices += f["kspace"].shape[0]
print(f"{len(chosen)} volumes, {n_slices} slices -> {OUT}")
for name in chosen:
    print(" ", name)
