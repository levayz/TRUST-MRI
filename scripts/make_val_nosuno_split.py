#!/usr/bin/env python
"""Validation list for baselines trained on our training set: 15 multicoil_val volumes with <= 20 coils, disjoint from the
suno test split (so checkpoint selection never sees the test volumes). Seeded; writes results/feature_varnet/val_nosuno15.txt."""
import os
import random
from collections import Counter

import h5py

val = sorted(os.listdir("fastmri/brain/multicoil_val"))
test = {l.strip() for l in open("fastmri/splits/suno_brain_test.txt") if l.strip()}
train = {l.strip() for l in open("fastmri/splits/multicoil_brain_train.txt") if l.strip()}
print("val & train:", len(set(val) & train), "| suno test volumes in val:", len(test & set(val)))
rng = random.Random(0)
cand = [v for v in val if v not in test and v not in train]   # 89 val filenames also appear in the training list
rng.shuffle(cand)
out = []
for v in cand:
    with h5py.File(f"fastmri/brain/multicoil_val/{v}", "r") as f:
        if f["kspace"].shape[1] <= 20:
            out.append(v)
    if len(out) == 15:
        break
os.makedirs("results/feature_varnet", exist_ok=True)
with open("results/feature_varnet/val_nosuno15.txt", "w") as f:
    f.write("\n".join(sorted(out)) + "\n")
print("val contrasts:", Counter(n.split("_")[2] for n in out), "| suno:", Counter(n.split("_")[2] for n in test))
