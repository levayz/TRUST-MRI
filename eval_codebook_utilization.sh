#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export HDF5_USE_FILE_LOCKING=FALSE

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BRAIN_ROOT="fastmri/brain/singlecoil_emulated_train"
KNEE_ROOT="fastmri/knee/singlecoil_train"
MEDITOK_CONFIG="/fast/lev/meditok/weights/meditok/config.json"
MEDITOK_CKPT="/fast/lev/meditok/weights/meditok/meditok_ckpt_v1.pth"
OUTPUT_DIR="results/codebook_utilization"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
BATCH_SIZE=16
NUM_WORKERS=4
CROP_SIZE=256
CHALLENGE="singlecoil"   # singlecoil | multicoil

python eval_codebook_utilization.py \
    --meditok_config  "${MEDITOK_CONFIG}" \
    --meditok_ckpt    "${MEDITOK_CKPT}"   \
    --brain_root      "${BRAIN_ROOT}"     \
    --knee_root       "${KNEE_ROOT}"      \
    --challenge       "${CHALLENGE}"      \
    --output_dir      "${OUTPUT_DIR}"     \
    --batch_size      "${BATCH_SIZE}"     \
    --num_workers     "${NUM_WORKERS}"    \
    --crop_size       "${CROP_SIZE}"      \
    "$@"
