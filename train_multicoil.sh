#!/usr/bin/env bash
# Train MulticoilRecon on fastMRI brain multicoil data.
#
# Usage:
#   bash train_multicoil.sh [extra args...]
#
# Example overrides:
#   bash train_multicoil.sh --batch_size 8 --max_steps 200000
#   bash train_multicoil.sh --overfit --max_steps 500 --batch_size 1

set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_PATH="fastmri/brain/multicoil_train"
VAL_PATH="fastmri/brain/multicoil_val"

MEDITOK_CONFIG="/fast/lev/meditok/weights/meditok/config.json"
MEDITOK_CKPT="/fast/lev/meditok/weights/meditok/meditok_ckpt_v1.pth"

OUTPUT_DIR="results_multicoil/brain/x8/debug"

# ---------------------------------------------------------------------------
# Mask / acceleration
# ---------------------------------------------------------------------------
CENTER_FRACTION=0.04
ACCELERATION=8       # choose from: 4, 8, 10, 16 (must match a cached subdir)

# ---------------------------------------------------------------------------
# Token cache
# Pre-compute once with precompute_tokens.py, then set USE_CACHE=true.
# Cache layout: {TOKEN_CACHE}/acc{ACCELERATION}_cf{CF}/{stem}.pt
# ---------------------------------------------------------------------------
TOKEN_CACHE_TRAIN="token_cache/brain/train"
TOKEN_CACHE_VAL="token_cache/brain/val"
USE_CACHE=false

# ---------------------------------------------------------------------------
# VCC (Virtual Coil Compression)
# Pre-compute once with precompute_vcc.py, then set VCC_DIR_TRAIN/VAL.
# Omit --vcc_matrix_dir to use raw physical coils (default).
# C'=8 recommended (see results_vcc_benchmark/results.csv).
# ---------------------------------------------------------------------------
VCC_DIR_TRAIN="vcc_matrices/brain/train"
VCC_DIR_VAL="vcc_matrices/brain/val"
N_VIRTUAL_COILS=4
USE_VCC=true

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
D=1024
N_LAYER=24
N_HEAD=16
DE=64
MAX_COILS=4
CROP_SIZE=256

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
# Set CUDA_VISIBLE_DEVICES to the GPUs you want to use.
# The number of GPUs is derived automatically — no need to set it separately.
CUDA_VISIBLE_DEVICES=3,4

BATCH_SIZE=16    # per-GPU batch size
LR=1e-4
LAMBDA_COIL=0.1
MAX_STEPS=100000
VAL_EVERY=1000
SAVE_EVERY=5000
LOG_EVERY=100
NUM_WORKERS=4
GRAD_CLIP=1.0
WEIGHT_DECAY=0.05
VIS_EVERY=200

# ---------------------------------------------------------------------------
# (Optional) Pre-compute tokens — run once per split, then leave commented
# Generates 25 mask variations for accelerations 4/8/10/16 with cf=0.04
# # ---------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=4 python precompute_tokens.py \
#     --data_root  "${TRAIN_PATH}"    --cache_dir "${TOKEN_CACHE_TRAIN}" \
#     --meditok_config "${MEDITOK_CONFIG}" --meditok_ckpt "${MEDITOK_CKPT}" \
#     --accelerations 4 8 10 16  --center_fraction 0.04  --n_masks 1 \
#     --batch_size 64  --slice_batch_size 4  --num_workers 4
# CUDA_VISIBLE_DEVICES=4 python precompute_tokens.py \
#     --data_root  "${VAL_PATH}"      --cache_dir "${TOKEN_CACHE_VAL}" \
#     --meditok_config "${MEDITOK_CONFIG}" --meditok_ckpt "${MEDITOK_CKPT}" \
#     --accelerations 4 8 10 16  --center_fraction 0.04  --n_masks 25 \
#     --batch_size 64  --slice_batch_size 4  --num_workers 4

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
CACHE_ARGS=""
if [ "${USE_CACHE}" = "true" ]; then
    CACHE_ARGS="--use_cache --token_cache_train ${TOKEN_CACHE_TRAIN} --token_cache_val ${TOKEN_CACHE_VAL}"
fi

VCC_ARGS=""
if [ "${USE_VCC}" = "true" ]; then
    VCC_ARGS="--vcc_matrix_dir ${VCC_DIR_TRAIN} --vcc_matrix_dir_val ${VCC_DIR_VAL} --n_virtual_coils ${N_VIRTUAL_COILS}"
fi

export CUDA_VISIBLE_DEVICES="3,4"
# Disable HDF5 POSIX file locking — prevents deadlock when multiple DDP
# worker processes open the same .h5 file concurrently.
export HDF5_USE_FILE_LOCKING=FALSE
# G292-Z45 GPUs connect via PCIe host bridge (NODE topology, no NVLink).
# NCCL's PCIe P2P path busy-waits indefinitely on this topology; force the
# host-memory shared-memory path instead.
export NCCL_P2P_DISABLE=1
MASTER_PORT="${MASTER_PORT:-29501}"
NGPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
if [ "${NGPUS}" -gt 1 ]; then
    LAUNCHER="torchrun --nproc_per_node ${NGPUS} --master_port ${MASTER_PORT}"
else
    LAUNCHER="python"
fi

${LAUNCHER} train_multicoil.py \
    --train_path       "${TRAIN_PATH}"       \
    --val_path         "${VAL_PATH}"         \
    --meditok_config   "${MEDITOK_CONFIG}"   \
    --meditok_ckpt     "${MEDITOK_CKPT}"     \
    --output_dir       "${OUTPUT_DIR}"       \
    --center_fraction  "${CENTER_FRACTION}"  \
    --acceleration     "${ACCELERATION}"     \
    --D                "${D}"                \
    --n_layer          "${N_LAYER}"          \
    --n_head           "${N_HEAD}"           \
    --De               "${DE}"              \
    --max_coils        "${MAX_COILS}"        \
    --crop_size        "${CROP_SIZE}"        \
    --batch_size       "${BATCH_SIZE}"       \
    --lr               "${LR}"              \
    --lambda_coil      "${LAMBDA_COIL}"      \
    --max_steps        "${MAX_STEPS}"        \
    --val_every        "${VAL_EVERY}"        \
    --save_every       "${SAVE_EVERY}"       \
    --log_every        "${LOG_EVERY}"        \
    --num_workers      "${NUM_WORKERS}"      \
    --grad_clip        "${GRAD_CLIP}"        \
    --weight_decay     "${WEIGHT_DECAY}"     \
    --vis_every        "${VIS_EVERY}"        \
    --no_scheduler \
    --simple       \
    ${CACHE_ARGS}  \
    ${VCC_ARGS}    \
    "$@"
