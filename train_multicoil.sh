#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=5,6
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_P2P_DISABLE=1
MASTER_PORT="${MASTER_PORT:-29505}"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR="results_multicoil/brain/x8/run5_img_loss_resume_from_run4"
SIMPLE=false
# PRETRAIN_BACKBONE="results_multicoil/brain/x8/run1/0075000.pt"
PRETRAIN_BACKBONE="/fast/lev/TRUST-MRI/results_multicoil/brain/x8/run3_8_coils/0050000.pt"
FREEZE_BACKBONE=true

# Optional: restrict val set to a fixed split file (one H5 filename per line).
# Leave empty to use the full val_path directory.
VAL_SPLIT_FILE="fastmri/splits/suno_brain_test.txt"   # e.g. "fastmri/splits/suno_brain_test.txt"

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
USE_CACHE=false

# ---------------------------------------------------------------------------
# VCC (Virtual Coil Compression)
# Pre-compute once with precompute_vcc.py, then set VCC_DIR_TRAIN/VAL.
# C'=8 recommended (see results_vcc_benchmark/results.csv).
# ---------------------------------------------------------------------------
N_VIRTUAL_COILS=8
USE_VCC=true

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
D=1024
N_LAYER=24
N_HEAD=16
DE=64
MAX_COILS=8
CROP_SIZE=256

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
# Set CUDA_VISIBLE_DEVICES to the GPUs you want to use.
# The number of GPUs is derived automatically — no need to set it separately.

BATCH_SIZE=8     # per-GPU batch size (reduced to fit img-loss decoder activations)
MIXED_PRECISION=bf16   # bf16 (default), fp16, or none (fp32)
LR=1e-4
LAMBDA_COIL=1
LAMBDA_IMG=1
MAX_STEPS=100000
VAL_EVERY=10000
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
# Hardcoded paths 
# # ---------------------------------------------------------------------------
TRAIN_PATH="fastmri/brain/multicoil_train"
VAL_PATH="fastmri/brain/multicoil_val"

MEDITOK_CONFIG="/fast/lev/meditok/weights/meditok/config.json"
MEDITOK_CKPT="/fast/lev/meditok/weights/meditok/meditok_ckpt_v1.pth"

VCC_DIR_TRAIN="vcc_matrices/brain/train"
VCC_DIR_VAL="vcc_matrices/brain/val"

TOKEN_CACHE_TRAIN="token_cache/brain/train"
TOKEN_CACHE_VAL="token_cache/brain/val"
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

SPLIT_ARGS=""
if [ -n "${VAL_SPLIT_FILE}" ]; then
    SPLIT_ARGS="--val_split_file ${VAL_SPLIT_FILE}"
fi

PRETRAIN_ARGS=""
if [ -n "${PRETRAIN_BACKBONE:-}" ]; then
    PRETRAIN_ARGS="--pretrain_backbone ${PRETRAIN_BACKBONE}"
    if [ "${FREEZE_BACKBONE:-false}" = "true" ]; then
        PRETRAIN_ARGS="${PRETRAIN_ARGS} --freeze_backbone"
    fi
fi

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
    --lambda_img       "${LAMBDA_IMG}"       \
    --max_steps        "${MAX_STEPS}"        \
    --val_every        "${VAL_EVERY}"        \
    --save_every       "${SAVE_EVERY}"       \
    --log_every        "${LOG_EVERY}"        \
    --num_workers      "${NUM_WORKERS}"      \
    --grad_clip        "${GRAD_CLIP}"        \
    --weight_decay     "${WEIGHT_DECAY}"     \
    --vis_every        "${VIS_EVERY}"        \
    --mixed_precision  "${MIXED_PRECISION}"  \
    --no_scheduler \
    $([ "${SIMPLE:-true}" = "true" ] && echo "--simple") \
    ${CACHE_ARGS}    \
    ${VCC_ARGS}      \
    ${SPLIT_ARGS}    \
    ${PRETRAIN_ARGS} \
    "$@"
