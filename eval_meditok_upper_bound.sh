#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export HDF5_USE_FILE_LOCKING=FALSE

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH="fastmri/brain/multicoil_val"
VCC_DIR="vcc_matrices/brain/val"
MEDITOK_CONFIG="/fast/lev/meditok/weights/meditok/config.json"
MEDITOK_CKPT="/fast/lev/meditok/weights/meditok/meditok_ckpt_v1.pth"
SPLIT_FILE="fastmri/splits/suno_brain_test.txt"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
N_VIRTUAL_COILS=8
BATCH_SIZE=1
NUM_WORKERS=4
ENC_CHUNK_SIZE=8   # coil images per MedITok forward pass; reduce if OOM
N_VIS_SCANS=4
NO_QUANTIZE=true   # set to true to skip VQ quantization (continuous upper bound)
EVAL_SIZE="1024"        # set to e.g. 512 to upsample before encode/decode, then downsample back

# ---------------------------------------------------------------------------
# Derived: output dir encodes the run configuration
# ---------------------------------------------------------------------------
QUANT_ARG=""
[ "${NO_QUANTIZE}" = "true" ] && QUANT_ARG="--no_quantize"

EVAL_SIZE_ARG=""
SIZE_TAG=""
if [ -n "${EVAL_SIZE}" ]; then
    EVAL_SIZE_ARG="--eval_size ${EVAL_SIZE}"
    SIZE_TAG="_sz${EVAL_SIZE}"
fi

QUANT_TAG=""
[ "${NO_QUANTIZE}" = "true" ] && QUANT_TAG="_no_quant"

OUTPUT_DIR="results/meditok_upper_bound/brain${SIZE_TAG}${QUANT_TAG}"

python eval_meditok_upper_bound.py \
    --data_path       "${DATA_PATH}"       \
    --vcc_matrix_dir  "${VCC_DIR}"         \
    --meditok_config  "${MEDITOK_CONFIG}"  \
    --meditok_ckpt    "${MEDITOK_CKPT}"    \
    --output_dir      "${OUTPUT_DIR}"      \
    --split_file      "${SPLIT_FILE}"      \
    --n_virtual_coils "${N_VIRTUAL_COILS}" \
    --batch_size      "${BATCH_SIZE}"      \
    --num_workers     "${NUM_WORKERS}"     \
    --enc_chunk_size  "${ENC_CHUNK_SIZE}"  \
    --n_vis_scans     "${N_VIS_SCANS}"     \
    ${QUANT_ARG}                           \
    ${EVAL_SIZE_ARG}                       \
    "$@"
