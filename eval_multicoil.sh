#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="results_multicoil/brain/x8/run1/0075000.pt"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH="fastmri/brain/multicoil_val"

# Optional: restrict evaluation to a fixed split file (one H5 filename per line).
# Leave empty to evaluate the full DATA_PATH directory.
DATA_SPLIT="fastmri/splits/suno_brain_test.txt"   # e.g. "fastmri/splits/suno_brain_test.txt"

MEDITOK_CONFIG="/fast/lev/meditok/weights/meditok/config.json"
MEDITOK_CKPT="/fast/lev/meditok/weights/meditok/meditok_ckpt_v1.pth"

# Output dir alongside checkpoint, e.g. checkpoints/multicoil/run1/eval/
OUTPUT_DIR="$(dirname "${CHECKPOINT}")/eval/$(basename "${CHECKPOINT}" .pt)"

# ---------------------------------------------------------------------------
# Mask / acceleration  (must match training settings)
# ---------------------------------------------------------------------------
CENTER_FRACTIONS=0.04
ACCELERATION=8
CROP_SIZE=256
N_VIS_SCANS=4       # center-slice images saved per eval run (0 = disabled)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
SPLIT_ARGS=""
if [ -n "${DATA_SPLIT}" ]; then
    SPLIT_ARGS="--split_file ${DATA_SPLIT}"
fi

python eval_multicoil.py \
    --data_path         "${DATA_PATH}"         \
    --checkpoint        "${CHECKPOINT}"        \
    --meditok_config    "${MEDITOK_CONFIG}"    \
    --meditok_ckpt      "${MEDITOK_CKPT}"      \
    --output_dir        "${OUTPUT_DIR}"        \
    --center_fractions  "${CENTER_FRACTIONS}"  \
    --acceleration      "${ACCELERATION}"      \
    --crop_size         "${CROP_SIZE}"         \
    --batch_size        4                      \
    --num_workers       4                      \
    --n_vis_scans       "${N_VIS_SCANS}"       \
    ${SPLIT_ARGS}                              \
    "$@"

echo "Results written to: ${OUTPUT_DIR}"
