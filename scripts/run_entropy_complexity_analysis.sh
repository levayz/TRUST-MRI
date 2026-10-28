#!/usr/bin/env bash
# Run token-entropy vs image-complexity analysis for all four settings.
# Adjust OUTBASE, DEVICE, NUM_MASKS, and MAX_SLICES as needed.
set -e

SCRIPT="$(dirname "$0")/analyze_token_entropy_complexity.py"
OUTBASE="/fast/lev/TRUST-MRI/results/entropy_complexity"
DEVICE="cuda"
NUM_MASKS=3
MAX_SLICES=100   # set to empty string to use all slices
SEED=42

run_one() {
    local DATASET=$1
    local ACCEL=$2
    local OUTDIR="${OUTBASE}/${DATASET}_R${ACCEL}"
    echo "====== ${DATASET} R=${ACCEL} ======"
    python "$SCRIPT" \
        --dataset "$DATASET" \
        --acceleration "$ACCEL" \
        --output_dir "$OUTDIR" \
        --num_random_masks "$NUM_MASKS" \
        --seed "$SEED" \
        --num_sanity_checks 4 \
        --save_detail \
        --save_scatter \
        ${MAX_SLICES:+--max_slices "$MAX_SLICES"} \
        --device "$DEVICE"
}

run_one knee  8
run_one knee  16
run_one brain 8
run_one brain 16

echo "All done. Results in ${OUTBASE}/"
