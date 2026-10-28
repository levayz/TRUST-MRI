#!/usr/bin/env bash
# Patch-level entropy investigation.
# Stage 1: all LES variants in parallel (4 GPUs) — brain + knee, R=8 + R=16
# Stage 2: all GEO variants in parallel (4 GPUs) after LES completes
#
# Usage:
#   cd /fast/lev/TRUST-MRI
#   bash experiments/experiment1/run_patch_entropy_experiment.sh
#
# Options:
#   --run_dir PATH         Output root (default: results/patch_entropy)
#   --gpus LIST            Comma-separated GPU IDs (default: 0,1,2,3)
#   --center_slices N      Center slices per scan (default: 15)
#   --batch_size N         LES batch size (default: 16)
#   --python PATH          Python executable (default: /opt/miniconda3/bin/python)

set -euo pipefail

RUN_DIR="results/patch_entropy"
GPUS="4,5,6,7"
CENTER_SLICES=15
BATCH_SIZE=16
PYTHON="/opt/miniconda3/bin/python"
CONFIG="experiments/experiment1/config.yaml"
SCRIPT="experiments/experiment1/run_patch_entropy_experiment.py"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_dir)       RUN_DIR="$2";       shift 2 ;;
        --gpus)          GPUS="$2";          shift 2 ;;
        --center_slices) CENTER_SLICES="$2"; shift 2 ;;
        --batch_size)    BATCH_SIZE="$2";    shift 2 ;;
        --python)        PYTHON="$2";        shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

IFS=',' read -ra GPU_LIST <<< "${GPUS}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

# launch appends directly to BG_PIDS and LABELS — no subshell, so wait works
BG_PIDS=()
LABELS=()

launch() {
    local label="$1" gpu="$2"; shift 2
    local log="${LOG_DIR}/${label}.log"
    echo "  [$(date +%H:%M:%S)]  GPU ${gpu}  ${label}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${SCRIPT}" "$@" >"${log}" 2>&1 &
    BG_PIDS+=($!)
    LABELS+=("${label}")
}

wait_all() {
    local failed=0
    for i in "${!BG_PIDS[@]}"; do
        if wait "${BG_PIDS[$i]}"; then
            echo "  ✓  ${LABELS[$i]}"
        else
            echo "  ✗  ${LABELS[$i]}  (see ${LOG_DIR}/${LABELS[$i]}.log)" >&2
            failed=1
        fi
    done
    BG_PIDS=(); LABELS=()
    (( failed == 0 )) || exit 1
}

echo "════════════════════════════════════════════════"
echo " Patch Entropy Experiment"
echo "════════════════════════════════════════════════"
echo "  run_dir      : ${RUN_DIR}"
echo "  gpus         : ${GPUS}"
echo "  center_slices: ${CENTER_SLICES}"
echo "  batch_size   : ${BATCH_SIZE} (LES_ZM only)"
echo ""

# ── Stage 1: LES_ZM ──────────────────────────────────────────────────────────
echo "=== Stage 1: LES_ZM ==="

launch brain_R16_LES_ZM "${GPU_LIST[0]}" \
    --config "${CONFIG}" --run_dir "${RUN_DIR}" \
    --dataset brain --acceleration 16 --method LES_ZM \
    --split_file fastmri/splits/suno_brain_test.txt \
    --center_slices "${CENTER_SLICES}" --batch_size "${BATCH_SIZE}" --device cuda:0

launch brain_R8_LES_ZM "${GPU_LIST[1]}" \
    --config "${CONFIG}" --run_dir "${RUN_DIR}" \
    --dataset brain --acceleration 8 --method LES_ZM \
    --split_file fastmri/splits/suno_brain_test.txt \
    --center_slices "${CENTER_SLICES}" --batch_size "${BATCH_SIZE}" --device cuda:0

launch knee_R16_LES_ZM "${GPU_LIST[2]}" \
    --config "${CONFIG}" --run_dir "${RUN_DIR}" \
    --dataset knee --acceleration 16 --method LES_ZM \
    --split_file fastmri/splits/adasense_test_split.txt \
    --center_slices "${CENTER_SLICES}" --batch_size "${BATCH_SIZE}" --device cuda:0

launch knee_R8_LES_ZM "${GPU_LIST[3]}" \
    --config "${CONFIG}" --run_dir "${RUN_DIR}" \
    --dataset knee --acceleration 8 --method LES_ZM \
    --split_file fastmri/splits/adasense_test_split.txt \
    --center_slices "${CENTER_SLICES}" --batch_size "${BATCH_SIZE}" --device cuda:0

echo ""
echo "Waiting for all LES_ZM jobs..."
wait_all
echo ""

# ── Stage 2: GEO ─────────────────────────────────────────────────────────────
echo "=== Stage 2: GEO ==="

launch brain_R16_GEO "${GPU_LIST[0]}" \
    --config "${CONFIG}" --run_dir "${RUN_DIR}" \
    --dataset brain --acceleration 16 --method GEO \
    --split_file fastmri/splits/suno_brain_test.txt \
    --center_slices "${CENTER_SLICES}" --device cuda:0

launch brain_R8_GEO "${GPU_LIST[1]}" \
    --config "${CONFIG}" --run_dir "${RUN_DIR}" \
    --dataset brain --acceleration 8 --method GEO \
    --split_file fastmri/splits/suno_brain_test.txt \
    --center_slices "${CENTER_SLICES}" --device cuda:0

launch knee_R16_GEO "${GPU_LIST[2]}" \
    --config "${CONFIG}" --run_dir "${RUN_DIR}" \
    --dataset knee --acceleration 16 --method GEO \
    --split_file fastmri/splits/adasense_test_split.txt \
    --center_slices "${CENTER_SLICES}" --device cuda:0

launch knee_R8_GEO "${GPU_LIST[3]}" \
    --config "${CONFIG}" --run_dir "${RUN_DIR}" \
    --dataset knee --acceleration 8 --method GEO \
    --split_file fastmri/splits/adasense_test_split.txt \
    --center_slices "${CENTER_SLICES}" --device cuda:0

echo ""
echo "Waiting for all GEO jobs..."
wait_all

echo ""
echo "════════════════════════════════════════════════"
echo " All jobs complete."
echo " Results: ${RUN_DIR}/patch_entropy/"
echo " Logs   : ${LOG_DIR}/"
echo "════════════════════════════════════════════════"
