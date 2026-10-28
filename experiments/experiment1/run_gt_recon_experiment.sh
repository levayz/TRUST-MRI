#!/usr/bin/env bash
# GT reconstruction experiment.
# Runs FAR with fully-sampled k-space mask on brain + knee test splits.
# Results saved under <run_dir>/patch_entropy/.../method=GT/ so that
# existing notebook visualization cells work with METHOD='GT'.
#
# Usage:
#   cd /fast/lev/TRUST-MRI
#   bash experiments/experiment1/run_gt_recon_experiment.sh
#
# Options:
#   --run_dir PATH         Output root (default: results/gt_recon)
#   --gpus LIST            Comma-separated GPU IDs (default: 4,5,6,7)
#   --center_slices N      Center slices per scan (default: 15)
#   --batch_size N         Slices per forward pass (default: 16)
#   --python PATH          Python executable (default: /opt/miniconda3/bin/python)

set -euo pipefail

RUN_DIR="results/gt_recon"
GPUS="4,5,6,7"
CENTER_SLICES=15
BATCH_SIZE=16
PYTHON="/opt/miniconda3/bin/python"
CONFIG="experiments/experiment1/config.yaml"
SCRIPT="experiments/experiment1/run_gt_recon_experiment.py"

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
echo " GT Reconstruction Experiment"
echo "════════════════════════════════════════════════"
echo "  run_dir      : ${RUN_DIR}"
echo "  gpus         : ${GPUS}"
echo "  center_slices: ${CENTER_SLICES}"
echo "  batch_size   : ${BATCH_SIZE}"
echo ""

launch brain_R8 "${GPU_LIST[0]}" \
    --config "${CONFIG}" --run_dir "${RUN_DIR}" \
    --dataset brain --acceleration 8 \
    --split_file fastmri/splits/suno_brain_test.txt \
    --batch_size "${BATCH_SIZE}" --device cuda:0

launch brain_R16 "${GPU_LIST[1]}" \
    --config "${CONFIG}" --run_dir "${RUN_DIR}" \
    --dataset brain --acceleration 16 \
    --split_file fastmri/splits/suno_brain_test.txt \
    --batch_size "${BATCH_SIZE}" --device cuda:0

launch knee_R8 "${GPU_LIST[2]}" \
    --config "${CONFIG}" --run_dir "${RUN_DIR}" \
    --dataset knee --acceleration 8 \
    --split_file fastmri/splits/adasense_test_split.txt \
    --batch_size "${BATCH_SIZE}" --device cuda:0

launch knee_R16 "${GPU_LIST[3]}" \
    --config "${CONFIG}" --run_dir "${RUN_DIR}" \
    --dataset knee --acceleration 16 \
    --split_file fastmri/splits/adasense_test_split.txt \
    --batch_size "${BATCH_SIZE}" --device cuda:0

echo ""
echo "Waiting for GT jobs..."
wait_all

# ── Sanity check: same experiment but with undersampled input (method=GT_US) ──
echo ""
echo "=== GT_US sanity check (undersampled input) ==="

launch brain_R8_US "${GPU_LIST[0]}" \
    --config "${CONFIG}" --run_dir "${RUN_DIR}" \
    --dataset brain --acceleration 8 \
    --split_file fastmri/splits/suno_brain_test.txt \
    --batch_size "${BATCH_SIZE}" --device cuda:0 --use_undersampled_input

launch brain_R16_US "${GPU_LIST[1]}" \
    --config "${CONFIG}" --run_dir "${RUN_DIR}" \
    --dataset brain --acceleration 16 \
    --split_file fastmri/splits/suno_brain_test.txt \
    --batch_size "${BATCH_SIZE}" --device cuda:0 --use_undersampled_input

launch knee_R8_US "${GPU_LIST[2]}" \
    --config "${CONFIG}" --run_dir "${RUN_DIR}" \
    --dataset knee --acceleration 8 \
    --split_file fastmri/splits/adasense_test_split.txt \
    --batch_size "${BATCH_SIZE}" --device cuda:0 --use_undersampled_input

launch knee_R16_US "${GPU_LIST[3]}" \
    --config "${CONFIG}" --run_dir "${RUN_DIR}" \
    --dataset knee --acceleration 16 \
    --split_file fastmri/splits/adasense_test_split.txt \
    --batch_size "${BATCH_SIZE}" --device cuda:0 --use_undersampled_input

echo ""
echo "Waiting for GT_US jobs..."
wait_all

echo ""
echo "════════════════════════════════════════════════"
echo " All jobs complete."
echo " GT    : ${RUN_DIR}/patch_entropy/.../method=GT/"
echo " GT_US : ${RUN_DIR}/patch_entropy/.../method=GT_US/"
echo " Logs  : ${LOG_DIR}/"
echo "════════════════════════════════════════════════"
