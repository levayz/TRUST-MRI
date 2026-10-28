#!/usr/bin/env bash
# Token accuracy experiment: LES_ZM vs GEO vs RANDOM.
# Each job runs all 3 methods sequentially (RANDOM is cheap, LES/GEO are serial per slice).
#
# Usage:
#   cd /fast/lev/TRUST-MRI
#   bash experiments/experiment1/run_token_accuracy.sh
#
# Options:
#   --run_dir PATH       Output root (default: results/token_accuracy)
#   --gpus LIST          Comma-separated GPU IDs (default: 0,1,2,3)
#   --batch_size N       Batch size for LES_ZM and RANDOM (default: 16)
#   --geo_batch_size N   Batch size for GEO (default: 4, GEO needs more memory)
#   --python PATH        Python executable (default: /opt/miniconda3/bin/python)

set -euo pipefail

RUN_DIR="results/token_accuracy"
GPUS="0,1,2,3"
BATCH_SIZE=16
GEO_BATCH_SIZE=4
PYTHON="/opt/miniconda3/bin/python"
CONFIG="experiments/experiment1/config.yaml"
SCRIPT="experiments/experiment1/run_token_accuracy.py"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_dir) RUN_DIR="$2"; shift 2 ;;
        --gpus)    GPUS="$2";    shift 2 ;;
        --batch_size)     BATCH_SIZE="$2";     shift 2 ;;
        --geo_batch_size) GEO_BATCH_SIZE="$2"; shift 2 ;;
        --python)         PYTHON="$2";         shift 2 ;;
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
echo " Token Accuracy Experiment"
echo "════════════════════════════════════════════════"
echo "  run_dir        : ${RUN_DIR}"
echo "  gpus           : ${GPUS}"
echo "  batch_size     : ${BATCH_SIZE} (LES_ZM, RANDOM)"
echo "  geo_batch_size : ${GEO_BATCH_SIZE} (GEO)"
echo ""

launch brain_R8  "${GPU_LIST[0]}" \
    --config "${CONFIG}" --run_dir "${RUN_DIR}" \
    --dataset brain --acceleration 8 \
    --split_file fastmri/splits/suno_brain_test.txt \
    --batch_size "${BATCH_SIZE}" --geo_batch_size "${GEO_BATCH_SIZE}" --device cuda:0

launch brain_R16 "${GPU_LIST[1]}" \
    --config "${CONFIG}" --run_dir "${RUN_DIR}" \
    --dataset brain --acceleration 16 \
    --split_file fastmri/splits/suno_brain_test.txt \
    --batch_size "${BATCH_SIZE}" --geo_batch_size "${GEO_BATCH_SIZE}" --device cuda:0

launch knee_R8   "${GPU_LIST[2]}" \
    --config "${CONFIG}" --run_dir "${RUN_DIR}" \
    --dataset knee --acceleration 8 \
    --split_file fastmri/splits/adasense_test_split.txt \
    --batch_size "${BATCH_SIZE}" --geo_batch_size "${GEO_BATCH_SIZE}" --device cuda:0

launch knee_R16  "${GPU_LIST[3]}" \
    --config "${CONFIG}" --run_dir "${RUN_DIR}" \
    --dataset knee --acceleration 16 \
    --split_file fastmri/splits/adasense_test_split.txt \
    --batch_size "${BATCH_SIZE}" --geo_batch_size "${GEO_BATCH_SIZE}" --device cuda:0

echo ""
echo "Waiting for all jobs..."
wait_all

echo ""
echo "════════════════════════════════════════════════"
echo " All jobs complete."
echo " Results: ${RUN_DIR}/token_accuracy/"
echo " Logs   : ${LOG_DIR}/"
echo "════════════════════════════════════════════════"
