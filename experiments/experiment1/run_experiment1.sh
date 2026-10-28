#!/usr/bin/env bash
# Run the full Experiment 1 pipeline.
#
# Stages:
#   1  Select slices (fast, ~1 min)
#   2  Normal trajectories  (LES / GEO × brain/knee × R8/R16)
#   3  Candidate counterfactuals  (LES / GEO × brain/knee × R8/R16)
#   3b Oracle active sampling     (mse / ssfd / token_acc × brain/knee × R8/R16)
#   4  Offline analyses (read-only, can all run in parallel)
#   5  Paper tables and plots
#
# Usage:
#   cd /fast/lev/TRUST-MRI
#   bash experiments/experiment1/run_experiment1.sh [options]
#
# Options:
#   --run_dir PATH     Output directory (default: results/exp1)
#   --device DEVICE    torch device string (default: cuda)
#   --stages LIST      Comma-separated list of stages to run (default: 1,2,3,3b,4,5)
#   --num_slices N     Limit slices per combo (default: all, use small N for smoke tests)
#   --jobs N           Max parallel jobs within a stage (default: 1)
#   --resume           Skip already-completed work instead of re-running it.
#                      Stage 1 is skipped when metadata files exist.
#                      Stages 2-3b always skip completed slices/steps internally.
#   -h / --help        Show this help

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=1
RUN_DIR="results/exp1"
DEVICE="cuda"
STAGES="1,2,3b,4,5,6"
NUM_SLICES="100"          # 100 slices = 1 center slice × 100 scans per dataset
MAX_JOBS=1
RESUME=false
PYTHON="/opt/miniconda3/bin/python"
CONFIG="experiments/experiment1/config.yaml"
SCRIPT_DIR="experiments/experiment1"

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_dir)   RUN_DIR="$2";   shift 2 ;;
        --device)    DEVICE="$2";    shift 2 ;;
        --stages)    STAGES="$2";    shift 2 ;;
        --num_slices) NUM_SLICES="$2"; shift 2 ;;
        --jobs)      MAX_JOBS="$2";  shift 2 ;;
        --python)    PYTHON="$2";    shift 2 ;;
        --resume)    RESUME=true;  shift ;;
        -h|--help)
            sed -n '/^# Run the full/,/^set -euo/{ /^set -euo/d; s/^# \?//; p }' "$0"
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Helpers ────────────────────────────────────────────────────────────────────
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

stage_enabled() {
    local stage="$1"
    [[ ",${STAGES}," == *",${stage},"* ]]
}

num_slices_flag() {
    if [[ -n "${NUM_SLICES}" ]]; then
        echo "--num_slices ${NUM_SLICES}"
    fi
}

# Run a command, writing stdout+stderr to a log file.
# Usage: run_job <log_name> <cmd...>
run_job() {
    local log_name="$1"; shift
    local log="${LOG_DIR}/${log_name}.log"
    echo "  [$(date +%H:%M:%S)]  $*  →  ${log}"
    "$@" >"${log}" 2>&1
}

# Launch a background job if MAX_JOBS > 1, otherwise run inline.
# Tracks PIDs in the global array BG_PIDS.
BG_PIDS=()
launch() {
    local log_name="$1"; shift
    local log="${LOG_DIR}/${log_name}.log"
    if (( MAX_JOBS > 1 )); then
        echo "  [$(date +%H:%M:%S)]  (bg) $*  →  ${log}"
        "$@" >"${log}" 2>&1 &
        BG_PIDS+=($!)
    else
        run_job "${log_name}" "$@"
    fi
}

# Wait for all background jobs and fail if any exited non-zero.
wait_jobs() {
    local failed=0
    for pid in "${BG_PIDS[@]+"${BG_PIDS[@]}"}"; do
        if ! wait "${pid}"; then
            echo "ERROR: background job PID ${pid} failed." >&2
            failed=1
        fi
    done
    BG_PIDS=()
    (( failed == 0 )) || exit 1
}

DATASETS=(brain knee)
ACCELERATIONS=(8 16)
NORMAL_METHODS=(LES GEO)
CF_METHODS=(LES GEO)
ORACLE_OBJECTIVES=(mse ssfd)
ORACLE_ACCELERATIONS=(16)          # oracle only at R=16 (most informative, saves ~50% compute)

echo "========================================================"
echo " Experiment 1: Entropy Calibration & Acquisition Utility"
echo "========================================================"
echo "  run_dir   : ${RUN_DIR}"
echo "  device    : ${DEVICE}"
echo "  stages    : ${STAGES}"
echo "  num_slices: ${NUM_SLICES:-all}"
echo "  max_jobs  : ${MAX_JOBS}"
echo "  resume    : ${RESUME}"
echo "  log_dir   : ${LOG_DIR}"
echo ""

# ═══════════════════════════════════════════════════════════════
# Stage 1 — Select slices
# ═══════════════════════════════════════════════════════════════
if stage_enabled "1"; then
    _stage1_done=false
    if [[ "${RESUME}" == "true" ]]; then
        # All four metadata files must exist to skip
        _all_meta=true
        for _ds in brain knee; do
            for _acc in 8 16; do
                if [[ ! -f "${RUN_DIR}/metadata/selected_slices_${_ds}_R${_acc}.json" ]]; then
                    _all_meta=false
                    break 2
                fi
            done
        done
        [[ "${_all_meta}" == "true" ]] && _stage1_done=true
    fi

    if [[ "${_stage1_done}" == "true" ]]; then
        echo "── Stage 1: skipping (metadata already exists, --resume set) ──"
    else
        echo "── Stage 1: select slices ──────────────────────────────"
        run_job "stage1_select_slices" \
            "${PYTHON}" "${SCRIPT_DIR}/select_slices_center_out.py" \
            --config "${CONFIG}" \
            --run_dir "${RUN_DIR}"
        echo "  Done."
    fi
    echo ""
fi

# ═══════════════════════════════════════════════════════════════
# Stage 2 — Normal trajectories
# ═══════════════════════════════════════════════════════════════
if stage_enabled "2"; then
    echo "── Stage 2: normal trajectories ────────────────────────"
    for ds in "${DATASETS[@]}"; do
        for acc in "${ACCELERATIONS[@]}"; do
            for method in "${NORMAL_METHODS[@]}"; do
                tag="traj_${ds}_R${acc}_${method}"
                launch "${tag}" \
                    "${PYTHON}" "${SCRIPT_DIR}/run_normal_trajectory.py" \
                    --config "${CONFIG}" \
                    --run_dir "${RUN_DIR}" \
                    --dataset "${ds}" \
                    --acceleration "${acc}" \
                    --method "${method}" \
                    --device "${DEVICE}" \
                    $(num_slices_flag)
                # Throttle: wait after every MAX_JOBS launches
            if (( MAX_JOBS > 1 && ${#BG_PIDS[@]} >= MAX_JOBS )); then
                    wait_jobs
                fi
            done
        done
    done
    wait_jobs
    echo "  Stage 2 done."
    echo ""
fi

# ═══════════════════════════════════════════════════════════════
# Stage 3 — Candidate counterfactuals
# ═══════════════════════════════════════════════════════════════
if stage_enabled "3"; then
    echo "── Stage 3: candidate counterfactuals ──────────────────"
    for ds in "${DATASETS[@]}"; do
        for acc in "${ORACLE_ACCELERATIONS[@]}"; do
            for method in "${CF_METHODS[@]}"; do
                tag="cf_${ds}_R${acc}_${method}"
                launch "${tag}" \
                    "${PYTHON}" "${SCRIPT_DIR}/run_candidate_counterfactuals.py" \
                    --config "${CONFIG}" \
                    --run_dir "${RUN_DIR}" \
                    --dataset "${ds}" \
                    --acceleration "${acc}" \
                    --base_method "${method}" \
                    --device "${DEVICE}" \
                    $(num_slices_flag)
                if (( MAX_JOBS > 1 && ${#BG_PIDS[@]} >= MAX_JOBS )); then
                    wait_jobs
                fi
            done
        done
    done
    wait_jobs
    echo "  Stage 3 done."
    echo ""
fi

# ═══════════════════════════════════════════════════════════════
# Stage 3b — Oracle active sampling
# ═══════════════════════════════════════════════════════════════
if stage_enabled "3b"; then
    echo "── Stage 3b: oracle active sampling ────────────────────"
    for ds in "${DATASETS[@]}"; do
        for acc in "${ORACLE_ACCELERATIONS[@]}"; do
            for obj in "${ORACLE_OBJECTIVES[@]}"; do
                tag="oracle_${ds}_R${acc}_${obj}"
                launch "${tag}" \
                    "${PYTHON}" "${SCRIPT_DIR}/run_oracle_active_sampling.py" \
                    --config "${CONFIG}" \
                    --run_dir "${RUN_DIR}" \
                    --dataset "${ds}" \
                    --acceleration "${acc}" \
                    --oracle_objective "${obj}" \
                    --device "${DEVICE}" \
                    $(num_slices_flag)
                if (( MAX_JOBS > 1 && ${#BG_PIDS[@]} >= MAX_JOBS )); then
                    wait_jobs
                fi
            done
        done
    done
    wait_jobs
    echo "  Stage 3b done."
    echo ""
fi

# ═══════════════════════════════════════════════════════════════
# Stage 4 — Offline analyses
# ═══════════════════════════════════════════════════════════════
if stage_enabled "4"; then
    echo "── Stage 4: offline analyses ───────────────────────────"

    for ds in "${DATASETS[@]}"; do
        for acc in "${ACCELERATIONS[@]}"; do
            for method in "${NORMAL_METHODS[@]}"; do

                launch "a1_entropy_token_error_${ds}_R${acc}_${method}" \
                    "${PYTHON}" "${SCRIPT_DIR}/analyze_entropy_token_error.py" \
                    --run_dir "${RUN_DIR}" \
                    --dataset "${ds}" \
                    --acceleration "${acc}" \
                    --method "${method}"

                launch "a2_calibration_${ds}_R${acc}_${method}" \
                    "${PYTHON}" "${SCRIPT_DIR}/analyze_calibration.py" \
                    --run_dir "${RUN_DIR}" \
                    --dataset "${ds}" \
                    --acceleration "${acc}" \
                    --method "${method}"

                launch "a3_patch_error_${ds}_R${acc}_${method}" \
                    "${PYTHON}" "${SCRIPT_DIR}/analyze_entropy_patch_error.py" \
                    --run_dir "${RUN_DIR}" \
                    --dataset "${ds}" \
                    --acceleration "${acc}" \
                    --method "${method}"

                launch "a6_entropy_recon_delta_${ds}_R${acc}_${method}" \
                    "${PYTHON}" "${SCRIPT_DIR}/analyze_entropy_reconstruction_delta.py" \
                    --run_dir "${RUN_DIR}" \
                    --dataset "${ds}" \
                    --acceleration "${acc}" \
                    --method "${method}"

                if (( MAX_JOBS > 1 && ${#BG_PIDS[@]} >= MAX_JOBS )); then
                    wait_jobs
                fi
            done

            for method in "${CF_METHODS[@]}"; do
                launch "a4_candidate_utility_${ds}_R${acc}_${method}" \
                    "${PYTHON}" "${SCRIPT_DIR}/analyze_candidate_line_utility.py" \
                    --run_dir "${RUN_DIR}" \
                    --dataset "${ds}" \
                    --acceleration "${acc}" \
                    --base_method "${method}"

                if (( MAX_JOBS > 1 && ${#BG_PIDS[@]} >= MAX_JOBS )); then
                    wait_jobs
                fi
            done

            launch "a5_oracle_${ds}_R${acc}" \
                "${PYTHON}" "${SCRIPT_DIR}/analyze_oracle_active_sampling.py" \
                --run_dir "${RUN_DIR}" \
                --dataset "${ds}" \
                --acceleration "${acc}"

            if (( MAX_JOBS > 1 && ${#BG_PIDS[@]} >= MAX_JOBS )); then
                wait_jobs
            fi
        done
    done

    # Stage 4 also runs entropy-validation analysis (requires Stage 6 data)
    launch "a7_entropy_validation" \
        "${PYTHON}" "${SCRIPT_DIR}/analyze_entropy_validation.py" \
        --run_dir "${RUN_DIR}"

    if (( MAX_JOBS > 1 && ${#BG_PIDS[@]} >= MAX_JOBS )); then
        wait_jobs
    fi

    wait_jobs
    echo "  Stage 4 done."
    echo ""
fi

# ═══════════════════════════════════════════════════════════════
# Stage 5 — Paper tables and plots
# ═══════════════════════════════════════════════════════════════
if stage_enabled "5"; then
    echo "── Stage 5: paper tables and plots ─────────────────────"
    run_job "stage5_tables_plots" \
        "${PYTHON}" "${SCRIPT_DIR}/make_experiment1_tables_and_plots.py" \
        --run_dir "${RUN_DIR}"
    echo "  Done."
    echo ""
fi

# ═══════════════════════════════════════════════════════════════
# Stage 6 — Entropy validation data generation
# ═══════════════════════════════════════════════════════════════
if stage_enabled "6"; then
    echo "── Stage 6: entropy validation data generation ─────────"
    NUM_SCANS_EXP=50

    for ds in "${DATASETS[@]}"; do
        for acc in "${ACCELERATIONS[@]}"; do
            launch "randmask_${ds}_R${acc}" \
                "${PYTHON}" "${SCRIPT_DIR}/run_random_mask_diagnostic.py" \
                --config "${CONFIG}" \
                --run_dir "${RUN_DIR}" \
                --dataset "${ds}" \
                --acceleration "${acc}" \
                --num_scans "${NUM_SCANS_EXP}" \
                --num_seeds 5 \
                --device "${DEVICE}"
            if (( MAX_JOBS > 1 && ${#BG_PIDS[@]} >= MAX_JOBS )); then
                wait_jobs
            fi
        done
    done
    wait_jobs

    for ds in "${DATASETS[@]}"; do
        for acc in "${ACCELERATIONS[@]}"; do
            for method in LES GEO; do
                launch "entacq_${ds}_R${acc}_${method}" \
                    "${PYTHON}" "${SCRIPT_DIR}/run_entropy_acquisition.py" \
                    --config "${CONFIG}" \
                    --run_dir "${RUN_DIR}" \
                    --dataset "${ds}" \
                    --acceleration "${acc}" \
                    --method "${method}" \
                    --num_scans "${NUM_SCANS_EXP}" \
                    --device "${DEVICE}"
                if (( MAX_JOBS > 1 && ${#BG_PIDS[@]} >= MAX_JOBS )); then
                    wait_jobs
                fi
            done
        done
    done
    wait_jobs
    echo "  Stage 6 done."
    echo ""
fi

echo "========================================================"
echo " Experiment 1 complete."
echo " Results: ${RUN_DIR}/results/"
echo " Logs:    ${LOG_DIR}/"
echo "========================================================"
