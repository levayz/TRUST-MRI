#!/usr/bin/env bash
# Experiment 1 benchmark: oracle + entropy-based acquisition methods.
#
# Specify exactly which methods to run with per-method flags.
# Nothing runs unless you ask for it.
#
# Usage:
#   cd /fast/lev/TRUST-MRI
#   # Run everything:
#   bash experiments/experiment1/run_oracle_benchmark.sh --oracle --les --geo --ps --tds --erse
#   # Run only TDS:
#   bash experiments/experiment1/run_oracle_benchmark.sh --tds
#   # Run TDS + LES + GEO (skip oracle and PS):
#   bash experiments/experiment1/run_oracle_benchmark.sh --tds --les --geo
#
# Options:
#   --oracle               Run greedy oracle (all --objectives)
#   --les                  Run LES acquisition
#   --geo                  Run GEO acquisition
#   --ps                   Run PS acquisition
#   --tds                  Run TDS acquisition
#   --gles                 Run GLES acquisition
#   --random               Run random line acquisition (ablation baseline)
#   --run_dir PATH         Output root          (default: results/oracle_benchmark)
#   --gpus LIST            Comma-sep GPU IDs    (default: 0,1,2,3)
#   --batch_size N         Candidates per fwd pass for oracle/LES  (default: 32)
#   --objectives LIST      Comma-sep oracle objectives (default: psnr,ssfd)
#   --datasets LIST        Comma-sep datasets   (default: brain,knee)
#   --acceleration N       R factor             (default: 16)
#   --brain_split FILE     Brain split file     (default: fastmri/splits/suno_brain_test.txt)
#   --knee_split  FILE     Knee  split file     (default: fastmri/splits/adasense_test_split.txt)
#   --num_scans N          First N scans only   (default: all)
#   --shards N             Split each oracle job across N parallel GPU workers (default: 1)
#   --stratified           Select scans stratified by contrast with center slices only
#   --scans_per_contrast N Scans per contrast group when --stratified (default: 8)
#   --center_slices N      Center slices per scan when --stratified (default: 15)
#   --seed N               Random seed for --stratified selection (default: 42)
#   --ps_n_samples N       Posterior samples per step (default: 8; ps_sweep default: 16)
#   --save_ps_samples      Save per-step posterior samples (PS only, implies --ps)
#   --save_per_step_recon  Save recon_t{t}.npy at every acquisition step
#   --ps_sweep             Run 4 PS config variants instead of default PS (implies --ps)
#   --tds_candidate_mode M  geo_topk | all  (default: geo_topk)
#   --tds_candidate_k N     Shortlist size for geo_topk  (default: 16)
#   --gles_m N              Top-M patches for GLES; 0 = all  (default: 8)
#   --erse_residual_mode M  pred_minus_current | pred | current  (default: pred_minus_current)
#   --erse_weight_mode M    soft_entropy | top_m_entropy | foreground_soft_entropy  (default: soft_entropy)
#   --erse_top_m N          Top-M patches for top_m_entropy mode  (default: 32)
#   --python PATH          Python executable    (default: /opt/miniconda3/bin/python)
#   -h / --help            Show this help

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
RUN_ORACLE=false
RUN_LES=false
RUN_GEO=false
RUN_PS=false
RUN_PS_KSPACE=false
RUN_TDS=false
RUN_GLES=false
RUN_ERSE=false
RUN_RANDOM=false

RUN_DIR="results/oracle_benchmark"
GPUS="4,5,6,7"
BATCH_SIZE=32
OBJECTIVES="psnr,ssfd"
DATASETS="brain,knee"
ACCELERATION=16
BRAIN_SPLIT="fastmri/splits/suno_brain_test.txt"
KNEE_SPLIT="fastmri/splits/adasense_test_split.txt"
NUM_SCANS=""
SHARDS=1
STRATIFIED=false
SCANS_PER_CONTRAST=8
CENTER_SLICES=15
SEED=42
PS_N_SAMPLES=""
SAVE_PS_SAMPLES=false
SAVE_PER_STEP_RECON=false
PS_SWEEP=false
TDS_CANDIDATE_MODE="geo_topk"
TDS_CANDIDATE_K=16
GLES_M=8
ERSE_RESIDUAL_MODE="pred_minus_current"
ERSE_WEIGHT_MODE="soft_entropy"
ERSE_TOP_M=32
PYTHON="/opt/miniconda3/envs/meditok/bin/python"
CONFIG="experiments/experiment1/config.yaml"
SCRIPT="experiments/experiment1/run_oracle_active_sampling.py"
ENTROPY_SCRIPT="experiments/experiment1/run_entropy_acquisition.py"

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --oracle)   RUN_ORACLE=true;  shift   ;;
        --les)      RUN_LES=true;     shift   ;;
        --geo)      RUN_GEO=true;     shift   ;;
        --ps)          RUN_PS=true;          shift   ;;
        --ps_kspace)   RUN_PS_KSPACE=true;    shift   ;;
        --tds)      RUN_TDS=true;     shift   ;;
        --gles)     RUN_GLES=true;    shift   ;;
        --erse)     RUN_ERSE=true;    shift   ;;
        --random)   RUN_RANDOM=true;  shift   ;;
        --run_dir)       RUN_DIR="$2";       shift 2 ;;
        --gpus)          GPUS="$2";          shift 2 ;;
        --batch_size)    BATCH_SIZE="$2";    shift 2 ;;
        --objectives)    OBJECTIVES="$2";    shift 2 ;;
        --datasets)      DATASETS="$2";      shift 2 ;;
        --acceleration)  ACCELERATION="$2";  shift 2 ;;
        --brain_split)   BRAIN_SPLIT="$2";   shift 2 ;;
        --knee_split)    KNEE_SPLIT="$2";    shift 2 ;;
        --num_scans)          NUM_SCANS="$2";          shift 2 ;;
        --shards)             SHARDS="$2";             shift 2 ;;
        --stratified)         STRATIFIED=true;         shift   ;;
        --scans_per_contrast) SCANS_PER_CONTRAST="$2"; shift 2 ;;
        --center_slices)      CENTER_SLICES="$2";      shift 2 ;;
        --seed)               SEED="$2";               shift 2 ;;
        --ps_n_samples)       PS_N_SAMPLES="$2";       shift 2 ;;
        --save_ps_samples)      SAVE_PS_SAMPLES=true; RUN_PS=true; shift ;;
        --save_per_step_recon)  SAVE_PER_STEP_RECON=true; shift ;;
        --ps_sweep)           PS_SWEEP=true;        RUN_PS=true; shift ;;
        --tds_candidate_mode) TDS_CANDIDATE_MODE="$2"; shift 2 ;;
        --tds_candidate_k)    TDS_CANDIDATE_K="$2";    shift 2 ;;
        --gles_m)             GLES_M="$2";             shift 2 ;;
        --erse_residual_mode) ERSE_RESIDUAL_MODE="$2"; shift 2 ;;
        --erse_weight_mode)   ERSE_WEIGHT_MODE="$2";   shift 2 ;;
        --erse_top_m)         ERSE_TOP_M="$2";         shift 2 ;;
        --python)             PYTHON="$2";             shift 2 ;;
        -h|--help)
            sed -n '/^# Experiment 1/,/^set -euo/{ /^set -euo/d; s/^# \?//; p }' "$0"
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Warn if nothing was requested
if [[ "${RUN_ORACLE}" != "true" && "${RUN_LES}" != "true" && "${RUN_GEO}" != "true" \
   && "${RUN_PS}" != "true" && "${RUN_PS_KSPACE}" != "true" && "${RUN_TDS}" != "true" \
   && "${RUN_GLES}" != "true" && "${RUN_ERSE}" != "true" && "${RUN_RANDOM}" != "true" ]]; then
    echo "Nothing to run. Pass at least one of: --oracle --les --geo --ps --ps_kspace --tds --gles --erse --random" >&2
    exit 1
fi

# ── Parse lists ────────────────────────────────────────────────────────────────
IFS=',' read -ra GPU_LIST      <<< "${GPUS}"
IFS=',' read -ra OBJ_LIST      <<< "${OBJECTIVES}"
IFS=',' read -ra DATASET_LIST  <<< "${DATASETS}"
IFS=',' read -ra ACCEL_LIST    <<< "${ACCELERATION}"
NUM_GPUS=${#GPU_LIST[@]}

# ── Helpers ────────────────────────────────────────────────────────────────────
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

split_for() {
    local ds="$1"
    case "${ds}" in
        brain) echo "${BRAIN_SPLIT}" ;;
        knee)  echo "${KNEE_SPLIT}"  ;;
        *) echo "Unknown dataset: ${ds}" >&2; exit 1 ;;
    esac
}

BG_PIDS=()
JOB_LABELS=()
JOB_IDX=0

launch_job() {
    local label="$1"; shift
    local gpu_id="${GPU_LIST[$((JOB_IDX % NUM_GPUS))]}"
    local log="${LOG_DIR}/${label}.log"

    echo "  [$(date +%H:%M:%S)]  GPU ${gpu_id}  ${label}"
    echo "                      log → ${log}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" "$@" >"${log}" 2>&1 &
    BG_PIDS+=($!)
    JOB_LABELS+=("${label}")
    JOB_IDX=$(( JOB_IDX + 1 ))
}

wait_all() {
    local failed=0
    for i in "${!BG_PIDS[@]}"; do
        if wait "${BG_PIDS[$i]}"; then
            echo "  ✓  ${JOB_LABELS[$i]}"
        else
            echo "  ✗  ${JOB_LABELS[$i]}  (see ${LOG_DIR}/${JOB_LABELS[$i]}.log)" >&2
            failed=1
        fi
    done
    BG_PIDS=()
    JOB_LABELS=()
    (( failed == 0 )) || exit 1
}

# ── Banner ─────────────────────────────────────────────────────────────────────
_methods=""
[[ "${RUN_ORACLE}" == "true" ]] && _methods="${_methods} oracle"
[[ "${RUN_LES}"    == "true" ]] && _methods="${_methods} LES"
[[ "${RUN_GEO}"    == "true" ]] && _methods="${_methods} GEO"
[[ "${RUN_PS}"          == "true" ]] && _methods="${_methods} PS"
[[ "${RUN_PS_KSPACE}"   == "true" ]] && _methods="${_methods} PS_kspace"
[[ "${RUN_TDS}"    == "true" ]] && _methods="${_methods} TDS"
[[ "${RUN_GLES}"   == "true" ]] && _methods="${_methods} GLES"
[[ "${RUN_ERSE}"   == "true" ]] && _methods="${_methods} ERSE"
[[ "${RUN_RANDOM}" == "true" ]] && _methods="${_methods} random"

echo "════════════════════════════════════════════════════════"
echo " Experiment 1 Benchmark"
echo "════════════════════════════════════════════════════════"
echo "  methods     :${_methods}"
echo "  run_dir     : ${RUN_DIR}"
echo "  gpus        : ${GPUS}  (${NUM_GPUS} available)"
echo "  datasets    : ${DATASETS}"
echo "  acceleration: R=${ACCELERATION/,/,R=}"
printf "  brain split : %s\n" "${BRAIN_SPLIT}"
printf "  knee  split : %s\n" "${KNEE_SPLIT}"
[[ -n "${NUM_SCANS}" ]] && echo "  num_scans   : ${NUM_SCANS}" || echo "  num_scans   : all"
[[ "${STRATIFIED}" == "true" ]] && \
    echo "  stratified  : ${SCANS_PER_CONTRAST} scans/contrast × ${CENTER_SLICES} center slices (seed=${SEED})"
[[ "${RUN_ORACLE}" == "true" ]] && echo "  objectives  : ${OBJECTIVES}  batch_size=${BATCH_SIZE}"
[[ "${RUN_TDS}"    == "true" ]] && echo "  tds         : candidate_mode=${TDS_CANDIDATE_MODE}, k=${TDS_CANDIDATE_K}"
[[ "${RUN_GLES}"   == "true" ]] && echo "  gles        : M=${GLES_M}"
[[ "${RUN_ERSE}"   == "true" ]] && echo "  erse        : residual=${ERSE_RESIDUAL_MODE}, weight=${ERSE_WEIGHT_MODE}, top_m=${ERSE_TOP_M}"
[[ "${RUN_PS}"     == "true" && "${PS_SWEEP}"       == "true" ]] && echo "  ps_sweep    : 4 config variants"
[[ "${RUN_PS}"     == "true" && "${SAVE_PS_SAMPLES}" == "true" ]] && echo "  ps          : saving per-step samples"
echo ""

# ── Compute scans per shard for a given split file ────────────────────────────
total_scans_in() {
    wc -l < "$1" | tr -d ' '
}

# ── Sweep configs (used when --ps --ps_sweep) ─────────────────────────────────
PS_SWEEP_CONFIGS=(
    "PS_T1_k100:1.0:100:1.0"
    "PS_T1_pure:1.0:0:1.0"
    "PS_T2_pure:2.0:0:1.0"
    "PS_T1_p95:1.0:0:0.95"
)

# ── Launch jobs ────────────────────────────────────────────────────────────────
for ACCELERATION in "${ACCEL_LIST[@]}"; do
for ds in "${DATASET_LIST[@]}"; do
    split_file="$(split_for "${ds}")"

    # Shard range for oracle
    if [[ "${SHARDS}" -gt 1 ]]; then
        total_scans="$(total_scans_in "${split_file}")"
        [[ -n "${NUM_SCANS}" ]] && total_scans=$(( NUM_SCANS < total_scans ? NUM_SCANS : total_scans ))
        scans_per_shard=$(( (total_scans + SHARDS - 1) / SHARDS ))
    fi

    stratified_flags=()
    if [[ "${STRATIFIED}" == "true" ]]; then
        stratified_flags=(
            --stratified
            --scans_per_contrast "${SCANS_PER_CONTRAST}"
            --center_slices      "${CENTER_SLICES}"
            --seed               "${SEED}"
        )
    fi

    num_scans_flag=()
    [[ -n "${NUM_SCANS}" ]] && num_scans_flag=(--num_scans "${NUM_SCANS}")

    # ── Oracle ────────────────────────────────────────────────────────────────
    if [[ "${RUN_ORACLE}" == "true" ]]; then
        for obj in "${OBJ_LIST[@]}"; do
            for (( shard=0; shard<SHARDS; shard++ )); do
                if [[ "${SHARDS}" -gt 1 ]]; then
                    label="oracle_${ds}_R${ACCELERATION}_${obj}_shard${shard}"
                    shard_flags=(--scan_offset "$(( shard * scans_per_shard ))" --num_scans "${scans_per_shard}")
                else
                    label="oracle_${ds}_R${ACCELERATION}_${obj}"
                    shard_flags=()
                    [[ -n "${NUM_SCANS}" ]] && shard_flags=(--num_scans "${NUM_SCANS}")
                fi

                launch_job "${label}" \
                    "${PYTHON}" "${SCRIPT}" \
                    --config    "${CONFIG}" \
                    --run_dir   "${RUN_DIR}" \
                    --dataset   "${ds}" \
                    --acceleration "${ACCELERATION}" \
                    --oracle_objective "${obj}" \
                    --split_file "${split_file}" \
                    --batch_size "${BATCH_SIZE}" \
                    --device    cuda \
                    "${shard_flags[@]+"${shard_flags[@]}"}" \
                    "${stratified_flags[@]+"${stratified_flags[@]}"}"
            done
        done
    fi

    # shared flag — appended to every entropy launch_job below
    per_step_recon_flag=()
    [[ "${SAVE_PER_STEP_RECON}" == "true" ]] && per_step_recon_flag=(--save_per_step_recon)

    # ── LES ───────────────────────────────────────────────────────────────────
    if [[ "${RUN_LES}" == "true" ]]; then
        launch_job "entropy_${ds}_R${ACCELERATION}_LES" \
            "${PYTHON}" "${ENTROPY_SCRIPT}" \
            --config    "${CONFIG}" \
            --run_dir   "${RUN_DIR}" \
            --dataset   "${ds}" \
            --acceleration "${ACCELERATION}" \
            --method    LES \
            --batch_size "${BATCH_SIZE}" \
            --compute_ssfd \
            --split_file "${split_file}" \
            "${num_scans_flag[@]+"${num_scans_flag[@]}"}" \
            "${stratified_flags[@]+"${stratified_flags[@]}"}" \
            "${per_step_recon_flag[@]+"${per_step_recon_flag[@]}"}"
    fi

    # ── GEO ───────────────────────────────────────────────────────────────────
    if [[ "${RUN_GEO}" == "true" ]]; then
        launch_job "entropy_${ds}_R${ACCELERATION}_GEO" \
            "${PYTHON}" "${ENTROPY_SCRIPT}" \
            --config    "${CONFIG}" \
            --run_dir   "${RUN_DIR}" \
            --dataset   "${ds}" \
            --acceleration "${ACCELERATION}" \
            --method    GEO \
            --compute_ssfd \
            --split_file "${split_file}" \
            "${num_scans_flag[@]+"${num_scans_flag[@]}"}" \
            "${stratified_flags[@]+"${stratified_flags[@]}"}" \
            "${per_step_recon_flag[@]+"${per_step_recon_flag[@]}"}"
    fi

    # ── PS (regular, save_ps_samples, or ps_sweep) ────────────────────────────
    if [[ "${RUN_PS}" == "true" ]]; then
        if [[ "${PS_SWEEP}" == "true" ]]; then
            sweep_n_samples="${PS_N_SAMPLES:-16}"
            for cfg_str in "${PS_SWEEP_CONFIGS[@]}"; do
                IFS=':' read -r tag temperature top_k top_p <<< "${cfg_str}"
                launch_job "entropy_${ds}_R${ACCELERATION}_${tag}" \
                    "${PYTHON}" "${ENTROPY_SCRIPT}" \
                    --config       "${CONFIG}" \
                    --run_dir      "${RUN_DIR}" \
                    --dataset      "${ds}" \
                    --acceleration "${ACCELERATION}" \
                    --method       PS \
                    --ps_method_tag "${tag}" \
                    --ps_n_samples "${sweep_n_samples}" \
                    --ps_temperature "${temperature}" \
                    --ps_top_k     "${top_k}" \
                    --ps_top_p     "${top_p}" \
                    --compute_ssfd \
                    --split_file   "${split_file}" \
                    "${num_scans_flag[@]+"${num_scans_flag[@]}"}" \
                    "${stratified_flags[@]+"${stratified_flags[@]}"}"
            done
        else
            ps_n_flag=()
            [[ -n "${PS_N_SAMPLES}" ]] && ps_n_flag=(--ps_n_samples "${PS_N_SAMPLES}")
            save_ps_flag=()
            [[ "${SAVE_PS_SAMPLES}" == "true" ]] && save_ps_flag=(--save_ps_samples)

            launch_job "entropy_${ds}_R${ACCELERATION}_PS" \
                "${PYTHON}" "${ENTROPY_SCRIPT}" \
                --config    "${CONFIG}" \
                --run_dir   "${RUN_DIR}" \
                --dataset   "${ds}" \
                --acceleration "${ACCELERATION}" \
                --method    PS \
                --compute_ssfd \
                --split_file "${split_file}" \
                "${ps_n_flag[@]+"${ps_n_flag[@]}"}" \
                "${num_scans_flag[@]+"${num_scans_flag[@]}"}" \
                "${stratified_flags[@]+"${stratified_flags[@]}"}" \
                "${save_ps_flag[@]+"${save_ps_flag[@]}"}" \
                "${per_step_recon_flag[@]+"${per_step_recon_flag[@]}"}"
        fi
    fi

    # ── PS_kspace (AdaSense-compatible: k-space variance of complex samples) ──
    if [[ "${RUN_PS_KSPACE}" == "true" ]]; then
        ps_n_flag=()
        [[ -n "${PS_N_SAMPLES}" ]] && ps_n_flag=(--ps_n_samples "${PS_N_SAMPLES}")
        save_ps_flag=()
        [[ "${SAVE_PS_SAMPLES}" == "true" ]] && save_ps_flag=(--save_ps_samples)

        launch_job "entropy_${ds}_R${ACCELERATION}_PS_kspace" \
            "${PYTHON}" "${ENTROPY_SCRIPT}" \
            --config    "${CONFIG}" \
            --run_dir   "${RUN_DIR}" \
            --dataset   "${ds}" \
            --acceleration "${ACCELERATION}" \
            --method    PS_kspace \
            --compute_ssfd \
            --split_file "${split_file}" \
            "${ps_n_flag[@]+"${ps_n_flag[@]}"}" \
            "${num_scans_flag[@]+"${num_scans_flag[@]}"}" \
            "${stratified_flags[@]+"${stratified_flags[@]}"}" \
            "${save_ps_flag[@]+"${save_ps_flag[@]}"}" \
            "${per_step_recon_flag[@]+"${per_step_recon_flag[@]}"}"
    fi

    # ── TDS ───────────────────────────────────────────────────────────────────
    if [[ "${RUN_TDS}" == "true" ]]; then
        launch_job "entropy_${ds}_R${ACCELERATION}_TDS" \
            "${PYTHON}" "${ENTROPY_SCRIPT}" \
            --config    "${CONFIG}" \
            --run_dir   "${RUN_DIR}" \
            --dataset   "${ds}" \
            --acceleration "${ACCELERATION}" \
            --method    TDS \
            --tds_candidate_mode "${TDS_CANDIDATE_MODE}" \
            --tds_candidate_k    "${TDS_CANDIDATE_K}" \
            --compute_ssfd \
            --split_file "${split_file}" \
            "${num_scans_flag[@]+"${num_scans_flag[@]}"}" \
            "${stratified_flags[@]+"${stratified_flags[@]}"}" \
            "${per_step_recon_flag[@]+"${per_step_recon_flag[@]}"}"
    fi

    # ── GLES ──────────────────────────────────────────────────────────────────
    if [[ "${RUN_GLES}" == "true" ]]; then
        launch_job "entropy_${ds}_R${ACCELERATION}_GLES" \
            "${PYTHON}" "${ENTROPY_SCRIPT}" \
            --config    "${CONFIG}" \
            --run_dir   "${RUN_DIR}" \
            --dataset   "${ds}" \
            --acceleration "${ACCELERATION}" \
            --method    GLES \
            --gles_m    "${GLES_M}" \
            --compute_ssfd \
            --split_file "${split_file}" \
            "${num_scans_flag[@]+"${num_scans_flag[@]}"}" \
            "${stratified_flags[@]+"${stratified_flags[@]}"}" \
            "${per_step_recon_flag[@]+"${per_step_recon_flag[@]}"}"
    fi

    # ── ERSE ──────────────────────────────────────────────────────────────────
    if [[ "${RUN_ERSE}" == "true" ]]; then
        launch_job "entropy_${ds}_R${ACCELERATION}_ERSE" \
            "${PYTHON}" "${ENTROPY_SCRIPT}" \
            --config    "${CONFIG}" \
            --run_dir   "${RUN_DIR}" \
            --dataset   "${ds}" \
            --acceleration "${ACCELERATION}" \
            --method    ERSE \
            --erse_residual_mode "${ERSE_RESIDUAL_MODE}" \
            --erse_weight_mode   "${ERSE_WEIGHT_MODE}" \
            --erse_top_m         "${ERSE_TOP_M}" \
            --compute_ssfd \
            --split_file "${split_file}" \
            "${num_scans_flag[@]+"${num_scans_flag[@]}"}" \
            "${stratified_flags[@]+"${stratified_flags[@]}"}" \
            "${per_step_recon_flag[@]+"${per_step_recon_flag[@]}"}"
    fi

    # ── Random (ablation baseline) ────────────────────────────────────────────
    if [[ "${RUN_RANDOM}" == "true" ]]; then
        launch_job "entropy_${ds}_R${ACCELERATION}_random" \
            "${PYTHON}" "${ENTROPY_SCRIPT}" \
            --config    "${CONFIG}" \
            --run_dir   "${RUN_DIR}" \
            --dataset   "${ds}" \
            --acceleration "${ACCELERATION}" \
            --method    random \
            --compute_ssfd \
            --split_file "${split_file}" \
            "${num_scans_flag[@]+"${num_scans_flag[@]}"}" \
            "${stratified_flags[@]+"${stratified_flags[@]}"}" \
            "${per_step_recon_flag[@]+"${per_step_recon_flag[@]}"}"
    fi
done
done

# ── Wait and report ────────────────────────────────────────────────────────────
echo ""
echo "Waiting for all jobs to finish..."
wait_all

echo ""
echo "════════════════════════════════════════════════════════"
echo " Done."
[[ "${RUN_ORACLE}" == "true" ]] && echo " Oracle  : ${RUN_DIR}/oracle_active_sampling/"
echo " Entropy : ${RUN_DIR}/entropy_acquisition/"
echo " Logs    : ${LOG_DIR}/"
echo "════════════════════════════════════════════════════════"
