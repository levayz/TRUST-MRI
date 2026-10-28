#!/usr/bin/env bash
# Latency benchmark for active k-space acquisition methods.
#
# Runs each requested method N_TRIALS times and reports mean ± std latency.
# N_WARMUP throwaway runs are done first to prime the GPU before timing starts.
#
# Usage:
#   cd /fast/lev/TRUST-MRI
#   bash scripts/run_benchmark_timing.sh --les --gles --tds
#   bash scripts/run_benchmark_timing.sh --geo --ps_kspace --steps 16
#
# Options:
#   --far              Include FAR baseline
#   --les              Benchmark LES
#   --geo              Benchmark GEO
#   --gles             Benchmark GLES
#   --ps_kspace        Benchmark PS_kspace
#   --tds              Benchmark TDS
#   --dataset NAME     brain | knee                 (default: brain)
#   --acceleration N   R factor                     (default: 8)
#   --steps N          Lines acquired per sequence  (default: 8)
#   --rank N           Lines acquired per step      (default: 1)
#   --n_warmup N       Throwaway runs before timing (default: 20)
#   --n_trials N       Timed runs per method        (default: 100)
#   --ps_n_samples N   Posterior samples per PS_kspace step (default: 8)
#   --gpu ID           CUDA device                  (default: 0)
#   --output FILE      Write table to file in addition to stdout
#   --python PATH      Python executable            (default: /opt/miniconda3/envs/meditok/bin/python)
#   -h / --help        Show this help

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
RUN_FAR=false
RUN_LES=false
RUN_GEO=false
RUN_GLES=false
RUN_PS_KSPACE=false
RUN_TDS=false

DATASET="brain"
ACCELERATION=8
STEPS=8
RANK=1
N_WARMUP=20
N_TRIALS=100
PS_N_SAMPLES=8
GPU="7"
OUTPUT=""
PYTHON="/opt/miniconda3/envs/meditok/bin/python"
SCRIPT="scripts/benchmark_timing.py"

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --far)        RUN_FAR=true;       shift   ;;
        --les)        RUN_LES=true;       shift   ;;
        --geo)        RUN_GEO=true;       shift   ;;
        --gles)       RUN_GLES=true;      shift   ;;
        --ps_kspace)  RUN_PS_KSPACE=true; shift   ;;
        --tds)        RUN_TDS=true;       shift   ;;
        --dataset)      DATASET="$2";      shift 2 ;;
        --acceleration) ACCELERATION="$2"; shift 2 ;;
        --steps)        STEPS="$2";        shift 2 ;;
        --rank)         RANK="$2";         shift 2 ;;
        --n_warmup)     N_WARMUP="$2";     shift 2 ;;
        --n_trials)     N_TRIALS="$2";     shift 2 ;;
        --ps_n_samples) PS_N_SAMPLES="$2"; shift 2 ;;
        --gpu)          GPU="$2";          shift 2 ;;
        --output)       OUTPUT="$2";       shift 2 ;;
        --python)       PYTHON="$2";       shift 2 ;;
        -h|--help)
            sed -n '/^# Latency benchmark/,/^set -euo/{ /^set -euo/d; s/^# \?//; p }' "$0"
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ "${RUN_FAR}" != "true" && "${RUN_LES}" != "true" && "${RUN_GEO}" != "true" \
   && "${RUN_GLES}" != "true" && "${RUN_PS_KSPACE}" != "true" && "${RUN_TDS}" != "true" ]]; then
    echo "Nothing to run. Pass at least one of: --far --les --geo --gles --ps_kspace --tds" >&2
    exit 1
fi

# ── Build methods list ────────────────────────────────────────────────────────
METHODS=()
[[ "${RUN_FAR}"       == "true" ]] && METHODS+=(FAR)
[[ "${RUN_LES}"       == "true" ]] && METHODS+=(LES)
[[ "${RUN_GEO}"       == "true" ]] && METHODS+=(GEO)
[[ "${RUN_GLES}"      == "true" ]] && METHODS+=(GLES)
[[ "${RUN_PS_KSPACE}" == "true" ]] && METHODS+=(PS_kspace)
[[ "${RUN_TDS}"       == "true" ]] && METHODS+=(TDS)

# ── Optional output flag ──────────────────────────────────────────────────────
OUTPUT_FLAG=()
[[ -n "${OUTPUT}" ]] && OUTPUT_FLAG=(--output "${OUTPUT}")

# ── Run ───────────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${SCRIPT}" \
    --dataset      "${DATASET}" \
    --acceleration "${ACCELERATION}" \
    --budget       "${STEPS}" \
    --rank         "${RANK}" \
    --n-warmup     "${N_WARMUP}" \
    --n-trials     "${N_TRIALS}" \
    --ps-n-samples "${PS_N_SAMPLES}" \
    --methods      "${METHODS[@]}" \
    --device       cuda \
    "${OUTPUT_FLAG[@]+"${OUTPUT_FLAG[@]}"}"
