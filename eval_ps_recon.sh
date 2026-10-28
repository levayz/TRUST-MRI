#!/usr/bin/env bash
# Posterior-sampling reconstruction ("ps_recon") evaluation for the single-coil GPT-L models.
#
# For every slice: reconstruct N_PS times with top-K token sampling on the SAME undersampling
# mask, average the magnitude images (= final reconstruction), and also reconstruct greedily
# (argmax) on that mask as a paired baseline. Masks are seeded from the filename (--use-seed),
# so they are identical across k values and across runs.
#
# Usage:
#   ./eval_ps_recon.sh [--subdir NAME] ORGAN:ACC:TOPK:GPU [ORGAN:ACC:TOPK:GPU ...]
#   sweep on brain:   ./eval_ps_recon.sh --subdir ps_recon_topk_sweep brain:8:5:1 brain:8:10:1 brain:16:5:3 ...
#   final knee runs:  ./eval_ps_recon.sh knee:8:10:1 knee:16:10:3
# Jobs that share a GPU run sequentially; different GPUs run in parallel.
# Output: checkpoints/<organ>/cart_1D/eval[/<subdir>]/cf_<CF>_acc_<acc>_val_cart_1d_random_ps_recon_top<K>_n<N_PS>_<split>/
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
export PYTHONPATH="$(pwd)"

PYTHON="${PYTHON:-/opt/miniconda3/envs/meditok/bin/python}"   # the only env with piq

# ── fixed evaluation config (env-overridable) ────────────────────────────────
CF=0.04
GEN_STEPS=1                       # single-step reconstruction
N_PS="${N_PS:-8}"                 # stochastic reconstructions per slice
TEMPERATURE="${TEMPERATURE:-1.0}"
TOPP="${TOPP:-1.0}"
PS_AVERAGE="${PS_AVERAGE:-magnitude}"
SEED="${SEED:-0}"                 # torch seed for the token draws
BATCH_SIZE="${BATCH_SIZE:-4}"     # effective generation batch = BATCH_SIZE * N_PS
GPT_MODEL="GPT-L"
NUM_CLASSES=8
N_SCANS_TO_SAVE="${N_SCANS_TO_SAVE:-1000}"
ENCODER_CKPT="${ENCODER_CKPT:-weights/meditok/meditok_simple_v1.pth}"   # MedITok weights used for encoding the input and decoding the tokens
TOK_TAG="${TOK_TAG:-}"            # output-dir suffix when swapping the tokenizer, e.g. ft4ep -> ..._tokft4ep

SUBDIR=""
JOBS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --subdir) SUBDIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) JOBS+=("$1"); shift ;;
  esac
done
[[ ${#JOBS[@]} -gt 0 ]] || { echo "no jobs given (ORGAN:ACC:TOPK:GPU)" >&2; exit 1; }

get_data_config() {
  case "$1" in
    brain) DATA_ROOT="fastmri/brain/singlecoil_emulated_all"; DATA_SPLIT="fastmri/splits/suno_brain_test.txt";   SPLIT_TAG="suno";     IMG_SIZE=256; COND_TOKENS=256 ;;
    knee)  DATA_ROOT="fastmri/knee/singlecoil_val";           DATA_SPLIT="fastmri/splits/adasense_test_split.txt"; SPLIT_TAG="adasense"; IMG_SIZE=320; COND_TOKENS=400 ;;
    *) echo "unknown organ: $1" >&2; return 1 ;;
  esac
}

run_job() {   # organ acc topk gpu
  local organ=$1 acc=$2 topk=$3 gpu=$4
  get_data_config "$organ"
  local ckpt="checkpoints/${organ}/cart_1D/x${acc}/best.pt"
  [[ -f "$ckpt" ]] || { echo "missing checkpoint $ckpt" >&2; return 1; }
  local base_outdir="checkpoints/${organ}/cart_1D/eval${SUBDIR:+/$SUBDIR}"
  local variant="cf_${CF}_acc_${acc}_val_cart_1d_random_ps_recon_top${topk}_n${N_PS}_${SPLIT_TAG}${TOK_TAG:+_tok${TOK_TAG}}"
  local outdir="${base_outdir}/${variant}"
  mkdir -p "$outdir"
  echo "$ckpt" > "$outdir/checkpoint.txt"
  echo "$ENCODER_CKPT" > "$outdir/tokenizer.txt"
  echo "  [$(date +%H:%M:%S)] GPU ${gpu}  ${organ} x${acc} top-k=${topk}  -> ${outdir}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" autoregressive/sample/reconstruct_far.py \
    --output-dir "$outdir" \
    --gpt-ckpt "$ckpt" \
    --center-fractions "$CF" \
    --accelerations "$acc" \
    --gpt-model "$GPT_MODEL" \
    --gpt-type c2i \
    --num-classes "$NUM_CLASSES" \
    --num-codebooks 8 \
    --vocab-size 32768 \
    --image-size "$IMG_SIZE" \
    --downsample-size 16 \
    --encoder-ckpt "$ENCODER_CKPT" \
    --encoder-config weights/meditok/config.json \
    --dataset mri_code \
    --data-root "$DATA_ROOT" \
    --data-split "$DATA_SPLIT" \
    --batch-size "$BATCH_SIZE" \
    --num-workers 4 \
    --precision bf16 \
    --cfg-scale 1.0 \
    --cfg-interval -1 \
    --far-mask-mode radial \
    --complex-mode cartesian \
    --cls-token-num 1 \
    --cond-token-num "$COND_TOKENS" \
    --n-steps "$GEN_STEPS" \
    --temperature "$TEMPERATURE" \
    --top-k "$topk" \
    --top-p "$TOPP" \
    --gt-mode raw \
    --n-scans-to-save "$N_SCANS_TO_SAVE" \
    --use-seed \
    --seed "$SEED" \
    --ps-n-samples "$N_PS" \
    --ps-average "$PS_AVERAGE" \
    --ps-greedy-baseline \
    > "${outdir}.log" 2>&1
}

# ── group jobs by GPU: each GPU runs its jobs sequentially, GPUs in parallel ──
declare -A PER_GPU=()
for j in "${JOBS[@]}"; do
  IFS=: read -r organ acc topk gpu <<< "$j"
  [[ -n "${organ:-}" && -n "${acc:-}" && -n "${topk:-}" && -n "${gpu:-}" ]] || { echo "bad job spec: $j (want ORGAN:ACC:TOPK:GPU)" >&2; exit 1; }
  PER_GPU[$gpu]+="${organ}:${acc}:${topk} "
done

PIDS=(); LABELS=()
for gpu in "${!PER_GPU[@]}"; do
  (
    status=0
    for j in ${PER_GPU[$gpu]}; do
      IFS=: read -r organ acc topk <<< "$j"
      if run_job "$organ" "$acc" "$topk" "$gpu"; then
        echo "  [$(date +%H:%M:%S)] ✓ GPU ${gpu} ${organ} x${acc} top-k=${topk}"
      else
        echo "  [$(date +%H:%M:%S)] ✗ GPU ${gpu} ${organ} x${acc} top-k=${topk}" >&2; status=1
      fi
    done
    exit $status
  ) &
  PIDS+=($!); LABELS+=("GPU ${gpu}")
done

FAILED=0
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}" || { echo "  ✗ ${LABELS[$i]} had failures" >&2; FAILED=1; }
done
(( FAILED == 0 )) || exit 1
echo "All done."
