#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="1"

# ── Checkpoints, accelerations, organs ───────────────────────────────────────
# Add one entry per run; arrays must be the same length.
CKPTS=(
  "" # ckpt here
)
ACCS=(
  8
  # 16
)
ORGANS=(
  knee
  # brain
)

BASE_OUTDIR="results/knee/x8/"

# ── Per-organ data config ─────────────────────────────────────────────────────
get_data_config() {
  local organ=$1
  case $organ in
    brain)
      DATA_ROOT="fastmri/brain/singlecoil_emulated_all"
      DATA_SPLIT="fastmri/splits/multicoil_brain_test.txt"
      IMG_SIZE=256
      COND_TOKENS=256
      ;;
    knee)
      DATA_ROOT="fastmri/knee/singlecoil_val"
      # DATA_SPLIT="fastmri/splits/adasense_test_split.txt"
      DATA_SPLIT="fastmri/splits/full_knee_val_split.txt"
      IMG_SIZE=320
      COND_TOKENS=400
      ;;
  esac
}

# ── Number of active-sampling steps ──────────────────────────────────────────
get_as_steps() {
  local res=$1 acc=$2
  awk -v res="$res" -v acc="$acc" -v cf="$CF" \
    'BEGIN { x = res * (1/acc - cf); print (x == int(x)) ? int(x) : int(x) + 1 }'
}

# ── Fixed evaluation config ───────────────────────────────────────────────────
CF=0.04
GEN_STEPS=1
TOPK=0
NUM_CLASSES=8
BATCH_SIZE=4
GPT_MODEL="GPT-L"

# ── Helper: run one evaluation variant ───────────────────────────────────────
run_variant() {
  local base_outdir=$1 ckpt=$2 acc=$3 organ=$4 variant=$5
  shift 5
  local outdir="$base_outdir/$variant"

  mkdir -p "$outdir"
  echo "$ckpt" > "$outdir/checkpoint.txt"

  python autoregressive/sample/reconstruct_far.py \
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
    --encoder-ckpt weights/meditok/meditok_simple_v1.pth \
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
    --temperature 1.0 \
    --top-k "$TOPK" \
    --top-p 1.0 \
    --gt-mode raw \
    --n-scans-to-save 1000 \
    "$@"
}

# ── Main loop ─────────────────────────────────────────────────────────────────
for i in "${!CKPTS[@]}"; do
  CKPT="${CKPTS[$i]}"
  ACC="${ACCS[$i]}"
  ORGAN="${ORGANS[$i]}"
  
  # Allow override via BASE_OUTDIR environment variable
  if [[ -z "${BASE_OUTDIR:-}" ]]; then
    BASE_OUTDIR="$(dirname "$(dirname "$CKPT")")/eval"
  fi

  get_data_config "$ORGAN"
  AS_STEPS=$(get_as_steps "$IMG_SIZE" "$ACC")

  if [[ "$CKPT" =~ best\.pt$ ]]; then
    V_PREFIX="cf_${CF}_acc_${ACC}_"
  else
    ITER=$(basename "$CKPT" .pt)
    V_PREFIX="cf_${CF}_acc_${ACC}_iter_${ITER}_"
  fi

  echo "==> [$((i+1))/${#CKPTS[@]}] organ=$ORGAN  acc=$ACC  as_steps=$AS_STEPS"

  # Plain reconstruction (no active sampling)
  # run_variant "$BASE_OUTDIR" "$CKPT" "$ACC" "$ORGAN" "${V_PREFIX}val"

  # LES — Latent Entropy Selection
  run_variant "$BASE_OUTDIR" "$CKPT" "$ACC" "$ORGAN" "${V_PREFIX}val_LES_${AS_STEPS}step" \
    --active-sampling --active-sampling-steps "$AS_STEPS" --active-sampling-fn "patch_logits"

  # GEO — Gradient-based Entropy Optimization
  run_variant "$BASE_OUTDIR" "$CKPT" "$ACC" "$ORGAN" "${V_PREFIX}val_GEO_${AS_STEPS}step" \
    --active-sampling --active-sampling-steps "$AS_STEPS" --active-sampling-fn "gradient"
done
