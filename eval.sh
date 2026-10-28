#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)"
GPUS=(6)

# ── Checkpoints, accelerations, organs ───────────────────────────────────────
# Add one entry per run; arrays must be the same length.
CKPTS=(
  # "checkpoints/brain/cart_1D/x8/best.pt"
  # "checkpoints/brain/cart_1D/x16/best.pt"
  "checkpoints/knee/cart_1D/x8/best.pt"
  # "checkpoints/knee/cart_1D/x16/best.pt"
)
ACCS=(8)
ORGANS=(knee)

# ── Per-organ data config ─────────────────────────────────────────────────────
get_data_config() {
  local organ=$1
  case $organ in
    brain)
      DATA_ROOT="fastmri/brain/singlecoil_emulated_all"
      # DATA_SPLIT="fastmri/splits/multicoil_brain_test.txt"
      DATA_SPLIT="fastmri/splits/suno_brain_test.txt"
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
# 1D Cartesian: budget = res*(1/acc - cf) lines; one step = one line.
get_as_steps() {
  local res=$1 acc=$2
  awk -v res="$res" -v acc="$acc" -v cf="$CF" \
    'BEGIN { x = res * (1/acc - cf); print (x == int(x)) ? int(x) : int(x) + 1 }'
}

# 2D VDS/Poisson: budget = n_target - n_acs, capped at 16 steps.
get_as_steps_2d() {
  local res=$1 acc=$2
  awk -v res="$res" -v acc="$acc" -v af="$ACS_FRAC" \
    'BEGIN {
      n_target = int(res*res/acc + 0.5)
      acs_half = int(res*af/2 + 0.5)
      n_acs    = (2*acs_half)^2
      budget   = n_target - n_acs
      if (budget < 1) budget = 1
      x = (budget < 16) ? budget : 16
      print x
    }'
}


# ── evaluation config ───────────────────────────────────────────────────
CF=0.04          # center fraction — used for 1D Cartesian AS budget
ACS_FRAC=0.025   # ACS square side as fraction of image size (must match --vds-acs-fraction in training)
GEN_STEPS=1
TOPK=0
NUM_CLASSES=8
BATCH_SIZE=4
GPT_MODEL="GPT-L"
MASK_TYPE="cart_1d"  # 1D Cartesian: cart_1d | equispaced    2D: vds2d | poisson

# Derive mask flags from MASK_TYPE
case $MASK_TYPE in
  cart_1d|equispaced)
    MASK_LABEL="cart_1d"
    INITIAL_MASK_ARGS=()
    AS_MASK_TYPE="cartesian"
    AS_MASK_EXTRA_ARGS=()
    ;;
  vds2d)
    MASK_LABEL="vds2d"
    INITIAL_MASK_ARGS=(--initial-mask-type vds2d --vds-decay 2.0 --vds-acs-fraction "$ACS_FRAC")
    AS_MASK_TYPE="2d"
    AS_MASK_EXTRA_ARGS=()
    ;;
  poisson)
    MASK_LABEL="poisson"
    INITIAL_MASK_ARGS=(--initial-mask-type poisson --poisson-min-dist-frac 0.03)
    AS_MASK_TYPE="poisson"
    AS_MASK_EXTRA_ARGS=(--poisson-active-min-dist 4.0)
    ;;
esac

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

# ── Main loop — launch all jobs in parallel, each on its own GPU ──────────────
PIDS=()
LABELS=()

for i in "${!CKPTS[@]}"; do
  CKPT="${CKPTS[$i]}"
  ACC="${ACCS[$i]}"
  ORGAN="${ORGANS[$i]}"
  GPU="${GPUS[$i]}"
  BASE_OUTDIR="$(dirname "$(dirname "$CKPT")")/eval"

  get_data_config "$ORGAN"
  AS_STEPS=$(get_as_steps "$IMG_SIZE" "$ACC")
  AS_STEPS_2D=$(get_as_steps_2d "$IMG_SIZE" "$ACC")

  if [[ "$CKPT" =~ best\.pt$ ]]; then
    V_PREFIX="cf_${CF}_acc_${ACC}_"
  else
    ITER=$(basename "$CKPT" .pt)
    V_PREFIX="cf_${CF}_acc_${ACC}_iter_${ITER}_"
  fi

  if [[ "$AS_MASK_TYPE" == "cartesian" ]]; then
    N_STEPS="$AS_STEPS"
  else
    N_STEPS="$AS_STEPS_2D"
  fi

  LABEL="${ORGAN}_x${ACC}"
  echo "  [$(date +%H:%M:%S)]  GPU ${GPU}  ${LABEL}  as_steps=${N_STEPS}"

  LOGFILE="${BASE_OUTDIR}/${V_PREFIX}val_${MASK_LABEL}_random_fullval.log"
  mkdir -p "$(dirname "$LOGFILE")"

  CUDA_VISIBLE_DEVICES="${GPU}" \
  run_variant "$BASE_OUTDIR" "$CKPT" "$ACC" "$ORGAN" "${V_PREFIX}val_${MASK_LABEL}_random_fullval" \
    "${INITIAL_MASK_ARGS[@]}" \
    >"${LOGFILE}" 2>&1 &

  PIDS+=($!)
  LABELS+=("${LABEL}")
done

echo ""
echo "Waiting for all jobs..."
FAILED=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "  ✓  ${LABELS[$i]}"
  else
    echo "  ✗  ${LABELS[$i]}" >&2
    FAILED=1
  fi
done
(( FAILED == 0 )) || exit 1
echo "All done."
