#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)"

# ── GPU / port ────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="1,2"
MASTER_PORT=29505
# ── Dataset: choose "knee" or "brain" ────────────────────────────────────────
ORGAN="knee"

case $ORGAN in
  knee)
    DATASET="fastmri/knee/singlecoil_train"
    CUSTOM_SPLIT="none"
    VAL_DATASET="fastmri/knee/singlecoil_val"
    CUSTOM_VAL_SPLIT="fastmri/splits/adasense_test_split.txt"
    RESULT_DIR="results/knee"
    IMG_SIZE=320
    COND_TOKENS=400
    END_SLICE_SKIP=0
    ;;
  brain)
    DATASET="fastmri/brain/singlecoil_emulated_all"
    CUSTOM_SPLIT="fastmri/splits/multicoil_brain_train.txt"
    VAL_DATASET="fastmri/brain/singlecoil_emulated_all"
    CUSTOM_VAL_SPLIT="fastmri/splits/suno_brain_test.txt"
    RESULT_DIR="results/brain"
    IMG_SIZE=256
    COND_TOKENS=256
    END_SLICE_SKIP=0
    ;;
esac

# ── Model / training config ───────────────────────────────────────────────────
GPT_MODEL="GPT-L"
BATCH_SIZE=8
CF=0.04          # center fraction of k-space always acquired (ignored for vds2d/poisson)
ACC=400       # one or more acceleration factors; mask func picks randomly per sample
MASK_TYPE="vds2d"  # 1D Cartesian: cart_1d | equispaced    2D: vds2d | poisson

# Map mask type to a short display label for directory names
case $MASK_TYPE in
  cart_1d|equispaced) MASK_LABEL="cart_1d" ;;
  *)                  MASK_LABEL="$MASK_TYPE" ;;
esac
RESULT_DIR="${RESULT_DIR}/${MASK_LABEL}"

torchrun --nproc_per_node=2 --master_port=$MASTER_PORT \
  autoregressive/train/train.py \
  --exp-name "run1_${GPT_MODEL}/cf_${CF}/acc_${ACC// /_}" \
  --results-dir "$RESULT_DIR" \
  --dataset mri_code \
  --data-root "$DATASET" \
  --data-split "$CUSTOM_SPLIT" \
  --val-data-root "$VAL_DATASET" \
  --val-data-split "$CUSTOM_VAL_SPLIT" \
  --encoder-ckpt weights/meditok/meditok_simple_v1.pth \
  --encoder-config weights/meditok/config.json \
  --gpt-model "$GPT_MODEL" \
  --gpt-type c2i \
  --num-classes 8 \
  --num-masks 1 \
  --num-codebooks 8 \
  --vocab-size 32768 \
  --image-size "$IMG_SIZE" \
  --downsample-size 16 \
  --center-fractions ${CF} \
  --accelerations ${ACC} \
  --mask-type "$MASK_TYPE" \
  --far-mask-mode radial \
  --complex-mode cartesian \
  --global-batch-size "$BATCH_SIZE" \
  --num-workers 4 \
  --lr 1e-4 \
  --epochs 200 \
  --log-every 100 \
  --ckpt-every 1000 \
  --viz-every 100 \
  --val-every 1 \
  --cls-token-num 1 \
  --cond-token-num "$COND_TOKENS" \
  --end-slice-skip "$END_SLICE_SKIP" \
  --encode-gt \
  --no-compile \
  --val-save-viz \
  --val-n-scans-to-save 2
