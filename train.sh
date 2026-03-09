#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)"

# ── GPU / port ────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="0,1"
MASTER_PORT=29503

# ── Dataset: choose "knee" or "brain" ────────────────────────────────────────
ORGAN="knee"

case $ORGAN in
  knee)
    DATASET="fastmri/knee/singlecoil_train"
    CUSTOM_SPLIT="none"
    VAL_DATASET="fastmri/knee/singlecoil_val"
    CUSTOM_VAL_SPLIT="fastmri/splits/adasense_test_split.txt"
    RESULT_DIR="results/kfar_knee"
    IMG_SIZE=320
    COND_TOKENS=400
    END_SLICE_SKIP=0
    ;;
  brain)
    DATASET="fastmri/brain/singlecoil_emulated_all"
    CUSTOM_SPLIT="fastmri/splits/multicoil_brain_train.txt"
    VAL_DATASET="fastmri/brain/singlecoil_emulated_all"
    CUSTOM_VAL_SPLIT="fastmri/splits/suno_brain_test.txt"
    RESULT_DIR="results/kfar_brain"
    IMG_SIZE=256
    COND_TOKENS=256
    END_SLICE_SKIP=0
    ;;
esac

# ── Model / training config ───────────────────────────────────────────────────
GPT_MODEL="GPT-L"
BATCH_SIZE=32
CF=0.04          # center fraction of k-space always acquired
ACC=8            # undersampling acceleration factor

torchrun --nproc_per_node=2 --master_port=$MASTER_PORT \
  autoregressive/train/train.py \
  --exp-name "run1_${GPT_MODEL}/cf_${CF}/acc_${ACC}" \
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
  --center-fractions "$CF" \
  --accelerations "$ACC" \
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
