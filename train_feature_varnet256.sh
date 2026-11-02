#!/usr/bin/env bash
# Feature VarNet baseline (Giannakopoulos et al., Sci. Rep. 2024; fastMRI fastmri_examples/feature_varnet) trained from scratch
# on the 256-grid SENSE setup: our training set (multicoil_brain_train.txt), random x8, cf 0.04, 256 crop, SSIM loss on RSS.
# No released Feature VarNet weights exist, so unlike the E2E-VarNet baseline (fine-tuned from the leaderboard weights) this starts
# from scratch with the official recipe: 12 cascades, 32 feature chans, batch 1, AdamW lr 3e-4, warmup then cosine, grad clip 1.
# Checkpoint selection on results/feature_varnet/val_nosuno15.txt (disjoint from the suno test split).
#   CUDA_VISIBLE_DEVICES=4 bash train_feature_varnet256.sh                       # -> results/feature_varnet256/run1/best.pt
#   RUN=smoke MAX_STEPS=30 VAL_MAX_VOLUMES=1 bash train_feature_varnet256.sh
set -euo pipefail
export PYTHONPATH="$(pwd)"
PY=${PY:-/opt/miniconda3/envs/meditok/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
RUN=${RUN:-run1}
VARIANT=${VARIANT:-fi_varnet}
DATA_ROOT=${DATA_ROOT:-fastmri/brain/multicoil_train}
SPLIT=${SPLIT:-fastmri/splits/multicoil_brain_train.txt}
SENS_DIRS=${SENS_DIRS:-fastmri/brain/multicoil_train_sens_maps_gpu,fastmri/brain/multicoil_train_sens_maps}
VAL_SPLIT=${VAL_SPLIT:-results/feature_varnet/val_nosuno15.txt}
VAL_MAX_VOLUMES=${VAL_MAX_VOLUMES:-0}
EPOCHS=${EPOCHS:-4}; MAX_STEPS=${MAX_STEPS:-0}; BS=${BS:-1}; LR=${LR:-3e-4}; CF=${CF:-0.04}; ACC=${ACC:-8}
CHANS=${CHANS:-32}; CASCADES=${CASCADES:-12}; WARMUP=${WARMUP:-0.02}
WORKERS=${WORKERS:-12}
OUT=${OUT:-results/feature_varnet256/$RUN}
EXTRA=${EXTRA:-}
mkdir -p "$OUT"
$PY train_varnet256.py --model feature_varnet --fv-variant "$VARIANT" --num-cascades "$CASCADES" --chans "$CHANS" --sens-chans 8 \
  --optimizer adamw --lr "$LR" --warmup-frac "$WARMUP" --grad-clip 1.0 \
  --data-root "$DATA_ROOT" --data-split "$SPLIT" --sens-dirs "$SENS_DIRS" \
  --val-data-root fastmri/brain/multicoil_val --val-data-split "$VAL_SPLIT" --val-max-volumes "$VAL_MAX_VOLUMES" \
  --val-sens-dirs fastmri/brain/multicoil_val_sens_maps_gpu,fastmri/brain/multicoil_val_sens_maps \
  --image-size 256 --center-fractions "$CF" --accelerations "$ACC" --mask-type cart_1d \
  --epochs "$EPOCHS" --max-steps "$MAX_STEPS" --batch-size "$BS" --num-workers "$WORKERS" \
  --output-dir "$OUT" $EXTRA 2>&1 | grep -vE "Warning|warn" | tee -a "$OUT/stdout.log"
