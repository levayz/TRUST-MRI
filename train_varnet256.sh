#!/usr/bin/env bash
# E2E-VarNet baseline on the 256-grid SENSE setup (random x8, cf 0.04, 256 crop), from the fastMRI leaderboard weights.
#   bash train_varnet256.sh                                   # -> results/varnet256/run1/best.pt
#   RUN=smoke SPLIT=results/sense/splits/one_volume.txt DATA_ROOT=fastmri/brain/multicoil_val SENS_DIRS=<val dirs> MAX_STEPS=30 bash train_varnet256.sh
set -euo pipefail
export PYTHONPATH="$(pwd)"
PY=${PY:-/opt/miniconda3/envs/meditok/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
RUN=${RUN:-run1}
DATA_ROOT=${DATA_ROOT:-fastmri/brain/multicoil_train}
SPLIT=${SPLIT:-fastmri/splits/multicoil_brain_train.txt}
SENS_DIRS=${SENS_DIRS:-fastmri/brain/multicoil_train_sens_maps_gpu,fastmri/brain/multicoil_train_sens_maps}
VAL_SPLIT=${VAL_SPLIT:-fastmri/splits/suno_brain_test.txt}
EPOCHS=${EPOCHS:-4}; MAX_STEPS=${MAX_STEPS:-0}; BS=${BS:-4}; LR=${LR:-1e-4}; CF=${CF:-0.04}; ACC=${ACC:-8}
INIT=${INIT:-weights/varnet/brain_leaderboard_state_dict.pt}
WORKERS=${WORKERS:-12}
OUT=${OUT:-results/varnet256/$RUN}
EXTRA=${EXTRA:-}
mkdir -p "$OUT"
$PY train_varnet256.py --data-root "$DATA_ROOT" --data-split "$SPLIT" --sens-dirs "$SENS_DIRS" \
  --val-data-root fastmri/brain/multicoil_val --val-data-split "$VAL_SPLIT" \
  --val-sens-dirs fastmri/brain/multicoil_val_sens_maps_gpu,fastmri/brain/multicoil_val_sens_maps \
  --image-size 256 --center-fractions "$CF" --accelerations "$ACC" --mask-type cart_1d --init-ckpt "$INIT" \
  --epochs "$EPOCHS" --max-steps "$MAX_STEPS" --batch-size "$BS" --lr "$LR" --num-workers "$WORKERS" \
  --output-dir "$OUT" $EXTRA 2>&1 | grep -vE "Warning|warn" | tee -a "$OUT/stdout.log"
