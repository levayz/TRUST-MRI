#!/usr/bin/env bash
# Stage 1 of the SENSE pipeline: train the sensitivity-map U-Net on 256-grid undersampled brain k-space.
#   bash train_sens_net.sh                      # full run  -> results/sens_net/run1/best.pt
#   RUN=overfit1 SPLIT=results/sense/splits/one_volume.txt EPOCHS=5 bash train_sens_net.sh
set -euo pipefail
export PYTHONPATH="$(pwd)"
PY=${PY:-/opt/miniconda3/envs/meditok/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}

RUN=${RUN:-run1}
SPLIT=${SPLIT:-fastmri/splits/multicoil_brain_train.txt}
VAL_SPLIT=${VAL_SPLIT:-fastmri/splits/suno_brain_test.txt}
EPOCHS=${EPOCHS:-10}
MAX_STEPS=${MAX_STEPS:-0}
BS=${BS:-8}
LR=${LR:-3e-4}
CF=${CF:-0.04}
ACC=${ACC:-8}
INIT=${INIT:-weights/varnet/brain_leaderboard_state_dict.pt}   # none = from scratch
CHANS=${CHANS:-8}
POOLS=${POOLS:-4}
TAU=${TAU:-0.05}
LAMBDA_MAG=${LAMBDA_MAG:-1.0}
LAMBDA_CPLX=${LAMBDA_CPLX:-0.0}
WORKERS=${WORKERS:-12}
OUT=${OUT:-results/sens_net/$RUN}
EXTRA=${EXTRA:-}

mkdir -p "$OUT"
$PY train_sens_net.py \
  --data-root fastmri/brain/multicoil_train --data-split "$SPLIT" \
  --sens-dirs fastmri/brain/multicoil_train_sens_maps_gpu,fastmri/brain/multicoil_train_sens_maps \
  --val-data-root fastmri/brain/multicoil_val --val-data-split "$VAL_SPLIT" \
  --val-sens-dirs fastmri/brain/multicoil_val_sens_maps_gpu,fastmri/brain/multicoil_val_sens_maps \
  --image-size 256 --center-fractions "$CF" --accelerations "$ACC" --mask-type cart_1d \
  --chans "$CHANS" --num-pools "$POOLS" --init-ckpt "$INIT" \
  --lambda-mag "$LAMBDA_MAG" --lambda-cplx "$LAMBDA_CPLX" --gauge-tau "$TAU" \
  --epochs "$EPOCHS" --max-steps "$MAX_STEPS" --batch-size "$BS" --lr "$LR" --schedule cosine \
  --mixed-precision bf16 --num-workers "$WORKERS" --val-every 1 \
  --output-dir "$OUT" $EXTRA 2>&1 | tee -a "$OUT/stdout.log"
