#!/usr/bin/env bash
# Evaluate a SENSE GPT on the suno split with per-coil DC (eval_sense_block.py): net vs oracle maps,
# with and without the oracle support applied to the reconstructions before scoring.
#   bash eval_sense.sh
#   GPT_CKPT=... MAPS_LIST="net" SUPPORT_LIST="none" bash eval_sense.sh
set -euo pipefail
export PYTHONPATH="$(pwd)"
PY=${PY:-/opt/miniconda3/envs/meditok/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}

GPT_CKPT=${GPT_CKPT:-results/brain_sense/cart_1d/run1_GPT-L/cf_0.04/acc_8_maps_net/checkpoints/best.pt}
CROP_SIZE=${CROP_SIZE:-256}
TOKEN_SIZE=${TOKEN_SIZE:-256}
if [ "$TOKEN_SIZE" = "256" ]; then
  ENCODER_CKPT=${ENCODER_CKPT:-/fast/lev/MediTok_official/outputs/ckpts/brain_mc_sense_cplxpair_dist_ep8/meditok_fastmri_best.pth}
  ENCODER_CONFIG=${ENCODER_CONFIG:-weights/meditok/config.json}
else
  ENCODER_CKPT=${ENCODER_CKPT:-weights/meditok/brain_mc_sense_cplxpair_res512_ep5_frozen.pth}
  ENCODER_CONFIG=${ENCODER_CONFIG:-weights/meditok/config_${TOKEN_SIZE}.json}
fi
SENS_CKPT=${SENS_CKPT:-results/sens_net/run1/best.pt}
ACC=${ACC:-8}
TAG=${TAG:-sensegpt}
PS_N=${PS_N:-8}
MAPS_LIST=${MAPS_LIST:-"net oracle"}
SUPPORT_LIST=${SUPPORT_LIST:-"none oracle"}
GT_REF=${GT_REF:-sense}                          # sense | rss | both
EXTRA=${EXTRA:-}

mkdir -p results/sense_block
for maps in $MAPS_LIST; do
  for sup in $SUPPORT_LIST; do
    out="results/sense_block/suno_acc${ACC}_${TAG}_${maps}"
    [ "$CROP_SIZE" != "256" ] && out="${out}_crop${CROP_SIZE}"
    [ "$sup" = "oracle" ] && out="${out}_support"
    $PY eval_sense_block.py --acceleration "$ACC" --gpt-ckpt "$GPT_CKPT" --encoder-ckpt "$ENCODER_CKPT" --encoder-config "$ENCODER_CONFIG" --token-size "$TOKEN_SIZE" --crop-size "$CROP_SIZE" \
      --sens-maps "$maps" --sens-ckpt "$SENS_CKPT" --eval-support "$sup" --ps-n-samples "$PS_N" --gt-ref "$GT_REF" \
      --output-dir "$out" $EXTRA 2>&1 | tee "$out.log"
  done
done
