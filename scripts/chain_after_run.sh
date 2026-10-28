#!/usr/bin/env bash
# Wait for a SENSE GPT training run to finish, freeze its best checkpoint as best_final.pt and run the full suno evaluation
# (8 posterior draws, both references, oracle support, net maps). Detach it right after launching a run so the GPUs never idle:
#   setsid nohup scripts/chain_after_run.sh <run_dir> <gpu> <eval_name> [extra eval_sense_block args] > /dev/null 2>&1 &
# Progress is appended to <run_dir>/chain.log; the evaluation log goes to results/sense_block/<eval_name>.log.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)
PY=${PY:-/opt/miniconda3/envs/meditok/bin/python}
R=$1; GPU=$2; NAME=$3; shift 3
EXP=${R#results/brain_sense/cart_1d/}
LOG=$R/chain.log
mkdir -p "$R"
echo "[$(date '+%F %T')] waiting for $EXP to finish (eval on GPU $GPU -> $NAME)" >> "$LOG"
until [ -f "$R/stdout.log" ] && grep -q "Done!" "$R/stdout.log"; do
  sleep 300
  if [ -f "$R/stdout.log" ] && ! pgrep -f -- "--exp-name $EXP" > /dev/null; then
    sleep 120   # a restart in progress?
    if ! pgrep -f -- "--exp-name $EXP" > /dev/null && ! grep -q "Done!" "$R/stdout.log"; then echo "[$(date '+%F %T')] no training process for $EXP and no Done!: giving up" >> "$LOG"; exit 1; fi
  fi
done
sleep 60
cp -n "$R/checkpoints/best.pt" "$R/checkpoints/best_final.pt"
OUT=results/sense_block/$NAME
echo "[$(date '+%F %T')] run finished; evaluating $R/checkpoints/best_final.pt on GPU $GPU -> $OUT" >> "$LOG"
CUDA_VISIBLE_DEVICES=$GPU $PY eval_sense_block.py --acceleration 8 --gpt-ckpt "$R/checkpoints/best_final.pt" \
  --encoder-ckpt weights/meditok/brain_mc_sense_cplxpair_res512_ep5_frozen.pth --encoder-config weights/meditok/config_512.json --token-size 512 \
  --sens-maps net --sens-ckpt results/sens_net/run1/best.pt --eval-support oracle --gt-ref both --ps-n-samples 8 --batch-size 2 \
  --output-dir "$OUT" "$@" > "$OUT.log" 2>&1
echo "[$(date '+%F %T')] evaluation exit $?" >> "$LOG"
sed 's/\x1b\[[0-9;]*m//g' "$OUT.log" | sed 's/^.*\] //' | grep -E "^(greedy_dc|latent_dc|latent_no_dc|ps_mean_dc|exp1_dc) " >> "$LOG"
