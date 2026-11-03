#!/usr/bin/env bash
# CVPR baseline queue (Nov 3 2026), all on one GPU, after the Feature VarNet x8 pipeline (results/feature_varnet256/run1) has scored:
#   A  (optional, RESELECT_E2E=1) E2E-VarNet x8, re-selected: leaderboard init, 4 epochs, model selection on val_nosuno15 (run1 selected on suno = test)
#   B  E2E-VarNet x16: init from the E2E x8 run1 best.pt (or A's), 4 epochs
#   C  Feature VarNet x16: init from the Feature VarNet x8 best_final.pt, 2 epochs, AdamW lr 1e-4
# Every model: random 1D masks, cf 0.04 (10 ACS lines at 256), selection on results/feature_varnet/val_nosuno15.txt (disjoint
# from suno and from the training list), then one suno reconstruction at its training acceleration + scoring (both references).
# Detach:  GPU=4 setsid nohup bash run_baseline_queue.sh > /dev/null 2>&1 &      Progress: $RES/baseline_queue.log
set -uo pipefail
cd "$(dirname "$0")"
GPU=${GPU:?set GPU}
RES=${RES:-/fast/lev/TRUST-MRI/results}
LOG=$RES/baseline_queue.log
VAL=results/feature_varnet/val_nosuno15.txt
PY=/opt/miniconda3/envs/meditok/bin/python
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

gpu_mb() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU"; }
wait_gpu() {   # GPU 4 has no holder: wait for < 2 GB, but after 30 min accept sharing if >= 24 GB are free
  local t=0
  until [ "$(gpu_mb)" -lt 2000 ]; do
    [ $t -ge 1800 ] && [ "$(gpu_mb)" -lt 25000 ] && { log "GPU $GPU shared ($(gpu_mb) MiB used), starting anyway"; return; }
    sleep 60; t=$((t + 60))
  done
}

train() {   # $1 out dir, rest: train_varnet256.py args
  local out=$1; shift
  mkdir -p "$out"; wait_gpu
  log "training -> $out"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$(pwd) $PY train_varnet256.py --data-root fastmri/brain/multicoil_train \
    --data-split fastmri/splits/multicoil_brain_train.txt --val-data-split "$VAL" --image-size 256 --center-fractions 0.04 \
    --mask-type cart_1d --num-workers 12 --output-dir "$out" "$@" 2>&1 | grep -vE "Warning|warn" >> "$out/stdout.log"
  log "training exit ${PIPESTATUS[0]}: $(sed 's/\x1b\[[0-9;]*m//g' "$out/stdout.log" | grep -h 'done. best' | tail -1 | sed 's/^.*\] //')"
  [ -f "$out/best.pt" ] && cp -n "$out/best.pt" "$out/best_final.pt"
}

evaluate() {   # $1 checkpoint, $2 acceleration, $3 tag
  local ck=$1 acc=$2 tag=$3 rd=$RES/varnet_baseline/$3
  [ -f "$ck" ] || { log "no $ck: skipping eval $tag"; return 1; }
  log "evaluating $ck at x$acc -> $tag"
  HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH= /fast/lev/miniconda3/envs/varnet/bin/python scripts/eval_varnet_baseline.py \
    --ckpt "$ck" --acceleration "$acc" --center-fractions 0.04 --mask-type random --grids grid256 --out "$rd" > "$rd.log" 2>&1
  PYTHONPATH=$(pwd) $PY scripts/score_varnet_baseline.py --recon-dir "$rd" --out-dir "$RES/sense_block/$tag" >> "$rd.log" 2>&1
  log "scored $tag:"; grep -E "grid256_(rss|sense) " "$rd.log" >> "$LOG"
}

log "queue start: waiting for the Feature VarNet x8 pipeline to score"
until grep -qE "score exit|giving up" "$RES/feature_varnet256/run1/pipeline.log"; do sleep 300; done
log "Feature VarNet x8 done"

# A is off by default (Nov 3: selecting on suno is accepted for the paper, so E2E x8 run1 stands); RESELECT_E2E=1 turns it on.
if [ "${RESELECT_E2E:-0}" = 1 ]; then
  A=$RES/varnet256/run2_valnosuno; E2E_X8=$A/best_final.pt
  train "$A" --model e2e --init-ckpt weights/varnet/brain_leaderboard_state_dict.pt --accelerations 8 --epochs 4 --batch-size 4 --lr 1e-4
  evaluate "$E2E_X8" 8 suno_acc8_varnet256_run2_valnosuno
else
  E2E_X8=$RES/varnet256/run1/best.pt
  log "E2E x8 re-selection skipped; E2E x16 starts from $E2E_X8"
fi

B=$RES/varnet256_x16/run1
if [ -f "$E2E_X8" ]; then
  train "$B" --model e2e --init-ckpt "$E2E_X8" --accelerations 16 --epochs 4 --batch-size 4 --lr 1e-4
  evaluate "$B/best_final.pt" 16 suno_acc16_varnet256_x16_run1
else
  log "no E2E x8 checkpoint: skipping E2E x16"
fi

C=$RES/feature_varnet256_x16/run1
train "$C" --model feature_varnet --fv-init-ckpt "$RES/feature_varnet256/run1/best_final.pt" --accelerations 16 --epochs 2 \
  --batch-size 1 --optimizer adamw --lr 1e-4 --grad-clip 1.0
evaluate "$C/best_final.pt" 16 suno_acc16_featvarnet256_x16_run1
log "queue done"
