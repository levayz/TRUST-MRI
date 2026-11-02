#!/usr/bin/env bash
# Feature VarNet baseline end to end: wait until the target GPU is free (autoresearch/gpu_wait.sh), train on our training set
# (train_feature_varnet256.sh), then reconstruct the suno split on the 256 grid with best.pt and score it (both references).
# Detach it:  GPU=4 RUN=run1 setsid nohup bash run_feature_varnet_pipeline.sh > /dev/null 2>&1 &
# Progress: $RES/feature_varnet256/$RUN/pipeline.log; scores: results/sense_block/suno_acc8_featvarnet256_$RUN/summary.json
set -uo pipefail
cd "$(dirname "$0")"
GPU=${GPU:?set GPU}; RUN=${RUN:-run1}
RES=${RES:-/fast/lev/TRUST-MRI/results}   # outputs go to the main checkout (results/ is gitignored; the worktree may be removed)
R=$RES/feature_varnet256/$RUN; TAG=suno_acc8_featvarnet256_$RUN
mkdir -p "$R"; LOG=$R/pipeline.log
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "waiting for GPU $GPU (autoresearch/gpu_wait.sh: asks gpu_hold.py to release it, then < 2 GB used by anything but the holder)"
# a plain nvidia-smi memory check never passes: our holder grabs the freed memory within seconds
AR_WAIT_MINUTES=${WAIT_MINUTES:-4320} bash autoresearch/gpu_wait.sh "$GPU" >> "$LOG" 2>&1 || { log "GPU $GPU not free after the wait: giving up"; exit 3; }
log "GPU $GPU free; training $R"
CUDA_VISIBLE_DEVICES=$GPU RUN=$RUN OUT=$R bash train_feature_varnet256.sh > /dev/null 2>&1
log "training exit $?; $(grep -h 'done. best' "$R/stdout.log" | tail -1)"
[ -f "$R/best.pt" ] || { log "no best.pt: giving up"; exit 1; }
cp -n "$R/best.pt" "$R/best_final.pt"
RD=$RES/varnet_baseline/$TAG
log "evaluating $R/best_final.pt on suno -> $RD"
HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH= /fast/lev/miniconda3/envs/varnet/bin/python scripts/eval_varnet_baseline.py \
  --ckpt "$R/best_final.pt" --acceleration 8 --center-fractions 0.04 --mask-type random --grids grid256 --out "$RD" > "$RD.log" 2>&1
log "eval exit $?"
PYTHONPATH=$(pwd) /opt/miniconda3/envs/meditok/bin/python scripts/score_varnet_baseline.py --recon-dir "$RD" --out-dir "$RES/sense_block/$TAG" >> "$RD.log" 2>&1
log "score exit $?"
grep -E "grid256_(rss|sense) " "$RD.log" >> "$LOG"
