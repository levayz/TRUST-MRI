#!/usr/bin/env bash
# E2E-VarNet baselines on the suno split under the SENSE-block conventions (our seeded random x8 mask, 256 crop,
# scored vs the oracle SENSE image and vs RSS). Reconstruction runs in the fastMRI env, scoring in the meditok env.
#   bash eval_varnet.sh                      # both checkpoints on GPU 5
#   CKPTS="path.ckpt:tag" bash eval_varnet.sh
set -euo pipefail
export PYTHONPATH="$(pwd)"
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
PY_VARNET=${PY_VARNET:-/fast/lev/miniconda3/envs/varnet/bin/python}
PY=${PY:-/opt/miniconda3/envs/meditok/bin/python}
ACC=${ACC:-8}
CKPTS=${CKPTS:-"/fast/lev/fastMRI/logs/varnet/varnet_brain_r8/checkpoints/best_ssim.ckpt:varnet_r8_bestssim weights/varnet/brain_leaderboard_state_dict.pt:varnet_leaderboard"}
EXTRA=${EXTRA:-}
for item in $CKPTS; do
  ckpt=${item%%:*}; tag=${item##*:}
  rd=results/varnet_baseline/suno_acc${ACC}_${tag}
  PYTHONPATH= $PY_VARNET scripts/eval_varnet_baseline.py --ckpt "$ckpt" --acceleration "$ACC" --out "$rd" $EXTRA 2>&1 | grep -vE "Warning|warn|pkg_resources" | tee "$rd.log"
  $PY scripts/score_varnet_baseline.py --recon-dir "$rd" --out-dir "results/sense_block/suno_acc${ACC}_${tag}" 2>&1 | grep -vE "Warning|warn" | tee -a "$rd.log"
done
