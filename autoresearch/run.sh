#!/usr/bin/env bash
# One experiment: train.py on the GPU(s) of AR_GPU under the frozen budget. Output goes to stdout (redirect it to run.log).
#   AR_GPU=0,2 bash run.sh > run.log 2>&1 ; grep '^val_psnr:' run.log
# Several GPUs (comma-separated) = one process per GPU through torchrun, the global batch split over them (data parallel).
# gpu_wait.sh first asks gpu_hold.py (if running) to free each GPU and waits while any process holds it, so a queued run
# never collides with a job.
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(cd .. && pwd)"
export CUDA_VISIBLE_DEVICES=${AR_GPU:-0,2}
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}     # NCCL peer-to-peer hangs between some GPU pairs on this box (as in train_sense.sh)
PY=${PY:-/opt/miniconda3/envs/meditok/bin/python}
MAX_MINUTES=${AR_MAX_MINUTES:-75}

IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
for pass in 1 2; do                          # the second pass renews every holder's grace period right before the launch
    for g in "${GPUS[@]}"; do bash gpu_wait.sh "$g" || exit 3; done
done
echo "$(date '+%F %T') starting on GPU(s) $CUDA_VISIBLE_DEVICES"
# hard kill at the training cap + 25 min (startup, evaluation); the summary block is then missing = crash
if [ "${#GPUS[@]}" -gt 1 ]; then
    exec timeout --signal=TERM --kill-after=60 $(( (MAX_MINUTES + 25) * 60 )) "$PY" -m torch.distributed.run \
        --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nproc_per_node="${#GPUS[@]}" train.py "$@"
else
    exec timeout --signal=KILL $(( (MAX_MINUTES + 25) * 60 )) "$PY" -u train.py "$@"
fi
