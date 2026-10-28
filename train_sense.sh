#!/usr/bin/env bash
# Stage 2 of the SENSE pipeline: train GPT-L to predict the clean SENSE tokens (cplxpair SENSE tokenizer) from the
# zero-filled SENSE image built with the stage-1 sensitivity net. No data consistency in training; per-coil DC
# at validation / inference (models/sense_ops.make_sense_dc_fn through reconstruct_far's dc_fn hook).
#   bash train_sense.sh                       # full run (2 GPUs, global batch 16, 10 epochs), warm start from the single-coil brain x8 GPT
#   CUDA_VISIBLE_DEVICES=2 NPROC=1 BS=8 OVERFIT=1 EPOCHS=2 bash train_sense.sh   # one-volume overfit smoke test
#   SENS_MAPS=oracle bash train_sense.sh      # map-error-free upper bound
#   WARM_START="" bash train_sense.sh         # from scratch
#   TOKEN_SIZE=512 BS=8 bash train_sense.sh   # 512-token grid with the res512 SENSE tokenizer (4x tokens: slower, more memory)
#   REFINE=1 TOKEN_SIZE=512 WARM_START=<512 best> bash train_sense.sh   # teacher-student refinement fine-tune
set -euo pipefail
export PYTHONPATH="$(pwd)"
PY=${PY:-/opt/miniconda3/envs/meditok/bin/python}      # the env with piq (autoregressive/utils/metrics.py)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,2}   # GPT-L needs ~17 GB per GPU at 8 slices/GPU
MASTER_PORT=${MASTER_PORT:-29531}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}   # less fragmentation on shared GPUs
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}      # NCCL P2P hangs between some GPU pairs on this box (e.g. 2,0): allreduce never completes
NPROC=${NPROC:-2}                                # ranks = GPUs listed above; global batch = BS

RUN=${RUN:-run1_GPT-L}
SENS_MAPS=${SENS_MAPS:-net}                      # net | oracle
SENS_CKPT=${SENS_CKPT:-results/sens_net/run1/best.pt}
GT_GAUGE=${GT_GAUGE:-oracle}                     # oracle | net
CROP_SIZE=${CROP_SIZE:-256}                      # acquisition grid (SENSE image size); 320 = the fastMRI/VarNet native phase-encode grid (13 ACS lines)
TOKEN_SIZE=${TOKEN_SIZE:-256}                    # tokenizer / GPT grid; 512 = bicubic-upsample the SENSE image (res512 SENSE tokenizer)
if [ "$TOKEN_SIZE" = "256" ]; then
  ENCODER_CKPT=${ENCODER_CKPT:-/fast/lev/MediTok_official/outputs/ckpts/brain_mc_sense_cplxpair_dist_ep8/meditok_fastmri_best.pth}
  ENCODER_CONFIG=${ENCODER_CONFIG:-weights/meditok/config.json}
else
  ENCODER_CKPT=${ENCODER_CKPT:-weights/meditok/brain_mc_sense_cplxpair_res512_ep5_frozen.pth}   # frozen copy of MediTok_official brain_mc_sense_cplxpair_res512_ep8 (epoch 5)
  ENCODER_CONFIG=${ENCODER_CONFIG:-weights/meditok/config_${TOKEN_SIZE}.json}
fi
COND_TOKENS=$(( (TOKEN_SIZE / 16) * (TOKEN_SIZE / 16) ))
WARM_START=${WARM_START-checkpoints/brain/cart_1D/x8/best.pt}   # "" = scratch
GPT_MODEL=${GPT_MODEL:-GPT-L}
EPOCHS=${EPOCHS:-10}                             # ~7k steps/epoch at global batch 8 (1.1 it/s/GPU)
BS=${BS:-16}                                     # global batch (split over NPROC ranks)
LR=${LR:-1e-4}
CF=${CF:-0.04}
ACC=${ACC:-8}
P_EMPTY=${P_EMPTY:-0.5}
WORKERS=${WORKERS:-8}
VAL_BS=${VAL_BS:-8}
OVERFIT=${OVERFIT:-0}
REFINE=${REFINE:-0}                              # 1 = teacher-student refinement (EMA teacher drafts -> DC -> refine pass at timestep 1)
LATENT=${LATENT:-0}                              # 1 = add a latent-regression head (CE + LAMBDA_LATENT * L2 on unit-norm latent chunks); validation on the latent path
LAMBDA_LATENT=${LAMBDA_LATENT:-5.0}
LATENT_HEAD=${LATENT_HEAD:-linear}               # latent head: linear | mlp (+ residual MLP on the hidden state) | fused (+ MLP on hidden state + token head's rollout)
EXTRA=${EXTRA:-}
REFINE_DRAFT=${REFINE_DRAFT:-sample}               # teacher draft: sample | greedy | mixed (per step)
DRAFTVIEW=${DRAFTVIEW:-0}                        # 1 = two-view refinement (--draft-cond): refine pass sees zero-filled tokens + draft tokens
MODEL_VERSION=${MODEL_VERSION:-1}                # 1 = legacy GPT; 2 = + timestep & sampling-mask conditioning; 2.1 = v2 + continuous input + interleaved Re/Im head
if [ "$REFINE" = "1" ]; then P_EMPTY=1.0; VAL_BS=${VAL_BS_REFINE:-2}; EXTRA="$EXTRA --refine-rounds 1 --refine-draft ${REFINE_DRAFT:-sample} --refine-teacher ema --ema --ema-decay 0.999"; [ "$DRAFTVIEW" = "1" ] && EXTRA="$EXTRA --draft-cond"; [ "$MODEL_VERSION" = "1" ] && MODEL_VERSION=2; fi   # refinement needs a visible timestep (v2+)
HEAD_IL=${HEAD_IL:-1}                              # v2.1 head: 1 = interleaved Re/Im chain, 0 = parallel asymmetric head
EXTRA="$EXTRA --model-version $MODEL_VERSION --head-interleaved $HEAD_IL"   # two-pass validation: small val batch

TRAIN_ROOT=fastmri/brain/multicoil_train
TRAIN_SPLIT=fastmri/splits/multicoil_brain_train.txt
SENS_DIRS=fastmri/brain/multicoil_train_sens_maps_gpu,fastmri/brain/multicoil_train_sens_maps
VAL_ROOT=fastmri/brain/multicoil_val
VAL_SPLIT=fastmri/splits/suno_brain_test.txt
VAL_SENS_DIRS=fastmri/brain/multicoil_val_sens_maps_gpu,fastmri/brain/multicoil_val_sens_maps
RESULT_DIR=results/brain_sense/cart_1d
EXP="${RUN}/cf_${CF}/acc_${ACC}_maps_${SENS_MAPS}"
if [ "$GT_GAUGE" != "oracle" ]; then EXP="${EXP}_gauge_${GT_GAUGE}"; fi
if [ "$TOKEN_SIZE" != "256" ]; then EXP="${EXP}_tok${TOKEN_SIZE}"; fi
if [ "$MODEL_VERSION" != "1" ]; then EXP="${EXP}_v${MODEL_VERSION}"; [ "$MODEL_VERSION" = "2.1" ] && [ "$HEAD_IL" = "0" ] && EXP="${EXP}asym"; fi
if [ "$REFINE" = "1" ]; then EXP="${EXP}_refine"; [ "$DRAFTVIEW" = "1" ] && EXP="${EXP}_2view"; fi
if [ "$LATENT" = "1" ]; then EXP="${EXP}_latent"; [ "$LATENT_HEAD" != "linear" ] && EXP="${EXP}${LATENT_HEAD}"; EXTRA="$EXTRA --latent-dim 64 --lambda-latent $LAMBDA_LATENT --val-rec-mode latent --latent-head-type $LATENT_HEAD"; fi
if [ "$CROP_SIZE" != "256" ]; then EXP="${EXP}_crop${CROP_SIZE}"; fi
if [ "$OVERFIT" = "1" ]; then
  TRAIN_ROOT=$VAL_ROOT; SENS_DIRS=$VAL_SENS_DIRS
  TRAIN_SPLIT=results/sense/splits/one_volume.txt; VAL_SPLIT=$TRAIN_SPLIT
  case "$EXTRA" in *overfit-n-samples*) EXTRA="$EXTRA --ckpt-every 100000";; *) EXTRA="$EXTRA --overfit-n-samples 1600 --ckpt-every 100000";; esac
  EXP="${EXP}_overfit"
fi

# warm start only when the experiment has no checkpoint yet (otherwise train.py auto-resumes from it)
CKPT_ARGS=""
if [ -n "$WARM_START" ] && ! ls "$RESULT_DIR/$EXP/checkpoints/"*.pt >/dev/null 2>&1; then
  CKPT_ARGS="--gpt-ckpt $WARM_START --load-weights-only"
fi
mkdir -p "$RESULT_DIR/$EXP"
echo "experiment: $RESULT_DIR/$EXP   maps=$SENS_MAPS   warm start: ${CKPT_ARGS:-none (resume/scratch)}"

$PY -m torch.distributed.run --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
  autoregressive/train/train.py \
  --exp-name "$EXP" \
  --results-dir "$RESULT_DIR" \
  --dataset sense \
  --data-root "$TRAIN_ROOT" --data-split "$TRAIN_SPLIT" --sens-dirs "$SENS_DIRS" \
  --val-data-root "$VAL_ROOT" --val-data-split "$VAL_SPLIT" --val-sens-dirs "$VAL_SENS_DIRS" \
  --sens-maps "$SENS_MAPS" --sens-ckpt "$SENS_CKPT" --gt-gauge "$GT_GAUGE" --max-coils 20 \
  --encoder-ckpt "$ENCODER_CKPT" \
  --encoder-config "$ENCODER_CONFIG" \
  --gpt-model "$GPT_MODEL" --gpt-type c2i \
  --num-classes 8 --num-masks 1 --num-codebooks 8 --vocab-size 32768 \
  --image-size "$TOKEN_SIZE" --downsample-size 16 --sense-crop-size "$CROP_SIZE" --sense-resize-mode bicubic \
  --center-fractions "$CF" --accelerations "$ACC" --mask-type cart_1d \
  --far-mask-mode radial --complex-mode cartesian --p-empty "$P_EMPTY" \
  --global-batch-size "$BS" --num-workers "$WORKERS" \
  --lr "$LR" --schedule cosine --epochs "$EPOCHS" \
  --log-every 100 --ckpt-every 1000 --viz-every 500 \
  --val-every 1 --val-batch-size "$VAL_BS" --val-save-viz --val-n-scans-to-save 4 \
  --cls-token-num 1 --cond-token-num "$COND_TOKENS" \
  --encode-gt --no-compile \
  $CKPT_ARGS $EXTRA 2>&1 | tee -a "$RESULT_DIR/$EXP/stdout.log"
