#!/usr/bin/env bash
# Launch run21 (MaskGIT-style commitment fine-tune of run11) once run19's chained evaluation has exited and a GPU pair is free
# (< 2 GB used on both; preference order 7,5,0), then chain its plain-DC evaluation with the hard-commit arms on the first GPU of the pair.
#   setsid nohup scripts/chain_run21_launch.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.."
R19=results/brain_sense/cart_1d/run19_v21_latent_long/cf_0.04/acc_8_maps_net_tok512_v2.1asym_latent
LOG=results/brain_sense/cart_1d/run21_launch_chain.log
echo "[$(date '+%F %T')] waiting for run19's evaluation to exit" >> $LOG
until grep -q "evaluation exit" $R19/chain.log 2>/dev/null; do sleep 600; done
pair=""
until [ -n "$pair" ]; do
  free=""
  for g in 7 5 0; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    [ "$used" -lt 2000 ] && free="$free $g"
  done
  set -- $free
  if [ $# -ge 2 ]; then pair="$1,$2"; else sleep 300; fi
done
echo "[$(date '+%F %T')] launching run21 on GPUs $pair" >> $LOG
R11=results/brain_sense/cart_1d/run11_v21_latent/cf_0.04/acc_8_maps_net_tok512_v2.1asym_latent/checkpoints/best_final.pt
D=results/brain_sense/cart_1d/run21_v21_latent_commitgt/cf_0.04/acc_8_maps_net_tok512_v2.1asym_commitgt_latent
mkdir -p $D
CUDA_VISIBLE_DEVICES=$pair NPROC=2 BS=8 EPOCHS=2 LR=5e-5 TOKEN_SIZE=512 MODEL_VERSION=2.1 HEAD_IL=0 LATENT=1 COMMIT=gt COMMIT_KEEP="0.2 0.9" \
  GT_CACHE=fastmri/brain/multicoil_train_gt_cache_crop256_tok512 RUN=run21_v21_latent_commitgt WARM_START=$R11 MASTER_PORT=29601 \
  setsid nohup bash train_sense_commit.sh > $D/launcher.log 2>&1 &
sleep 5
g1=${pair%%,*}
exec scripts/chain_after_run.sh $D $g1 suno_acc8_sensegpt512_v21asym_latent_commitgt_final_dcacq_net_support --dc-path acq --commit-schedule 0.55 0.8 --commit-view codes --hybrid-radii 0.5
