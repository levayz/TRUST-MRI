#!/usr/bin/env bash
# Calibration of the autoresearch proxy on one GPU (idempotent: finished steps are skipped on a rerun):
#   0. conditioning cache (build_cond_cache.py) if incomplete, then a check of 16 slices against the slow path
#   1. harness check: run19 best.pt on the suno split through model.py + sample.py + prepare.evaluate (must match its log)
#   2. timing: 120 steps of GPT-B and of GPT-L without evaluation -> steps/s, VRAM -> SAMPLE_BUDGET for ~25 min of training
#   3. noise floor: the baseline with 3 seeds
#   4. transfer tests: changes with known full-scale rankings (latent head off, interleaved head, image loss on)
# Output: calib/*.log, calib/budget.txt, calib/table.txt. Run detached: nohup bash calibrate.sh > calib/calibrate.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
export AR_GPU=${AR_GPU:-1}
PY=/opt/miniconda3/envs/meditok/bin/python
OUT=calib; mkdir -p $OUT
RUN19=../results/brain_sense/cart_1d/run19_v21_latent_long/cf_0.04/acc_8_maps_net_tok512_v2.1asym_latent
BEST=$RUN19/checkpoints/best.pt
COND_CACHE=$(PYTHONPATH=.. $PY -c "import prepare; print(prepare.COND_CACHE_DIR)")
run_env() { PYTHONPATH=.. CUDA_VISIBLE_DEVICES=$AR_GPU HDF5_USE_FILE_LOCKING=FALSE "$@"; }

echo "$(date '+%F %T') == 0. conditioning cache"
if [ ! -e "$COND_CACHE/_complete" ]; then
    bash gpu_wait.sh $AR_GPU || exit 3
    run_env $PY -u build_cond_cache.py > $OUT/build_cond_cache.log 2>&1 || { echo "cache build failed"; tail -20 $OUT/build_cond_cache.log; exit 1; }
fi
cat "$COND_CACHE/_complete"
if ! grep -q "^cache check PASSED" $OUT/verify_cache.log 2>/dev/null; then
    bash gpu_wait.sh $AR_GPU || exit 3
    run_env $PY -u build_cond_cache.py --verify 16 > $OUT/verify_cache.log 2>&1
fi
grep "^verified" $OUT/verify_cache.log; grep -q "^cache check PASSED" $OUT/verify_cache.log || { echo "cache check failed"; tail -5 $OUT/verify_cache.log; exit 1; }

echo "$(date '+%F %T') == 1. harness check"
if ! grep -q "^split" $OUT/check_harness.log 2>/dev/null; then
    bash gpu_wait.sh $AR_GPU || exit 3
    sed -E 's/\x1b\[[0-9;]*m//g' $RUN19/stdout.log | grep -E "New best model|alt_rec: PSNR" | tail -2 > $OUT/expected_run19.txt
    run_env $PY -u check_harness.py --ckpt $BEST --split suno > $OUT/check_harness.log 2>&1
fi
grep -E "^split|^loaded" $OUT/check_harness.log; cat $OUT/expected_run19.txt | cut -c1-160

echo "$(date '+%F %T') == 2. timing"
timed() { grep -q "^peak_vram_mb" $OUT/$1.log 2>/dev/null && grep -q "cached_inputs True" $OUT/$1.log; }
timed timing_gptb || AR_TAG=timing_gptb AR_SAMPLE_BUDGET=960 AR_SKIP_EVAL=1 AR_MAX_MINUTES=30 bash run.sh > $OUT/timing_gptb.log 2>&1
timed timing_gptl || AR_TAG=timing_gptl AR_DEPTH=24 AR_DIM=1024 AR_NHEAD=16 AR_SAMPLE_BUDGET=960 AR_SKIP_EVAL=1 AR_MAX_MINUTES=30 bash run.sh > $OUT/timing_gptl.log 2>&1
for f in timing_gptb timing_gptl; do echo "$f: $(grep -oE '[0-9.]+ steps/s' $OUT/$f.log | tail -1)  $(grep -E '^peak_vram_mb' $OUT/$f.log)"; done
rate=$(grep -oE "[0-9.]+ steps/s" $OUT/timing_gptb.log | tail -1 | awk '{print $1}')
if [ -z "$rate" ]; then echo "GPT-B timing run failed, stopping"; tail -30 $OUT/timing_gptb.log; exit 1; fi
budget=$($PY -c "r=float('$rate'); b=int(round(25*60*r*8/1000))*1000; print(max(8000, min(48000, b)))")
echo "GPT-B $rate steps/s at batch 8 -> SAMPLE_BUDGET $budget (about 25 min of training)" | tee $OUT/budget.txt
export AR_SAMPLE_BUDGET=$budget AR_MAX_MINUTES=60

# a run counts as done when its log has a finite val_psnr at the current budget
done_run() { grep -qE "^val_psnr: +[0-9]" $OUT/$1.log 2>/dev/null && grep -q "budget $AR_SAMPLE_BUDGET samples" $OUT/$1.log; }
echo "$(date '+%F %T') == 3. noise floor"
for s in 0 1 2; do done_run base_s$s || AR_TAG=base_s$s AR_SEED=$s bash run.sh > $OUT/base_s$s.log 2>&1; echo "base_s$s: $(grep -E '^val_psnr' $OUT/base_s$s.log)"; done

echo "$(date '+%F %T') == 4. transfer tests"
done_run nolatent    || AR_TAG=nolatent AR_LAMBDA_LATENT=0 bash run.sh > $OUT/nolatent.log 2>&1
done_run interleaved || AR_TAG=interleaved AR_HEAD_INTERLEAVED=1 bash run.sh > $OUT/interleaved.log 2>&1
done_run imgloss     || AR_TAG=imgloss AR_LAMBDA_IMG=1.0 bash run.sh > $OUT/imgloss.log 2>&1

{ printf "%-12s %-10s %-9s %-10s %-10s %-9s %s\n" run val_psnr val_ssim train_min vram_gb status arms
  for f in base_s0 base_s1 base_s2 nolatent interleaved imgloss; do
    p=$(grep -E '^val_psnr' $OUT/$f.log | awk '{print $2}'); s=$(grep -E '^val_ssim' $OUT/$f.log | awk '{print $2}')
    t=$(grep -E '^training_seconds' $OUT/$f.log | awk '{printf "%.1f", $2/60}'); v=$(grep -E '^peak_vram_mb' $OUT/$f.log | awk '{printf "%.1f", $2/1024}')
    st=$(grep -E '^status' $OUT/$f.log | awk '{print $2}'); a=$(grep -E '^arm_' $OUT/$f.log | tr '\n' ' ')
    printf "%-12s %-10s %-9s %-10s %-10s %-9s %s\n" $f "${p:-crash}" "${s:--}" "${t:--}" "${v:--}" "${st:--}" "$a"
  done; } | tee $OUT/table.txt
echo "$(date '+%F %T') == done"
