#!/usr/bin/env bash
# Regenerate results/sense_table.md from the sens-map check and the SENSE block evals.
set -euo pipefail
PY=${PY:-/opt/miniconda3/envs/meditok/bin/python}
SENS_CHECK=${SENS_CHECK:-results/sens_check/trained_final}
OUT=${OUT:-results/sense_table.md}
{
echo "# SENSE pipeline results (brain, suno split, 15 volumes / 240 slices, x8, cf 0.04, random 1D masks)"
echo
echo "Generated $(date '+%Y-%m-%d %H:%M'). Pipeline: trained sensitivity U-Net (ACS lines of the undersampled k-space) -> SENSE combine -> Re/Im tokens"
echo "(cplxpair SENSE MedITok) -> GPT-L (warm start from the single-coil brain x8 GPT) -> decode -> expand -> per-coil DC (inference only)."
echo
echo "## Sensitivity maps vs oracle ESPIRiT (\`$SENS_CHECK/summary.txt\`; per-epoch history in \`results/sens_net/run1/history.json\`)"
echo
echo "| estimator | agreement (support) | agreement (bright) | model residual | combined PSNR vs oracle (support) | vs RSS | tokenizer round trip (re/im) vs oracle |"
echo "|---|---|---|---|---|---|---|"
$PY - "$SENS_CHECK" <<'PYEOF'
import json, sys
m = json.load(open(f"{sys.argv[1]}/metrics.json"))
d = m["map_diagnostics"]; s = m["summary"]["split"]
for n in ("varnet256_trained", "varnet_native", "espirit_acs", "varnet256"):
    a = s[f"{n}__combined__vs_oracle_support"]; b = s[f"{n}__combined__vs_rss"]; t = s.get(f"{n}__raw_reim__vs_oracle_support")
    tt = f"{t['psnr']:.2f} / {t['ssim']:.4f}" if t else "-"
    print(f"| {n} | {d[n]['agree_support']['mean']:.4f} | {d[n]['agree_bright']['mean']:.4f} | {d[n]['model_resid']['mean']:.4f} | {a['psnr']:.2f} / {a['ssim']:.4f} | {b['psnr']:.2f} / {b['ssim']:.4f} | {tt} |")
t = s.get("oracle__raw_reim__vs_oracle_support")
if t:
    print(f"| oracle | 1 | 1 | - | - | - | {t['psnr']:.2f} / {t['ssim']:.4f} |")
h = json.load(open("results/sens_net/run1/history.json"))
print()
print("Stage-1 validation per epoch (suno): " + "; ".join(f"ep{e['epoch']}: agree {e['agree_support']:.4f}, {e['vs_oracle_support']['psnr_mean']:.2f} dB" for e in h))
PYEOF
echo
echo "## Reconstruction (\`eval_sense_block.py\`; PSNR/SSIM/LPIPS on |x| vs oracle SENSE |x0|; 'support' = reconstructions multiplied by the oracle ESPIRiT support before scoring)"
echo
$PY scripts/summarize_sense_block.py --glob 'suno_acc8_*' --arms zero_filled greedy_no_dc greedy_dc exp1_dc latent_dc ps_mean_dc gt_reim_dc --md | grep -E "^\| run|^\|---|tokpretrained |zeroshot_tokpre_net_support|sensegpt"
echo
echo "## Metrics vs the RSS image (fully sampled coils; no support masking; runs made with --gt-ref both)"
echo
$PY scripts/summarize_sense_block.py --glob 'suno_acc8_*' --ref rss --arms zero_filled greedy_dc exp1_dc latent_dc ps_mean_dc gt_reim_dc --md | grep -E "^\| run|^\|---|tokpretrained |zeroshot_tokpre|sensegpt"
echo
echo "## E2E-VarNet baselines (scripts/eval_varnet_baseline.py + scripts/score_varnet_baseline.py; our seeded random x8 mask; native = VarNet run on the native k-space grid then cropped to 256, grid256 = run on the 256-grid measurement (outside its training distribution); _rss = RSS of the predicted coils, _sense = SENSE combine with the oracle maps)"
echo
echo "vs SENSE (oracle support):"
echo
$PY scripts/summarize_sense_block.py --glob 'suno_acc8_varnet_*' --arms native_rss native_sense grid256_rss grid256_sense --md
echo
echo "vs RSS:"
echo
$PY scripts/summarize_sense_block.py --glob 'suno_acc8_varnet_*' --ref rss --arms native_rss native_sense grid256_rss grid256_sense --md
echo
echo "## Other accelerations (same x8-trained models)"
echo
$PY scripts/summarize_sense_block.py --glob 'suno_acc[0-9]*_*' --arms zero_filled greedy_dc ps_mean_dc gt_reim_dc --md | grep -vE "suno_acc8_" | grep -E "^\| run|^\|---|sensegpt|tokpretrained "
echo
echo "Rows: \`tokpretrained\` = zero-shot single-coil GPT + pretrained tokenizer, oracle maps (existing baseline); \`zeroshot_tokpre_net_support\` = same with net maps;"
echo "\`sensegpt_ep<N>_*\` = SENSE GPT after N epochs (\`results/brain_sense/cart_1d/run1_GPT-L/cf_0.04/acc_8_maps_net/checkpoints/best_ep<N>.pt\`; \`_sensfinal\` = final stage-1 net instead of \`best_ep1.pt\`)."
echo "\`gt_reim_dc\` = tokenizer round trip of x0 + DC (ceiling). PS = 8 draws, top-k 4096, magnitude mean. \`latent_dc\` = latent-regression head decoded without quantisation (runs with \`_latent\`), one pass. Regenerate: \`bash scripts/make_sense_table.sh\`."
} > "$OUT"
echo "written $OUT"
