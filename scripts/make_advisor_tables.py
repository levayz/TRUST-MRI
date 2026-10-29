#!/usr/bin/env python
"""Regenerate sections 1-3 of results/advisor_summary_2026-10-26.md from the metric files (plain-DC path for every 512-token model)."""
import json
from pathlib import Path
SB = Path("results/sense_block")
def m(d, arm):
    a = json.load(open(SB / d / f"metrics_{arm}.json")); r = json.load(open(SB / d / f"metrics_{arm}_vs_rss.json"))
    return dict(ps=a["psnr_mean"], ss=a["ssim_mean"], lp=a["lpips_mean"], di=a["dists_mean"], rp=r["psnr_mean"], rs=r["ssim_mean"])
def v(d, key):
    s = json.load(open(SB / d / "summary.json")); a = json.load(open(SB / d / f"metrics_{key}.json"))
    return dict(ps=s[key]["psnr_mean"], ss=s[key]["ssim_mean"], lp=a["lpips_mean"], di=a["dists_mean"], rp=s["vs_rss"][key]["psnr_mean"], rs=s["vs_rss"][key]["ssim_mean"])
c = lambda x: f"{x['ps']:.2f} / {x['ss']:.3f}"; cr = lambda x: f"{x['rp']:.2f} / {x['rs']:.3f}"
R = "suno_acc8_sensegpt512_v21asym_latent_final_dcacq_net_support"; E32 = "suno_acc8_sensegpt512_v21asym_latent_final_equispaced_32lines_dcacq_net_support"; E41 = "suno_acc8_sensegpt512_v21asym_latent_final_equispaced_41lines_dcacq_net_support"
VR, VE32, VE41 = "suno_acc8_varnet256_r8", "suno_acc8_varnet256_r8_equispaced_32lines", "suno_acc8_varnet256_r8_equispaced_41lines"
out = []
out.append("""## 1. Ours vs VarNet on the 256 grid, same sampling budget (10 ACS columns, 32 columns in total nominal)

PSNR / SSIM, suno (15 volumes, 240 slices), x8. "vs SENSE" = against the oracle-map SENSE image (our target); "vs RSS" = against the RSS of the fully sampled coils (VarNet's target); compare within a column. Ours = the best model of the ablation chain (section 3, row 5: run11, v2.1 + latent head), four decodings, per-coil DC applied on the acquisition grid. VarNet = the leaderboard 12-cascade E2E-VarNet fine-tuned 4 epochs on exactly this 256-grid measurement with random masks (the pretrained checkpoint collapses to 20-23 dB here), two outputs. Random mask = the training mask of both models (23-39 columns per volume, mean 30). Equispaced mask = fastMRI's equispaced-fraction mask at the same budget (31-32 columns), seen by neither model in training.

| Model | Random: vs SENSE | Random: vs RSS | Equispaced: vs SENSE | Equispaced: vs RSS |
|---|---|---|---|---|""")
rows = [("zero-filled SENSE image", m(R, "zero_filled"), m(E32, "zero_filled")),
        ("ours = run11 (ablation row 5); greedy tokens + DC", m(R, "greedy_dc"), m(E32, "greedy_dc")),
        ("same model, token path: posterior mean of 8 sampled token reconstructions + DC", m(R, "ps_mean_dc"), m(E32, "ps_mean_dc")),
        ("**same model, latent head: one pass + DC (best PSNR)**", m(R, "latent_dc"), m(E32, "latent_dc")),
        ("same model, hybrid: latent head below k-space radius 0.5, one sampled token reconstruction above, + DC (best perceptual)", m(R, "hybrid_s_r0.5_dc"), m(E32, "hybrid_s_r0.5_dc")),
        ("VarNet fine-tuned on this measurement, SENSE-combined output (oracle maps)", v(VR, "grid256_sense"), v(VE32, "grid256_sense")),
        ("VarNet fine-tuned on this measurement, RSS output", v(VR, "grid256_rss"), v(VE32, "grid256_rss"))]
for name, a, b in rows:
    out.append(f"| {name} | {c(a)} | {cr(a)} | {c(b)} | {cr(b)} |")
out.append("| tokenizer ceiling: round trip of the ground truth, no DC (mask-independent) | 39.54 / 0.973 | 33.01 / 0.708 | same | same |")
la, va, le, ve = m(R, "latent_dc"), v(VR, "grid256_sense"), m(E32, "latent_dc"), v(VE32, "grid256_sense")
out.append(f"""
Gap to VarNet's SENSE output vs SENSE: {va['ps'] - la['ps']:.2f} dB under the random mask ({la['ps']:.2f} vs {va['ps']:.2f}) and {ve['ps'] - le['ps']:.2f} dB under the equispaced mask ({le['ps']:.2f} vs {ve['ps']:.2f}). VarNet's RSS output wins the vs-RSS column by construction (RSS is its training target); the SENSE ground truth itself scores only 34.8 dB vs RSS (27-39 per volume), the ceiling of that column for any SENSE-target method.

**Data-consistency path (changed Oct 26).** Until Oct 26 every "+ DC" number of our 512-token models applied the per-coil DC on the 256 grid inside a resize round trip (downsample from 512, DC, bicubic upsample, antialiased downsample for scoring); the resize after DC is a low-pass. All "ours" numbers in this file now apply the DC directly on the acquisition grid to the decoded image (`eval_sense_block.py --dc-path acq`). On the 256 grid this changes PSNR by -0.2 (greedy) to +0.2 dB (sampled) and improves LPIPS by about 0.025; on the 320 grid the old low-pass was worth +0.4 dB PSNR at a perceptual cost, so the 320 rows below drop by that much against the earlier report.

**Perceptual metrics (vs SENSE; LPIPS and DISTS, lower is better)**, including the hybrid decoding found on Oct 26: the latent one pass below a normalised k-space radius r, a token decoding above it, merged in k-space before DC. The latent head's blur sits in the high frequencies and the token path's errors in the low ones.

| Decoding of run11 (random mask) | PSNR | SSIM | LPIPS | DISTS |
|---|---|---|---|---|""")
for name, arm in [("greedy tokens + DC", "greedy_dc"), ("latent head, one pass + DC", "latent_dc"), ("token path, posterior mean of 8 + DC", "ps_mean_dc"), ("hybrid: latent below r = 0.7, one sample above + DC", "hybrid_s_r0.7_dc"), ("hybrid: latent below r = 0.5, one sample above + DC", "hybrid_s_r0.5_dc")]:
    x = m(R, arm); out.append(f"| {name} | {x['ps']:.2f} | {x['ss']:.3f} | {x['lp']:.3f} | {x['di']:.3f} |")
x = v(VR, "grid256_sense"); out.append(f"| VarNet fine-tuned, SENSE-combined output | {x['ps']:.2f} | {x['ss']:.3f} | {x['lp']:.3f} | {x['di']:.3f} |")
pe, he, vle = m(E32, "ps_mean_dc"), m(E32, "hybrid_s_r0.5_dc"), v(VE32, "grid256_sense")
out.append(f"""
Under the equispaced mask the same ordering holds: posterior mean of 8 {pe['lp']:.3f} / {pe['di']:.3f}, hybrid {he['lp']:.3f} / {he['di']:.3f}, VarNet SENSE output {vle['lp']:.3f} / {vle['di']:.3f}. So on the 256 grid our model is 0.7-0.8 dB behind VarNet in PSNR and level with it or ahead on the perceptual metrics, depending on how far along the sampled side the decoding goes.
""")
# ---- section 2
R16 = "suno_acc8_sensegpt320_v21asym_latent_final_dcacq_net_crop320_support"
g, p8, l, h = m(R16, "greedy_dc"), m(R16, "ps_mean_dc"), m(R16, "latent_dc"), m(R16, "hybrid_s_r0.5_dc")
R18 = "suno_acc8_sensegpt320_v21asym_latent_long_final_dcacq_net_crop320_support"
g8, p88, l8, h8 = m(R18, "greedy_dc"), m(R18, "ps_mean_dc"), m(R18, "latent_dc"), m(R18, "hybrid_s_r0.5_dc")
R18E = "suno_acc8_sensegpt320_v21asym_latent_long_final_equispaced_40lines_dcacq_net_crop320_support"
p8e, l8e = m(R18E, "ps_mean_dc"), m(R18E, "latent_dc")
out.append(f"""## 2. Other grids and budgets (not comparable row for row with section 1)

### 2a. Native 320 grid, native budget (13 ACS columns, 40 columns in total nominal), scored at 320

| Model | vs SENSE | vs RSS | LPIPS / DISTS vs SENSE |
|---|---|---|---|
| zero-filled SENSE image | 25.36 / 0.777 | 24.69 / 0.533 | 0.234 / 0.280 |
| SENSE GPT, 1024-token tokenizer, trained at 320, posterior mean of 8 (Oct 19; old DC path) | 32.50 / 0.914 | 29.63 / 0.582 | 0.112 / 0.138 |
| run16 = run11's model fine-tuned 3 epochs at 320: greedy tokens + DC | {c(g)} | {cr(g)} | {g['lp']:.3f} / {g['di']:.3f} |
| run16, latent head, one pass + DC | {c(l)} | {cr(l)} | {l['lp']:.3f} / {l['di']:.3f} |
| run16, token path, posterior mean of 8 + DC | {c(p8)} | {cr(p8)} | {p8['lp']:.3f} / {p8['di']:.3f} |
| run16, hybrid r = 0.5 + DC | {c(h)} | {cr(h)} | {h['lp']:.3f} / {h['di']:.3f} |
| run18 = run16 trained 10 more epochs at LR 1e-4 (the longer schedule): greedy tokens + DC | {c(g8)} | {cr(g8)} | {g8['lp']:.3f} / {g8['di']:.3f} |
| run18, latent head, one pass + DC | {c(l8)} | {cr(l8)} | {l8['lp']:.3f} / {l8['di']:.3f} |
| **run18, token path, posterior mean of 8 + DC** | **{c(p88)}** | {cr(p88)} | {p88['lp']:.3f} / {p88['di']:.3f} |
| run18, hybrid r = 0.5 + DC | {c(h8)} | {cr(h8)} | {h8['lp']:.3f} / {h8['di']:.3f} |
| run18, equispaced-fraction mask (about 40 columns, VarNet's training mask), latent head one pass + DC | {c(l8e)} | {cr(l8e)} | {l8e['lp']:.3f} / {l8e['di']:.3f} |
| run18, equispaced-fraction mask, token path, posterior mean of 8 + DC | {c(p8e)} | {cr(p8e)} | {p8e['lp']:.3f} / {p8e['di']:.3f} |
| VarNet, leaderboard weights, random mask, RSS output | 34.25 / 0.909 | 35.01 / 0.930 | 0.092 / 0.122 |
| VarNet, 8 cascades trained at R=8 with equispaced-fraction masks, random mask, RSS output | 33.08 / 0.898 | 33.75 / 0.920 | |
| VarNet, 8 cascades trained at R=8 with equispaced-fraction masks, equispaced-fraction mask, RSS output | 35.16 / 0.914 | 36.11 / 0.936 | |
| tokenizer ceiling (round trip + DC) | 38.19 / 0.969 | 32.08 / 0.628 | |

Gap to the leaderboard VarNet with random masks: 1.75 dB for the v1 model, {34.25 - p8['ps']:.2f} dB for run16 and {34.25 - p88['ps']:.2f} dB for run18 (posterior mean of 8); perceptually run18 is ahead of the leaderboard VarNet ({p88['lp']:.3f} / {p88['di']:.3f} vs 0.092 / 0.122). Under the equispaced-fraction mask, which the 8-cascade VarNet was trained on and our model never saw, run18 scores {p8e['ps']:.2f} against that VarNet's 35.16 (RSS output); under the random mask, our training distribution, run18 is ahead of it (33.29 vs 33.08). The standard fastMRI equispaced mask (51-52 columns) gives run18 34.52 with the posterior mean of 8. With the old DC path run16 scored 33.18 (one pass) / 33.14 (posterior mean) at LPIPS 0.113 / 0.107. Run18, the same model trained 10 more epochs at a hotter schedule, is running.

### 2b. Leaderboard VarNet on its native grid, cropped to 256 (the rows of the last meeting's table: native budget of 40 columns, not ours)

| Model | vs SENSE | vs RSS |
|---|---|---|
| fastMRI leaderboard weights, native grid, random mask, cropped to 256 | 32.73 / 0.887 | 33.48 / 0.914 |
| fastMRI leaderboard weights, native grid, equispaced mask, cropped to 256 | 34.64 / 0.904 | 35.69 / 0.932 |
| fastMRI leaderboard weights, native grid, our 256-grid budget (10 ACS, 33 columns in total), random mask | 29.58 / 0.850 | - |
| fastMRI leaderboard weights, native grid, our 256-grid budget (10 ACS, 33 columns in total), equispaced mask | 31.68 / 0.881 | - |

### 2c. Larger budget on the 256 grid: the standard fastMRI equispaced mask at R=8 (every 8th column + 10 ACS = 41 columns, 28 percent more than section 1)

| Model | vs SENSE | vs RSS | LPIPS / DISTS |
|---|---|---|---|""")
for name, x in [("ours (run11), latent one pass + DC", m(E41, "latent_dc")), ("ours (run11), posterior mean of 8 + DC", m(E41, "ps_mean_dc")), ("ours (run11), hybrid r = 0.5 + DC", m(E41, "hybrid_s_r0.5_dc")), ("VarNet fine-tuned on the 256 measurement, SENSE-combined output", v(VE41, "grid256_sense")), ("VarNet fine-tuned on the 256 measurement, RSS output", v(VE41, "grid256_rss"))]:
    out.append(f"| {name} | {c(x)} | {cr(x)} | {x['lp']:.3f} / {x['di']:.3f} |")
out.append("""
Why the old leaderboard rows read higher than section 1: they carry the native budget (40 columns at 320 width, cropped afterwards), and the equispaced one adds the mask advantage. On our exact budget the pretrained model drops to 29.6 / 31.7 dB (2b), and the fine-tuned one to 32.1 / 32.6 dB (section 1).
""")
# ---- section 3
r3, r4, r5 = "suno_acc8_sensegpt512_final_dcacq_net_support", "suno_acc8_sensegpt512_v21asym_final_dcacq_net_support", R
g3, p3 = m(r3, "greedy_dc"), m(r3, "ps_mean_dc"); g4, p4 = m(r4, "greedy_dc"), m(r4, "ps_mean_dc"); g5, p5, l5 = m(r5, "greedy_dc"), m(r5, "ps_mean_dc"), m(r5, "latent_dc")
out.append(f"""## 3. Ablation: what each component did (256 grid, random x8 mask, PSNR vs SENSE, full suno evaluation, plain DC)

Cumulative chain of the kept components, each row = previous row + one change. Recipe for every fine-tune: 3 epochs, cosine from 5e-5, batch 8, warm start from the previous row (the base GPT-L body is the single-coil brain model throughout; rows 1-3 trained 10 epochs at 1e-4). "greedy" = argmax tokens + DC; "mean of 8" = posterior mean of 8 sampled token reconstructions + DC; "one pass" = the latent head decoded without sampling + DC. Rows 0-2 use 256 tokens and have no resize step, so their numbers are unchanged by the DC-path switch.

| # | Component added | greedy tokens | token path, mean of 8 | latent, one pass | gain (mean of 8 unless noted) |
|---|---|---|---|---|---|
| 0 | single-coil GPT + pretrained tokenizer, applied zero-shot to SENSE images (256 tokens) | 25.16 | 25.70 | - | reference |
| 1 | + training on SENSE images (same pretrained tokenizer) | 27.24 | 28.22 | - | +2.5 |
| 2 | + tokenizer fine-tuned on SENSE images (complex pair, 256 tokens) | 28.08 | 28.57 | - | +0.35 (+0.85 greedy) |
| 3 | + 1024 tokens (slice upsampled 2x to 512 for a tokenizer trained at 512) | {g3['ps']:.2f} | {p3['ps']:.2f} | - | +{p3['ps'] - 28.57:.1f} |
| 4 | + v2.1: continuous latent input instead of token IDs; stream-aware token head (separate tables for the previous Re and Im codes, concatenated and projected instead of summed); mask and timestep conditioning | {g4['ps']:.2f} | {p4['ps']:.2f} | - | +{p4['ps'] - p3['ps']:.2f} (+{g4['ps'] - g3['ps']:.2f} greedy) |
| 5 | + latent-regression head (regress the tokenizer latent, decode without quantisation) | {g5['ps']:.2f} | {p5['ps']:.2f} | **{l5['ps']:.2f}** | +{l5['ps'] - g5['ps']:.2f} one pass over greedy; one pass {l5['ps'] - p5['ps']:+.2f} vs the mean of 8 |

Same thing on the latent head alone (row 5's one pass): on the v1 model (row 3) it gives 30.92 (old DC path), on v2.1 (row 4) {l5['ps']:.2f}, so the two gains add.

Side branch, not in the chain: row 4's model trained 10 more epochs with a hotter schedule gives 30.63 greedy and 31.22 mean of 8 on the old DC path (+0.24 over row 4 on that path). It has not been combined with the latent head at 256; the combination is running at the 320 grid (run18).

Components tried and dropped (each from the best model at the time; difference to its control; old DC path, the differences are what matter):

| Component | result | control | effect |
|---|---|---|---|
| mask + timestep conditioning on the token model (v2) | 29.88 (validation) | 29.92 | 0; the model shrank the gates |
| 16-step interleaved Re/Im token head instead of the 8-step stream-aware head | 30.34 / 30.95 | 30.47 / 30.98 | -0.13 greedy, 0 mean of 8 |
| teacher-student refinement, first version (timestep invisible to the model) | 29.19 refined (validation) | 29.98 pass 1 | -0.8 |
| two-view refinement after the timestep fix | 29.98 refined (validation) | 30.00 pass 1 | 0 |
| image-domain loss through the frozen decoder | 30.04 mean of 8 | 30.68 | -0.6 |
| latent head with a residual MLP on the hidden state | 31.32 one pass | 31.27 | +0.05 (noise) |
| fused latent head reading the token head's committed codes | 31.31 one pass | 31.27 | +0.04 (noise) |
| privileged-teacher distillation (EMA teacher at x2-x4; KL on the token distribution and L2 on the latent, 3 epochs; plain DC) | 31.40 one pass / 31.33 mean of 8 | 31.35 / 31.29 | +0.05 (noise; the same as 3 more epochs without the teacher) |
| progressive commitment (pass 2 and 3 re-predict the positions not committed after pass 1; commitment = expected embedding at the most confident positions by the latent spread; 2 epochs; plain DC) | 31.39 one pass after the commit passes | 31.38 pass 1 of the same model, 31.35 run11 | 0; the draft gate decayed to 0.19: the committed view is the model's own belief and adds no evidence |
| sensitivity net instead of oracle maps (the whole pipeline) | within 0.1 of oracle maps | | 0 |

Decoding-side findings of Oct 26 (no training): the plain acquisition-grid DC (+0.08 dB and -0.025 LPIPS on the latent path at 256) and the hybrid decoding (section 1). Running: the longer schedule (10 epochs at LR 1e-4) on row 5's model at 256 (run19); the same at 320 (run18) is done, see section 2a.
""")
text = "\n".join(out) + "\n"
p = Path("results/advisor_summary_2026-10-26.md"); s = p.read_text()
head, _, rest = s.partition("## 1. "); _, _, tail = rest.partition("## 4. ")
p.write_text(head + text + "## 4. " + tail)
print("sections 1-3 regenerated")
