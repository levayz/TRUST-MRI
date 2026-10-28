# SENSE transformer vs E2E-VarNet, and the week of Oct 18-25

Setting: fastMRI brain multicoil, suno split (15 volumes, 240 slices), x8 random 1-D Cartesian masks, cf 0.04. PSNR in dB vs the oracle SENSE image (our target) and vs the RSS image (VarNet's target); compare within a column. "+ DC" = per-coil data consistency after decoding. Full tables: results/sense_summary_256.md, results/sense_summary_320.md.

## 1. Ours vs VarNet on the 256 grid, same sampling budget (10 ACS columns, 32 columns in total nominal)

PSNR / SSIM, suno (15 volumes, 240 slices), x8. "vs SENSE" = against the oracle-map SENSE image (our target); "vs RSS" = against the RSS of the fully sampled coils (VarNet's target); compare within a column. Ours = the best model of the ablation chain (section 3, row 5: run11, v2.1 + latent head), four decodings, per-coil DC applied on the acquisition grid. VarNet = the leaderboard 12-cascade E2E-VarNet fine-tuned 4 epochs on exactly this 256-grid measurement with random masks (the pretrained checkpoint collapses to 20-23 dB here), two outputs. Random mask = the training mask of both models (23-39 columns per volume, mean 30). Equispaced mask = fastMRI's equispaced-fraction mask at the same budget (31-32 columns), seen by neither model in training.

| Model | Random: vs SENSE | Random: vs RSS | Equispaced: vs SENSE | Equispaced: vs RSS |
|---|---|---|---|---|
| zero-filled SENSE image | 23.57 / 0.682 | 23.31 / 0.557 | 23.54 / 0.681 | 23.29 / 0.557 |
| ours = run11 (ablation row 5); greedy tokens + DC | 30.37 / 0.872 | 28.80 / 0.676 | 30.86 / 0.879 | 29.18 / 0.684 |
| same model, token path: posterior mean of 8 sampled token reconstructions + DC | 31.29 / 0.889 | 29.59 / 0.704 | 31.72 / 0.894 | 29.91 / 0.711 |
| **same model, latent head: one pass + DC (best PSNR)** | 31.35 / 0.886 | 29.48 / 0.683 | 31.83 / 0.892 | 29.83 / 0.690 |
| same model, hybrid: latent head below k-space radius 0.5, one sampled token reconstruction above, + DC (best perceptual) | 30.98 / 0.876 | 29.27 / 0.679 | 31.44 / 0.883 | 29.61 / 0.687 |
| VarNet fine-tuned on this measurement, SENSE-combined output (oracle maps) | 32.06 / 0.892 | 30.14 / 0.691 | 32.61 / 0.899 | 30.55 / 0.699 |
| VarNet fine-tuned on this measurement, RSS output | 31.69 / 0.877 | 31.83 / 0.903 | 32.16 / 0.884 | 32.35 / 0.911 |
| tokenizer ceiling: round trip of the ground truth, no DC (mask-independent) | 39.54 / 0.973 | 33.01 / 0.708 | same | same |

Gap to VarNet's SENSE output vs SENSE: 0.71 dB under the random mask (31.35 vs 32.06) and 0.77 dB under the equispaced mask (31.83 vs 32.61). VarNet's RSS output wins the vs-RSS column by construction (RSS is its training target); the SENSE ground truth itself scores only 34.8 dB vs RSS (27-39 per volume), the ceiling of that column for any SENSE-target method.

**Data-consistency path (changed Oct 26).** Until Oct 26 every "+ DC" number of our 512-token models applied the per-coil DC on the 256 grid inside a resize round trip (downsample from 512, DC, bicubic upsample, antialiased downsample for scoring); the resize after DC is a low-pass. All "ours" numbers in this file now apply the DC directly on the acquisition grid to the decoded image (`eval_sense_block.py --dc-path acq`). On the 256 grid this changes PSNR by -0.2 (greedy) to +0.2 dB (sampled) and improves LPIPS by about 0.025; on the 320 grid the old low-pass was worth +0.4 dB PSNR at a perceptual cost, so the 320 rows below drop by that much against the earlier report.

**Perceptual metrics (vs SENSE; LPIPS and DISTS, lower is better)**, including the hybrid decoding found on Oct 26: the latent one pass below a normalised k-space radius r, a token decoding above it, merged in k-space before DC. The latent head's blur sits in the high frequencies and the token path's errors in the low ones.

| Decoding of run11 (random mask) | PSNR | SSIM | LPIPS | DISTS |
|---|---|---|---|---|
| greedy tokens + DC | 30.37 | 0.872 | 0.122 | 0.136 |
| latent head, one pass + DC | 31.35 | 0.886 | 0.131 | 0.154 |
| token path, posterior mean of 8 + DC | 31.29 | 0.889 | 0.122 | 0.139 |
| hybrid: latent below r = 0.7, one sample above + DC | 31.16 | 0.881 | 0.118 | 0.139 |
| hybrid: latent below r = 0.5, one sample above + DC | 30.98 | 0.876 | 0.115 | 0.131 |
| VarNet fine-tuned, SENSE-combined output | 32.06 | 0.892 | 0.124 | 0.147 |

Under the equispaced mask the same ordering holds: posterior mean of 8 0.116 / 0.135, hybrid 0.111 / 0.127, VarNet SENSE output 0.117 / 0.143. So on the 256 grid our model is 0.7-0.8 dB behind VarNet in PSNR and level with it or ahead on the perceptual metrics, depending on how far along the sampled side the decoding goes.

## 2. Other grids and budgets (not comparable row for row with section 1)

### 2a. Native 320 grid, native budget (13 ACS columns, 40 columns in total nominal), scored at 320

| Model | vs SENSE | vs RSS | LPIPS / DISTS vs SENSE |
|---|---|---|---|
| zero-filled SENSE image | 25.36 / 0.777 | 24.69 / 0.533 | 0.234 / 0.280 |
| SENSE GPT, 1024-token tokenizer, trained at 320, posterior mean of 8 (Oct 19; old DC path) | 32.50 / 0.914 | 29.63 / 0.582 | 0.112 / 0.138 |
| run16 = run11's model fine-tuned 3 epochs at 320: greedy tokens + DC | 31.85 / 0.908 | 29.23 / 0.585 | 0.092 / 0.114 |
| run16, latent head, one pass + DC | 32.81 / 0.917 | 29.75 / 0.584 | 0.094 / 0.129 |
| run16, token path, posterior mean of 8 + DC | 33.08 / 0.923 | 30.06 / 0.612 | 0.087 / 0.112 |
| run16, hybrid r = 0.5 + DC | 32.49 / 0.911 | 29.61 / 0.584 | 0.087 / 0.121 |
| run18 = run16 trained 10 more epochs at LR 1e-4 (the longer schedule): greedy tokens + DC | 32.11 / 0.910 | 29.38 / 0.585 | 0.088 / 0.111 |
| run18, latent head, one pass + DC | 32.99 / 0.919 | 29.84 / 0.585 | 0.092 / 0.127 |
| **run18, token path, posterior mean of 8 + DC** | **33.29 / 0.925** | 30.16 / 0.611 | 0.085 / 0.109 |
| run18, hybrid r = 0.5 + DC | 32.67 / 0.913 | 29.71 / 0.585 | 0.084 / 0.117 |
| run18, equispaced-fraction mask (about 40 columns, VarNet's training mask), latent head one pass + DC | 33.33 / 0.922 | 30.02 / 0.589 | 0.089 / 0.123 |
| run18, equispaced-fraction mask, token path, posterior mean of 8 + DC | 33.57 / 0.928 | 30.32 / 0.616 | 0.083 / 0.106 |
| VarNet, leaderboard weights, random mask, RSS output | 34.25 / 0.909 | 35.01 / 0.930 | 0.092 / 0.122 |
| VarNet, 8 cascades trained at R=8 with equispaced-fraction masks, random mask, RSS output | 33.08 / 0.898 | 33.75 / 0.920 | |
| VarNet, 8 cascades trained at R=8 with equispaced-fraction masks, equispaced-fraction mask, RSS output | 35.16 / 0.914 | 36.11 / 0.936 | |
| tokenizer ceiling (round trip + DC) | 38.19 / 0.969 | 32.08 / 0.628 | |

Gap to the leaderboard VarNet with random masks: 1.75 dB for the v1 model, 1.17 dB for run16 and 0.96 dB for run18 (posterior mean of 8); perceptually run18 is ahead of the leaderboard VarNet (0.085 / 0.109 vs 0.092 / 0.122). Under the equispaced-fraction mask, which the 8-cascade VarNet was trained on and our model never saw, run18 scores 33.57 against that VarNet's 35.16 (RSS output); under the random mask, our training distribution, run18 is ahead of it (33.29 vs 33.08). The standard fastMRI equispaced mask (51-52 columns) gives run18 34.52 with the posterior mean of 8. With the old DC path run16 scored 33.18 (one pass) / 33.14 (posterior mean) at LPIPS 0.113 / 0.107. Run18, the same model trained 10 more epochs at a hotter schedule, is running.

### 2b. Leaderboard VarNet on its native grid, cropped to 256 (the rows of the last meeting's table: native budget of 40 columns, not ours)

| Model | vs SENSE | vs RSS |
|---|---|---|
| fastMRI leaderboard weights, native grid, random mask, cropped to 256 | 32.73 / 0.887 | 33.48 / 0.914 |
| fastMRI leaderboard weights, native grid, equispaced mask, cropped to 256 | 34.64 / 0.904 | 35.69 / 0.932 |
| fastMRI leaderboard weights, native grid, our 256-grid budget (10 ACS, 33 columns in total), random mask | 29.58 / 0.850 | - |
| fastMRI leaderboard weights, native grid, our 256-grid budget (10 ACS, 33 columns in total), equispaced mask | 31.68 / 0.881 | - |

### 2c. Larger budget on the 256 grid: the standard fastMRI equispaced mask at R=8 (every 8th column + 10 ACS = 41 columns, 28 percent more than section 1)

| Model | vs SENSE | vs RSS | LPIPS / DISTS |
|---|---|---|---|
| ours (run11), latent one pass + DC | 32.74 / 0.906 | 30.49 / 0.711 | 0.110 / 0.137 |
| ours (run11), posterior mean of 8 + DC | 32.56 / 0.907 | 30.55 / 0.731 | 0.104 / 0.124 |
| ours (run11), hybrid r = 0.5 + DC | 32.30 / 0.897 | 30.26 / 0.708 | 0.100 / 0.117 |
| VarNet fine-tuned on the 256 measurement, SENSE-combined output | 33.46 / 0.910 | 31.17 / 0.711 | 0.104 / 0.131 |
| VarNet fine-tuned on the 256 measurement, RSS output | 32.90 / 0.894 | 33.21 / 0.922 | 0.109 / 0.136 |

Why the old leaderboard rows read higher than section 1: they carry the native budget (40 columns at 320 width, cropped afterwards), and the equispaced one adds the mask advantage. On our exact budget the pretrained model drops to 29.6 / 31.7 dB (2b), and the fine-tuned one to 32.1 / 32.6 dB (section 1).

## 3. Ablation: what each component did (256 grid, random x8 mask, PSNR vs SENSE, full suno evaluation, plain DC)

Cumulative chain of the kept components, each row = previous row + one change. Recipe for every fine-tune: 3 epochs, cosine from 5e-5, batch 8, warm start from the previous row (the base GPT-L body is the single-coil brain model throughout; rows 1-3 trained 10 epochs at 1e-4). "greedy" = argmax tokens + DC; "mean of 8" = posterior mean of 8 sampled token reconstructions + DC; "one pass" = the latent head decoded without sampling + DC. Rows 0-2 use 256 tokens and have no resize step, so their numbers are unchanged by the DC-path switch.

| # | Component added | greedy tokens | token path, mean of 8 | latent, one pass | gain (mean of 8 unless noted) |
|---|---|---|---|---|---|
| 0 | single-coil GPT + pretrained tokenizer, applied zero-shot to SENSE images (256 tokens) | 25.16 | 25.70 | - | reference |
| 1 | + training on SENSE images (same pretrained tokenizer) | 27.24 | 28.22 | - | +2.5 |
| 2 | + tokenizer fine-tuned on SENSE images (complex pair, 256 tokens) | 28.08 | 28.57 | - | +0.35 (+0.85 greedy) |
| 3 | + 1024 tokens (slice upsampled 2x to 512 for a tokenizer trained at 512) | 29.76 | 30.81 | - | +2.2 |
| 4 | + v2.1: continuous latent input instead of token IDs; stream-aware token head (separate tables for the previous Re and Im codes, concatenated and projected instead of summed); mask and timestep conditioning | 30.28 | 31.16 | - | +0.35 (+0.52 greedy) |
| 5 | + latent-regression head (regress the tokenizer latent, decode without quantisation) | 30.37 | 31.29 | **31.35** | +0.98 one pass over greedy; one pass +0.06 vs the mean of 8 |

Same thing on the latent head alone (row 5's one pass): on the v1 model (row 3) it gives 30.92 (old DC path), on v2.1 (row 4) 31.35, so the two gains add.

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
| sensitivity net instead of oracle maps (the whole pipeline) | within 0.1 of oracle maps | | 0 |

Decoding-side findings of Oct 26 (no training): the plain acquisition-grid DC (+0.08 dB and -0.025 LPIPS on the latent path at 256) and the hybrid decoding (section 1). Running: the longer schedule (10 epochs at LR 1e-4) on row 5's model at 256 (run19) and on the 320-grid model (run18).

## 4. What was done this week

| Dates | Experiment | Result (vs SENSE, 256 grid unless noted) | Takeaway |
|---|---|---|---|
| Oct 18-19 | Native 320 grid: sensitivity net retrained at 320, v1 model fine-tuned at 320, VarNet baselines on the same line budgets | 32.50 at 320 vs 30.68 at 256; leaderboard VarNet 34.25 | +1.8 dB from the 13 ACS lines and the larger grid; 1.75 dB gap to VarNet |
| Oct 18-19 | Latent-regression head on the v1 model: regress the tokenizer's continuous latent, decode without quantisation | one pass 30.92 vs greedy 30.02 and 8-draw mean 30.68 | +0.9 dB in a single pass; beats sampling |
| Oct 19-20 | Ablation: same recipe with the pretrained tokenizer | greedy 27.24, 8-draw mean 28.22 | of the 5 dB over zero-shot (25.16), tokenizer fine-tuning gives 2.8 dB, the 512-token grid 1.9 dB of that |
| Oct 18, 21-23 | Teacher-student refinement: first version hurt (29.19 vs 29.98); root cause found (timestep token invisible to the model); fixed and rerun as two-view refinement | refined 29.98 vs pass 1 30.00 | neutral: the draft carries no new evidence and the model cannot mark its own errors; shelved in favour of progressive commitment |
| Oct 22 | v2: sampling-mask and timestep conditioning on the token model | 29.88 vs 29.92 | no gain; the model shrank the gates |
| Oct 22-23 | v2.1: continuous pre-quantisation latents as input instead of token IDs; asymmetric Re/Im head (concat + projection) | greedy 30.47, 8-draw mean 30.98 (v1: 30.02 / 30.68) | +0.3 to +0.45 dB |
| Oct 23-24 | v2.1 + latent head | one pass 31.27, 8-draw mean 31.12 | gains stack; best transformer, 0.8 dB below the matched VarNet |
| Oct 23-25 | Longer schedule: v2.1 for 10 more epochs at LR 1e-4 | greedy 30.63, 8-draw mean 31.22 (3 epochs: 30.47 / 30.98) | +0.2 dB; flat after epoch 9, so the 3-epoch budget costs about 0.2 dB |
| Oct 24-25 | 16-step interleaved Re/Im token head | greedy 30.34, 8-draw mean 30.95 | no gain, token accuracy identical to the 8-step head; keep the 8-step head |
| Oct 24-26 | Fused latent head (reads the token head's committed codes) and its MLP-only control | one pass 31.31 and 31.32 vs 31.27 for the linear head; 8-draw mean 31.17 vs 31.12 | +0.05 dB for both, within noise: the token head's codes add nothing; the linear head stays |
| Oct 25-26 | v2.1 + latent head at the native 320 grid (mask embedding resampled to 320 columns), 3 epochs from the best 256 model | one pass 32.81, 8-draw mean 33.08 at 320 (v1 token model there: 32.50 with 8 draws); LPIPS / DISTS 0.087 / 0.112 against 0.092 / 0.122 for the leaderboard VarNet at 34.25 dB | +0.6 dB over v1 at 320; level with VarNet perceptually at 1.2 dB lower PSNR |
| Oct 26 | Decoding without retraining: per-coil DC applied directly on the acquisition grid instead of inside a resize round trip; hybrid decoding (latent head below a k-space radius, one sampled token reconstruction above) | latent one pass 31.35 (+0.08 dB, LPIPS 0.156 to 0.131); hybrid 30.98 at LPIPS 0.115 against VarNet's 32.06 at 0.124 | the resize after DC was a low-pass; the hybrid gives the best perceptual numbers at a 0.4 dB PSNR cost against the one pass |
| Oct 26-27 | Privileged-teacher distillation: an EMA teacher that sees the same scan at x2-x4 trains the x8 student toward its token distribution (KL) and its latent (L2), 3 epochs from the best 256 model | one pass 31.40, 8-draw mean 31.33 against 31.35 / 31.29 for the control | +0.05 dB, the same as 3 more epochs without the teacher; dropped at these weights |
| Oct 26-28 | Longer schedule at 320: run16 trained 10 more epochs at LR 1e-4 | posterior mean of 8 33.29 (run16: 33.08), one pass 32.99 (32.81), greedy 32.11 (31.85); LPIPS / DISTS 0.085 / 0.109 against the leaderboard VarNet's 0.092 / 0.122 at 34.25 dB | +0.2 dB, as the longer schedule gave at 256; gap to the leaderboard VarNet 0.96 dB with better perceptual metrics |
| Oct 24 | Paper-style architecture figure (pipeline, tokenizer, transformer, heads); run queue agreed; automatic evaluation chains after every run | | |

## 5. Running and next

1. Running: the longer schedule (10 epochs at LR 1e-4) on the best 256 model (run19, ends Oct 29 around noon; validation one pass 31.00, 31.13, 31.16, 31.22, 31.29 over epochs 1-5 against 31.23 for the 3-epoch model). Its full evaluation (plain DC, hybrid arms) is chained after it. The same schedule at 320 (run18) is done, see section 2a; its equispaced-mask evaluations are running.
2. Progressive-commitment refinement (keep confident positions, re-predict the rest; confidence from the latent spread), commitment scheme to be discussed; first step is a validation diagnostic of the confidence score against posterior-sampling disagreement.
