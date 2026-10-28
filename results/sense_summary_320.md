# SENSE transformer vs E2E-VarNet on the native 320 grid (brain, suno split, x8)

Companion to `results/sense_summary_256.md`. Everything here is scored on the **320x320 centre crop of the native fastMRI grid**
(the grid the fastMRI VarNet checkpoints were trained on), 15 volumes / 240 slices, random 1D Cartesian masks seeded by filename,
cf 0.04 -> **13 ACS lines, ~40 sampled lines in total** (vs 10 ACS / 32 lines nominal on the 256 grid). PSNR / SSIM, per-volume PSNR with the
ground-truth range. "vs SENSE" = |reconstruction| x oracle ESPIRiT support vs |x0| (oracle SENSE combine); "vs RSS" = no support, vs the RSS of the fully sampled coils.

| method | vs SENSE | vs RSS |
|---|---|---|
| zero-filled SENSE (net maps) | 25.36 / 0.777 | 24.69 / 0.533 |
| ours (320 grid): greedy + DC | 31.58 / 0.900 | 28.99 / 0.559 |
| ours (320 grid): expected embedding + DC | 32.07 / 0.905 | 29.25 / 0.558 |
| **ours (320 grid): PS8 + DC** | **32.50 / 0.914** | 29.63 / 0.582 |
| ours v2.1 + latent head at 320 (run16, Oct 26): greedy + DC | 32.34 / 0.908 | 29.40 / 0.559 |
| run16: expected embedding + DC | 32.82 / 0.911 | 29.62 / 0.557 |
| **run16: latent head one pass + DC** | **33.18 / 0.914** | 29.81 / 0.555 |
| run16: token path, posterior mean of 8 + DC | 33.14 / 0.920 | 29.94 / 0.580 |
| run16, plain acquisition-grid DC (Oct 26 standard): greedy + DC | 31.85 / 0.908 | 29.23 / 0.585 |
| run16, plain DC: expected embedding + DC | 32.47 / 0.914 | 29.56 / 0.585 |
| run16, plain DC: latent one pass + DC | 32.81 / 0.917 | 29.75 / 0.584 |
| run16, plain DC: posterior mean of 8 + DC | 33.08 / 0.923 | 30.06 / 0.612 |
| run16, plain DC: hybrid r = 0.5 + DC | 32.49 / 0.911 | 29.61 / 0.584 |
| ours v2.1 + latent head at 320, 10 more epochs at LR 1e-4 (run18, Oct 28), plain DC: greedy + DC | 32.11 / 0.910 | 29.38 / 0.585 |
| run18, plain DC: expected embedding + DC | 32.72 / 0.917 | 29.69 / 0.585 |
| run18, plain DC: latent one pass + DC | 32.99 / 0.919 | 29.84 / 0.585 |
| **run18, plain DC: posterior mean of 8 + DC** | **33.29 / 0.925** | 30.16 / 0.611 |
| run18, plain DC: hybrid r = 0.5, one sample above + DC | 32.67 / 0.913 | 29.71 / 0.585 |
| run18, plain DC: hybrid r = 0.5, mean of 8 samples above + DC | 33.16 / 0.920 | 29.94 / 0.585 |
| run18, equispaced-fraction mask (about 40 columns in total, the native budget), plain DC: greedy + DC | 32.44 / 0.914 | 29.58 / 0.590 |
| same: latent one pass + DC | 33.33 / 0.922 | 30.02 / 0.589 |
| same: posterior mean of 8 + DC | 33.57 / 0.928 | 30.32 / 0.616 |
| same: hybrid r = 0.5, one sample above + DC | 32.99 / 0.916 | 29.89 / 0.590 |
| run18, standard fastMRI equispaced mask (every 8th column + 13 ACS = 51-52 columns), plain DC: greedy + DC | 33.55 / 0.926 | 30.29 / 0.614 |
| same: latent one pass + DC | 34.36 / 0.933 | 30.65 / 0.613 |
| same: posterior mean of 8 + DC | 34.52 / 0.938 | 30.93 / 0.639 |
| same: hybrid r = 0.5, one sample above + DC | 33.96 / 0.927 | 30.51 / 0.614 |
| ceiling: tokenizer round trip of x0 + DC | 38.19 / 0.969 | 32.08 / 0.628 |
| VarNet, fastMRI leaderboard weights (12 cascades), random mask, RSS out | 34.25 / 0.909 | **35.01 / 0.930** |
| VarNet, our R=8 fine-tune (8 cascades), random mask, RSS out | 33.08 / 0.898 | 33.75 / 0.920 |
| VarNet, our R=8 fine-tune (8 cascades), equispaced mask, RSS out | 35.16 / 0.914 | 36.11 / 0.936 |

## Notes

- **run16** (Oct 25-26): run11's model (v2.1 + latent head) fine-tuned 3 epochs at the 320 crop from its 256-grid checkpoint, the sampling-mask embedding resampled from 256 to 320 columns, sens-net run320. Validation (latent path): 32.89 / 33.04 / 33.07. Eval dir `results/sense_block/suno_acc8_sensegpt320_v21asym_latent_final_net_crop320_support/` (token-path DC; LPIPS / DISTS 0.113 / 0.145 one pass, 0.107 / 0.133 posterior mean; leaderboard VarNet 0.092 / 0.122). Gap to the leaderboard VarNet vs SENSE: 1.07 dB on the old DC path. Plain-DC evaluation (`..._dcacq_net_crop320_support`, the table standard from Oct 26): greedy 31.85, latent 32.81, posterior mean of 8 33.08 (gap 1.17 dB), LPIPS / DISTS 0.094 / 0.129 (one pass) and 0.087 / 0.112 (posterior mean) against 0.092 / 0.122 for the leaderboard VarNet: perceptually level. At 320 the old resize path was worth +0.4 dB PSNR at a perceptual cost.
- **run18** (Oct 26-28): run16's model trained 10 more epochs at LR 1e-4 (cosine) at the 320 crop, GPUs 6+7, batch 8. Validation one pass by epoch: 32.76, 32.93, 32.98, 33.02, 33.10, 33.21, 33.24, 33.26, 33.28, 33.27 (run16 final 33.07): the hotter restart dips first and passes run16 at epoch 6. Full evaluation (plain DC) `results/sense_block/suno_acc8_sensegpt320_v21asym_latent_long_final_dcacq_net_crop320_support/`: LPIPS / DISTS vs SENSE greedy 0.088 / 0.111, expected embedding 0.095 / 0.129, latent one pass 0.092 / 0.127, posterior mean of 8 0.085 / 0.109, hybrid one sample 0.084 / 0.117, hybrid mean of 8 0.092 / 0.126 (leaderboard VarNet 0.092 / 0.122). Gain over run16: +0.26 greedy, +0.18 one pass, +0.21 posterior mean; gap to the leaderboard VarNet 0.96 dB (posterior mean of 8). Equispaced masks (Oct 28, `..._equispaced_40lines_...` and `..._equispaced_std_...`): equispaced-fraction mask at the native budget (39-41 columns) LPIPS / DISTS greedy 0.087 / 0.109, one pass 0.089 / 0.123, posterior mean of 8 0.083 / 0.106, hybrid 0.081 / 0.112; standard fastMRI equispaced mask (51-52 columns) greedy 0.078 / 0.101, one pass 0.079 / 0.113, posterior mean of 8 0.074 / 0.097, hybrid 0.072 / 0.102. The random training mask has 31-50 columns (mean 39) on this split.

- **ours (320 grid)** = `results/brain_sense/cart_1d/run4_crop320/.../checkpoints/best_final.pt`: GPT-L warm-started from the 256-grid
  512-token model, 3 epochs at the 320 crop (SENSE image at 320, bicubic-resized to the 512 tokenizer input, scored back at 320),
  LR 5e-5 cosine, random masks only, no DC during training. Sensitivity maps from the sens-net fine-tuned on the 320 grid
  (`results/sens_net/run320/best.pt`, agreement 0.9972 with the oracle maps). Per-epoch validation (greedy + DC): 31.27 / 31.39 / 31.45.
  Eval dir: `results/sense_block/suno_acc8_sensegpt320_final_net_crop320_support/`.
- **PS8** = posterior sampling, 8 draws, magnitude mean, then per-coil DC. **expected embedding** = softmax-weighted codebook rows decoded without sampling.
- **ceiling** = encode/decode the oracle SENSE image with the same res512 complex-pair tokenizer (320 -> 512 -> 320) + DC. It is the best this
  tokenizer can give on this grid; the transformer is 5.7 dB below it vs SENSE.
- **VarNet rows** are the fastMRI E2E-VarNet run on the native k-space with the same seeded masks and line budget, output = RSS of the predicted
  coils, cropped to 320 (`results/sense_block/suno_acc8_varnet_*_crop320/`). Equispaced = the fastMRI-style equispaced mask with the same ACS width.
- **Why the vs-RSS gap is larger than the vs-SENSE gap**: our target is the SENSE image, VarNet's is RSS. Even a perfect round trip of the
  SENSE image scores only 32.08 vs RSS on this grid, so vs RSS the comparison measures the SENSE/RSS reference mismatch more than reconstruction quality.
  The like-for-like number is vs SENSE: 32.50 (ours) vs 34.25 (leaderboard VarNet) with random masks, a 1.75 dB gap.
- Compared with the 256 grid (10 ACS / 32 lines nominal), the same recipe gains ~1.8 dB (PS8 + DC 30.68 -> 32.50 vs SENSE), mostly from the extra ACS lines and the larger grid.
