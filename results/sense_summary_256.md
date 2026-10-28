# SENSE reconstruction on the 256x256 grid: ceilings, transformer variants, VarNet (suno, x8)

Columns: PSNR / SSIM. "vs SENSE" = against the oracle-map SENSE image (reconstruction x oracle support). "vs RSS" = against the RSS of the fully sampled coils (no support).

| # | row | vs SENSE | vs RSS |
|---|---|---|---|
| 1 | zero-filled SENSE image (net maps) | 23.57 / 0.682 | 23.31 / 0.557 |
| | **Tokenizer ceilings (plain round trip of the ground truth, no DC)** | | |
| 2 | pretrained MedITok, complex pair at 256 | 29.48 / 0.714 | 29.11 / 0.790 |
| 3 | SENSE-finetuned, complex pair, trained with 256 inputs; tokenizer input 256, scored at 256 | 34.81 / 0.936 | 31.36 / 0.685 |
| 4 | SENSE-finetuned, complex pair, trained with 512 inputs (2x upsample of the 256 slice); tokenizer input 512, scored at 256 | 39.54 / 0.973 | 33.01 / 0.708 |
| | **Pipeline ceilings (round trip + per-coil DC with the final estimated maps)** | | |
| 5 | pretrained MedITok at 256 | 32.14 / 0.897 | 30.36 / 0.735 |
| 6 | SENSE-finetuned 256 (input 256, scored at 256) | 35.31 / 0.934 | 32.02 / 0.734 |
| 7 | SENSE-finetuned 512 (input 512, scored at 256) | 39.54 / 0.975 | 33.81 / 0.766 |
| | **Transformer (GPT-L) variants, net maps** | | |
| 8 | zero-shot single-coil GPT + pretrained tokenizer (256), greedy + DC | 25.16 / 0.728 | 24.78 / 0.616 |
| 9 | same, posterior mean of 8 samples + DC | 25.70 / 0.750 | 25.38 / 0.666 |
| 10 | SENSE GPT trained on SENSE images with the pretrained tokenizer (256), greedy + DC (ablation) | 27.24 / 0.799 | 26.50 / 0.652 |
| 11 | same, posterior mean of 8 + DC | 28.22 / 0.824 | 27.45 / 0.691 |
| 12 | SENSE GPT 256 (tokenizer of row 3), greedy + DC | 28.08 / 0.829 | 27.11 / 0.650 |
| 13 | SENSE GPT 256, expected-embedding decode + DC | 28.34 / 0.836 | 27.31 / 0.654 |
| 14 | SENSE GPT 256, posterior mean of 8 + DC | 28.57 / 0.837 | 27.59 / 0.669 |
| 15 | SENSE GPT 512 (tokenizer of row 4), greedy, no DC | 28.74 / 0.838 | 27.31 / 0.575 |
| 16 | SENSE GPT 512, greedy + DC | 30.02 / 0.862 | 28.50 / 0.657 |
| 17 | SENSE GPT 512, expected-embedding decode + DC | 30.39 / 0.868 | 28.75 / 0.660 |
| 18 | SENSE GPT 512, posterior mean of 8 + DC | 30.68 / 0.876 | 29.09 / 0.678 |
| 19 | SENSE GPT 512 + latent-regression head (3 more epochs), greedy tokens + DC | 30.14 / 0.864 | 28.58 / 0.658 |
| 20 | **same model, latent head decoded without quantisation + DC, one pass** | **30.92 / 0.874** | 29.13 / 0.661 |
| 21 | same model, token path: posterior mean of 8 sampled token reconstructions + DC | 30.79 / 0.878 | **29.15 / 0.679** |
| 22 | SENSE GPT 512 + image-domain loss (1 epoch), posterior mean + DC | 30.04 / 0.862 | not scored |
| 23 | SENSE GPT 512 + teacher-student refinement (2 epochs), refined pass greedy + DC | 29.19 / 0.806 (training-time validation) | not scored |
| 24 | v2: token input + mask and timestep conditioning (1 epoch from the 512 model), greedy + DC | 29.88 / 0.821 (training-time validation) | not scored |
| 25 | two-view teacher-student refinement with a visible timestep (2 epochs), refined pass greedy + DC | 29.98 / 0.822 (pass 1 alone 30.00; training-time validation) | not scored |
| 26 | v2.1: continuous encoder-latent input + asymmetric parallel head + mask/timestep conditioning (3 epochs from the 512 model), greedy + DC | 30.47 / 0.870 | 28.82 / 0.662 |
| 27 | v2.1, expected-embedding decode + DC | 30.80 / 0.874 | 29.04 / 0.664 |
| 28 | v2.1, posterior mean of 8 + DC | 30.98 / 0.881 | 29.29 / 0.681 |
| 29 | v2.1 + latent-regression head (3 more epochs), greedy tokens + DC | 30.57 / 0.872 | 28.88 / 0.663 |
| 30 | **same model, latent head decoded without quantisation + DC, one pass (best transformer)** | **31.27 / 0.879** | 29.36 / 0.665 |
| 31 | same model, token path: posterior mean of 8 sampled token reconstructions + DC | 31.12 / 0.883 | **29.37 / 0.682** |
| | **E2E-VarNet** | | |
| 32 | VarNet fine-tuned on this 256-grid setup (matched), RSS output | 31.69 / 0.877 | **31.83 / 0.903** |
| 33 | same model, SENSE-combined output (oracle maps) | **32.06 / 0.892** | 30.14 / 0.691 |
| 33a | VarNet of row 32 tested with an equispaced mask of the same nominal budget (equispaced-fraction, 10 ACS, 31-32 columns), RSS output | 32.16 / 0.884 | 32.35 / 0.911 |
| 33b | same, SENSE-combined output (oracle maps) | 32.61 / 0.899 | 30.55 / 0.699 |
| 33c | VarNet of row 32 tested with the standard fastMRI equispaced mask at R=8 (every 8th column + 10 ACS = 40-41 columns), RSS output | 32.90 / 0.894 | 33.21 / 0.922 |
| 33d | same, SENSE-combined output (oracle maps) | 33.46 / 0.910 | 31.17 / 0.711 |
| | **v2.1 follow-ups (Oct 25)** | | |
| 34 | v2.1 continued 10 more epochs at LR 1e-4 (run12, 13 epochs in total), greedy tokens + DC | 30.63 / 0.873 | 28.94 / 0.664 |
| 35 | same model, posterior mean of 8 + DC | 31.22 / 0.885 | 29.44 / 0.683 |
| 36 | v2.1 with the 16-step interleaved Re/Im head (run13, 3 epochs), greedy tokens + DC | 30.34 / 0.868 | 28.73 / 0.660 |
| 37 | same model, posterior mean of 8 + DC | 30.95 / 0.881 | 29.27 / 0.681 |
| 38 | v2.1 + fused latent head (run14: residual MLP reading the hidden state, the token head's chain state and the codes it committed to; 3 epochs from row 30's model), latent path one pass + DC | 31.31 / 0.880 | 29.38 / 0.665 |
| 39 | same model, token path: posterior mean of 8 sampled token reconstructions + DC | 31.17 / 0.884 | 29.41 / 0.683 |
| 40 | v2.1 + latent head with a residual MLP on the hidden state only (run15, the control for row 38; 3 epochs from row 30's model), latent path one pass + DC | 31.32 / 0.880 | 29.40 / 0.665 |
| 41 | same model, token path: posterior mean of 8 sampled token reconstructions + DC | 31.17 / 0.884 | 29.41 / 0.682 |
| | **Mask transfer (Oct 26): the row-30 model (trained on random masks) tested with equispaced masks** | | |
| 42 | equispaced-fraction mask, same nominal budget (10 ACS, 31-32 columns), latent path one pass + DC | 31.68 / 0.884 | 29.65 / 0.670 |
| 43 | same model, token path: posterior mean of 8 sampled token reconstructions + DC | 31.48 / 0.888 | 29.64 / 0.688 |
| 44 | standard fastMRI equispaced mask at R=8 (40-41 columns), latent path one pass + DC | 32.44 / 0.897 | 30.20 / 0.687 |
| 45 | same model, token path: posterior mean of 8 sampled token reconstructions + DC | 32.17 / 0.900 | 30.16 / 0.704 |
| | **Hybrid decoding (Oct 26): the row-30 model, latent one pass below a k-space radius r, a token decoding above it, merged in k-space, then DC** | | |
| 46 | r = 0.5, high frequencies from the complex mean of 4 posterior samples | 31.28 / 0.884 | 29.45 / 0.667 |
| 47 | r = 0.3, high frequencies from the complex mean of 4 posterior samples | 31.24 / 0.883 | 29.42 / 0.666 |
| 48 | r = 0.5, high frequencies from one posterior sample | 30.99 / 0.876 | 29.27 / 0.662 |
| 49 | r = 0.3, high frequencies from one posterior sample | 30.74 / 0.870 | 29.12 / 0.658 |
| 50 | r = 0.3, high frequencies from the greedy tokens | 30.97 / 0.879 | 29.24 / 0.663 |
| | **Plain acquisition-grid DC (Oct 26, the table standard from here on; `--dc-path acq`)** | | |
| 51 | SENSE GPT 512 (row 16's model), greedy + DC | 29.76 / 0.862 | 28.37 / 0.669 |
| 52 | same, posterior mean of 8 + DC | 30.81 / 0.881 | 29.27 / 0.700 |
| 53 | v2.1 (row 26's model), greedy + DC | 30.28 / 0.871 | 28.75 / 0.675 |
| 54 | same, posterior mean of 8 + DC | 31.16 / 0.887 | 29.51 / 0.703 |
| 55 | v2.1 + latent head (row 30's model), greedy tokens + DC | 30.37 / 0.872 | 28.80 / 0.676 |
| 56 | same, expected-embedding decode + DC | 30.87 / 0.882 | 29.13 / 0.682 |
| 57 | same, token path: posterior mean of 8 + DC | 31.29 / 0.889 | 29.59 / 0.704 |
| 58 | **same, latent head one pass + DC** | 31.35 / 0.886 | 29.48 / 0.683 |
| 59 | same, hybrid: latent below k-space radius 0.5, one sample above, + DC | 30.98 / 0.876 | 29.27 / 0.679 |
| 60 | same, hybrid r = 0.7 + DC | 31.16 / 0.881 | 29.37 / 0.681 |
| 61 | row 30's model, equispaced mask (32 columns): greedy + DC | 30.86 / 0.879 | 29.18 / 0.684 |
| 62 | same: posterior mean of 8 + DC | 31.72 / 0.894 | 29.91 / 0.711 |
| 63 | same: latent one pass + DC | 31.83 / 0.892 | 29.83 / 0.690 |
| 64 | same: hybrid r = 0.5 + DC | 31.44 / 0.883 | 29.61 / 0.687 |
| 65 | row 30's model, fastMRI equispaced mask (41 columns): greedy + DC | 31.86 / 0.895 | 29.94 / 0.706 |
| 66 | same: posterior mean of 8 + DC | 32.56 / 0.907 | 30.55 / 0.731 |
| 67 | same: latent one pass + DC | 32.74 / 0.906 | 30.49 / 0.711 |
| 68 | same: hybrid r = 0.5 + DC | 32.30 / 0.897 | 30.26 / 0.708 |
| 69 | v2.1 + latent head + privileged-teacher distillation (row 30's model, 3 more epochs with an EMA teacher at x2-x4; run17): greedy + DC | 30.29 / 0.872 | 28.75 / 0.676 |
| 70 | same: posterior mean of 8 + DC | 31.33 / 0.890 | 29.61 / 0.705 |
| 71 | same: latent one pass + DC | 31.40 / 0.887 | 29.51 / 0.684 |
| 72 | same: hybrid r = 0.5 + DC | 31.03 / 0.877 | 29.30 / 0.679 |

## Notes

**Data and measurement.** fastMRI brain multicoil, suno split (15 volumes of `multicoil_val`, 240 slices). Per slice the native k-space is inverse-FFT'd, centre-cropped to the ESPIRiT map grid (320x320) and then to 256x256; the measurement is the k-space of that 256 image under a random 1D Cartesian mask, cf 0.04 (10 ACS columns), R=8 (32 columns nominal = 256/8; the random mask realises 23-39 columns per volume, mean 30.3 over the 15 suno volumes), seeded by the filename, identical for every row. Every row measures and is scored on this 256 grid; the pretrained VarNets evaluated on their native grid (different measurement) are kept out of this table and listed in results/sense_table.md.

**References.** SENSE = sum_c conj(S_c) x_c / sum_c |S_c|^2 with the oracle sigpy ESPIRiT maps, zero outside the ESPIRiT support (~35-40% of pixels). RSS = sqrt(sum_c |x_c|^2) of the fully sampled 256 coils; brighter than the SENSE image wherever the maps do not fully explain the coils, with background noise. The two columns are different targets; compare within a column. SENSE-type outputs score higher vs SENSE, RSS-type outputs vs RSS. Measured mismatch (Oct 26): the SENSE ground truth scored vs RSS gives 34.83 dB (27-39 per volume), the RSS image with the support applied scored vs SENSE 41.97 dB; RSS carries 2.2 percent of its energy outside the support and is 2 percent brighter inside. 34.8 dB is therefore the ceiling of the vs-RSS column for SENSE-target reconstructions.

**Support.** For the "vs SENSE" column every reconstruction is multiplied by the oracle support before scoring (the support is the only oracle information used; maps come from the sensitivity net). Without it PSNR changes by <0.1 dB and SSIM drops (e.g. row 18: 30.62 / 0.831), because net maps have no support and DC puts measured noise where the reference is zero. The "vs RSS" column never uses the support.

**Metrics.** PSNR per volume (data range = ground-truth volume range), SSIM per slice averaged, both via `autoregressive/utils/metrics.py`, images in physical units; no percentile clipping.

**Data consistency path (changed Oct 26).** For the 512-token models the "+ DC" arms up to row 50 apply the per-coil DC on the 256 grid inside a resize round trip (downsample from 512, DC, bicubic upsample, antialiased downsample for scoring). The resize after DC is a low-pass: on the row-30 model it costs 0.08 dB and 0.025 LPIPS on the latent path and gains 0.2 dB on the greedy path at a large perceptual cost (greedy LPIPS 0.146 vs 0.122). From row 51 on, and in results/advisor_summary_2026-10-26.md, every "+ DC" arm of a 512-token model applies the plain DC on the 256 grid to the decoded image (`eval_sense_block.py --dc-path acq`); 256-token rows have no resize and are unaffected. Plain-DC numbers for the row-30 model: greedy 30.37 / 0.872, posterior mean of 8 31.29 / 0.889, latent one pass 31.35 / 0.886 (vs SENSE); LPIPS / DISTS 0.122 / 0.136, 0.122 / 0.139, 0.131 / 0.154 against VarNet's 0.124 / 0.147 (dir `results/sense_block/suno_acc8_sensegpt512_v21asym_latent_final_dcacq_net_support/`).

Rows 51-72 are the plain-DC numbers (perceptual metrics: row 51: LPIPS 0.130 / DISTS 0.141; row 52: LPIPS 0.128 / DISTS 0.145; row 53: LPIPS 0.124 / DISTS 0.137; row 54: LPIPS 0.123 / DISTS 0.141; row 55: LPIPS 0.122 / DISTS 0.136; row 56: LPIPS 0.133 / DISTS 0.156; row 57: LPIPS 0.122 / DISTS 0.139; row 58: LPIPS 0.131 / DISTS 0.154; row 59: LPIPS 0.115 / DISTS 0.131; row 60: LPIPS 0.118 / DISTS 0.139; row 61: LPIPS 0.118 / DISTS 0.133; row 62: LPIPS 0.116 / DISTS 0.135; row 63: LPIPS 0.125 / DISTS 0.149; row 64: LPIPS 0.111 / DISTS 0.127; row 65: LPIPS 0.106 / DISTS 0.124; row 66: LPIPS 0.104 / DISTS 0.124; row 67: LPIPS 0.110 / DISTS 0.137; row 68: LPIPS 0.100 / DISTS 0.117; row 69: LPIPS 0.123 / DISTS 0.138; row 70: LPIPS 0.121 / DISTS 0.139; row 71: LPIPS 0.130 / DISTS 0.154; row 72: LPIPS 0.114 / DISTS 0.130; VarNet SENSE output 0.124 / 0.147).

**Resolution.** Every metric in this file is computed on the 256x256 grid. "512" always refers to the tokenizer's input size: the 256 slice is bicubic-upsampled to 512 before the tokenizer/GPT and the decoded output is bilinear-downsampled to 256 before DC and scoring.

**Tokenizers.** All three take the complex SENSE image divided by its max magnitude and tokenise the real and imaginary parts as two separate grayscale images (8 product codebooks x 4096 entries each). Row 2: the pretrained MedITok. Row 3: `brain_mc_sense_cplxpair_dist_ep8`, fine-tuned on 256x256 SENSE slices (16x16 = 256 tokens per part). Row 4: `brain_mc_sense_cplxpair_res512_ep8` (epoch-5 checkpoint), fine-tuned on the same 256 slices bicubic-upsampled to 512x512 (32x32 = 1024 tokens per part), output bilinear-downsampled to 256. No MRI data exist at 512; it is a token-budget change.

**Ceilings.** "Tokenizer ceiling" (rows 2-4) = encode-decode of the ground-truth SENSE image, nothing else, no support, no DC; identical to the MediTok_official upsample-curve numbers where both exist (34.81 at 256). "Pipeline ceiling" (rows 5-7) = the same round trip followed by the support and the per-coil DC with the final estimated maps, i.e. what a perfect token prediction would score in our pipeline. For the 512 tokenizer DC adds nothing (its round-trip error is already below the DC correction).

**Sensitivity maps.** Rows 8-31 use maps from our trained sensitivity U-Net (`results/sens_net/run1/best.pt`, agreement 0.995 with the oracle maps); replacing them by oracle maps changes row 18 by <0.1 dB. VarNet rows use VarNet's own sensitivity net inside the network; "SENSE-combined output" means its predicted coil images combined with the oracle maps, "RSS output" its native RSS.

**Transformer rows.** GPT-L (426M) predicts the clean tokens from the tokens of the zero-filled SENSE image (cross-entropy), warm-started from the single-coil brain GPT. "greedy" = argmax tokens; "expected-embedding" = decode the softmax-weighted average of codebook vectors; "posterior mean of 8" = 8 sampled reconstructions (top-k 4096, T=1), each with DC, magnitude-averaged. "+ DC" = decode -> (downsample to 256 for the 512 model) -> expand with the maps -> replace acquired k-space lines per coil -> combine. Rows 22 and 23 are negative results (image loss through the frozen decoder; refinement pass on the model's own DC'd draft); both started from the row-16/18 checkpoint. Rows 19-21: the 512 model continued for 3 epochs (LR 5e-5) with an extra linear head that regresses the tokenizer's continuous 64-d latent (unit-norm 8-d chunks, L2 loss weighted 5x next to the token CE); at inference the predicted latent is decoded directly by the tokenizer decoder, no sampling. One pass (row 20) beats the 8-draw posterior mean of the token-only model (row 18) and the PS of the same model vs SENSE; vs RSS they tie. Rows 10-11 (ablation, Oct 20): the same 256-token recipe as rows 12-14 but with the pretrained MedITok, i.e. the tokenizer of rows 2/5; isolates training on SENSE images (+2.1 dB greedy, +2.6 dB PS over the zero-shot rows 8-9) from the tokenizer finetuning (+0.8 / +0.35 dB for 256, a further +1.9 / +2.1 dB for 512). Rows 24-28 (Oct 22-23): model versions. v2 adds the sampling mask (column vector -> linear -> added to every token) and a visible timestep embedding to the token model: no gain, and the model shrank the gates. v2.1 replaces the input token lookup by the tokenizer's continuous pre-quantisation latents (Re|Im concatenated -> MLP) and removes the Re/Im symmetry of the head's conditioning (per-stream tables, concat + projection): +0.3 dB greedy / +0.2 dB PS over v1 at the same budget, best transformer so far. Row 25: the teacher-student refinement rerun after fixing the timestep token (which the model could not see in row 23): the refine pass no longer hurts but gives no gain. Rows 29-31 (Oct 24): the latent head added to v2.1 (3 epochs from row 26's checkpoint): the two gains stack, 31.27 dB in a single pass, 0.35 dB above the same head on v1 (row 20) and 0.8 dB above the greedy token path of the same model. Rows 34-37 (Oct 25): two follow-ups on the v2.1 token model (rows 26-28). Rows 34-35: run10's checkpoint trained for 10 more epochs with a hotter cosine schedule (peak LR 1e-4): +0.16 dB greedy, +0.24 dB posterior mean over rows 26/28, flat from epoch 9, so the 3-epoch budget cost about 0.2 dB. Rows 36-37: the token head's chain interleaved as Re_0, Im_0, Re_1, ... (16 steps, per-stream tables, step embeddings) instead of predicting Re_i and Im_i jointly at each of 8 steps: token accuracy identical to the 8-step head (teacher-forced 16 percent, generated 14 percent), 0.13 dB lower greedy, equal posterior mean; the 8-step head stays. Rows 38-39 (Oct 25): the fused latent head, which also sees what the token head committed to: +0.04 dB over row 30 (31.31 vs 31.27), within noise; the token head's codes carry no information the body's hidden state does not already have. Rows 40-41: the control with the same MLP reading the hidden state alone scores the same (31.32), so the +0.05 dB of both is readout capacity plus 3 more epochs, not the token head. The linear latent head stays. Rows 46-50 (Oct 26, hybrid decoding; `results/sense_block/suno_acc8_sensegpt512_v21asym_latent_final_hybrid_net_support/`): the latent head's error is in the high frequencies (it regresses a conditional mean), the token path's in the low ones (quantisation). Merging them in k-space with a cosine-ramp low-pass of normalised radius r keeps the latent path's PSNR and restores the texture: LPIPS / DISTS go from 0.156 / 0.186 (row 30) to 0.124 / 0.145 (row 46, VarNet's SENSE output: 32.06 dB, 0.124 / 0.147), 0.121 / 0.140 (row 47), 0.115 / 0.131 (row 48) and 0.112 / 0.119 (row 49), i.e. rows 48-49 beat VarNet perceptually at 1.1-1.3 dB lower PSNR. Rows 42-45 (Oct 26): the same model under the equispaced masks of rows 33a-33d (greedy token path: 31.00 and 31.86): +0.4 dB at the same nominal budget and +1.2 dB with the 41-column fastMRI mask, against +0.55 and +1.4 dB for the fine-tuned VarNet, so the gap to VarNet's SENSE output is 0.8 / 0.9 / 1.0 dB under the three masks.

**VarNet.** Rows 32-33: the 12-cascade fastMRI leaderboard E2E-VarNet fine-tuned 4 epochs on exactly the 256-grid measurement above (SSIM loss on RSS), converged. The pretrained checkpoints (your 8-cascade R=8 model, the leaderboard 12-cascade model) collapse to 21-23 dB on this 256-grid measurement and are not shown; their native-grid numbers are in results/sense_table.md. Rows 33a-33b (Oct 25): the same fine-tuned VarNet (random masks in training) evaluated with fastMRI's equispaced-fraction mask at the same nominal budget (10 ACS, 31-32 columns vs the random masks' realised 23-39, mean 30.3): +0.5 dB; `results/sense_block/suno_acc8_varnet256_r8_equispaced_32lines/`. Rows 33c-33d: the standard fastMRI equispaced mask at R=8 (every 8th column plus the 10 ACS columns = 40-41 columns, i.e. a 28 percent larger budget): +1.4 dB; `results/sense_block/suno_acc8_varnet256_r8_equispaced_41lines/`.

Sources: `results/sense_block/<run>/summary.json`; regenerate the full run table with `bash scripts/make_sense_table.sh`.
