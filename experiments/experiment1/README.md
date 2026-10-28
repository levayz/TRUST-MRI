# Experiment 1: Entropy Calibration & Acquisition Utility

This directory contains the full pipeline for validating token entropy as an
uncertainty signal for active MRI k-space acquisition.

## Quick start

```bash
cd /fast/lev/TRUST-MRI

# Full pipeline (all stages)
bash experiments/experiment1/run_experiment1.sh --run_dir results/exp1

# Entropy validation only (Stages 6 + analysis)
bash experiments/experiment1/run_experiment1.sh \
    --run_dir results/exp1_entropy --stages 6
python experiments/experiment1/analyze_entropy_validation.py \
    --run_dir results/exp1_entropy

# Smoke test (1 scan, fast)
bash experiments/experiment1/run_experiment1.sh \
    --run_dir results/exp_smoke --stages 6 --num_slices 1
```

## Pipeline stages

| Stage | Name | Description |
|-------|------|-------------|
| 1 | Select slices | Pick 1 center slice × 100 scans per (dataset, R) combo |
| 2 | Normal trajectories | Run LES/GEO acquisition on selected slices; save per-step reconstructions |
| 3 | Candidate counterfactuals | Score every unsampled line at 3 checkpoints (early/mid/late) |
| 3b | Oracle active sampling | Greedy oracle that always picks the best line by MSE or SSFD |
| 4 | Offline analyses | Read-only post-processing; produces all tables and intermediate CSVs |
| 5 | Paper tables & plots | Assemble final figures from Stage 4 outputs |
| 6 | Entropy validation | Generate data for Experiments A and B (see below) |

Stages 4 and 5 consume outputs from earlier stages. Stage 4's `a7` analysis
(`analyze_entropy_validation.py`) requires Stage 6 to have completed first.

## Options for `run_experiment1.sh`

| Option | Default | Description |
|--------|---------|-------------|
| `--run_dir PATH` | `results/exp1` | Root output directory |
| `--device DEVICE` | `cuda` | PyTorch device string |
| `--stages LIST` | `1,2,3b,4,5,6` | Comma-separated list of stages to run |
| `--num_slices N` | 100 | Limit slices per combo (use a small number for smoke tests) |
| `--jobs N` | 1 | Max parallel jobs within a stage |
| `--resume` | off | Skip already-completed slices/steps instead of re-running |
| `--python PATH` | `/opt/miniconda3/bin/python` | Python executable |

## Scan and slice selection

**Stages 1–3b** use `select_slices.py` / `select_slices_center_out.py`:
- Reads the first 100 scans from a fixed split file (`fastmri/splits/exp1_{dataset}_100scans.txt`)
- Split files were generated once with `seed=42` from the full validation set
  (100/460 brain scans, 100/199 knee scans)
- `--num_slices N` limits how many slices per (dataset, R, method) combo are processed
- One center slice is selected per scan

**Stage 6** (entropy validation) uses `select_slices_fixed_fractions()`:
- Reads the same split files but takes only the first `num_scans=50` scans
- Selects 3 slices per scan at volume positions L/4, L/2, and 3L/4
- → 150 slices per (dataset, R) combination

## Entropy validation experiments (Stage 6)

Stage 6 implements two complementary experiments that validate token entropy
as a calibrated uncertainty signal.

### Experiment A — Random-mask diagnostic (`run_random_mask_diagnostic.py`)

**Goal:** Show that patch entropy predicts reconstruction error on random
(non-entropy-guided) masks, eliminating circularity concerns.

**Method:**
- For each scan/slice: sample `num_seeds=5` independent random orderings of
  the outer k-space columns (columns outside the center mask)
- For each seed, incrementally reveal `t = 1 … B` columns and record
  per-patch entropy and per-patch reconstruction MSE at every step
- Pool all (slice, seed, step) observations, bin patches into 5 entropy
  quintiles (Q1=low … Q5=high), and report mean normalized patch MSE per bin

**Expected result:** Monotone increase in patch MSE from Q1 to Q5, showing
that high-entropy patches are harder to reconstruct.

### Experiment B — Acquisition curve (`run_entropy_acquisition.py`)

**Goal:** Show that entropy-guided acquisition (LES or GEO) reduces both token
entropy and reconstruction error as more k-space lines are collected.

**Method:**
- Start from a center-only mask (step 0), add one line per step using either
  LES (logit-based, fast) or GEO (gradient-based, slower)
- Record mean entropy and MSE at every step, normalized by their step-0 values
- Plot both quantities vs normalized acquisition progress τ = t/B (so R=8 and
  R=16 are comparable on the same x-axis)

**Expected result:** Both normalized entropy and normalized MSE decrease
monotonically toward B, confirming that the entropy signal drives the model
toward lower-uncertainty, higher-quality reconstructions.

### Acquisition budgets

| Setting | Image width W | Budget B = W//R − ⌊W×0.04⌋ |
|---------|--------------|------------------------------|
| Brain R=8 | 256 | 22 |
| Brain R=16 | 256 | 6 |
| Knee R=8 | 320 | 28 |
| Knee R=16 | 320 | 8 |

### Output figure

`analyze_entropy_validation.py` produces a 2×2 figure:

```
(a) Knee   — entropy-bin patch MSE (Exp A)
(b) Brain  — entropy-bin patch MSE (Exp A)
(c) Knee   — normalized entropy + MSE vs τ (Exp B)
(d) Brain  — normalized entropy + MSE vs τ (Exp B)
```

Visual encoding: color = R value (blue=8, orange=16); linestyle = solid/dotted
for LES/GEO entropy curves, dashed for MSE curve (LES only).

## Output directory structure

```
<run_dir>/
├── metadata/
│   └── selected_slices_{dataset}_R{R}.json          # Stage 1
├── trajectories/
│   └── dataset={d}/R={r}/method={m}/
│       └── sample_id={id}/                           # Stage 2
├── candidate_counterfactuals/
│   └── dataset={d}/R={r}/base_method={m}/           # Stage 3
├── oracle_active_sampling/
│   └── dataset={d}/R={r}/objective={obj}/           # Stage 3b
├── random_mask_diagnostic/
│   └── dataset={d}/R={r}/
│       ├── metadata.parquet                          # Stage 6 Exp A
│       └── sample_id={id}/seed={s}/
│           ├── entropy_map_t{t:03d}.npy
│           ├── patch_error_map_t{t:03d}.npy
│           └── mask_t{t:03d}.npy
├── entropy_acquisition/
│   └── dataset={d}/R={r}/method={m}/
│       ├── metadata.parquet                          # Stage 6 Exp B
│       └── sample_id={id}/
│           ├── entropy_map_t{t:03d}.npy
│           ├── patch_error_map_t{t:03d}.npy
│           ├── mask_t{t:03d}.npy
│           └── kspace_scores_t{t:03d}.npy
├── results/
│   ├── plots/
│   │   ├── figure_entropy_validation.png             # Stage 4/6 analysis
│   │   └── *.png                                     # Stage 5 paper figures
│   └── tables/
│       ├── entropy_validation_summary.csv
│       └── *.csv
└── logs/
    └── *.log                                         # one log file per job
```

## Analysis scripts (Stage 4)

| Script | Analysis |
|--------|----------|
| `analyze_entropy_token_error.py` | Per-token entropy vs token prediction error |
| `analyze_entropy_patch_error.py` | Patch-level entropy vs patch reconstruction error |
| `analyze_calibration.py` | Entropy calibration curves |
| `analyze_candidate_line_utility.py` | Candidate line utility (counterfactual analysis) |
| `analyze_oracle_active_sampling.py` | Oracle vs heuristic acquisition comparison |
| `analyze_entropy_reconstruction_delta.py` | Δentropy vs ΔPSNR/ΔSSIM across steps |
| `analyze_entropy_validation.py` | Entropy validation figure (requires Stage 6) |

## Checkpoints and data paths

All paths in `config.yaml` are relative to the TRUST-MRI repository root.
Override per-run with CLI flags:

```bash
bash experiments/experiment1/run_experiment1.sh \
    --run_dir results/custom \
    --stages 6
```

Override checkpoint or data path for a single script:

```bash
python experiments/experiment1/run_entropy_acquisition.py \
    --config experiments/experiment1/config.yaml \
    --run_dir results/exp1_entropy \
    --dataset brain --acceleration 16 --method LES \
    --checkpoint path/to/custom.pt \
    --data_path /data/fastmri/brain/val
```
