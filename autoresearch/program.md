# autoresearch for TRUST-MRI

An AI agent runs its own research loop on the SENSE GPT of TRUST-MRI: edit the model, the training recipe or the sampler,
train for a fixed budget on one GPU, score, keep or discard, repeat. The human edits this file; the agent edits
`model.py`, `train.py` and `sample.py`.

## Starting a campaign (the human does only this)

Open a tmux window on the GPU machine (so the session survives a closed laptop), go to the main checkout, and start the
agent with permissions off, naming the GPU and the campaign tag in the first sentence:

    tmux new -s ar
    cd /fast/lev/TRUST-MRI && claude --dangerously-skip-permissions
    > Have a look at autoresearch/program.md and let's kick off a new campaign on GPUs 0 and 1, tag oct30. Do the setup first.

One agent, one campaign, both GPUs on every experiment (data parallel, which makes each run shorter). Detach with Ctrl-b d,
reattach with `tmux attach -t ar`. Everything below is the agent's job. For a status report, type "status report" into the
session: the agent answers with the results table, what it kept and why, what failed, and what it would try next.

## Setup (the agent)

1. **Take the run tag the user gave** (or make one from today's date, e.g. `oct30`); the branch `autoresearch/<tag>` must
   not exist yet (`git branch --list 'autoresearch/*'`). Tell the user the tag.
2. **Commit `autoresearch/` on the current branch if it has uncommitted changes** (`git status --porcelain autoresearch/`);
   commit only files under `autoresearch/` and `.gitignore`, nothing else of the user's working tree.
3. **Create the campaign worktree**: `bash autoresearch/setup_campaign.sh <tag>`. It makes the branch and a worktree at
   `.worktrees/<tag>` inside the main checkout, links the data and weights that live outside git, and writes the header of
   `results.tsv`. From here on work ONLY inside `.worktrees/<tag>/autoresearch/`; the main checkout hosts the user's own
   training runs and is never edited or reset.
4. **Read the in-scope files** in `.worktrees/<tag>/autoresearch/`:
   - `program.md` (this file)
   - `prepare.py`: the frozen harness: problem constants, data, frozen networks, budget, evaluation, summary. Do not modify.
   - `model.py`: the model. Editable.
   - `train.py`: hyperparameters, optimizer, losses, training loop. Editable.
   - `sample.py`: inference: how the trained model produces the scored reconstruction. Editable.
   The other scripts (`run.sh`, `gpu_wait.sh`, `gpu_hold.py`, `build_cond_cache.py`, `check_harness.py`, `calibrate.sh`,
   `setup_campaign.sh`) are harness tools: use them, do not modify them.
5. **Check the GPUs**: `nvidia-smi`. The experiment GPUs are the ones the user named, comma-separated in `AR_GPU` for
   every run (e.g. `AR_GPU=0,1`; `run.sh` starts one process per GPU and splits the batch over them). `run.sh` waits while
   another process holds one of them. Never use a GPU the user did not name.
6. **Keep the GPUs reserved between runs**: for each GPU, if `kill -0 $(cat /fast/lev/gpu_hold/gpu_hold_<gpu>.pid)` fails,
   start its holder from the worktree's `autoresearch/`:
   `nohup /opt/miniconda3/envs/meditok/bin/python -u gpu_hold.py <gpu> >> /fast/lev/gpu_hold/gpu_hold_<gpu>.log 2>&1 &`.
   It occupies the GPU's memory whenever none of our runs is on it and frees it when `run.sh` asks, so another user's job
   queue cannot take the GPU in the gaps. Leave it running for the whole campaign; never start a second holder for a GPU.
7. **Confirm and go.** The first run is always the baseline, the files as they are.

## Experimentation

One experiment = one GPU, from scratch, `prepare.SAMPLE_BUDGET` training samples (steps x batch), then evaluation on the
autoresearch validation split. Launch it from `.worktrees/<tag>/autoresearch/`:

    AR_GPU=0,1 bash run.sh > run.log 2>&1

**What you CAN do:** modify `model.py`, `train.py`, `sample.py`. Everything in them is fair game: architecture, model
size, heads, losses, optimizer, schedule, batch size, augmentation, the decoding path, sampling, data consistency usage.

**What you CANNOT do:**
- Modify `prepare.py`, `run.sh`, `gpu_wait.sh`, `gpu_hold.py`, `build_cond_cache.py`, `splits/`, the caches under
  `/fast/lev/datasets`, or any file outside `autoresearch/`. The data, the tokenizer, the sensitivity net, the DC operator,
  the metric and the budget are the ground truth.
- Set any `AR_*` environment variable (those are calibration hooks for the human). Change the constants in the files.
- Touch the suno test split (`prepare.SUNO_SPLIT`): it is reserved for the human's confirmation runs.
- Install packages. Only what the `meditok` environment already has.
- Train, replace or bypass the sensitivity net. It is trained, frozen and good enough; the maps it gives are part of the
  problem. The same holds for the tokenizer: the model sees its latents and the decoder renders the output, both frozen.

**The goal: the highest `val_psnr`.** It is the mean over the 15 validation volumes of the volume PSNR of the magnitude
reconstruction after per-coil data consistency. `val_ssim` is the secondary metric; report it, do not optimize for it alone.

**Budget.** `SAMPLE_BUDGET` samples, and the training loop must finish within `MAX_MINUTES` of wall clock (a run over it is a
failure: status `timeout`). Bigger models and bigger batches are allowed as long as they fit; VRAM is a soft constraint,
stay under about 40 GB. Note that the batch size trades steps against gradient quality at a fixed number of samples.

**Inputs.** By default (`CACHED_INPUTS = True` in `train.py`) the trainer reads the conditioning from a cache: for every
training slice the tokenizer latent of its zero-filled SENSE input under one fixed random mask, the mask columns, the
ground-truth image, tokens and latents (`prepare.cached_train_loader` / `prepare.cached_inputs`). The budget is below one
epoch, so no slice is seen twice and the fixed mask costs nothing; the baseline runs at about 2.5 steps/s. The slow path
(`CACHED_INPUTS = False`: `prepare.train_loader` / `prepare.sense_inputs`) encodes the input on the GPU each step, about
3x slower, and is the way to try ideas that change the input itself (extra revealed k-space, other masks, input noise).
A non-finite loss or gradient drops that update (`skipped_updates` in the summary); more than 20 of them end the run with
status `diverged`.

**Noise floor.** The baseline recipe was run three times with different seeds at this budget with a global batch of 8:
28.50, 28.49 and 28.44 dB (token path 27.40, 27.54 and 27.41 dB). The same recipe on two GPUs lands on the same score
(seed 2: 28.44 on one GPU, 28.44 on two). The file now sets the global batch to 16 (8 per GPU, the faster setting on two
GPUs); its score is expected to be the same within the noise, and the first experiment, the untouched baseline, shows what
it is. A change must beat the better of 28.50 dB and that baseline to count as an improvement; a difference below 0.1 dB is
noise. An improvement inside the noise is not an improvement.

**Simplicity criterion.** All else being equal, simpler is better. A small gain that adds ugly complexity is not worth it;
removing something and getting equal or better results is a great outcome. A gain of 0.05 dB from deleting code: keep. A
gain of 0.05 dB from 40 lines of hacks: probably not.

**Context.** This proxy stands in for the full-scale line (GPT-L, 24 layers, 1024 wide, trained for epochs on 2 GPUs,
31.4 dB on the test split; VarNet256 at the same setting is 31.7 dB). Ideas kept here get ported to the full-scale
recipe by the human and confirmed there. Known full-scale rankings: the latent head is worth about 0.1 dB and its
decoded path beats the token path by 0.7 dB; the asymmetric head beats the interleaved head by 0.1 dB; distillation
from a privileged teacher gave nothing; an image-domain loss gave 1.0 dB after one epoch.

**Calibration (what this proxy says about known changes).** The budget is one epoch, 56,224 samples: 7,028 steps at
batch 8, about 48 min for the GPT-B baseline on one GPU (2.5 steps/s) and about 44 min on two (2.7 steps/s: with 4
samples per GPU the second GPU adds little; at 8 per GPU, a global batch of 16, two GPUs run 1.9 steps/s, about 31 min per
epoch). GPT-L (24 layers, 1024 wide) runs at half that speed and does not fit the cap at this budget. Known changes were measured at a budget of 31,000 samples (baseline
27.73 dB there): latent head off 24.03 dB (the token path alone; full scale: -0.1 dB); interleaved head 27.72 dB in 1.2x
the time (full scale: -0.1 dB; indistinguishable from the baseline); the image loss at full strength scored 27.71 dB
there, no gain, where full scale gave +1.0 dB after one epoch (a disagreement to keep in mind), and it runs at 0.9 steps/s,
2.8x the baseline time, so at full strength it does not fit the cap. The proxy sits early in training
(cross-entropy 5.5 at the end versus 4.4 for the converged full-scale line), so it amplifies whatever speeds up early
learning: the latent regression loss is worth 3.7 dB here and 0.1 dB at full scale, and the latent path leads the token
path by 1.4 dB here versus 0.7 dB at full scale. A gain that comes from the regression path alone deserves suspicion; a
gain on both paths is the real thing.

## Output format

The script ends with a summary block:

```
---
val_psnr:         26.1234
val_ssim:         0.7890
status:           ok
train_steps:      2000
samples_seen:     16000
batch_size:       8
training_seconds: 1500.3
eval_seconds:     140.2
peak_vram_mb:     21034.5
num_params_M:     111.2
arm_tokens_psnr:  25.4321
```

(`skipped_updates: N` is printed just above the block.)

`grep "^val_psnr:" run.log` extracts the metric. `arm_*` lines are the other decoding paths, scored for information.

## Logging results

Log every experiment to `results.tsv` (tab-separated, NOT comma-separated). Header and 7 columns:

```
commit	val_psnr	val_ssim	train_min	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_psnr (e.g. 26.1234), 0.0000 for crashes and timeouts
3. val_ssim, 0.0000 for crashes
4. training minutes (training_seconds / 60, one decimal)
5. peak memory in GB (peak_vram_mb / 1024, one decimal)
6. status: `keep`, `discard`, `crash`, `timeout`, `diverged`
7. short description of what the experiment tried

## The experiment loop

LOOP FOREVER:

1. Look at the git state: the current branch and commit.
2. Change `model.py`, `train.py` or `sample.py` with one experimental idea.
3. `git commit` (only files under `autoresearch/`).
4. Run: `AR_GPU=0,1 bash run.sh > run.log 2>&1` (redirect everything; never tee or let the output flood your context).
5. Read out: `grep "^val_psnr:\|^status:\|^peak_vram_mb:\|^training_seconds:" run.log`.
6. Empty grep = crash. `tail -n 50 run.log` for the trace. Fix trivial bugs and rerun; skip ideas that are fundamentally
   broken (log `crash`).
7. Record the result in `results.tsv` (do not commit it).
8. Better `val_psnr` than the current best (above the noise floor): keep, the branch advances.
9. Equal or worse: `git reset --hard HEAD~1`, run inside `.worktrees/<tag>` (the worktree holds nothing but this campaign).

**Timing.** A run should take about `MAX_MINUTES` plus a few minutes of evaluation. `run.sh` kills it at the cap plus 25
minutes; a run that printed no summary is a failure.

**GPUs.** `run.sh` (through `gpu_wait.sh`) asks the holders to free the GPUs, then prints "waiting: GPU ... in use" lines
and waits (up to 12 hours) while another job holds one of them. That is normal on this shared machine; do not pick another
GPU on your own, and never kill a process you did not start. A multi-GPU run that dies with an NCCL or "unused parameters"
error is a code problem in your change (every parameter must get a gradient each step), not a GPU problem.

**NEVER STOP.** Once the loop has begun, do not pause to ask whether to continue. The human may be asleep. If you run out of
ideas, think harder: re-read the files, combine near-misses, try the bolder architectural changes. The loop runs until the
human interrupts you.
