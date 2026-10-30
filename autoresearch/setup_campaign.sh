#!/usr/bin/env bash
# Set up one autoresearch campaign (the agent runs this, from the main checkout):
#   bash autoresearch/setup_campaign.sh <tag>        e.g. oct30
# Creates the branch autoresearch/<tag> and a git worktree for it at .worktrees/<tag> (inside the main checkout, ignored by
# git), links the data and weights that live outside git, and initialises results.tsv. Idempotent: rerunning on an existing
# worktree only re-links and reports. The main checkout and its running launchers are never touched.
set -euo pipefail
tag=${1:?usage: setup_campaign.sh <tag>}
main=$(git rev-parse --show-toplevel)
cd "$main"
wt="$main/.worktrees/$tag"
if git show-ref --verify --quiet "refs/heads/autoresearch/$tag" && [ ! -d "$wt" ]; then
    echo "branch autoresearch/$tag exists but has no worktree at $wt; pick a new tag or remove the branch" >&2; exit 1
fi
[ -n "$(git status --porcelain autoresearch/)" ] && { echo "autoresearch/ has uncommitted changes in the main checkout; commit them first" >&2; exit 1; }
mkdir -p "$main/.worktrees"
if [ ! -d "$wt" ]; then
    git worktree add -b "autoresearch/$tag" "$wt" HEAD
    echo "worktree $wt on branch autoresearch/$tag (from $(git rev-parse --short HEAD))"
fi
cd "$wt"
for d in fastmri weights checkpoints; do [ -e "$d" ] || ln -s "$main/$d" "$d"; done
mkdir -p results/sens_net
[ -e results/sens_net/run1 ] || ln -s "$main/results/sens_net/run1" results/sens_net/run1
[ -e autoresearch/results.tsv ] || printf "commit\tval_psnr\tval_ssim\ttrain_min\tmemory_gb\tstatus\tdescription\n" > autoresearch/results.tsv
echo "ready: cd $wt/autoresearch"
echo "  branch $(git rev-parse --abbrev-ref HEAD), results.tsv $(wc -l < autoresearch/results.tsv) line(s), GT cache $(ls /fast/lev/datasets/fastmri/brain/multicoil_train_gt_cache_crop256_tok512 | wc -l) volumes"
