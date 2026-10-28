"""
MedITok codebook utilization analysis for brain and knee MRI.

For each dataset (brain, knee) and their combination:
  - Tokenizes all GT magnitude slices with MedITok
  - Counts codeword usage per codebook (8 codebooks, 4096 entries each)
  - Reports: active rate, entropy, perplexity per codebook
  - Saves: metrics.json + histogram plots

Supports both multicoil (reads target_rss via build_multicoil_dataset) and
singlecoil (reads reconstruction_esc / reconstruction_rss directly from H5).

Usage:
    # singlecoil
    python eval_codebook_utilization.py \\
        --meditok_config  checkpoints/meditok_config.json \\
        --meditok_ckpt    checkpoints/meditok.pt \\
        --brain_root      fastmri/brain/singlecoil_emulated_val \\
        --knee_root       fastmri/knee/singlecoil_val \\
        --challenge       singlecoil \\
        --output_dir      results/codebook_utilization

    # multicoil
    python eval_codebook_utilization.py \\
        --meditok_config  checkpoints/meditok_config.json \\
        --meditok_ckpt    checkpoints/meditok.pt \\
        --brain_root      fastmri/brain/multicoil_val \\
        --knee_root       fastmri/knee/multicoil_val \\
        --output_dir      results/codebook_utilization
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from dataset.data.subsample import create_mask_for_mask_type
from dataset.multicoil_data import build_multicoil_dataset
from eval_multicoil import DotDict, get_logger
from models.meditok import MedITok


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MedITok codebook utilization")

    p.add_argument("--meditok_config", required=True)
    p.add_argument("--meditok_ckpt",   required=True)
    p.add_argument("--output_dir",     required=True)

    p.add_argument("--brain_root",  default=None,
                   help="Brain dataset directory (omit to skip)")
    p.add_argument("--knee_root",   default=None,
                   help="Knee dataset directory (omit to skip)")
    p.add_argument("--challenge",   default="singlecoil",
                   choices=["singlecoil", "multicoil"],
                   help="Dataset format (default: singlecoil)")

    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--crop_size",   type=int, default=256)
    p.add_argument("--device",      default="cuda")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Singlecoil dataset
# ---------------------------------------------------------------------------

class SinglecoilDataset(Dataset):
    """Reads precomputed magnitude images from fastMRI singlecoil H5 files.

    Reads reconstruction_esc (emulated single coil) if present, otherwise
    reconstruction_rss. Returns center-cropped, per-slice normalized images.
    """

    def __init__(self, root: str, crop_size: int = 256):
        self.crop = crop_size
        self.samples: list[tuple[str, int]] = []   # (filepath, slice_idx)

        for h5path in sorted(Path(root).glob("*.h5")):
            with h5py.File(h5path, "r") as f:
                if "reconstruction_esc" in f:
                    n_slices = f["reconstruction_esc"].shape[0]
                elif "reconstruction_rss" in f:
                    n_slices = f["reconstruction_rss"].shape[0]
                else:
                    continue
            for s in range(n_slices):
                self.samples.append((str(h5path), s))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, sl = self.samples[idx]
        with h5py.File(path, "r") as f:
            if "reconstruction_esc" in f:
                img = f["reconstruction_esc"][sl]   # (H, W) float32
            else:
                img = f["reconstruction_rss"][sl]   # (H, W) float32

        img = torch.from_numpy(img.astype(np.float32))  # (H, W)

        # Center crop to crop_size × crop_size
        H, W = img.shape
        ch = self.crop
        h0 = max((H - ch) // 2, 0)
        w0 = max((W - ch) // 2, 0)
        img = img[h0:h0 + ch, w0:w0 + ch]

        # Pad if smaller than crop_size (rare edge case)
        pH = max(ch - img.shape[0], 0)
        pW = max(ch - img.shape[1], 0)
        if pH > 0 or pW > 0:
            img = F.pad(img, (pW // 2, pW - pW // 2, pH // 2, pH - pH // 2))

        # Normalize to [0, 1] per slice
        img = img / (img.max() + 1e-8)

        return {"target_rss": img}


# ---------------------------------------------------------------------------
# MedITok loader
# ---------------------------------------------------------------------------

def load_meditok(config_path: str, ckpt_path: str, device: torch.device) -> MedITok:
    with open(config_path) as f:
        cfg = json.load(f)
    model = MedITok(DotDict(cfg)).to(device).eval()

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "trainer" in state:
        state = state["trainer"]["model"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    if any(k.startswith("core.") for k in state):
        state = {k[len("core."):]: v for k, v in state.items() if k.startswith("core.")}

    model.load_state_dict(state, strict=False)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_counts(
    loader: DataLoader,
    meditok: MedITok,
    num_codebooks: int,
    per_cb_vocab: int,
    device: torch.device,
    logger,
    label: str,
) -> torch.Tensor:
    """Return counts tensor of shape (num_codebooks, per_cb_vocab)."""
    counts = torch.zeros(num_codebooks, per_cb_vocab, dtype=torch.int64)
    use_cuda = device.type == "cuda"

    for batch_idx, batch in enumerate(loader):
        rss = batch["target_rss"].to(device)   # (B, H, W) in [0, 1]

        # MedITok expects (B, 3, H, W); replicate grayscale across channels
        img = rss.unsqueeze(-1).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)

        with torch.autocast("cuda" if use_cuda else "cpu", dtype=torch.bfloat16):
            idx = meditok.img_to_idx(img)      # (B, num_codebooks, L)

        idx = idx.long()
        for k in range(num_codebooks):
            counts[k] += torch.bincount(
                idx[:, k].flatten().cpu(), minlength=per_cb_vocab
            )

        if (batch_idx + 1) % 50 == 0:
            n = (batch_idx + 1) * loader.batch_size
            logger.info(f"  [{label}] {n} slices …")

    return counts


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(counts: torch.Tensor) -> list[dict]:
    """Compute utilization metrics per codebook from a (K, V) count tensor."""
    K, V = counts.shape
    results = []
    for k in range(K):
        c = counts[k].float()
        total = c.sum().item()
        active = int((c > 0).sum().item())
        p = c / total
        p_nz = p[p > 0]
        entropy = float(-(p_nz * p_nz.log2()).sum().item())
        perplexity = float(2 ** entropy)
        results.append({
            "active_count":   active,
            "active_rate":    round(active / V, 4),
            "entropy":        round(entropy, 4),
            "perplexity":     round(perplexity, 2),
            "max_perplexity": V,
        })
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {"brain": "#2196F3", "knee": "#FF5722", "combined": "#4CAF50"}


def plot_histograms(
    counts_by_split: dict[str, torch.Tensor],
    out_dir: Path,
    num_codebooks: int,
) -> None:
    """Save per-codebook histogram PNGs and a summary figure."""
    splits = [s for s in ("brain", "knee", "combined") if s in counts_by_split]

    for k in range(num_codebooks):
        fig, ax = plt.subplots(figsize=(10, 4))
        for split in splits:
            c = counts_by_split[split][k].float().cpu().numpy()
            sorted_c = np.sort(c)[::-1]
            ax.plot(sorted_c, label=split, color=COLORS[split], linewidth=1.0, alpha=0.85)
        ax.set_yscale("log")
        ax.set_xlabel("Codeword rank (sorted by frequency)")
        ax.set_ylabel("Usage count")
        ax.set_title(f"Codebook {k} — codeword frequency (log scale)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"histogram_cb{k}.png", dpi=150)
        plt.close(fig)

    # Summary: 8-panel figure (2 rows × 4 cols)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for k, ax in enumerate(axes.flat):
        for split in splits:
            c = counts_by_split[split][k].float().cpu().numpy()
            sorted_c = np.sort(c)[::-1]
            ax.plot(sorted_c, label=split, color=COLORS[split], linewidth=0.9, alpha=0.85)
        ax.set_yscale("log")
        ax.set_title(f"Codebook {k}", fontsize=10)
        ax.set_xlabel("Rank", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8)
    fig.suptitle("MedITok codebook utilization — frequency per codeword (sorted, log scale)")
    fig.tight_layout()
    fig.savefig(out_dir / "summary.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    logger = get_logger("eval_codebook_util")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.brain_root is None and args.knee_root is None:
        raise ValueError("Provide at least one of --brain_root or --knee_root")

    # ------------------------------------------------------------------
    # MedITok
    # ------------------------------------------------------------------
    meditok = load_meditok(args.meditok_config, args.meditok_ckpt, device)
    logger.info("MedITok loaded and frozen")

    with open(args.meditok_config) as f:
        enc_cfg = json.load(f)
    num_codebooks = enc_cfg["num_codebooks"]
    per_cb_vocab  = enc_cfg["vocab_size"] // num_codebooks
    logger.info(f"num_codebooks={num_codebooks}, per_cb_vocab={per_cb_vocab}")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    splits_to_run: dict[str, DataLoader] = {}

    for name, root in [("brain", args.brain_root), ("knee", args.knee_root)]:
        if root is None:
            continue

        if args.challenge == "singlecoil":
            dataset = SinglecoilDataset(root, crop_size=args.crop_size)
        else:
            mask_func = create_mask_for_mask_type(
                "equispaced", center_fractions=[0.08], accelerations=[4],
            )
            dataset = build_multicoil_dataset(
                root      = root,
                mask_func = mask_func,
                crop_size = (args.crop_size, args.crop_size),
                use_seed  = True,
                max_coils = 1,
            )

        loader = DataLoader(
            dataset,
            batch_size  = args.batch_size,
            shuffle     = False,
            num_workers = args.num_workers,
            pin_memory  = True,
        )
        splits_to_run[name] = loader
        logger.info(f"{name} ({args.challenge}): {len(dataset)} slices")

    # ------------------------------------------------------------------
    # Count codeword usage per split
    # ------------------------------------------------------------------
    counts_by_split: dict[str, torch.Tensor] = {}

    for split, loader in splits_to_run.items():
        logger.info(f"Encoding {split} …")
        counts_by_split[split] = collect_counts(
            loader, meditok, num_codebooks, per_cb_vocab, device, logger, split
        )
        logger.info(f"  done — {counts_by_split[split].sum().item():,} tokens counted")

    if len(counts_by_split) > 1:
        counts_by_split["combined"] = sum(counts_by_split.values())

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    all_metrics: dict[str, list[dict]] = {}
    for split, counts in counts_by_split.items():
        all_metrics[split] = compute_metrics(counts)

    for split, cb_metrics in all_metrics.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"  {split.upper()} — codebook utilization")
        logger.info(f"  {'CB':>4}  {'Active':>8}  {'Active%':>8}  {'Entropy':>8}  {'Perplexity':>12}  {'MaxPerp':>8}")
        for k, m in enumerate(cb_metrics):
            logger.info(
                f"  {k:>4}  {m['active_count']:>8}  {m['active_rate']*100:>7.1f}%"
                f"  {m['entropy']:>8.3f}  {m['perplexity']:>12.1f}  {m['max_perplexity']:>8}"
            )

    out_json = out_dir / "metrics.json"
    with open(out_json, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\nmetrics saved → {out_json}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    logger.info("Generating histograms …")
    plot_histograms(counts_by_split, out_dir, num_codebooks)
    logger.info(f"plots saved → {out_dir}")


if __name__ == "__main__":
    main()
