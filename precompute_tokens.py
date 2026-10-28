"""
Pre-tokenize the multicoil MRI dataset using the frozen MedITok encoder.

Generates N_MASKS undersampling variations per slice for each requested
acceleration factor, all with the same center-fraction.

Output layout  (one subdir per acceleration):
    {cache_dir}/acc{acc}_cf{cf_tag}/
        {fname_stem}.pt
            tok_us_re  : (S, N_MASKS, C, K, L)  int16
            tok_us_im  : (S, N_MASKS, C, K, L)  int16
            tok_gt_rss : (S, K, L)              int16   (mask-independent)
            masks      : (S, N_MASKS, W)         bool    (1-D k-space masks)
            num_coils  : (S,)                    int16
            slice_nums : (S,)                    int32

Storage estimate (int16 tokens, bool masks):
    per slice per acceleration ≈ N_MASKS × (2 × C × K × L × 2 + W) bytes
    e.g. 25 masks, C=20, K=8, L=256, W=256 → ~3.9 MB/slice/acc

Run once:
    CUDA_VISIBLE_DEVICES=0 python precompute_tokens.py \\
        --data_root       fastmri/brain/multicoil_train \\
        --cache_dir       token_cache/brain/train \\
        --meditok_config  weights/meditok/config.json \\
        --meditok_ckpt    weights/meditok/meditok_ckpt_v1.pth
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from dataset.data.subsample import create_mask_for_mask_type
from dataset.data.transforms import apply_mask, root_sum_of_squares, complex_abs
from dataset.data.fftc import ifft2c_new as ifft2c
from dataset.multicoil_data import build_multicoil_dataset
from models.meditok import MedITok, DotDict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cf_tag(cf: float) -> str:
    """0.04 → 'cf040'"""
    return f"cf{int(round(cf * 1000)):03d}"


def acc_subdir(cache_dir: Path, acc: int, cf: float) -> Path:
    return cache_dir / f"acc{acc}_{cf_tag(cf)}"


def load_meditok(config_path: str, ckpt_path: str, device: torch.device) -> MedITok:
    with open(config_path) as f:
        cfg = json.load(f)
    meditok = MedITok(DotDict(cfg)).to(device).eval()
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "trainer" in sd:
        sd = sd["trainer"]["model"]
    elif isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    if any(k.startswith("core.") for k in sd):
        sd = {k[len("core."):]: v for k, v in sd.items() if k.startswith("core.")}
    result = meditok.load_state_dict(sd, strict=False)
    if result.missing_keys:
        print(f"WARNING: MedITok missing keys ({len(result.missing_keys)}): {result.missing_keys[:5]}")
    for p in meditok.parameters():
        p.requires_grad_(False)
    return meditok


@torch.no_grad()
def encode_imgs(meditok: MedITok, imgs_bhw: torch.Tensor, batch_size: int) -> torch.Tensor:
    """(N, H, W) → (N, K, L) int16 tokens, chunked."""
    results = []
    for chunk in imgs_bhw.split(batch_size, dim=0):
        img4 = chunk.unsqueeze(-1).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)
        z = meditok.quant_proj(meditok.encoder(img4))
        results.append(meditok.quantizer.f_to_idx(z))
    return torch.cat(results, dim=0).to(torch.int16)   # (N, K, L)


def rss_from_coils(x: torch.Tensor) -> torch.Tensor:
    """(C, H, W, 2) → (H, W) in [0, 1]"""
    return root_sum_of_squares(complex_abs(x), dim=0)


def save_volume(out_path: Path, slices: list):
    """Stack and save accumulated slice data sorted by slice_num."""
    slices.sort(key=lambda s: s["slice_num"])
    torch.save({
        "tok_us_re":  torch.stack([s["tok_us_re"]  for s in slices]),   # (S,n,C,K,L) int16
        "tok_us_im":  torch.stack([s["tok_us_im"]  for s in slices]),   # (S,n,C,K,L) int16
        "tok_gt_rss": torch.stack([s["tok_gt_rss"] for s in slices]),   # (S,K,L)     int16
        "masks":      torch.stack([s["masks"]      for s in slices]),   # (S,n,W)     bool
        "num_coils":  torch.tensor([s["num_coils"] for s in slices], dtype=torch.int16),
        "slice_nums": torch.tensor([s["slice_num"] for s in slices], dtype=torch.int32),
    }, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",        required=True)
    parser.add_argument("--cache_dir",        required=True)
    parser.add_argument("--meditok_config",   required=True)
    parser.add_argument("--meditok_ckpt",     required=True)
    parser.add_argument("--accelerations",    type=int,   nargs="+", default=[4, 8, 10, 16])
    parser.add_argument("--center_fraction",  type=float, default=0.04)
    parser.add_argument("--n_masks",          type=int,   default=25,
                        help="number of different undersampling masks per slice")
    parser.add_argument("--batch_size",       type=int,   default=64,
                        help="images per MedITok forward pass")
    parser.add_argument("--slice_batch_size", type=int,   default=4,
                        help="slices per DataLoader batch (more = better CPU parallelism)")
    parser.add_argument("--num_workers",      type=int,   default=4)
    parser.add_argument("--device",           default="cuda")
    parser.add_argument("--max_coils",        type=int,   default=20)
    parser.add_argument("--overwrite",        action="store_true")
    args = parser.parse_args()

    device    = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir)
    cf        = args.center_fraction
    n_masks   = args.n_masks

    # Storage estimate
    C, K, L, W = args.max_coils, 8, 256, 256
    bytes_per_slice = n_masks * (2 * C * K * L * 2 + W)   # int16 tokens + bool masks
    print(f"Accelerations : {args.accelerations}  cf={cf}  n_masks={n_masks}")
    print(f"Est. storage  : {bytes_per_slice/1e6:.1f} MB/slice/acc  "
          f"× {len(args.accelerations)} accs  (int16 tokens)")

    # Create output subdirs
    for acc in args.accelerations:
        acc_subdir(cache_dir, acc, cf).mkdir(parents=True, exist_ok=True)

    print(f"\nLoading MedITok ...")
    meditok = load_meditok(args.meditok_config, args.meditok_ckpt, device)
    print("MedITok loaded.")

    # Build base dataset (use any acceleration — we only need x_gt and kspace_gt)
    base_mask_func = create_mask_for_mask_type(
        "random", center_fractions=[cf], accelerations=[args.accelerations[0]]
    )
    dataset = build_multicoil_dataset(
        root=args.data_root, mask_func=base_mask_func,
        max_coils=args.max_coils, use_seed=True,
    )
    loader = DataLoader(dataset, batch_size=args.slice_batch_size,
                        num_workers=args.num_workers, shuffle=False, pin_memory=True)

    # One mask_func per acceleration (same cf)
    mask_funcs = {
        acc: create_mask_for_mask_type("random", center_fractions=[cf], accelerations=[acc])
        for acc in args.accelerations
    }

    print(f"Dataset: {len(dataset)} slices  →  {cache_dir}")

    # Per-acceleration slice accumulators  {stem: [slice_dict, ...]}
    acc_volumes = {acc: {} for acc in args.accelerations}
    saved_vols = {acc: 0 for acc in args.accelerations}
    skipped_vols = {acc: 0 for acc in args.accelerations}
    encoded_slices = 0
    t0 = time.perf_counter()
    current_stem = None

    def flush_volume(acc: int, stem: str):
        out = acc_subdir(cache_dir, acc, cf) / f"{stem}.pt"
        save_volume(out, acc_volumes[acc].pop(stem))
        saved_vols[acc] += 1

    def flush_all_accs(stem: str):
        """Save completed volume for all accelerations and free memory."""
        for acc in args.accelerations:
            if stem in acc_volumes[acc]:
                flush_volume(acc, stem)

    for batch in tqdm(loader, desc="Tokenising"):
        batch_size = len(batch["fname"])

        # -------------------------------------------------------------------
        # Process each slice in the batch for bookkeeping / flush logic,
        # but accumulate all their images to encode in one MedITok call
        # per acceleration.
        # -------------------------------------------------------------------

        # Per-acceleration: list of (slice_idx_in_batch, re_imgs, im_imgs, masks_list, meta)
        # We first collect CPU work (masking/IFFT) for all slices × accs,
        # then do one big GPU encode per acceleration.

        # Step 1: CPU work — masking + IFFT for all slices and all accs
        # acc → list of per-slice dicts with raw image tensors + meta
        acc_slice_data = {acc: [] for acc in args.accelerations}
        rss_imgs = []   # GT RSS images for all slices in batch

        for i in range(batch_size):
            fname     = batch["fname"][i]
            slice_num = int(batch["slice_num"][i])
            stem      = Path(fname).stem
            num_coils = int(batch["num_coils"][i])

            # Volume boundary — flush previous volume
            if stem != current_stem:
                if current_stem is not None:
                    flush_all_accs(current_stem)
                current_stem = stem

            all_cached = all(
                (acc_subdir(cache_dir, acc, cf) / f"{stem}.pt").exists()
                for acc in args.accelerations
            )
            if all_cached and not args.overwrite:
                rss_imgs.append(None)
                for acc in args.accelerations:
                    acc_slice_data[acc].append(None)
                    skipped_vols[acc] += 1
                continue

            kspace_gt = batch["kspace_gt"][i]          # CPU
            x_gt      = batch["x_gt"][i].to(device)   # GPU for RSS
            H_img, W_img = x_gt.shape[1], x_gt.shape[2]

            # GT RSS
            rss = rss_from_coils(x_gt[:num_coils])
            rss_imgs.append((rss * 2.0 - 1.0))   # (H, W) on GPU

            # Per-acceleration masking
            for acc in args.accelerations:
                out_path = acc_subdir(cache_dir, acc, cf) / f"{stem}.pt"
                if out_path.exists() and not args.overwrite:
                    acc_slice_data[acc].append(None)
                    continue

                mf = mask_funcs[acc]
                all_re, all_im, all_masks = [], [], []
                for m in range(n_masks):
                    seed = tuple(map(ord, fname)) + (slice_num, m, acc)
                    kspace_us, mask, _ = apply_mask(kspace_gt, mf, seed=seed)
                    x_us = ifft2c(kspace_us)[:num_coils].to(device)
                    all_re.append(x_us[:, :, :, 0])
                    all_im.append(x_us[:, :, :, 1])
                    all_masks.append(mask.squeeze().bool().cpu())

                acc_slice_data[acc].append({
                    "fname":     fname,
                    "slice_num": slice_num,
                    "stem":      stem,
                    "num_coils": num_coils,
                    "H_img":     H_img,
                    "W_img":     W_img,
                    "re_stk":    torch.stack(all_re),   # (n_masks, C, H, W)
                    "im_stk":    torch.stack(all_im),
                    "masks":     torch.stack(all_masks), # (n_masks, W)
                })

        # Step 2: Encode GT RSS for all slices in one call
        valid_rss = [(i, r) for i, r in enumerate(rss_imgs) if r is not None]
        if valid_rss:
            rss_stack = torch.stack([r for _, r in valid_rss]).unsqueeze(1)  # (N,1,H,W)
            # reshape to (N,H,W) then encode
            rss_flat = rss_stack.squeeze(1)
            tok_rss_all = encode_imgs(meditok, rss_flat, args.batch_size)  # (N,K,L)
            rss_tok_map = {i: tok_rss_all[j] for j, (i, _) in enumerate(valid_rss)}
        else:
            rss_tok_map = {}

        # Step 3: Encode US coils per acceleration in one big call
        for acc in args.accelerations:
            valid = [(i, d) for i, d in enumerate(acc_slice_data[acc]) if d is not None]
            if not valid:
                continue

            # Flatten all slices' re+im images together
            all_re_flat, all_im_flat = [], []
            counts = []   # n_masks * C per slice
            for _, d in valid:
                C  = d["num_coils"]
                re = d["re_stk"].reshape(-1, d["H_img"], d["W_img"])  # (n_masks*C, H, W)
                im = d["im_stk"].reshape(-1, d["H_img"], d["W_img"])
                all_re_flat.append(re)
                all_im_flat.append(im)
                counts.append(n_masks * C)

            re_cat = torch.cat(all_re_flat)   # (sum(n_masks*C), H, W)
            im_cat = torch.cat(all_im_flat)
            tok_both = encode_imgs(meditok, torch.cat([re_cat, im_cat]), args.batch_size)
            half = tok_both.shape[0] // 2
            tok_re_cat = tok_both[:half]
            tok_im_cat = tok_both[half:]

            Kk, Ll = tok_re_cat.shape[1], tok_re_cat.shape[2]

            # Split back per slice
            re_splits = torch.split(tok_re_cat, counts)
            im_splits = torch.split(tok_im_cat, counts)

            for j, (i, d) in enumerate(valid):
                C  = d["num_coils"]
                tok_re = re_splits[j].view(n_masks, C, Kk, Ll).cpu()
                tok_im = im_splits[j].view(n_masks, C, Kk, Ll).cpu()
                tok_gt_rss = rss_tok_map[i].cpu()

                slice_dict = {
                    "slice_num":  d["slice_num"],
                    "tok_us_re":  tok_re,
                    "tok_us_im":  tok_im,
                    "tok_gt_rss": tok_gt_rss,
                    "masks":      d["masks"],
                    "num_coils":  C,
                }
                stem = d["stem"]
                if stem not in acc_volumes[acc]:
                    acc_volumes[acc][stem] = []
                acc_volumes[acc][stem].append(slice_dict)

        encoded_slices += batch_size

    # Flush the last volume
    if current_stem is not None:
        flush_all_accs(current_stem)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.0f}s  ({encoded_slices / (elapsed + 1e-6):.1f} slices/s)")
    for acc in args.accelerations:
        print(f"  acc{acc}: saved={saved_vols[acc]}  skipped≈{skipped_vols[acc]}")


if __name__ == "__main__":
    main()
