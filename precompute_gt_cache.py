"""Cache the ground-truth SENSE tokens and latent targets per slice, so the SENSE GPT trainer skips its tokenizer encoder passes on the
ground truth (4 of the 6 encoder passes of a training step; the tokenizer, not GPT-L, dominates the step time).

Per volume <out_dir>/<volume>.h5:
    tokens (S, 2, K, L) int16   codes of the max-normalised SENSE ground truth (oracle maps) on the token grid, Re and Im streams
    latent (S, 2, L, D) float16 the same encoder features with each codebook chunk L2-normalised (the latent head's target)
    slice  (S,) int32
attrs: resolution, token_size, resize_mode, encoder_ckpt, encoder_config, max_coils
The image path is exactly the trainer's: SenseSliceDataset gt_c_image (crop, oracle combine, / max|x0|) -> tok_upsample -> encoder in fp32.
Volumes with more than --max-coils coils are skipped like the trainer skips them.

  CUDA_VISIBLE_DEVICES=5 python precompute_gt_cache.py --data-root fastmri/brain/multicoil_train --data-split fastmri/splits/multicoil_brain_train.txt \
      --sens-dirs fastmri/brain/multicoil_train_sens_maps_gpu,fastmri/brain/multicoil_train_sens_maps --out-dir fastmri/brain/multicoil_train_gt_cache_crop256_tok512
"""
import argparse, os, sys, time
from pathlib import Path
import h5py, numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoregressive.sample.reconstruct_far import load_meditok  # noqa: E402
from dataset.sense_data import SenseSliceDataset  # noqa: E402
from dataset.data.subsample import create_mask_for_mask_type  # noqa: E402
from models.sense_ops import tok_upsample, encode_features, features_to_tokens, features_to_latent  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--data-split", default=None)
    p.add_argument("--sens-dirs", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--token-size", type=int, default=512)
    p.add_argument("--resize-mode", default="bicubic")
    p.add_argument("--max-coils", type=int, default=20)
    p.add_argument("--encoder-ckpt", default="weights/meditok/brain_mc_sense_cplxpair_res512_ep5_frozen.pth")
    p.add_argument("--encoder-config", default="weights/meditok/config_512.json")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-volumes", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    meditok = load_meditok(args.encoder_config, args.encoder_ckpt, device)
    mask_func = create_mask_for_mask_type("random", [0.04], [8])            # required by the dataset, unused here
    ds = SenseSliceDataset(args.data_root, args.sens_dirs, mask_func, resolution=args.crop_size, max_coils=args.max_coils,
                           use_seed=True, custom_split=args.data_split)
    if not args.overwrite:   # skip volumes that are already cached
        done = {f.name for f in ds.map_paths if (out_dir / f.name).exists()}
        ds.examples = [(f, s) for f, s in ds.examples if f.name not in done]
        print(f"{len(done)} volumes already cached, {len(ds.examples)} slices to do")
    if args.max_volumes > 0:
        keep = sorted({f for f, _ in ds.examples})[:args.max_volumes]
        ds.examples = [(f, s) for f, s in ds.examples if f in keep]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    attrs = dict(resolution=args.crop_size, token_size=args.token_size, resize_mode=args.resize_mode, encoder_ckpt=args.encoder_ckpt,
                 encoder_config=args.encoder_config, max_coils=args.max_coils)
    buf = {}   # fname -> {slice: (tokens, latent)}

    def flush(fname):
        items = buf.pop(fname)
        slices = sorted(items)
        tokens = np.stack([items[s][0] for s in slices]); latent = np.stack([items[s][1] for s in slices])
        tmp = out_dir / (fname + ".tmp")
        with h5py.File(tmp, "w") as f:
            f.create_dataset("tokens", data=tokens, chunks=(1,) + tokens.shape[1:])
            f.create_dataset("latent", data=latent, chunks=(1,) + latent.shape[1:])
            f.create_dataset("slice", data=np.asarray(slices, dtype=np.int32))
            for k, v in attrs.items(): f.attrs[k] = v
        os.replace(tmp, out_dir / fname)

    expected = {}
    for f, s in ds.examples: expected[f.name] = expected.get(f.name, 0) + 1
    t0 = time.time(); n = 0; n_vol = 0
    for batch in loader:
        gt = tok_upsample(batch["gt_c_image"].to(device, non_blocking=True), args.token_size, args.resize_mode)   # (B, tok, tok, 2)
        feats = [encode_features(meditok, gt[..., c]) for c in range(2)]
        tokens = torch.stack([features_to_tokens(meditok, f) for f in feats], dim=1).to(torch.int16).cpu().numpy()   # (B, 2, K, L)
        latent = torch.stack([features_to_latent(f) for f in feats], dim=1).half().cpu().numpy()                    # (B, 2, L, D)
        for b in range(tokens.shape[0]):
            fname, s = batch["fname"][b], int(batch["slice"][b])
            buf.setdefault(fname, {})[s] = (tokens[b], latent[b])
            if len(buf[fname]) == expected[fname]:
                flush(fname); n_vol += 1
        n += tokens.shape[0]
        if n % (args.batch_size * 100) == 0:
            el = time.time() - t0
            print(f"{n}/{len(ds)} slices, {n_vol} volumes written, {n / el:.1f} slices/s, eta {(len(ds) - n) / max(n / el, 1e-6) / 3600:.2f} h", flush=True)
    assert not buf, f"incomplete volumes left in the buffer: {list(buf)[:3]}"
    print(f"done: {n} slices, {n_vol} volumes in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
