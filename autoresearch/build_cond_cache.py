"""Build the conditioning cache of the autoresearch trainer (harness; one-off, about 90 min on one GPU, resumable).
For every training slice: one fixed random 1D mask (seed prepare.cond_cache_seed), the sens-net maps from its ACS lines,
the zero-filled SENSE input on the token grid, its tokenizer latent (prepare.sense_inputs, exactly what the slow path feeds
the model), the mask columns and the max-normalised SENSE ground truth on the acquisition grid. One HDF5 file per volume
under prepare.COND_CACHE_DIR, written atomically; volumes already present are skipped; `_complete` marks a full cache.
    PYTHONPATH=.. CUDA_VISIBLE_DEVICES=1 python build_cond_cache.py            # build (or resume)
    PYTHONPATH=.. CUDA_VISIBLE_DEVICES=1 python build_cond_cache.py --verify 16   # recompute 16 random cached slices, compare
"""
import argparse
import os
import sys
import time
import pathlib

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import prepare


class SeededMaskDataset(torch.utils.data.Dataset):
    """SenseSliceDataset items with the fixed per-slice mask of the cache in place of the fresh draw."""

    def __init__(self, inner):
        self.inner = inner
        self.mask_func = prepare._mask_func()

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        out = self.inner[i]
        res = prepare.CROP_SIZE
        mask, _ = self.mask_func((res, res, 2), seed=prepare.cond_cache_seed(out["fname"], out["slice"]))
        out["mask"] = mask.repeat(res, 1, 1).float()                                    # (H, W, 1)
        return out


def write_volume(out_dir, name, items):
    """items: {slice: (cond f16 (2,L,64), cols bool (W,), img f16 (H,W,2))} for all slices 0..S-1 of the volume."""
    S = len(items)
    assert sorted(items) == list(range(S)), f"{name}: slices {sorted(items)[:5]}..."
    tmp = out_dir / (name + ".tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("cond_latent", data=np.stack([items[s][0] for s in range(S)]), dtype="f2")
        f.create_dataset("mask_cols", data=np.stack([items[s][1] for s in range(S)]), dtype="bool")
        f.create_dataset("gt_image", data=np.stack([items[s][2] for s in range(S)]), dtype="f2")
        f.create_dataset("slice", data=np.arange(S, dtype=np.int32))
        f.attrs.update(dict(mask_type=prepare.MASK_TYPE, acceleration=prepare.ACCELERATION, center_fraction=prepare.CENTER_FRACTION,
                            resolution=prepare.CROP_SIZE, token_size=prepare.TOKEN_SIZE, resize_mode=prepare.RESIZE_MODE,
                            max_coils=prepare.MAX_COILS, sens_ckpt=str(prepare.SENS_CKPT.relative_to(prepare.REPO)),
                            encoder_ckpt=str(prepare.ENCODER_CKPT.relative_to(prepare.REPO)), seed="cond_cache_seed(fname, slice)"))
    os.replace(tmp, out_dir / name)


def build(args):
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    tokenizer = prepare.load_tokenizer(device)
    sens_net = prepare.load_sens_net(device)
    inner = prepare.train_dataset()
    n_slices = {}
    for f, s in inner.examples:
        n_slices[f.name] = n_slices.get(f.name, 0) + 1
    vols = list(n_slices)                                                                 # in dataset order (sorted)
    if args.limit:
        vols = vols[:args.limit]
    todo_vols = [v for v in vols if not (out_dir / v).exists()]
    idx = [i for i, (f, s) in enumerate(inner.examples) if f.name in set(todo_vols)]
    print(f"{len(vols)} volumes, {len(todo_vols)} to build ({len(idx)} slices), {len(vols) - len(todo_vols)} already cached", flush=True)
    loader = DataLoader(Subset(SeededMaskDataset(inner), idx), batch_size=args.batch_size, shuffle=False,
                        num_workers=prepare.NUM_WORKERS, pin_memory=True, drop_last=False)
    buf, done, t0 = {}, 0, time.time()
    for bi, batch in enumerate(loader):
        inp = prepare.sense_inputs(batch, device, sens_net, tokenizer)
        cond = inp["cond_latent"].half().cpu().numpy()
        cols = inp["mask_cols"].bool().cpu().numpy()
        img = inp["gt_image"].half().cpu().numpy()
        for b in range(cond.shape[0]):
            name, s = batch["fname"][b], int(batch["slice"][b])
            buf.setdefault(name, {})[s] = (cond[b], cols[b], img[b])
            if len(buf[name]) == n_slices[name]:
                write_volume(out_dir, name, buf.pop(name))
                done += 1
        if bi % 50 == 0:
            el = time.time() - t0
            rate = (bi + 1) * args.batch_size / el
            print(f"{time.strftime('%F %T')} batch {bi}/{len(loader)}  {done} volumes written  {rate:.1f} slices/s  "
                  f"eta {(len(idx) - (bi + 1) * args.batch_size) / rate / 60:.0f} min", flush=True)
    assert not buf, f"incomplete volumes left in the buffer: {list(buf)[:3]}"
    missing = [v for v in n_slices if not (out_dir / v).exists()]
    if not missing:
        (out_dir / "_complete").write_text(f"{len(n_slices)} volumes, {len(inner.examples)} slices, {time.strftime('%F %T')}\n")
        print(f"cache complete: {len(n_slices)} volumes", flush=True)
    else:
        print(f"{len(missing)} volumes still missing (limit run?)", flush=True)


@torch.no_grad()
def verify(args):
    """Recompute `args.verify` random cached slices through the slow path with the cache's mask and compare."""
    device = torch.device("cuda")
    tokenizer = prepare.load_tokenizer(device)
    sens_net = prepare.load_sens_net(device)
    cached = prepare.CondCacheDataset(cond_dir=args.out, allow_partial=True)
    inner = prepare.train_dataset()
    pos = {(f.name, s): i for i, (f, s) in enumerate(inner.examples)}
    seeded = SeededMaskDataset(inner)
    rng = np.random.RandomState(0)
    picks = rng.choice(len(cached), size=min(args.verify, len(cached)), replace=False)
    worst = dict(cond=0.0, img=0.0, cols=0)
    sq, n = 0.0, 0
    for i in picks:
        c = cached[int(i)]
        item = seeded[pos[(c["fname"], c["slice"])]]
        batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in item.items()}
        batch["num_coils"] = torch.tensor([item["num_coils"]])
        batch["gt_cached"] = torch.tensor([True])
        inp = prepare.sense_inputs(batch, device, sens_net, tokenizer)
        d = inp["cond_latent"][0].cpu() - c["cond_latent"].float()
        worst["cond"] = max(worst["cond"], d.abs().max().item())
        sq, n = sq + d.pow(2).sum().item(), n + d.numel()
        worst["img"] = max(worst["img"], (inp["gt_image"][0].cpu() - c["gt_image"].float()).abs().max().item())
        worst["cols"] = max(worst["cols"], int((inp["mask_cols"][0].cpu().bool() != c["mask_cols"]).sum()))
    rms = (sq / max(1, n)) ** 0.5
    # cond_latent: the slow path itself is only reproducible across batch compositions to about 3e-2 max / 1e-3 rms (TF32 in
    # the sens-net and the tokenizer encoder); with the build's own batch composition the cache matches to fp16 rounding (2e-4)
    ok = rms < 5e-3 and worst["cond"] < 1e-1 and worst["img"] < 5e-3 and worst["cols"] == 0
    print(f"verified {len(picks)} slices: cond_latent diff rms {rms:.2e}, max {worst['cond']:.2e} (TF32 noise of the slow path), "
          f"max |gt_image diff| {worst['img']:.2e}, mask column mismatches {worst['cols']}", flush=True)
    print("cache check", "PASSED" if ok else "FAILED", flush=True)
    assert ok, "cache does not match the slow path"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(prepare.COND_CACHE_DIR))
    ap.add_argument("--limit", type=int, default=0, help="only the first N volumes (test)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--verify", type=int, default=0, help="recompute N random cached slices and compare instead of building")
    a = ap.parse_args()
    verify(a) if a.verify else build(a)
