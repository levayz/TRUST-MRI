#!/usr/bin/env python
"""E2E-VarNet baseline on the suno split under the TRUST-MRI SENSE-block conventions.
Run with the fastMRI env: /fast/lev/miniconda3/envs/varnet/bin/python (needs the `fastmri` package) and WITHOUT the repo root on
sys.path / PYTHONPATH: the data symlink `fastmri/` in the repo root shadows the package (lives under scripts/ for that reason).

Per volume/slice (native k-space (C,H,W), oracle ESPIRiT maps (C,Hr,Wr)):
  refs      coils256 = crop(crop(ifft2c(k), map grid), 256);  rss_ref = RSS(coils256);  x0_ref = |combine(coils256, maps256)|, support
  native    random 1D mask (cf, R) seeded by the filename on the NATIVE width -> VarNet -> per-coil k-space -> ifft2c -> crop 256
            -> arms native_rss (RSS of the predicted coils) and native_sense (|combine with the oracle maps|)
  grid256   the same mask seed on the 256 grid (identical columns to the SENSE-GPT evals): k256 = fft2c(coils256) * mask -> VarNet
            -> arms grid256_rss, grid256_sense   (the sens-net sees 10 ACS lines on the 256 grid, outside its training distribution)
Writes <out>/h5/<volume>.h5 with (S,256,256) float32 arrays per arm + refs; score with scripts/score_varnet_baseline.py.
"""
import argparse
import importlib.util
import json
import os
import pathlib
import time

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py
import numpy as np
import torch
from fastmri.models import VarNet

_spec = importlib.util.spec_from_file_location("trust_subsample", "/fast/lev/TRUST-MRI/dataset/data/subsample.py")
_sub = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_sub)
create_mask_for_mask_type = _sub.create_mask_for_mask_type


def fft2c(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def ifft2c(k):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def crop(x, shape):
    h, w = shape
    H, W = x.shape[-2:]
    if H < h or W < w:
        ph, pw = max(h - H, 0), max(w - W, 0)
        x = torch.nn.functional.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        H, W = x.shape[-2:]
    top, left = (H - h) // 2, (W - w) // 2
    return x[..., top:top + h, left:left + w]


def combine(coils, maps):
    ss = (maps.abs() ** 2).sum(1)
    support = ss > 1e-6
    num = (maps.conj() * coils).sum(1)
    return torch.where(support, num / ss.clamp_min(1e-6), torch.zeros_like(num)), support


def load_varnet(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "hyper_parameters" in ck:           # Lightning checkpoint (fastmri_examples/varnet)
        hp = ck["hyper_parameters"]
        sd = {k[len("varnet."):]: v for k, v in ck["state_dict"].items() if k.startswith("varnet.")}
        model = VarNet(num_cascades=hp["num_cascades"], sens_chans=hp["sens_chans"], sens_pools=hp["sens_pools"], chans=hp["chans"], pools=hp["pools"])
        info = {"epoch": ck.get("epoch"), "global_step": ck.get("global_step"), **{k: hp[k] for k in ("num_cascades", "chans", "sens_chans")}}
    else:                                                           # bare state dict (fastMRI leaderboard)
        sd = ck["state_dict"] if "state_dict" in ck else ck
        n_casc = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("cascades."))
        chans = sd["cascades.0.model.unet.down_sample_layers.0.layers.0.weight"].shape[0]
        sens_chans = sd["sens_net.norm_unet.unet.down_sample_layers.0.layers.0.weight"].shape[0]
        model = VarNet(num_cascades=n_casc, sens_chans=sens_chans, sens_pools=4, chans=chans, pools=4)
        info = {"num_cascades": n_casc, "chans": chans, "sens_chans": sens_chans}
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval(), info


@torch.no_grad()
def varnet_kspace(model, k_masked, mask_cols, nlf):
    """k_masked (B,C,H,W) complex, mask_cols (W,) bool -> predicted per-coil k-space (B,C,H,W) complex."""
    kr = torch.view_as_real(k_masked).float()
    mask5 = mask_cols.view(1, 1, 1, -1, 1).expand(kr.shape[0], 1, 1, -1, 1).contiguous()
    sens = model.sens_net(kr, mask5, nlf)
    kp = kr.clone()
    for cascade in model.cascades:
        kp = cascade(kp, kr, mask5, sens)
    return torch.view_as_complex(kp.contiguous())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default="fastmri/brain/multicoil_val")
    ap.add_argument("--sens-dirs", default="fastmri/brain/multicoil_val_sens_maps_gpu,fastmri/brain/multicoil_val_sens_maps")
    ap.add_argument("--split", default="fastmri/splits/suno_brain_test.txt")
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--acceleration", type=float, default=8)
    ap.add_argument("--acs-lines", type=int, default=0, help=">0: force this many ACS columns on the native grid (cf = acs/W per volume)")
    ap.add_argument("--total-lines", type=int, default=0, help=">0: force this many sampled columns on the native grid (R = W/total per volume)")
    ap.add_argument("--mask-type", default="random", help="random (SENSE-block convention) | equispaced | equispaced_fraction (VarNet training convention)")
    ap.add_argument("--center-fractions", type=float, default=0.04)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grids", nargs="+", default=["native", "grid256"])
    ap.add_argument("--max-volumes", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    cs = args.crop_size
    out = pathlib.Path(args.out); (out / "h5").mkdir(parents=True, exist_ok=True)
    model, info = load_varnet(args.ckpt, device)
    print("VarNet:", info, flush=True)
    def _acc(a):   # the equispaced mask slices with the acceleration: it must be an int when it is integer-valued
        return int(a) if float(a).is_integer() else a
    mask_func = create_mask_for_mask_type(args.mask_type, [args.center_fractions], [_acc(args.acceleration)])
    names = [l.strip() for l in open(args.split) if l.strip()]
    if args.max_volumes:
        names = names[:args.max_volumes]
    t0 = time.time()
    for name in names:
        with h5py.File(pathlib.Path(args.data_root) / name, "r") as f:
            ksp = f["kspace"][()]
        sens = None
        for d in args.sens_dirs.split(","):
            p = pathlib.Path(d) / name
            if p.exists():
                with h5py.File(p, "r") as f:
                    sens = f["sens_maps"][()]
                break
        assert sens is not None, name
        S, C, H, W = ksp.shape
        seed = tuple(map(ord, name))
        if args.acs_lines > 0 or args.total_lines > 0:   # match a line budget on the native grid (e.g. the 256-grid 10 ACS / 33 lines)
            cf_nat = (args.acs_lines / W) if args.acs_lines > 0 else args.center_fractions
            acc_nat = (W / args.total_lines) if args.total_lines > 0 else args.acceleration
            mask_func_nat = create_mask_for_mask_type(args.mask_type, [cf_nat], [_acc(acc_nat)])
        else:
            mask_func_nat = mask_func
        m_nat, nlf_nat = mask_func_nat((1, H, W, 2), seed=seed); m_nat = m_nat.reshape(-1).bool().to(device)
        m_256, nlf_256 = mask_func((1, cs, cs, 2), seed=seed); m_256 = m_256.reshape(-1).bool().to(device)
        res = {k: [] for k in ("rss_ref", "x0_ref", "support")}
        for g in args.grids:
            res[f"{g}_rss"] = []; res[f"{g}_sense"] = []
        for s0 in range(0, S, args.batch_size):
            sl = list(range(s0, min(S, s0 + args.batch_size)))
            k = torch.from_numpy(ksp[sl]).to(device)                                    # (B,C,H,W)
            mp = torch.from_numpy(sens[sl]).to(device)
            coils256 = crop(crop(ifft2c(k), mp.shape[-2:]), (cs, cs))
            maps256 = crop(mp, (cs, cs))
            x0, support = combine(coils256, maps256)
            res["rss_ref"].append((coils256.abs() ** 2).sum(1).sqrt().cpu().numpy())
            res["x0_ref"].append(x0.abs().cpu().numpy()); res["support"].append(support.cpu().numpy())
            if "native" in args.grids:
                kp = varnet_kspace(model, k * m_nat, m_nat, int(nlf_nat))
                pc = crop(crop(ifft2c(kp), mp.shape[-2:]), (cs, cs))
                res["native_rss"].append((pc.abs() ** 2).sum(1).sqrt().cpu().numpy())
                res["native_sense"].append(combine(pc, maps256)[0].abs().cpu().numpy())
            if "grid256" in args.grids:
                k256 = fft2c(coils256) * m_256
                kp = varnet_kspace(model, k256, m_256, int(nlf_256))
                pc = ifft2c(kp)
                res["grid256_rss"].append((pc.abs() ** 2).sum(1).sqrt().cpu().numpy())
                res["grid256_sense"].append(combine(pc, maps256)[0].abs().cpu().numpy())
        with h5py.File(out / "h5" / name, "w") as f:
            for key, v in res.items():
                f.create_dataset(key, data=np.concatenate(v, 0))
            f.attrs["mask_native_cols"] = int(m_nat.sum()); f.attrs["mask_256_cols"] = int(m_256.sum())
            f.attrs["nlf_native"] = int(nlf_nat); f.attrs["nlf_256"] = int(nlf_256)
        print(f"{name}: {S} slices, C={C}, native {H}x{W} ({int(m_nat.sum())} lines, {int(nlf_nat)} ACS) | 256 grid ({int(m_256.sum())} lines) "
              f"[{time.time() - t0:.0f}s]", flush=True)
    with open(out / "info.json", "w") as f:
        json.dump({"ckpt": args.ckpt, "model": info, "acceleration": args.acceleration, "center_fractions": args.center_fractions, "mask_type": args.mask_type, "acs_lines": args.acs_lines, "total_lines": args.total_lines,
                   "split": args.split, "crop_size": cs, "grids": args.grids, "n_volumes": len(names)}, f, indent=2)
    print("done ->", out)


if __name__ == "__main__":
    main()
