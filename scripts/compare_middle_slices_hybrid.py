#!/usr/bin/env python
"""Middle slice of every suno volume with the HYBRID decodings: zero-filled input | ground truth | VarNet | ours latent one pass | hybrid r=0.5 mean of 4 samples | hybrid r=0.3 one sample
(SENSE-combined output), with per-slice PSNR vs the oracle SENSE image (data range = the volume's ground-truth maximum, support applied,
as in the tables). Random x8 mask seeded by the file name = the mask of every table row. VarNet images come from the saved evaluation
(results/varnet_baseline/<run>/h5); ours are reconstructed here.

Usage: PYTHONPATH=$(pwd) python scripts/compare_middle_slices.py --gpt-ckpt <run11 best_final.pt> --out results/figs/compare_run11_vs_varnet
"""
import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace
import h5py, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoregressive.sample.reconstruct_far import load_meditok, load_gpt
from autoregressive.models.gpt_kar import infer_model_args_from_state_dict
from autoregressive.models.generate import reconstruct_far
from dataset.data.subsample import create_mask_for_mask_type
from models.sense_ops import (acs_num_low_freqs, c2r, center_crop_or_pad, crop_native_to_grid, estimate_maps, fft2c, ifft2c,
                              make_sense_dc_fn, r2c, sense_combine, support_from_maps, tok_downsample, tok_upsample, wrap_dc_fn_resized)
from models.varnet_sens import load_sens_net

p = argparse.ArgumentParser()
p.add_argument("--gpt-ckpt", required=True)
p.add_argument("--varnet-h5", default="results/varnet_baseline/suno_acc8_varnet256_r8/h5")
p.add_argument("--varnet-key", default="grid256_sense")
p.add_argument("--ours-metrics", default="results/sense_block/suno_acc8_sensegpt512_v21asym_latent_final_net_support/metrics_latent_dc.json")
p.add_argument("--varnet-metrics", default="results/sense_block/suno_acc8_varnet256_r8/metrics_grid256_sense.json")
p.add_argument("--split", default="fastmri/splits/suno_brain_test.txt")
p.add_argument("--out", required=True)
p.add_argument("--n-samples", type=int, default=4)
p.add_argument("--hybrid-width", type=float, default=0.1)
a = p.parse_args()
cs, tok = 256, 512
device = torch.device("cuda"); dtype = torch.bfloat16
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
meditok = load_meditok("weights/meditok/config_512.json", "weights/meditok/brain_mc_sense_cplxpair_res512_ep5_frozen.pth", device)
sd = torch.load(a.gpt_ckpt, map_location="cpu", weights_only=False)["model"]
gpt_args = SimpleNamespace(gpt_model="GPT-L", vocab_size=32768, image_size=tok, downsample_size=16, num_classes=8, cls_token_num=1, cond_token_num=(tok // 16) ** 2,
                           gpt_type="c2i", num_codebooks=8, num_output_layer=1, vq_embed_path=None, probe_model=False, compile=False, gpt_ckpt=a.gpt_ckpt,
                           **infer_model_args_from_state_dict(sd, default_mask_width=cs))
del sd
gpt = load_gpt(gpt_args, device, dtype)
sens_net = load_sens_net("results/sens_net/run1/best.pt", device=device, trainable=False)
nlf = acs_num_low_freqs(cs, 0.04); mask_func = create_mask_for_mask_type("random", [0.04], [8])
far_kwargs = dict(n_steps=1, far_mask_mode="radial", complex_mode="cartesian", center_fractions=0.04)
greedy = {"temperature": 1.0, "top_k": 0, "top_p": 1.0}
names = [l.strip() for l in open(a.split) if l.strip()]
m_ours = json.load(open(a.ours_metrics)); m_var = json.load(open(a.varnet_metrics))
up = lambda t: tok_upsample(t, tok, "bicubic"); down = lambda t: tok_downsample(t, cs)
rows = []
for name in names:
    with h5py.File(Path("fastmri/brain/multicoil_val") / name, "r") as f: ksp = f["kspace"][()]
    for d in ["fastmri/brain/multicoil_val_sens_maps_gpu", "fastmri/brain/multicoil_val_sens_maps"]:
        if (Path(d) / name).exists():
            with h5py.File(Path(d) / name, "r") as f: sens = f["sens_maps"][()]
            break
    S = ksp.shape[0]; mid = S // 2
    m, _ = mask_func((1, cs, cs, 2), seed=tuple(map(ord, name)))
    mask_cols = (m.reshape(-1) > 0).to(device); mask_hw = mask_cols[None, :].expand(cs, cs)
    with torch.no_grad():
        # ground truth of the whole volume (physical units) for the data range
        gt_vol = []
        for s in range(S):
            mp_ = torch.from_numpy(sens[s]).to(device)
            c = crop_native_to_grid(ifft2c(torch.from_numpy(ksp[s]).to(device)), mp_.shape[-2:], cs)[None]
            mp = center_crop_or_pad(mp_, (cs, cs))[None]
            gt_vol.append((sense_combine(c, mp).abs() * support_from_maps(mp).float())[0])
            if s == mid:
                coils, maps_or = c, mp
        gt_vol = torch.stack(gt_vol); rng = float(gt_vol.max())
        support = support_from_maps(maps_or)
        x0 = sense_combine(coils, maps_or); mv = x0.abs().flatten(1).max(1).values.clamp_min(1e-12)[:, None, None]
        x0n, coils_n = x0 / mv, coils / mv[:, None]
        k_meas = fft2c(coils_n) * mask_hw; mask_b = mask_cols[None, :].expand(1, cs)
        maps = estimate_maps(sens_net, k_meas, mask_b, num_coils=None, num_low_freqs=nlf)
        x_zf = sense_combine(ifft2c(k_meas), maps)
        dc_tok = wrap_dc_fn_resized(make_sense_dc_fn(k_meas, maps, mask_b), cs, tok, "bicubic")
        sample_mask = mask_hw[None, :, :, None].expand(1, cs, cs, 1).float().contiguous()
        with torch.autocast("cuda", dtype=dtype):
            rec, aux = reconstruct_far(gpt, meditok, up(c2r(x_zf)), sample_mask, sampling_kwargs=greedy, dc_fn=dc_tok, timestep_offset=0, rec_mode="latent", **far_kwargs)
            n = a.n_samples
            _, aux_s = reconstruct_far(gpt, meditok, up(c2r(x_zf)).repeat(n, 1, 1, 1), sample_mask.repeat(n, 1, 1, 1), sampling_kwargs={"temperature": 1.0, "top_k": 4096, "top_p": 1.0}, dc_fn=dc_tok, timestep_offset=0, **far_kwargs)
        lat = r2c(down(aux["decoded_no_dc"].float()))[0]; smp = r2c(down(aux_s["decoded_no_dc"].float()))   # with rec_mode=latent, decoded_no_dc IS the latent image (alt_no_dc = the greedy token path); (H,W), (n,H,W) complex, no DC
        yy, xx = torch.meshgrid(torch.arange(cs, device=device), torch.arange(cs, device=device), indexing="ij")
        rho = torch.sqrt(((yy - cs // 2) / (cs / 2)) ** 2 + ((xx - cs // 2) / (cs / 2)) ** 2)
        def lowpass(r, w=a.hybrid_width):
            return (rho <= r).float() + ((rho > r) & (rho < r + w)).float() * 0.5 * (1 + torch.cos(torch.pi * (rho - r) / w))
        dc256 = make_sense_dc_fn(k_meas, maps, mask_b)
        def dc(z):                                                                                  # (n,H,W) complex -> per-coil DC
            return r2c(dc256(c2r(z)))
        def hybrid(r, hf):
            W = lowpass(r); return dc(ifft2c(W * fft2c(lat) + (1 - W) * fft2c(hf))[None])[0]
        gt = (x0n.abs() * support.float() * mv)[0].cpu().numpy()                                   # physical units
        zf = (x_zf.abs() * support.float() * mv)[0].cpu().numpy()
        ours = (dc(lat[None]).abs() * support.float() * mv)[0].cpu().numpy()                      # plain DC on the 256 grid (--dc-path acq)
        hyb_s = (hybrid(0.5, smp[0]).abs() * support.float()[0] * mv[0]).cpu().numpy()
        hyb_m = (dc(smp).abs().mean(0, keepdim=True)[0] * support.float()[0] * mv[0]).cpu().numpy()   # posterior mean of the DC'd samples
    with h5py.File(Path(a.varnet_h5) / name, "r") as f:
        var = f[a.varnet_key][mid] * f["support"][mid]; gt_h5 = f["x0_ref"][mid] * f["support"][mid]
    gt_mismatch = float(np.abs(gt_h5 - gt).max() / gt.max())
    psnr = lambda r: float(10 * np.log10(rng ** 2 / np.mean((r - gt) ** 2)))
    pz, po, pv, ps_, pm_ = psnr(zf), psnr(ours), psnr(var), psnr(hyb_s), psnr(hyb_m)
    key = Path(name).stem; vo, vv = m_ours[key]["psnr"], m_var[key]["psnr"]
    rows.append(dict(name=name, slice=mid, n_slices=S, psnr_zf=pz, psnr_ours=po, psnr_hybrid_s=ps_, psnr_hybrid_m=pm_, psnr_varnet=pv, vol_ours=vo, vol_varnet=vv, gt_mismatch=gt_mismatch, lines=int(mask_cols.sum())))
    print(f"{name:36s} slice {mid:2d}/{S}  ZF {pz:.2f}  latent {po:.2f}  hybrid-s(0.5) {ps_:.2f}  ps-mean4 {pm_:.2f}  VarNet {pv:.2f}", flush=True)
    vmax = float(np.percentile(gt, 99.8))
    fig, ax = plt.subplots(2, 3, figsize=(11, 7.6))
    panels = [(zf, f"zero-filled input  {pz:.2f} dB"), (gt, "ground truth (SENSE)"), (var, f"VarNet fine-tuned, SENSE out  {pv:.2f} dB"),
              (ours, f"ours, latent one pass + DC  {po:.2f} dB"), (hyb_m, f"ours, posterior mean of 4 + DC  {pm_:.2f} dB"), (hyb_s, f"ours, hybrid r=0.5 + DC  {ps_:.2f} dB")]
    for axi, (img, title) in zip(ax.ravel(), panels):
        axi.imshow(img, cmap="gray", vmin=0, vmax=vmax); axi.set_title(title, fontsize=9); axi.axis("off")
    fig.suptitle(f"{name}  slice {mid} of {S}  x8 random mask ({int(mask_cols.sum())} columns)   per-volume PSNR: latent {vo:.2f}, VarNet {vv:.2f}", fontsize=10)
    fig.tight_layout(); fig.savefig(out / f"{Path(name).stem}.png", dpi=110); plt.close(fig)
json.dump(rows, open(out / "psnr_middle_slices.json", "w"), indent=1)
print(f"mean over the 15 middle slices: ZF {np.mean([r['psnr_zf'] for r in rows]):.2f}  latent {np.mean([r['psnr_ours'] for r in rows]):.2f}  hybrid-s(0.5) {np.mean([r['psnr_hybrid_s'] for r in rows]):.2f}  ps-mean4 {np.mean([r['psnr_hybrid_m'] for r in rows]):.2f}  VarNet {np.mean([r['psnr_varnet'] for r in rows]):.2f}")
