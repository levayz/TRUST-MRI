"""Confidence diagnostic for progressive commitment (queue item 5), no training.

On validation slices the SENSE GPT's pass 1 (greedy token path) gives per-position logits over the 16 chunks (2 streams x 8
codebooks). Three per-position confidence scores are compared against per-position error targets, to decide which score should
pick the positions whose codes are kept ("committed") between passes:
  conf_spread : concentration of the expected codebook embedding, |mu|^2 = 1 - spread, averaged over the 16 chunks
  conf_top1   : top-1 probability averaged over the 16 chunks
  conf_ent    : negative entropy of the chunk distributions, averaged (in nats)
  conf_draw   : agreement of N posterior draws with the greedy code, averaged over draws and chunks
Targets (lower = better position):
  err_lat     : squared distance between the codebook latents of the greedy codes and of the ground-truth codes (summed over chunks)
  err_latpre  : squared distance between the greedy codes' latent and the ground truth's pre-quantisation latent (the latent head's target)
  err_patch   : mean squared error of the decoded greedy image (no DC) against the ground truth over the position's 32 x 32 patch
  acc         : fraction of the 16 chunks whose greedy code equals the ground-truth code
Reports Spearman correlations and keep-top-f curves: for each slice keep the f most confident positions; ratio = mean error of the
kept positions / mean error of all positions (oracle = keep the f lowest-error positions; random = 1).
Usage (shared GPU, batch 1):
  CUDA_VISIBLE_DEVICES=5 python scripts/diag_commit_confidence.py --gpt-ckpt <run11 best_final.pt> --max-volumes 5 --output-dir results/diag_commit_confidence/run11
"""
import argparse, json, sys, math
from pathlib import Path
from types import SimpleNamespace
import h5py, numpy as np, torch, torch.nn.functional as F
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoregressive.sample.reconstruct_far import load_meditok, load_gpt, encode  # noqa: E402
from autoregressive.models.generate import reconstruct_far  # noqa: E402
from autoregressive.models.gpt_kar import infer_model_args_from_state_dict, codebook_from_quantizer  # noqa: E402
from dataset.data.subsample import create_mask_for_mask_type  # noqa: E402
from models.sense_ops import (acs_num_low_freqs, c2r, center_crop_or_pad, crop_native_to_grid, estimate_maps, fft2c, ifft2c,  # noqa: E402
                              make_sense_dc_fn, sense_combine, support_from_maps, tok_upsample, wrap_dc_fn_resized, latent_target)
from models.varnet_sens import load_sens_net  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="fastmri/brain/multicoil_val")
    p.add_argument("--sens-dirs", default="fastmri/brain/multicoil_val_sens_maps_gpu,fastmri/brain/multicoil_val_sens_maps")
    p.add_argument("--split", default="fastmri/splits/suno_brain_test.txt")
    p.add_argument("--max-volumes", type=int, default=5)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--token-size", type=int, default=512)
    p.add_argument("--acceleration", type=int, default=8)
    p.add_argument("--center-fractions", type=float, default=0.04)
    p.add_argument("--gpt-ckpt", required=True)
    p.add_argument("--gpt-model", default="GPT-L")
    p.add_argument("--encoder-ckpt", default="weights/meditok/brain_mc_sense_cplxpair_res512_ep5_frozen.pth")
    p.add_argument("--encoder-config", default="weights/meditok/config_512.json")
    p.add_argument("--sens-ckpt", default="results/sens_net/run1/best.pt")
    p.add_argument("--n-draws", type=int, default=8)
    p.add_argument("--top-k", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--keep-fracs", type=float, nargs="*", default=[0.2, 0.4, 0.6, 0.75])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64); rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / math.sqrt((ra * ra).sum() * (rb * rb).sum() + 1e-12))


@torch.no_grad()
def main():
    args = parse_args()
    cs, tok = args.crop_size, args.token_size
    device = torch.device("cuda")
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    meditok = load_meditok(args.encoder_config, args.encoder_ckpt, device)
    gpt_args = SimpleNamespace(gpt_model=args.gpt_model, vocab_size=32768, image_size=tok, downsample_size=16, num_classes=8, cls_token_num=1,
                               cond_token_num=(tok // 16) ** 2, gpt_type="c2i", num_codebooks=8, num_output_layer=1, vq_embed_path=None,
                               probe_model=False, gpt_ckpt=args.gpt_ckpt, compile=False)
    _sd = torch.load(args.gpt_ckpt, map_location="cpu"); _sd = _sd.get("model", _sd)
    for k, v in infer_model_args_from_state_dict(_sd, default_mask_width=cs).items():
        setattr(gpt_args, k, v)
    del _sd
    gpt = load_gpt(gpt_args, device, torch.bfloat16)
    if gpt_args.latent_head_type == 'fused':
        gpt.set_latent_codebook(meditok.quantizer)
    sens_net = load_sens_net(args.sens_ckpt, device=device, trainable=False)
    E = codebook_from_quantizer(meditok.quantizer).to(device)                      # (K, V, d) unit-norm rows
    K, V, d = E.shape
    nlf = acs_num_low_freqs(cs, args.center_fractions)
    mask_func = create_mask_for_mask_type("random", [args.center_fractions], [args.acceleration])
    far_kwargs = dict(n_steps=1, far_mask_mode="radial", complex_mode="cartesian", center_fractions=args.center_fractions)
    greedy_kw = {"temperature": 1.0, "top_k": 0, "top_p": 1.0}
    draw_kw = {"temperature": args.temperature, "top_k": args.top_k, "top_p": 1.0}
    names = [l.strip() for l in open(args.split) if l.strip()][:args.max_volumes]
    L = (tok // 16) ** 2; P = tok // (tok // 16)                                   # positions per slice, pixels per patch side
    conf = {k: [] for k in ("spread", "top1", "ent", "draw")}
    err = {k: [] for k in ("lat", "latpre", "patch", "acc")}
    slice_ids = []
    fg_all = []
    mid_maps = None

    for name in tqdm(names, "volumes"):
        with h5py.File(Path(args.data_root) / name, "r") as f:
            ksp = f["kspace"][()]
        sens = None
        for dname in args.sens_dirs.split(","):
            pth = Path(dname) / name
            if pth.exists():
                with h5py.File(pth, "r") as f:
                    sens = f["sens_maps"][()]
                break
        assert sens is not None, name
        S = ksp.shape[0]
        m, _ = mask_func((1, cs, cs, 2), seed=tuple(map(ord, name)))
        mask_cols = (m.reshape(-1) > 0).to(device)
        mask_hw = mask_cols[None, :].expand(cs, cs)
        for s0 in range(0, S, args.batch_size):
            sl = list(range(s0, min(S, s0 + args.batch_size))); B = len(sl)
            xs, mps = [], []
            for s in sl:
                mp_ = torch.from_numpy(sens[s]).to(device)
                xs.append(crop_native_to_grid(ifft2c(torch.from_numpy(ksp[s]).to(device)), mp_.shape[-2:], cs))
                mps.append(center_crop_or_pad(mp_, (cs, cs)))
            coils, maps_or = torch.stack(xs), torch.stack(mps)
            x0 = sense_combine(coils, maps_or)
            support = support_from_maps(maps_or)                                              # (B, cs, cs) bool
            mv = x0.abs().flatten(1).max(1).values.clamp_min(1e-12)[:, None, None]
            x0n, coils_n = x0 / mv, coils / mv[:, None]
            k_meas = fft2c(coils_n) * mask_hw
            mask_b = mask_cols[None, :].expand(B, cs)
            maps = estimate_maps(sens_net, k_meas, mask_b, num_coils=None, num_low_freqs=nlf)
            x_zf = sense_combine(ifft2c(k_meas), maps)
            dc_fn = make_sense_dc_fn(k_meas, maps, mask_b)
            dc_tok = wrap_dc_fn_resized(dc_fn, cs, tok, "bicubic")
            up = lambda t: tok_upsample(t, tok, "bicubic")
            masked_image = up(c2r(x_zf))
            gt_tok = up(c2r(x0n))                                                          # (B, tok, tok, 2)
            sample_mask = mask_hw[None, :, :, None].expand(B, cs, cs, 1).float().contiguous()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                gt_codes = torch.stack([encode(meditok, gt_tok[..., c].float()) for c in range(2)], dim=1).long()   # (B, 2, K, L)
                z_gt = torch.stack([latent_target(meditok, gt_tok[..., c].float()) for c in range(2)], dim=1)      # (B, 2, L, K*d)
                _, aux = reconstruct_far(gpt, meditok, masked_image, sample_mask, sampling_kwargs=greedy_kw, dc_fn=dc_tok, timestep_offset=0, **far_kwargs)
                g_codes = aux["codes"].long()                                                                      # (B, 2, K, L)
                logits = aux["logits_1d"].float()                                                                  # (B, L, 2, K, V)
                dec = aux["decoded_no_dc"].float()                                                                 # (B, tok, tok, 2)
                draws = []
                for _ in range(args.n_draws):
                    _, auxd = reconstruct_far(gpt, meditok, masked_image, sample_mask, sampling_kwargs=draw_kw, dc_fn=dc_tok, timestep_offset=0, **far_kwargs)
                    draws.append(auxd["codes"].long())
            probs = torch.softmax(logits, dim=-1)                                                                  # (B, L, 2, K, V)
            mu = torch.einsum("blskv,kvd->blskd", probs, E)                                                        # expected embedding per chunk
            c_spread = (mu ** 2).sum(-1).mean((2, 3))                                                              # (B, L)
            c_top1 = probs.max(-1).values.mean((2, 3))
            c_ent = -(-(probs * (probs.clamp_min(1e-12)).log()).sum(-1)).mean((2, 3))
            agree = torch.stack([(dr == g_codes).float() for dr in draws]).mean(0)                                  # (B, 2, K, L)
            c_draw = agree.mean((1, 2))                                                                            # (B, L)
            def lat(codes):                                                                                        # (B, 2, K, L) -> (B, 2, L, K*d)
                return torch.cat([E[k][codes[:, :, k]] for k in range(K)], dim=-1)
            lg, lgt = lat(g_codes), lat(gt_codes)
            e_lat = ((lg - lgt) ** 2).sum(-1).sum(1)                                                                # (B, L)
            e_latpre = ((lg - z_gt) ** 2).sum(-1).sum(1)
            e_patch = F.avg_pool2d(((dec - gt_tok) ** 2).sum(-1)[:, None], P)[:, 0].reshape(B, L)
            a_chunk = (g_codes == gt_codes).float().mean((1, 2))                                                   # (B, L)
            sup_tok = F.interpolate(support[:, None].float(), size=(tok, tok), mode="nearest")
            fg = (F.avg_pool2d(sup_tok, P)[:, 0].reshape(B, L) > 0.5)                                            # (B, L) foreground positions
            fg_all.append(fg.cpu().numpy())
            for k, v in zip(("spread", "top1", "ent", "draw"), (c_spread, c_top1, c_ent, c_draw)):
                conf[k].append(v.cpu().numpy())
            for k, v in zip(("lat", "latpre", "patch", "acc"), (e_lat, e_latpre, e_patch, a_chunk)):
                err[k].append(v.cpu().numpy())
            slice_ids += [(name, s) for s in sl]
            if mid_maps is None and S // 2 in sl:
                i = sl.index(S // 2)
                mid_maps = {"name": name, "slice": S // 2, "conf_spread": c_spread[i].cpu().numpy(), "conf_draw": c_draw[i].cpu().numpy(),
                            "err_patch": e_patch[i].cpu().numpy(), "err_lat": e_lat[i].cpu().numpy(), "fg": fg[i].cpu().numpy(), "side": int(math.sqrt(L))}

    conf = {k: np.concatenate(v) for k, v in conf.items()}; err = {k: np.concatenate(v) for k, v in err.items()}
    fg_all = np.concatenate(fg_all)
    n_slices = conf["spread"].shape[0]
    summary = {"gpt_ckpt": args.gpt_ckpt, "volumes": names, "slices": n_slices, "positions_per_slice": L, "n_draws": args.n_draws,
               "mean_chunk_accuracy": float(err["acc"].mean()), "foreground_fraction": float(fg_all.mean()), "regions": {}}
    lines = [f"confidence diagnostic on {n_slices} slices of {len(names)} volumes, {L} positions each, {args.n_draws} draws; mean chunk accuracy "
             f"{err['acc'].mean():.3f}; foreground (patch inside the ESPIRiT support) = {100 * fg_all.mean():.0f} percent of the positions"]

    def analyse(region, sel):                                      # sel (n_slices, L) bool: the positions the analysis is restricted to
        out = {"spearman": {}, "keep_top": {}}
        for ck in conf:
            out["spearman"][ck] = {ek: spearman(conf[ck][sel], err[ek][sel]) for ek in err}
        for f in args.keep_fracs:
            row = {ek: {} for ek in err}; ov = {ck: [] for ck in conf}
            for i in range(n_slices):
                m = sel[i]
                if m.sum() < 8: continue
                kk = max(1, int(round(f * m.sum())))
                base = {ek: err[ek][i][m].mean() for ek in err}
                o_idx = np.argsort(err["patch"][i][m])[:kk]
                for ck in conf:
                    c = conf[ck][i][m]; e_idx = np.argsort(-c)[:kk]
                    for ek in err:
                        row[ek].setdefault(ck, []).append(err[ek][i][m][e_idx].mean() / (base[ek] + 1e-12))
                    ov[ck].append(len(set(o_idx) & set(e_idx)) / kk)
                for ek in err:
                    e = err[ek][i][m]; oi = np.argsort(e if ek != "acc" else -e)[:kk]
                    row[ek].setdefault("oracle", []).append(e[oi].mean() / (base[ek] + 1e-12))
            res = {ek: {ck: float(np.mean(v)) for ck, v in row[ek].items()} for ek in err}
            res["overlap_with_oracle_patch"] = {ck: float(np.mean(v)) for ck, v in ov.items()}; res["overlap_with_oracle_patch"]["random"] = f
            out["keep_top"][f"{f:g}"] = res
        summary["regions"][region] = out
        lines.extend(["", f"=== {region} ===", "Spearman rank correlation (confidence vs error; negative = confident positions have lower error; vs acc positive is good)",
                      "conf      " + "".join(f"{ek:>10s}" for ek in err)])
        for ck in conf:
            lines.append(f"{ck:10s}" + "".join(f"{out['spearman'][ck][ek]:10.3f}" for ek in err))
        lines.append("keep-top-f: mean error of the kept positions / mean error of all positions of the region, per slice (oracle keeps the lowest-error positions; random = 1.00)")
        for f, row in out["keep_top"].items():
            lines.append(f"-- keep {float(f) * 100:.0f} percent")
            lines.append("target    " + "".join(f"{ck:>10s}" for ck in list(conf) + ["oracle"]))
            for ek in err:
                lines.append(f"{ek:10s}" + "".join(f"{row[ek][ck]:10.3f}" for ck in list(conf) + ["oracle"]))
            lines.append("overlap with the oracle selection (lowest patch error): " + ", ".join(f"{k} {v:.2f}" for k, v in row["overlap_with_oracle_patch"].items()))

    analyse("all positions", np.ones_like(fg_all, dtype=bool))
    analyse("foreground only", fg_all)
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=1)
    np.savez_compressed(out_dir / "per_position.npz", fg=fg_all, **{f"conf_{k}": v for k, v in conf.items()}, **{f"err_{k}": v for k, v in err.items()})
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
    if mid_maps is not None:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            n = mid_maps["side"]; fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
            fgm = mid_maps["fg"].reshape(n, n)
            for a, (k, t) in zip(ax, (("conf_spread", "confidence: |mu|^2 (foreground)"), ("conf_draw", "confidence: draw agreement (foreground)"), ("err_patch", "error: patch MSE"), ("err_lat", "error: latent distance"))):
                v = mid_maps[k].reshape(n, n).astype(float)
                if k.startswith("conf"): v = np.where(fgm, v, np.nan)
                im = a.imshow(v, cmap="viridis"); a.set_title(t, fontsize=9); a.axis("off"); plt.colorbar(im, ax=a, fraction=0.046)
            fig.suptitle(f"{mid_maps['name']} slice {mid_maps['slice']}", fontsize=9); fig.tight_layout(); fig.savefig(out_dir / "middle_slice_maps.png", dpi=110)
        except Exception as e:  # noqa: BLE001
            print("no figure:", e)


if __name__ == "__main__":
    main()
