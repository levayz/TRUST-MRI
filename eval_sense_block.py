#!/usr/bin/env python
"""Brain multicoil SENSE block (suno split): ZF, greedy, posterior sampling and tokenizer ceilings on
SENSE-combined images, with per-coil data consistency. TRUST-MRI conventions (256 crop, per-slice max,
no clipping, fastMRI per-volume PSNR/SSIM via compute_reconstruction_metrics).

Per slice:
  coils   x_c = crop(ifft2c(k_c)) to the ESPIRiT-map grid, then to --crop-size; oracle maps S_c cropped alike
  GT      x0  = sum_c conj(S_c) x_c / sum_c |S_c|^2 on the oracle map support      (SENSE combine, oracle maps)
  scale   everything divided by max|x0| of the slice (as the single-coil dataset does with max_val)
  mask    1D Cartesian random (cf, R) seeded from the filename -> the SAME columns as the single-coil runs
  meas    y_c = M fft2c(x_c)
  maps    --sens-maps oracle: the ESPIRiT maps;  --sens-maps net: maps estimated from the ACS lines of y_c by the
          sensitivity net (--sens-ckpt, models/varnet_sens.py), RSS-normalised, coil-0 gauge
  ZF      combine(ifft2c(y_c)) with the chosen maps
  DC(x)   per-coil: k_c = fft2c(S_c x); k_c <- y_c on M; x <- combine(ifft2c(k_c))   (chosen maps)
  --eval-support oracle: reconstructions are multiplied by the oracle support before scoring (the oracle x0 is 0
          outside it; net maps have no support, so ZF/DC with them carry background there)
Arms (magnitude vs |x0|):
  zero_filled | greedy_no_dc, greedy_dc | ps_mean_no_dc, ps_mean_dc (magnitude mean of N draws), ps_single_dc
  gt_reim (encode-decode x0 as re/im), gt_reim_dc, gt_mag (encode-decode 2|x0|-1)
The GPT (--gpt-ckpt) can be the single-coil brain model applied zero-shot or a SENSE-trained model
(autoregressive/train/train.py --dataset sense); the tokenizer (--encoder-ckpt) is used for both the GPT's
input tokens and the decoding, so it must match the GPT's training tokenizer.
"""
import argparse
import math
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from autoregressive.sample.reconstruct_far import load_meditok, load_gpt, enc_dec_image, encode, decode_codes  # noqa: E402
from autoregressive.models.generate import reconstruct_far, reconstruct_commit  # noqa: E402
from autoregressive.models.gpt_kar import infer_model_args_from_state_dict  # noqa: E402
from autoregressive.utils.metrics import compute_reconstruction_metrics  # noqa: E402
from dataset.data.subsample import create_mask_for_mask_type  # noqa: E402
from models.sense_ops import (acs_num_low_freqs, c2r, center_crop_or_pad, crop_native_to_grid, estimate_maps,  # noqa: E402
                              fft2c, ifft2c, make_sense_dc_fn, r2c, sense_combine, support_from_maps, tok_downsample,
                              tok_upsample, wrap_dc_fn_resized, decode_expected)
from models.varnet_sens import load_sens_net  # noqa: E402
from utils.logger import create_logger, log_args  # noqa: E402

BASE_ARMS = ("zero_filled", "greedy_no_dc", "greedy_dc", "ps_mean_no_dc", "ps_mean_dc", "ps_single_dc", "gt_reim", "gt_reim_dc", "gt_mag")
# extra arms: exp{T}_no_dc / exp{T}_dc = expected-embedding decoding of the greedy logits (softmax temperature T);
#             iter{N}_dc = N rounds of GPT -> per-coil DC (the DC'd reconstruction is re-tokenised as the next input)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="fastmri/brain/multicoil_val")
    p.add_argument("--sens-dirs", default="fastmri/brain/multicoil_val_sens_maps_gpu,fastmri/brain/multicoil_val_sens_maps")
    p.add_argument("--split", default="fastmri/splits/suno_brain_test.txt")
    p.add_argument("--max-volumes", type=int, default=0)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--acceleration", type=float, required=True)
    p.add_argument("--center-fractions", type=float, default=0.04)
    p.add_argument("--mask-type", default="random", choices=["random", "equispaced", "equispaced_fraction"], help="test-time mask: random (training distribution) | equispaced (every R-th column + ACS) | equispaced_fraction (fastMRI VarNet convention, N/R columns in total)")
    p.add_argument("--gpt-ckpt", default=None, help="GPT-L checkpoint (single-coil brain model or SENSE-trained); omit for ZF + tokenizer rows only")
    p.add_argument("--gpt-model", default="GPT-L")
    p.add_argument("--encoder-ckpt", default="weights/meditok/meditok_simple_v1.pth")
    p.add_argument("--encoder-config", default="weights/meditok/config.json")
    p.add_argument("--token-size", type=int, default=0, help="tokenizer / GPT grid (0 = --crop-size); e.g. 512 with the res512 SENSE tokenizer: the SENSE image is bicubic-upsampled before tokenisation and the decoded image bilinear-downsampled back")
    p.add_argument("--resize-mode", default="bicubic")
    p.add_argument("--sens-maps", default="oracle", choices=["oracle", "net"], help="maps for ZF and DC (x0 is always oracle)")
    p.add_argument("--sens-ckpt", default=None, help="sensitivity net checkpoint (train_sens_net.py best.pt) for --sens-maps net")
    p.add_argument("--eval-support", default="none", choices=["none", "oracle"], help="multiply reconstructions by the oracle support before scoring")
    p.add_argument("--gt-ref", default="sense", choices=["sense", "rss", "both"], help="reference for the metrics: oracle SENSE |x0| (support applies), RSS of the fully sampled coils (no support), or both (writes metrics_<arm>_vs_rss.json and summary['vs_rss'])")
    p.add_argument("--ps-n-samples", type=int, default=8)
    p.add_argument("--exp-temperatures", type=float, nargs="*", default=[1.0], help="expected-embedding decoding of the greedy logits at these softmax temperatures ([] = off)")
    p.add_argument("--dc-path", default="token", choices=["token", "acq"], help="where the final per-coil DC of the *_dc arms is applied: token = on the token grid inside a resize round trip (down -> DC -> up -> down; the historical path, a low-pass after DC) | acq = plain DC on the acquisition grid from the decoded (no-DC) images")
    p.add_argument("--hybrid-radii", type=float, nargs="*", default=[], help="hybrid decoding: low frequencies (normalised k-space radius <= r) from the latent one pass, high frequencies from a token decoding (g = greedy, s = one posterior sample, m = mean of the samples), merged in k-space before DC; arms hybrid_{g,s,m}_r<r>_dc")
    p.add_argument("--hybrid-width", type=float, default=0.1, help="width of the cosine ramp of the hybrid low-pass window (normalised radius)")
    p.add_argument("--iters", type=int, default=1, help=">1: iterated greedy reconstruction (GPT -> DC -> re-tokenise), arms iter2_dc..iterN_dc")
    p.add_argument("--refine-rounds", type=int, default=0, help="models trained with --refine-rounds: N refinement passes (timestep k) after the greedy pass; arms refine{k}_dc; exp/PS arms are computed on the last pass")
    p.add_argument("--commit-schedule", type=float, nargs="*", default=[], help="two-view models trained with --commit: progressive-commitment passes at these committed fractions after the greedy pass; arms commit{k}_greedy_dc / commit{k}_latent_dc")
    p.add_argument("--commit-mode", default="conf", choices=["conf", "random"])
    p.add_argument("--commit-view", default="expected", choices=["expected", "codes"], help="codes = MaskGIT-style hard commitment of sampled codes (models trained with --commit gt): adds commit_single_dc, commit_ps_mean_dc, commit_latent_dc, commit_latent_mean_dc and hybrid_c* arms")
    p.add_argument("--top-k", type=int, default=4096)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=2, help="slices per batch (x ps-n-samples rows for PS)")
    p.add_argument("--ps-masks-dir", default=None, help="masks/ dir of the single-coil ps_recon run to verify identical columns")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cs = args.crop_size
    tok = args.token_size or cs
    device = torch.device("cuda")
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(out_dir); log_args(args, logger)

    meditok = load_meditok(args.encoder_config, args.encoder_ckpt, device)
    gpt = None
    if args.gpt_ckpt:
        gpt_args = SimpleNamespace(gpt_model=args.gpt_model, vocab_size=32768, image_size=tok, downsample_size=16, num_classes=8,
                                   cls_token_num=1, cond_token_num=(tok // 16) ** 2, gpt_type="c2i", num_codebooks=8,
                                   num_output_layer=1, vq_embed_path=None, probe_model=False, gpt_ckpt=args.gpt_ckpt, compile=False)
        _ck = torch.load(args.gpt_ckpt, map_location="cpu")
        _sd = _ck.get("model", _ck)
        for k, v in infer_model_args_from_state_dict(_sd, default_mask_width=cs).items():
            setattr(gpt_args, k, v)
        del _ck, _sd
        gpt = load_gpt(gpt_args, device, torch.bfloat16)
        if gpt_args.latent_head_type == 'fused':
            gpt.set_latent_codebook(meditok.quantizer)
        if gpt_args.latent_dim:
            logger.info(f"GPT has a latent-regression head (dim {gpt_args.latent_dim}, type {gpt_args.latent_head_type}): arms latent_no_dc / latent_dc added")
    sens_net = None
    if args.sens_maps == "net":
        assert args.sens_ckpt, "--sens-ckpt is required with --sens-maps net"
        sens_net = load_sens_net(args.sens_ckpt, device=device, trainable=False)
        logger.info(f"sensitivity net: {args.sens_ckpt} ({sum(p.numel() for p in sens_net.parameters()):,} params)")
    nlf = acs_num_low_freqs(cs, args.center_fractions)
    _acc = int(args.acceleration) if float(args.acceleration).is_integer() else args.acceleration   # the equispaced mask slices with it
    mask_func = create_mask_for_mask_type(args.mask_type, [args.center_fractions], [_acc])
    if args.mask_type != "random":
        logger.info(f"sampling mask: {args.mask_type} (R={_acc}, cf {args.center_fractions}); the random mask is the training distribution")
    far_kwargs = dict(n_steps=1, far_mask_mode="radial", complex_mode="cartesian", center_fractions=args.center_fractions)
    greedy_kw = {"temperature": 1.0, "top_k": 0, "top_p": 1.0}
    ps_kw = {"temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p}

    arms = list(BASE_ARMS) + (["latent_no_dc", "latent_dc"] if (gpt is not None and getattr(gpt, 'latent_head', None) is not None) else []) + [f"refine{k}_no_dc" for k in range(1, args.refine_rounds + 1)] + [f"refine{k}_dc" for k in range(1, args.refine_rounds + 1)] \
        + [a for t in args.exp_temperatures for a in (f"exp{t:g}_no_dc", f"exp{t:g}_dc")] + [f"iter{i}_dc" for i in range(2, args.iters + 1)] + [f"commit{k}_{src}_dc" for k in range(1, len(args.commit_schedule) + 1) for src in ("greedy", "latent")] \
        + (["commit_single_dc", "commit_ps_mean_dc", "commit_latent_dc", "commit_latent_mean_dc"] + [f"hybrid_{src}_r{r:g}_dc" for r in args.hybrid_radii for src in ("c", "cm")] if (args.commit_schedule and args.commit_view == "codes") else []) \
        + [f"hybrid_{src}_r{r:g}_{k}" for r in args.hybrid_radii for src in ("g", "s", "m") for k in ("dc", "dcs")] + (["latent_dc256", "greedy_dc256"] if args.hybrid_radii else [])
    if args.hybrid_radii:
        assert gpt is not None and getattr(gpt, 'latent_head', None) is not None, "--hybrid-radii needs a model with a latent head"
        yy, xx = torch.meshgrid(torch.arange(cs, device=device), torch.arange(cs, device=device), indexing="ij")
        rho = torch.sqrt(((yy - cs // 2) / (cs / 2)) ** 2 + ((xx - cs // 2) / (cs / 2)) ** 2)            # normalised radius, 0 at the k-space centre
        def lowpass(r):                                                                                   # 1 inside r, cosine ramp to 0 at r + width
            w = args.hybrid_width
            return ((rho <= r).float() + ((rho > r) & (rho < r + w)).float() * 0.5 * (1 + torch.cos(math.pi * (rho - r) / w)))
        hybrid_windows = {r: lowpass(r) for r in args.hybrid_radii}
    names = [l.strip() for l in open(args.split) if l.strip()]
    if args.max_volumes > 0:
        names = names[:args.max_volumes]
    bufs = {}
    bufs_rss = {}
    refs = {"sense": args.gt_ref in ("sense", "both"), "rss": args.gt_ref in ("rss", "both")}
    mask_check = {"dir": args.ps_masks_dir, "volumes_checked": 0, "mismatches": 0, "missing": 0}
    agreement = {}   # volume -> per-slice map agreement with the oracle (net mode)

    def add(arm, key, sl_list, mags_phys, gt_phys, store=None):
        b = (bufs if store is None else store).setdefault(arm, {}).setdefault(key, {"rec": [], "gt": [], "slice": []})
        for i, s in enumerate(sl_list):
            b["rec"].append(mags_phys[i]); b["gt"].append(gt_phys[i]); b["slice"].append(s)

    for name in tqdm(names, "volumes"):
        key = Path(name).stem
        with h5py.File(Path(args.data_root) / name, "r") as f:
            ksp = f["kspace"][()]                                  # (S, C, H, W) complex64
        sens = None
        for d in args.sens_dirs.split(","):
            p = Path(d) / name
            if p.exists():
                with h5py.File(p, "r") as f:
                    sens = f["sens_maps"][()]                      # (S, C, Hr, Wr) complex64
                break
        assert sens is not None, f"no sens maps for {name}"
        S = ksp.shape[0]
        m, _ = mask_func((1, cs, cs, 2), seed=tuple(map(ord, name)))
        mask_cols = (m.reshape(-1) > 0).to(device)                                   # (W,)
        if args.ps_masks_dir:
            mp = Path(args.ps_masks_dir) / f"{key}_mask.pt"
            if mp.exists():
                saved = torch.load(mp)[0, 0, :, 0].bool().to(device)
                mask_check["volumes_checked"] += 1
                mask_check["mismatches"] += int(not torch.equal(saved, mask_cols))
            else:
                mask_check["missing"] += 1
        mask_hw = mask_cols[None, :].expand(cs, cs)                                 # (H, W) bool

        for s0 in range(0, S, args.batch_size):
            sl = list(range(s0, min(S, s0 + args.batch_size)))
            B = len(sl)
            xs, mps = [], []
            for s in sl:
                mp_ = torch.from_numpy(sens[s]).to(device)
                xs.append(crop_native_to_grid(ifft2c(torch.from_numpy(ksp[s]).to(device)), mp_.shape[-2:], cs))
                mps.append(center_crop_or_pad(mp_, (cs, cs)))
            coils, maps_or = torch.stack(xs), torch.stack(mps)                        # (B, C, H, W)
            support = support_from_maps(maps_or)                                      # (B, H, W) bool

            x0 = sense_combine(coils, maps_or)                                        # (B, H, W) complex, physical units
            max_val = x0.abs().flatten(1).max(1).values.clamp_min(1e-12)
            mv = max_val[:, None, None]
            x0n, coils_n = x0 / mv, coils / mv[:, None]
            k_meas = fft2c(coils_n) * mask_hw
            mask_b = mask_cols[None, :].expand(B, cs)                                  # (B, W)
            if sens_net is not None:
                maps = estimate_maps(sens_net, k_meas, mask_b, num_coils=None, num_low_freqs=nlf)
                ag = (maps.conj() * maps_or).sum(1).abs()
                agreement.setdefault(key, []).extend(float(ag[b][support[b]].mean()) for b in range(B))
            else:
                maps = maps_or
            x_zf = sense_combine(ifft2c(k_meas), maps)
            dc_fn = make_sense_dc_fn(k_meas, maps, mask_b)
            dc_tok = wrap_dc_fn_resized(dc_fn, cs, tok, args.resize_mode)                 # DC for token-grid images
            up = lambda t: tok_upsample(t, tok, args.resize_mode)
            down = lambda t: tok_downsample(t, cs)

            def dc(xhat):                                                             # per-coil hard DC, complex in/out
                return r2c(dc_fn(c2r(xhat)))

            def dc_arm(no_dc_tok, rec_tok):                                           # the *_dc arm of a token-grid decoding
                return dc(r2c(down(no_dc_tok))) if args.dc_path == "acq" else r2c(down(rec_tok))

            sup = support.float() if args.eval_support == "oracle" else torch.ones_like(support, dtype=torch.float32)
            gt_phys = x0.abs().cpu().numpy()
            rss_phys = (coils.abs() ** 2).sum(1).sqrt().cpu().numpy()                 # RSS of the fully sampled coils, physical units

            def rec_m(arm, mag):
                if refs["sense"]:
                    add(arm, key, sl, (mag * sup * mv).cpu().numpy(), gt_phys)
                if refs["rss"]:
                    add(arm, key, sl, (mag * mv).cpu().numpy(), rss_phys, store=bufs_rss)   # no support: RSS has background too

            def rec_c(arm, z):
                rec_m(arm, z.abs())

            rec_c("zero_filled", x_zf)
            rt = r2c(down(enc_dec_image(meditok, up(c2r(x0n)))))
            rec_c("gt_reim", rt); rec_c("gt_reim_dc", dc(rt))
            if tok == cs:   # magnitude round trip only meaningful on the native grid (and only for magnitude-trained tokenizers)
                dec = decode_codes(meditok, encode(meditok, x0n.abs() * 2 - 1))
                rec_m("gt_mag", ((dec.mean(1) + 1) / 2).clamp(0, 1))

            if gpt is not None:
                masked_image = up(c2r(x_zf))                                               # token grid
                sample_mask = mask_hw[None, :, :, None].expand(B, cs, cs, 1).float().contiguous()   # acquisition grid
                rec, aux = reconstruct_far(gpt, meditok, masked_image, sample_mask, sampling_kwargs=greedy_kw, dc_fn=dc_tok, timestep_offset=0, **far_kwargs)
                rec_c("greedy_no_dc", r2c(down(aux["decoded_no_dc"]))); rec_c("greedy_dc", dc_arm(aux["decoded_no_dc"], rec))
                if aux.get("alt_rec") is not None:   # latent-regression head
                    rec_c("latent_no_dc", r2c(down(aux["alt_no_dc"]))); rec_c("latent_dc", dc_arm(aux["alt_no_dc"], aux["alt_rec"]))
                    if args.hybrid_radii:
                        hy_lat, hy_grd = r2c(down(aux["alt_no_dc"])), r2c(down(aux["decoded_no_dc"]))       # (B, H, W) complex, no DC
                        # the standard *_dc arms apply DC on the token grid (down -> DC -> up, then down for scoring): the resize round trip
                        # after DC is a low-pass. *_dc256 = plain DC on the acquisition grid, for comparison.
                        rec_c("latent_dc256", dc(hy_lat)); rec_c("greedy_dc256", dc(hy_grd))
                ps_input, ps_toff = masked_image, 0
                for k in range(1, args.refine_rounds + 1):            # refinement passes on the DC'd previous output
                    ps_input, ps_toff = rec, k
                    if getattr(gpt, 'draft_cond', False):   # two-view model: zero-filled input + previous reconstruction as the second view
                        rec, aux = reconstruct_far(gpt, meditok, masked_image, sample_mask, sampling_kwargs=greedy_kw, dc_fn=dc_tok, timestep_offset=k, draft_image=rec, **far_kwargs)
                    else:
                        rec, aux = reconstruct_far(gpt, meditok, rec, sample_mask, sampling_kwargs=greedy_kw, dc_fn=dc_tok, timestep_offset=k, **far_kwargs)
                    rec_c(f"refine{k}_no_dc", r2c(down(aux["decoded_no_dc"]))); rec_c(f"refine{k}_dc", r2c(down(rec)))
                if args.commit_schedule:                                # progressive commitment passes after the greedy pass
                    assert getattr(gpt, "draft_cond", False), "--commit-schedule needs a two-view model trained with --commit"
                    _, auxc = reconstruct_commit(gpt, meditok, masked_image, sample_mask, sampling_kwargs=greedy_kw, dc_fn=dc_tok,
                                                 commit_schedule=args.commit_schedule, commit_mode=args.commit_mode, commit_view=args.commit_view, **far_kwargs)
                    for k, ps_ in enumerate(auxc["commit_passes"], 1):
                        rec_c(f"commit{k}_greedy_dc", dc_arm(ps_["main_no_dc"], ps_["main"]))
                        if ps_.get("alt_no_dc") is not None:
                            rec_c(f"commit{k}_latent_dc", dc_arm(ps_["alt_no_dc"], ps_["alt"]))
                if args.exp_temperatures:
                    lg = aux["logits"]                                                   # (B, Hp, Wp, 2, K, V) greedy chain
                    lg = lg.reshape(lg.shape[0], -1, *lg.shape[3:]).permute(0, 2, 3, 1, 4)    # (B, 2, K, L, V)
                    for t in args.exp_temperatures:
                        x_exp = decode_expected(meditok, lg, t)                            # (B, tok, tok, 2)
                        rec_c(f"exp{t:g}_no_dc", r2c(down(x_exp))); rec_c(f"exp{t:g}_dc", dc(r2c(down(x_exp))) if args.dc_path == "acq" else r2c(down(dc_tok(x_exp))))
                    del lg
                x_it = rec
                for it in range(2, args.iters + 1):
                    x_it, _ = reconstruct_far(gpt, meditok, x_it, sample_mask, sampling_kwargs=greedy_kw, dc_fn=dc_tok, **far_kwargs)
                    rec_c(f"iter{it}_dc", r2c(down(x_it)))
                n = args.ps_n_samples
                rec, aux = reconstruct_far(gpt, meditok, ps_input.repeat(n, 1, 1, 1), sample_mask.repeat(n, 1, 1, 1),
                                           sampling_kwargs=ps_kw, dc_fn=dc_tok, timestep_offset=ps_toff, **far_kwargs)
                smp = r2c(down(aux["decoded_no_dc"])).reshape(n, B, cs, cs)
                smp_dc = dc(smp.reshape(n * B, cs, cs)).reshape(n, B, cs, cs) if args.dc_path == "acq" else r2c(down(rec)).reshape(n, B, cs, cs)
                rec_m("ps_mean_no_dc", smp.abs().mean(0)); rec_m("ps_mean_dc", smp_dc.abs().mean(0)); rec_c("ps_single_dc", smp_dc[0])
                if args.commit_schedule and args.commit_view == "codes":
                    # MaskGIT-style hard commitment on n sampled draws: coherent samples, their magnitude mean, the last pass's latent head
                    n = args.ps_n_samples
                    _, auxs = reconstruct_commit(gpt, meditok, masked_image.repeat(n, 1, 1, 1), sample_mask.repeat(n, 1, 1, 1), sampling_kwargs=ps_kw, dc_fn=dc_tok,
                                                 commit_schedule=args.commit_schedule, commit_mode=args.commit_mode, commit_view="codes", **far_kwargs)
                    last = auxs["commit_passes"][-1]
                    csmp = r2c(down(last["main_no_dc"])).reshape(n, B, cs, cs)                      # committed codes decoded, no DC
                    clat = r2c(down(last["alt_no_dc"])).reshape(n, B, cs, cs) if last.get("alt_no_dc") is not None else None
                    csmp_dc = dc(csmp.reshape(n * B, cs, cs)).reshape(n, B, cs, cs) if args.dc_path == "acq" else r2c(down(last["main"])).reshape(n, B, cs, cs)
                    rec_c("commit_single_dc", csmp_dc[0]); rec_m("commit_ps_mean_dc", csmp_dc.abs().mean(0))
                    if clat is not None:
                        clat_dc = dc(clat.reshape(n * B, cs, cs)).reshape(n, B, cs, cs) if args.dc_path == "acq" else r2c(down(last["alt"])).reshape(n, B, cs, cs)
                        rec_c("commit_latent_dc", clat_dc[0]); rec_m("commit_latent_mean_dc", clat_dc.abs().mean(0))
                    if args.hybrid_radii:
                        k_lat_c = fft2c(hy_lat)
                        for src, img in (("c", csmp[0]), ("cm", csmp.mean(0))):
                            k_src = fft2c(img)
                            for r, W in hybrid_windows.items():
                                rec_c(f"hybrid_{src}_r{r:g}_dc", dc(ifft2c(W * k_lat_c + (1 - W) * k_src)))
                    del auxs
                if args.hybrid_radii:
                    # hybrid decoding: latent one pass below the radius, a token decoding above it, merged in k-space, then per-coil DC
                    k_lat = fft2c(hy_lat)
                    for src, img in (("g", hy_grd), ("s", smp[0]), ("m", smp.mean(0))):
                        k_src = fft2c(img)
                        for r, W in hybrid_windows.items():
                            h = ifft2c(W * k_lat + (1 - W) * k_src)
                            rec_c(f"hybrid_{src}_r{r:g}_dc", dc(h))                                     # plain DC on the acquisition grid
                            rec_c(f"hybrid_{src}_r{r:g}_dcs", r2c(down(dc_tok(up(c2r(h))))))              # the standard arms' path: DC + resize round trip

    summary = {"dc_path": args.dc_path, "acceleration": args.acceleration, "center_fractions": args.center_fractions, "split": args.split,
               "gpt_ckpt": args.gpt_ckpt, "gpt_model": args.gpt_model, "encoder_ckpt": args.encoder_ckpt,
               "sens_maps": args.sens_maps, "sens_ckpt": args.sens_ckpt, "eval_support": args.eval_support,
               "crop_size": cs, "token_size": tok, "resize_mode": args.resize_mode, "encoder_config": args.encoder_config,
               "ps": {"n_samples": args.ps_n_samples, "top_k": args.top_k, "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed},
               "exp_temperatures": args.exp_temperatures, "iters": args.iters, "refine_rounds": args.refine_rounds, "commit_schedule": args.commit_schedule, "commit_mode": args.commit_mode, "commit_view": args.commit_view,
               "mask_check": mask_check,
               "gt_ref": args.gt_ref,
               "n_volumes": len((bufs or bufs_rss)["zero_filled"]), "n_slices": sum(len(v["slice"]) for v in (bufs or bufs_rss)["zero_filled"].values())}
    if agreement:
        summary["map_agreement_support"] = {"mean": float(np.mean([a for v in agreement.values() for a in v])),
                                            "per_volume": {k: float(np.mean(v)) for k, v in agreement.items()}}
        logger.info(f"net maps: mean agreement with the oracle inside the support {summary['map_agreement_support']['mean']:.4f}")
    for arm in arms:
        if arm not in bufs:
            continue
        mtr = compute_reconstruction_metrics(bufs[arm], device)
        with open(out_dir / f"metrics_{arm}.json", "w") as f:
            json.dump(mtr, f, indent=2)
        summary[arm] = {k: float(v) for k, v in mtr.items() if k.endswith("_mean")}
        logger.info(f"{arm:14s} PSNR {mtr['psnr_mean']:.3f}  SSIM {mtr['ssim_mean']:.4f}  LPIPS {mtr['lpips_mean']:.4f}  DISTS {mtr['dists_mean']:.4f}  NMSE {mtr['nmse_mean']:.4f}")
    if bufs_rss:
        summary["vs_rss"] = {}
        for arm in arms:
            if arm not in bufs_rss:
                continue
            mtr = compute_reconstruction_metrics(bufs_rss[arm], device)
            with open(out_dir / f"metrics_{arm}_vs_rss.json", "w") as f:
                json.dump(mtr, f, indent=2)
            summary["vs_rss"][arm] = {k: float(v) for k, v in mtr.items() if k.endswith("_mean")}
            logger.info(f"[vs RSS] {arm:14s} PSNR {mtr['psnr_mean']:.3f}  SSIM {mtr['ssim_mean']:.4f}  LPIPS {mtr['lpips_mean']:.4f}  DISTS {mtr['dists_mean']:.4f}  NMSE {mtr['nmse_mean']:.4f}")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"mask check: {mask_check}")
    logger.info(f"{summary['n_volumes']} volumes, {summary['n_slices']} slices -> {out_dir}")


if __name__ == "__main__":
    main()
