#!/usr/bin/env python
"""Diagnostic for the teacher-student refinement model (train.py --refine-rounds): does the second pass use its input?

For a few suno slices it feeds the trained student different conditioning tokens at timestep 0 / 1 and reports
  * the teacher-forced CE of the clean (GT) tokens,
  * the token agreement between the conditioning tokens and the GT tokens (how much the input already "knows"),
  * the top-1 token accuracy of the student's prediction,
  * the PSNR (per slice, oracle support, vs |x0|) of the decoded + DC'd reconstruction.
Inputs: zero-filled image (pass 1), the student's own greedy+DC output (pass 1 -> refine), the EMA teacher's sampled+DC
draft (what training used), and the ground-truth image itself (if the model uses its input at timestep 1, feeding the GT
must give ~0 CE and the tokenizer ceiling).

Usage (meditok env, from the repo root):
  PYTHONPATH=$(pwd) python scripts/diag_refine.py --gpt-ckpt <run3 best.pt> --out results/sense_block/diag_refine
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoregressive.sample.reconstruct_far import load_meditok, load_gpt, encode  # noqa: E402
from autoregressive.models.generate import reconstruct_far  # noqa: E402
from dataset.data.subsample import create_mask_for_mask_type  # noqa: E402
from models.sense_ops import (acs_num_low_freqs, c2r, center_crop_or_pad, crop_native_to_grid, estimate_maps, fft2c,  # noqa: E402
                              ifft2c, make_sense_dc_fn, r2c, sense_combine, support_from_maps, tok_downsample,
                              tok_upsample, wrap_dc_fn_resized, latent_target)
from models.varnet_sens import load_sens_net  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpt-ckpt", required=True)
    p.add_argument("--encoder-ckpt", default="weights/meditok/brain_mc_sense_cplxpair_res512_ep5_frozen.pth")
    p.add_argument("--encoder-config", default="weights/meditok/config_512.json")
    p.add_argument("--sens-ckpt", default="results/sens_net/run1/best.pt")
    p.add_argument("--data-root", default="fastmri/brain/multicoil_val")
    p.add_argument("--sens-dirs", default="fastmri/brain/multicoil_val_sens_maps_gpu,fastmri/brain/multicoil_val_sens_maps")
    p.add_argument("--split", default="fastmri/splits/suno_brain_test.txt")
    p.add_argument("--max-volumes", type=int, default=3)
    p.add_argument("--slices-per-volume", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--token-size", type=int, default=512)
    p.add_argument("--resize-mode", default="bicubic")
    p.add_argument("--acceleration", type=int, default=8)
    p.add_argument("--center-fractions", type=float, default=0.04)
    p.add_argument("--draft-samples", type=int, default=1, help="teacher drafts per slice (first is used; all report PSNR)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/sense_block/diag_refine")
    return p.parse_args()


def psnr_slices(rec_mag, gt_mag):
    """per-slice PSNR, data range = per-slice GT max (rec/gt: (B,H,W) tensors)."""
    out = []
    for r, g in zip(rec_mag, gt_mag):
        mse = ((r - g) ** 2).mean()
        out.append(float(10 * torch.log10(g.max() ** 2 / mse.clamp_min(1e-20))))
    return out


def main():
    a = parse_args()
    cs, tok = a.crop_size, a.token_size
    device = torch.device("cuda")
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    meditok = load_meditok(a.encoder_config, a.encoder_ckpt, device)
    gpt_args = SimpleNamespace(gpt_model="GPT-L", vocab_size=32768, image_size=tok, downsample_size=16, num_classes=8, cls_token_num=1,
                               cond_token_num=(tok // 16) ** 2, gpt_type="c2i", num_codebooks=8, num_output_layer=1, vq_embed_path=None,
                               probe_model=False, gpt_ckpt=a.gpt_ckpt, compile=False, latent_dim=0)
    ck = torch.load(a.gpt_ckpt, map_location="cpu", weights_only=False)
    _sd = ck["model"]
    gpt_args.model_version = 2.1 if "in_proj.0.weight" in _sd else (2 if "mask_embed.weight" in _sd else 1)
    gpt_args.mask_width = int(_sd["mask_embed.weight"].shape[1]) if "mask_embed.weight" in _sd else cs
    gpt_args.draft_cond = "draft_ln.weight" in _sd
    student = load_gpt(gpt_args, device, torch.bfloat16)
    teacher = None
    if "ema" in ck:
        teacher = load_gpt(gpt_args, device, torch.bfloat16)
        teacher.load_state_dict(ck["ema"], strict=True); teacher.eval()
    del ck
    sens_net = load_sens_net(a.sens_ckpt, device=device, trainable=False)
    nlf = acs_num_low_freqs(cs, a.center_fractions)
    mask_func = create_mask_for_mask_type("random", [a.center_fractions], [a.acceleration])
    far_kwargs = dict(n_steps=1, far_mask_mode="radial", complex_mode="cartesian", center_fractions=a.center_fractions)
    greedy = {"temperature": 1.0, "top_k": 0, "top_p": 1.0}
    sample_kw = {"temperature": 1.0, "top_k": 32768 // 8, "top_p": 1.0}   # train.py --refine-draft sample

    def tokens(img_bhw2):                                                     # (B,H,W,2) token grid -> (B,2,K,L)
        return torch.cat((encode(meditok, img_bhw2[..., 0].float()).unsqueeze(1), encode(meditok, img_bhw2[..., 1].float()).unsqueeze(1)), 1)

    names = [l.strip() for l in open(a.split) if l.strip()][: a.max_volumes]
    ce, acc_in, acc_pred, ps, ps_nodc = {}, {}, {}, {}, {}
    rows = []
    for name in names:
        with h5py.File(Path(a.data_root) / name, "r") as f:
            ksp = f["kspace"][()]
        sens = None
        for d in a.sens_dirs.split(","):
            pth = Path(d) / name
            if pth.exists():
                with h5py.File(pth, "r") as f:
                    sens = f["sens_maps"][()]
                break
        assert sens is not None, name
        S = ksp.shape[0]
        mid = S // 2
        sl_all = list(range(max(0, mid - a.slices_per_volume // 2), min(S, mid - a.slices_per_volume // 2 + a.slices_per_volume)))
        m, _ = mask_func((1, cs, cs, 2), seed=tuple(map(ord, name)))
        mask_cols = (m.reshape(-1) > 0).to(device)
        mask_hw = mask_cols[None, :].expand(cs, cs)
        for s0 in range(0, len(sl_all), a.batch_size):
            sl = sl_all[s0:s0 + a.batch_size]; B = len(sl)
            xs, mps = [], []
            for s in sl:
                mp_ = torch.from_numpy(sens[s]).to(device)
                xs.append(crop_native_to_grid(ifft2c(torch.from_numpy(ksp[s]).to(device)), mp_.shape[-2:], cs))
                mps.append(center_crop_or_pad(mp_, (cs, cs)))
            coils, maps_or = torch.stack(xs), torch.stack(mps)
            support = support_from_maps(maps_or)
            x0 = sense_combine(coils, maps_or)
            max_val = x0.abs().flatten(1).max(1).values.clamp_min(1e-12); mv = max_val[:, None, None]
            x0n, coils_n = x0 / mv, coils / mv[:, None]
            k_meas = fft2c(coils_n) * mask_hw
            mask_b = mask_cols[None, :].expand(B, cs)
            maps = estimate_maps(sens_net, k_meas, mask_b, num_coils=None, num_low_freqs=nlf)
            x_zf = sense_combine(ifft2c(k_meas), maps)
            dc_fn = make_sense_dc_fn(k_meas, maps, mask_b)
            dc_tok = wrap_dc_fn_resized(dc_fn, cs, tok, a.resize_mode)
            up = lambda t: tok_upsample(t, tok, a.resize_mode)
            down = lambda t: tok_downsample(t, cs)
            sample_mask = mask_hw[None, :, :, None].expand(B, cs, cs, 1).float()
            sup = support.float()
            gt_mag = (x0n.abs() * sup)

            masked_image = up(c2r(x_zf))
            gt_up = up(c2r(x0n))
            gt_tok = tokens(gt_up)
            zf_tok = tokens(masked_image)

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                # pass 1 (student, greedy, t=0) + DC
                rec1, aux1 = reconstruct_far(student, meditok, masked_image, sample_mask, sampling_kwargs=greedy, dc_fn=dc_tok, timestep_offset=0, **far_kwargs)
                # teacher draft as in training: sampled tokens, t=0, + DC
                drafts = []
                for _ in range(a.draft_samples):
                    d_img, _ = reconstruct_far(teacher if teacher is not None else student, meditok, masked_image, sample_mask, sampling_kwargs=sample_kw, dc_fn=dc_tok, timestep_offset=0, **far_kwargs)
                    drafts.append(d_img)
                draft = drafts[0]
                # teacher greedy (t=0) for reference
                tg, _ = reconstruct_far(teacher if teacher is not None else student, meditok, masked_image, sample_mask, sampling_kwargs=greedy, dc_fn=dc_tok, timestep_offset=0, **far_kwargs)
                inputs = {"zf_t0": (masked_image, 0), "zf_t1": (masked_image, 1),
                          "pass1_t1": (rec1, 1), "pass1_t0": (rec1, 0),
                          "draft_t1": (draft, 1), "draft_t0": (draft, 0),
                          "gt_t1": (gt_up, 1), "gt_t0": (gt_up, 0)}
                for key, (img, t) in inputs.items():
                    cond = tokens(img)
                    tt = torch.full((B,), t, device=device, dtype=torch.long)
                    mv = getattr(student, 'model_version', 1)
                    mk = {"sample_mask_cols": sample_mask[:, 0, :, 0].float()} if mv >= 2 else {}
                    if mv >= 2.1:   # continuous latents instead of tokens
                        mk["cond_latent"] = torch.stack([latent_target(meditok, img[..., c].float()) for c in range(2)], dim=1)
                        logits, loss, *_ = student(timestep=tt, targets=gt_tok.long(), **mk)
                    else:
                        logits, loss, *_ = student(cond_tokens=cond.long(), timestep=tt, targets=gt_tok.long(), **mk)
                    ce.setdefault(key, []).extend([float(loss)] * B)
                    acc_in.setdefault(key, []).extend([float((cond == gt_tok).float().mean())] * B)
                    # teacher-forced top-1 accuracy of the prediction
                    pred = logits.argmax(-1)                                          # (B,2,K,L)
                    if pred.shape == gt_tok.shape:
                        acc_pred.setdefault(key, []).extend([float((pred == gt_tok).float().mean())] * B)
                    # generated reconstruction from this input (greedy) + DC
                    rec, aux = reconstruct_far(student, meditok, img, sample_mask, sampling_kwargs=greedy, dc_fn=dc_tok, timestep_offset=t, **far_kwargs)
                    ps.setdefault(key, []).extend(psnr_slices(r2c(down(rec)).abs() * sup, gt_mag))
                    ps_nodc.setdefault(key, []).extend(psnr_slices(r2c(down(aux["decoded_no_dc"])).abs() * sup, gt_mag))
                # reference arms
                ps.setdefault("zero_filled", []).extend(psnr_slices(x_zf.abs() * sup, gt_mag))
                ps.setdefault("pass1_dc", []).extend(psnr_slices(r2c(down(rec1)).abs() * sup, gt_mag))
                ps_nodc.setdefault("pass1_dc", []).extend(psnr_slices(r2c(down(aux1["decoded_no_dc"])).abs() * sup, gt_mag))
                ps.setdefault("teacher_greedy_dc", []).extend(psnr_slices(r2c(down(tg)).abs() * sup, gt_mag))
                for i, d_img in enumerate(drafts):
                    ps.setdefault(f"teacher_draft{i}_dc", []).extend(psnr_slices(r2c(down(d_img)).abs() * sup, gt_mag))
                from autoregressive.sample.reconstruct_far import enc_dec_image
                rt = enc_dec_image(meditok, gt_up)
                ps.setdefault("gt_roundtrip", []).extend(psnr_slices(r2c(down(rt)).abs() * sup, gt_mag))
                ps.setdefault("gt_roundtrip_dc", []).extend(psnr_slices(r2c(down(dc_tok(rt, sample_mask))).abs() * sup, gt_mag))
                # token agreement of the inputs with the GT tokens
                for key, img in {"zf": masked_image, "pass1": rec1, "draft": draft, "gt_roundtrip": rt}.items():
                    acc_in.setdefault(f"tok_{key}", []).extend([float((tokens(img) == gt_tok).float().mean())] * B)
            rows.append({"volume": name, "slices": sl})
            print(f"{name} slices {sl}: zf {np.mean(ps['zero_filled'][-B:]):.2f}  pass1 {np.mean(ps['pass1_dc'][-B:]):.2f}  draft {np.mean(ps['teacher_draft0_dc'][-B:]):.2f}  "
                  f"refine(pass1,t1) {np.mean(ps['pass1_t1'][-B:]):.2f}  refine(gt,t1) {np.mean(ps['gt_t1'][-B:]):.2f}  CE zf_t0 {np.mean(ce['zf_t0'][-B:]):.3f} pass1_t1 {np.mean(ce['pass1_t1'][-B:]):.3f} gt_t1 {np.mean(ce['gt_t1'][-B:]):.3f}", flush=True)

    summary = {"ce": {k: float(np.mean(v)) for k, v in ce.items()}, "input_token_agreement": {k: float(np.mean(v)) for k, v in acc_in.items()},
               "pred_top1_acc": {k: float(np.mean(v)) for k, v in acc_pred.items()}, "psnr_dc": {k: float(np.mean(v)) for k, v in ps.items()},
               "psnr_no_dc": {k: float(np.mean(v)) for k, v in ps_nodc.items()}, "n_slices": len(ps["zero_filled"]), "gpt_ckpt": a.gpt_ckpt, "teacher": "ema" if teacher is not None else "student"}
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print("\n== conditioning -> student (teacher-forced CE of GT tokens | input token agreement w/ GT | top-1 pred acc | PSNR greedy+DC | no DC)")
    for k in ("zf_t0", "zf_t1", "pass1_t0", "pass1_t1", "draft_t0", "draft_t1", "gt_t0", "gt_t1"):
        print(f"  {k:9s} CE {summary['ce'][k]:.3f} | in-agree {summary['input_token_agreement'][k]:.3f} | pred-acc {summary['pred_top1_acc'].get(k, float('nan')):.3f} | PSNR {summary['psnr_dc'][k]:.2f} | {summary['psnr_no_dc'][k]:.2f}")
    print("== reference PSNR (+DC):", {k: round(v, 2) for k, v in summary["psnr_dc"].items() if k in ("zero_filled", "pass1_dc", "teacher_greedy_dc", "teacher_draft0_dc", "gt_roundtrip", "gt_roundtrip_dc")})
    print("== input token agreement with GT tokens:", {k: round(v, 3) for k, v in summary["input_token_agreement"].items() if k.startswith("tok_")})


if __name__ == "__main__":
    main()
