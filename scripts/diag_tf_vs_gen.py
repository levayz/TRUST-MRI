#!/usr/bin/env python
"""Teacher-forcing vs generation consistency check for a SENSE GPT checkpoint (any model version).

On a few slices it compares, position by position: (a) the teacher-forced logits/top-1 accuracy of the GT tokens, (b) the
logits produced by the step-wise sampler (`reconstruct_far`, greedy) and the accuracy of the generated tokens, and (c) whether
the step-0 logits (which cannot depend on previously generated codes) are identical in both paths. A model that memorised the
slices should have ~100% teacher-forced accuracy; if generation accuracy is far lower while step-0 logits match, the output
head's step-wise path disagrees with its teacher-forced path.

Usage: PYTHONPATH=$(pwd) python scripts/diag_tf_vs_gen.py --gpt-ckpt <ckpt> --split results/sense/splits/one_volume.txt
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoregressive.sample.reconstruct_far import load_meditok, load_gpt, encode  # noqa: E402
from autoregressive.models.generate import reconstruct_far  # noqa: E402
from autoregressive.models.gpt_kar import infer_model_args_from_state_dict  # noqa: E402
from dataset.data.subsample import create_mask_for_mask_type  # noqa: E402
from models.sense_ops import (acs_num_low_freqs, c2r, center_crop_or_pad, crop_native_to_grid, estimate_maps, fft2c, ifft2c,  # noqa: E402
                              make_sense_dc_fn, r2c, sense_combine, support_from_maps, tok_downsample, tok_upsample,
                              wrap_dc_fn_resized, latent_target)
from models.varnet_sens import load_sens_net  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpt-ckpt", required=True)
    p.add_argument("--encoder-ckpt", default="weights/meditok/brain_mc_sense_cplxpair_res512_ep5_frozen.pth")
    p.add_argument("--encoder-config", default="weights/meditok/config_512.json")
    p.add_argument("--sens-ckpt", default="results/sens_net/run1/best.pt")
    p.add_argument("--data-root", default="fastmri/brain/multicoil_val")
    p.add_argument("--sens-dirs", default="fastmri/brain/multicoil_val_sens_maps_gpu,fastmri/brain/multicoil_val_sens_maps")
    p.add_argument("--split", default="results/sense/splits/one_volume.txt")
    p.add_argument("--slices", type=int, nargs="+", default=[6, 7])
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--token-size", type=int, default=512)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    a = p.parse_args()
    cs, tok = a.crop_size, a.token_size
    device = torch.device("cuda")
    dtype = torch.float32 if a.dtype == "float32" else torch.bfloat16
    meditok = load_meditok(a.encoder_config, a.encoder_ckpt, device)
    ck = torch.load(a.gpt_ckpt, map_location="cpu", weights_only=False); sd = ck["model"]
    gpt_args = SimpleNamespace(gpt_model="GPT-L", vocab_size=32768, image_size=tok, downsample_size=16, num_classes=8, cls_token_num=1,
                               cond_token_num=(tok // 16) ** 2, gpt_type="c2i", num_codebooks=8, num_output_layer=1, vq_embed_path=None,
                               probe_model=False, gpt_ckpt=a.gpt_ckpt, compile=False,
                               **infer_model_args_from_state_dict(sd, default_mask_width=cs))
    del ck, sd
    gpt = load_gpt(gpt_args, device, dtype)
    if gpt_args.latent_head_type == 'fused':
        gpt.set_latent_codebook(meditok.quantizer)
    print(f"model version {gpt_args.model_version}, head interleaved: {getattr(gpt.output, 'interleaved', False)}, dtype {dtype}")
    sens_net = load_sens_net(a.sens_ckpt, device=device, trainable=False)
    nlf = acs_num_low_freqs(cs, 0.04)
    mask_func = create_mask_for_mask_type("random", [0.04], [8])
    far_kwargs = dict(n_steps=1, far_mask_mode="radial", complex_mode="cartesian", center_fractions=0.04)
    greedy = {"temperature": 1.0, "top_k": 0, "top_p": 1.0}
    name = [l.strip() for l in open(a.split) if l.strip()][0]
    with h5py.File(Path(a.data_root) / name, "r") as f:
        ksp = f["kspace"][()]
    sens = None
    for d in a.sens_dirs.split(","):
        if (Path(d) / name).exists():
            with h5py.File(Path(d) / name, "r") as f:
                sens = f["sens_maps"][()]
            break
    m, _ = mask_func((1, cs, cs, 2), seed=tuple(map(ord, name)))
    mask_cols = (m.reshape(-1) > 0).to(device); mask_hw = mask_cols[None, :].expand(cs, cs)
    sl = a.slices; B = len(sl)
    xs, mps = [], []
    for s in sl:
        mp_ = torch.from_numpy(sens[s]).to(device)
        xs.append(crop_native_to_grid(ifft2c(torch.from_numpy(ksp[s]).to(device)), mp_.shape[-2:], cs)); mps.append(center_crop_or_pad(mp_, (cs, cs)))
    coils, maps_or = torch.stack(xs), torch.stack(mps)
    support = support_from_maps(maps_or)
    x0 = sense_combine(coils, maps_or); mv = x0.abs().flatten(1).max(1).values.clamp_min(1e-12)[:, None, None]
    x0n, coils_n = x0 / mv, coils / mv[:, None]
    k_meas = fft2c(coils_n) * mask_hw; mask_b = mask_cols[None, :].expand(B, cs)
    maps = estimate_maps(sens_net, k_meas, mask_b, num_coils=None, num_low_freqs=nlf)
    x_zf = sense_combine(ifft2c(k_meas), maps)
    dc_tok = wrap_dc_fn_resized(make_sense_dc_fn(k_meas, maps, mask_b), cs, tok, "bicubic")
    up = lambda t: tok_upsample(t, tok, "bicubic"); down = lambda t: tok_downsample(t, cs)
    sample_mask = mask_hw[None, :, :, None].expand(B, cs, cs, 1).float().contiguous()
    masked_image, gt_up = up(c2r(x_zf)), up(c2r(x0n))
    gt_tok = torch.cat((encode(meditok, gt_up[..., 0]).unsqueeze(1), encode(meditok, gt_up[..., 1]).unsqueeze(1)), 1)   # (B,2,K,L)
    gt_mag = x0n.abs() * support.float()

    def psnr(rec):
        out = []
        for r, g in zip(r2c(down(rec)).abs() * support.float(), gt_mag):
            out.append(float(10 * torch.log10(g.max() ** 2 / ((r - g) ** 2).mean().clamp_min(1e-20))))
        return sum(out) / len(out)

    with torch.no_grad():
        kw = {"timestep": torch.zeros(B, dtype=torch.long, device=device), "targets": gt_tok.long()}
        if gpt_args.model_version >= 2:
            kw["sample_mask_cols"] = sample_mask[:, 0, :, 0].float()
        if gpt_args.model_version >= 2.1:
            kw["cond_latent"] = torch.stack([latent_target(meditok, masked_image[..., c].float()) for c in range(2)], 1)
        else:
            kw["cond_tokens"] = torch.cat((encode(meditok, masked_image[..., 0]).unsqueeze(1), encode(meditok, masked_image[..., 1]).unsqueeze(1)), 1).long()
        logits_tf, loss_tf = gpt(**kw)[:2]                                                  # (B,2,K,L,V)
        acc_tf = (logits_tf.argmax(-1) == gt_tok).float()
        print(f"teacher forcing: CE {float(loss_tf):.3f}  top-1 acc Re {float(acc_tf[:, 0].mean()):.3f} Im {float(acc_tf[:, 1].mean()):.3f}  per codebook {[round(float(acc_tf[:, :, k].mean()), 2) for k in range(8)]}")
        rec, aux = reconstruct_far(gpt, meditok, masked_image, sample_mask, sampling_kwargs=greedy, dc_fn=dc_tok, timestep_offset=0, **far_kwargs)
        lg = aux["logits"]                                                                   # (B,Hp,Wp,2,K,V)
        Hp = lg.shape[1]
        lg = lg.reshape(B, Hp * Hp, 2, 8, -1).permute(0, 2, 3, 1, 4)                          # (B,2,K,L,V)
        gen_tok = lg.argmax(-1)
        acc_gen = (gen_tok == gt_tok).float()
        print(f"generation (greedy): token acc vs GT Re {float(acc_gen[:, 0].mean()):.3f} Im {float(acc_gen[:, 1].mean()):.3f}  per codebook {[round(float(acc_gen[:, :, k].mean()), 2) for k in range(8)]}")
        print(f"generation PSNR vs GT (support): no DC {psnr(aux['decoded_no_dc']):.2f}  DC {psnr(rec):.2f}")
        # step 0 (Re code 0) cannot depend on generated codes: must match teacher forcing exactly
        d0 = (lg[:, 0, 0].float() - logits_tf[:, 0, 0].float()).abs().max()
        same0 = (lg[:, 0, 0].argmax(-1) == logits_tf[:, 0, 0].argmax(-1)).float().mean()
        print(f"step 0 (Re codebook 0): max |gen - tf| logits {float(d0):.4f}, argmax agreement {float(same0):.4f}")
        # Im code 0 given the *generated* Re code 0: compare where the generated Re_0 equals GT Re_0 (then TF and gen conditions agree)
        agree_re0 = gen_tok[:, 0, 0] == gt_tok[:, 0, 0]
        if agree_re0.any():
            dd = (lg[:, 1, 0].float() - logits_tf[:, 1, 0].float()).abs().amax(-1)[agree_re0]
            print(f"Im codebook 0 where Re_0 was generated correctly ({int(agree_re0.sum())} positions): max |gen - tf| logits {float(dd.max()):.4f}")
        # decode the teacher-forced argmax tokens: what the head 'knows' under teacher forcing
        from autoregressive.sample.reconstruct_far import decode_complex_codes, rgb_to_grayscale
        tf_img = rgb_to_grayscale(decode_complex_codes(meditok, logits_tf.argmax(-1)))
        print(f"decoded teacher-forced argmax tokens: PSNR no DC {psnr(tf_img):.2f}; GT tokens round trip: {psnr(rgb_to_grayscale(decode_complex_codes(meditok, gt_tok))):.2f}")


if __name__ == "__main__":
    main()
