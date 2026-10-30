"""Harness correctness check: evaluate a full-scale checkpoint through model.py + sample.py + prepare.evaluate and compare with
the number its own training log reports. run19's best.pt (GPT-L, v2.1 asym, latent head) on the suno split must give the
'PSNR:' of the matching 'New best model!' line of its stdout.log (latent path) and the '[ps_recon] alt_rec' PSNR (token path).
    PYTHONPATH=.. CUDA_VISIBLE_DEVICES=4 python check_harness.py --ckpt <best.pt> [--split suno|autoresearch]
Also prints the step rate of GPT-L for the calibration."""
import argparse, sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import prepare, sample
from model import build_model, num_params
import torch

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--split", default="suno", choices=["suno", "autoresearch"])
p.add_argument("--depth", type=int, default=24); p.add_argument("--dim", type=int, default=1024); p.add_argument("--n-head", type=int, default=16)
args = p.parse_args()
device = torch.device("cuda")
sd = torch.load(args.ckpt, map_location="cpu")["model"]
interleaved = "output.merge.weight" not in sd
latent_dim = int(sd["latent_head.1.weight"].shape[0] // 2) if "latent_head.1.weight" in sd else 0
model = build_model(args.depth, args.dim, args.n_head, latent_dim=latent_dim, head_interleaved=interleaved,
                    block_size=prepare.NUM_POS, mask_width=int(sd["mask_embed.weight"].shape[1]))
model.load_state_dict(sd, strict=True)          # the parameter names of model.py must match the full-scale model exactly
model = model.to(device).eval()
print(f"loaded {args.ckpt}: {num_params(model) / 1e6:.1f}M params, latent_dim {latent_dim}, interleaved {interleaved}", flush=True)
tokenizer = prepare.load_tokenizer(device)
sens_net = prepare.load_sens_net(device)
split = prepare.SUNO_SPLIT if args.split == "suno" else prepare.VAL_SPLIT
t0 = time.time()
m = prepare.evaluate(model, tokenizer, sens_net, sample.reconstruct, device, loader=prepare.val_loader(split))
print(f"split {args.split}: val_psnr {m['psnr_mean']:.4f} ssim {m['ssim_mean']:.4f}  "
      + " ".join(f"arm_{k}_psnr {v['psnr_mean']:.4f}" for k, v in m["arms"].items()) + f"  ({time.time() - t0:.0f} s)", flush=True)
