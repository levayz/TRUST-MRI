"""
Save example figures for the MedITok round trip of the SENSE-combined image
(recommended recipe: ACS-phase demodulation, real channel 2r-1, Im dropped).

One PNG per scan, centre slice, four columns:
    |x0| combined image  |  MedITok reconstruction  |  |difference|  |  ESPIRiT support mask
plus the support mask alone as <stem>_sliceNN_support.png / .npy

Usage:
    python scripts/save_demod_examples.py \\
        --scans file_brain_AXFLAIR_210_6001917.h5 file_brain_AXT1POST_208_2080550.h5 ... \\
        --output_dir results/demod_check/examples
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from eval_demod_ceiling import fft2c, ifft2c, center_crop_or_pad, acs_col_range   # noqa: E402
from eval_meditok_upper_bound import load_meditok, encode_decode_batch            # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="fastmri/brain/multicoil_val")
    p.add_argument("--sens_dir", default="fastmri/brain/multicoil_val_sens_maps_gpu")
    p.add_argument("--meditok_config", default="/fast/lev/meditok/weights/meditok/config.json")
    p.add_argument("--meditok_ckpt", default="/fast/lev/meditok/weights/meditok/meditok_ckpt_v1.pth")
    p.add_argument("--scans", nargs="+", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--center_fraction", type=float, default=0.04)
    p.add_argument("--slice", type=int, default=-1, help="-1 = centre slice")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    meditok = load_meditok(args.meditok_config, args.meditok_ckpt, device)
    cs = args.crop_size
    c0, c1 = acs_col_range(cs, args.center_fraction)
    acs_mask = torch.zeros(cs, device=device); acs_mask[c0:c1] = 1.0

    for fname in args.scans:
        with h5py.File(Path(args.data_path) / fname, "r") as f:
            ksp_np = f["kspace"][()]
        with h5py.File(Path(args.sens_dir) / fname, "r") as f:
            sens_np = f["sens_maps"][()]
        S, C, H, W = ksp_np.shape
        s = S // 2 if args.slice < 0 else args.slice

        ksp = torch.from_numpy(ksp_np[s]).to(device)
        sm = center_crop_or_pad(torch.from_numpy(sens_np[s]).to(device), (cs, cs))
        x = center_crop_or_pad(center_crop_or_pad(ifft2c(ksp), sens_np.shape[-2:]), (cs, cs))
        ss = (sm.abs() ** 2).sum(0); support = ss > 1e-6

        def combine(coils):
            num = (sm.conj() * coils).sum(0)
            return torch.where(support, num / ss.clamp_min(1e-6), torch.zeros_like(num))

        x0 = combine(x)                                             # combined complex image
        phi = torch.angle(combine(ifft2c(fft2c(x) * acs_mask[None, None, :])))   # ACS phase
        mx = float(x0.abs().max())
        z = x0 / mx * torch.exp(-1j * phi)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            r_hat = encode_decode_batch(meditok, (2 * z.real - 1).clamp(-1, 1).unsqueeze(0)).float()[0]
        rec = ((r_hat + 1) / 2).clamp_min(0) * mx                  # |x_hat| (phase reattachment does not change it)

        gt = x0.abs().cpu().numpy(); rc = rec.cpu().numpy(); sup = support.cpu().numpy()
        dr = float(gt.max())
        psnr_sup = peak_signal_noise_ratio(gt, rc * sup, data_range=dr)
        ssim_sup = structural_similarity(gt, rc * sup, data_range=dr)
        psnr_all = peak_signal_noise_ratio(gt, rc, data_range=dr)
        diff = np.abs(rc - gt)

        fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
        axes[0].imshow(gt, cmap="gray", vmin=0, vmax=dr); axes[0].set_title("SENSE-combined |x0|  (oracle maps)", fontsize=10)
        axes[1].imshow(rc, cmap="gray", vmin=0, vmax=dr)
        axes[1].set_title(f"MedITok round trip (demod, 2r-1, Im dropped)\nPSNR {psnr_sup:.2f} dB / SSIM {ssim_sup:.3f} in support"
                          f"   |   PSNR {psnr_all:.2f} dB full frame", fontsize=9)
        im = axes[2].imshow(diff, cmap="magma", vmin=0, vmax=0.25 * dr)
        axes[2].set_title(f"|difference|  (colour scale 0 to 25% of GT max; max = {100 * diff.max() / dr:.0f}%)", fontsize=9)
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02)
        axes[3].imshow(sup, cmap="gray", vmin=0, vmax=1)
        axes[3].contour(sup, levels=[0.5], colors="red", linewidths=0.6)
        axes[3].set_title(f"ESPIRiT support mask  ({100 * sup.mean():.0f}% of frame)\nmetrics \"in support\" use this mask", fontsize=9)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{Path(fname).stem}   slice {s}/{S}   {C} coils", fontsize=10, y=1.03)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = out_dir / f"{Path(fname).stem}_slice{s:02d}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
        plt.imsave(out_dir / f"{Path(fname).stem}_slice{s:02d}_support.png", sup.astype(np.uint8) * 255, cmap="gray", vmin=0, vmax=255)
        np.save(out_dir / f"{Path(fname).stem}_slice{s:02d}_support.npy", sup)
        print(f"{out}   PSNR(support) {psnr_sup:.2f}  SSIM {ssim_sup:.3f}  PSNR(full) {psnr_all:.2f}")


if __name__ == "__main__":
    main()
