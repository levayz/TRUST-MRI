import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import piq


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.array(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)


def compute_reconstruction_metrics(scan_buf, device):
    """
    Compute PSNR, SSIM, LPIPS, DISTS, and NMSE metrics for reconstructed scans.
    
    Args:
        scan_buf: Dictionary mapping scan keys to {"rec": list, "gt": list, "slice": list}
                  where rec/gt are numpy arrays per slice
        device: torch.device for computation
    
    Returns:
        metrics: Dictionary containing per-scan metrics and aggregate means:
                 {
                     scan_key: {"psnr": float, "ssim": float, "lpips": float, "dists": float, "nmse": float, "num_slices": int},
                     ...
                     "psnr_mean": float,
                     "ssim_mean": float,
                     "lpips_mean": float,
                     "dists_mean": float,
                     "nmse_mean": float
                 }
    """
    metrics = {}
    psnr_list = []
    ssim_list = []
    lpips_list = []
    dists_list = []
    nmse_list = []
    
    metric_lpips = piq.LPIPS(reduction='none', replace_pooling=True).to(device)
    metric_dists = piq.DISTS(reduction='none').to(device)

    for k, v in scan_buf.items():
        order = sorted(range(len(v["slice"])), key=lambda idx: v["slice"][idx])
        rec_stack = np.array([v["rec"][idx] for idx in order])
        gt_stack = np.array([v["gt"][idx] for idx in order])

        val_min, val_max = gt_stack.min(), gt_stack.max()
        data_range = val_max - val_min if val_max != val_min else 1.0
        
        scan_psnr = peak_signal_noise_ratio(gt_stack, rec_stack, data_range=data_range)
        
        ssim_scores = []
        for i in range(len(rec_stack)):
            slice_ssim = structural_similarity(
                gt_stack[i], 
                rec_stack[i], 
                data_range=data_range
            )
            ssim_scores.append(slice_ssim)
        scan_ssim = np.mean(ssim_scores)

        scale = 1.0 / data_range
        rec_norm = (rec_stack - val_min) * scale
        gt_norm = (gt_stack - val_min) * scale
        
        rec_norm = np.clip(rec_norm, 0, 1)
        gt_norm = np.clip(gt_norm, 0, 1)

        rec_t = torch.from_numpy(rec_norm).unsqueeze(1).repeat(1, 3, 1, 1).float().to(device)
        gt_t = torch.from_numpy(gt_norm).unsqueeze(1).repeat(1, 3, 1, 1).float().to(device)

        with torch.no_grad():
            lpips_scores = metric_lpips(rec_t, gt_t)
            dists_scores = metric_dists(rec_t, gt_t)
        
        scan_lpips = lpips_scores.mean().item()
        scan_dists = dists_scores.mean().item()

        scan_nmse = float(nmse(gt_stack, rec_stack))

        metrics[k] = {
            "psnr": scan_psnr,
            "ssim": scan_ssim,
            "lpips": scan_lpips,
            "dists": scan_dists,
            "nmse": scan_nmse,
            "num_slices": len(v["slice"]),
        }
        psnr_list.append(scan_psnr)
        ssim_list.append(scan_ssim)
        lpips_list.append(scan_lpips)
        dists_list.append(scan_dists)
        nmse_list.append(scan_nmse)

    metrics["psnr_mean"] = np.mean(psnr_list)
    metrics["ssim_mean"] = np.mean(ssim_list)
    metrics["lpips_mean"] = np.mean(lpips_list)
    metrics["dists_mean"] = np.mean(dists_list)
    metrics["nmse_mean"] = np.mean(nmse_list)

    return metrics
