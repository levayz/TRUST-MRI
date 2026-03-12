
import h5py
import numpy as np
import torch
import pathlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import fmin_l_bfgs_b
from data import transforms as T

def emulate_single_coil_fast(kspace_volume, subsample_pixels=20000):
    """
    Optimized Strict Tygert Implementation.
    1. Computes global weights z using a random SUBSET of pixels (fast).
    2. Applies weights z to the FULL volume on GPU (high quality).
    """
    # 1. Move to Image Domain (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kspace_volume = kspace_volume.to(device)
    
    # ifft2 handles (Slices, Coils, H, W, 2)
    image_volume = T.ifft2(kspace_volume)
    
    # 2. Prepare Data for Optimization (CPU is fine for small subset)
    # Convert to Complex Numpy for SciPy LBFGS
    vol_numpy = torch.view_as_complex(image_volume).detach().cpu().numpy()
    num_slices, num_coils, h, w = vol_numpy.shape
    
    # Flatten: (Total_Pixels, Coils)
    # Total_Pixels = Slices * H * W
    A_full = vol_numpy.transpose(0, 2, 3, 1).reshape(-1, num_coils)
    
    # 3. SUBSAMPLING (The Speed Hack)
    # Quick magnitude check to find 'brain' pixels (ignore background)
    magnitudes = np.linalg.norm(A_full, axis=1)
    # Threshold: Top 10% intensity * 0.1 (Adaptive noise floor)
    threshold = np.percentile(magnitudes, 90) * 0.1 
    mask = magnitudes > threshold
    
    valid_indices = np.where(mask)[0]
    
    # Safety: If volume is empty/noise, fall back to random
    if len(valid_indices) < subsample_pixels:
        valid_indices = np.arange(len(A_full))
        
    # Select random subset
    selected_indices = np.random.choice(valid_indices, subsample_pixels, replace=False)

    # Create small A matrix for optimization: (N_small, Coils)
    A_small = A_full[selected_indices]
    
    # Compute RSS target 'b' only for these pixels
    b_small = np.sqrt(np.sum(np.abs(A_small)**2, axis=1))

    # 4. Optimization (Hellinger Metric on Subset)
    def objective(z_real_imag):
        z = z_real_imag[:num_coils] + 1j * z_real_imag[num_coils:]
        Ax = A_small @ z
        loss = np.sum((np.sqrt(np.abs(Ax)) - np.sqrt(b_small))**2)
        return loss

    # Initial Guess
    z0_complex, _, _, _ = np.linalg.lstsq(A_small, b_small, rcond=None)
    z0 = np.concatenate([z0_complex.real, z0_complex.imag])

    # Run LBFGS (Fast on small matrix)
    z_opt_raw, min_val, info = fmin_l_bfgs_b(objective, z0, approx_grad=True, maxiter=20)
    z_opt = z_opt_raw[:num_coils] + 1j * z_opt_raw[num_coils:]
    
    # 5. Apply weights to FULL volume (GPU)
    # FIXED: Use .to(dtype) instead of .complex()
    z_torch = torch.from_numpy(z_opt).unsqueeze(1).to(device).to(torch.complex64)
    
    # image_volume is complex64 on GPU
    img_vol_complex = torch.view_as_complex(image_volume)
    
    # Permute to (Slices, H, W, Coils) to multiply by (Coils, 1)
    # Result: (Slices, H, W, 1)
    esc_vol_complex = torch.matmul(img_vol_complex.permute(0, 2, 3, 1), z_torch).squeeze(-1)
    
    # 6. Return to K-space
    esc_vol_torch = torch.view_as_real(esc_vol_complex)
    kspace_esc = T.fft2(esc_vol_torch)
    
    return kspace_esc.cpu()

# --- Visualization Helper ---
def visualize_volume_comparison(original_vol, esc_vol, fname):
    """
    Visualizes the middle slice of the volume.
    original_vol: (Slices, Coils, H, W, 2) Tensor
    esc_vol: (Slices, H, W, 2) Tensor
    """
    mid_slice = original_vol.shape[0] // 2
    
    # 1. Original RSS (Middle Slice)
    k_slice = original_vol[mid_slice] # (Coils, H, W, 2)
    img_slice = T.ifft2(k_slice)
    rss_img = T.rss(T.complex_abs(img_slice), dim=0).numpy()
    
    # 2. ESC Magnitude (Middle Slice)
    k_esc_slice = esc_vol[mid_slice] # (H, W, 2)
    img_esc_slice = T.ifft2(k_esc_slice)
    esc_img = T.complex_abs(img_esc_slice).numpy()
    
    # 3. Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = axes[0].imshow(rss_img, cmap='gray', vmin=0, vmax=rss_img.max())
    axes[0].set_title(f"Original RSS\n{fname}")
    axes[0].axis('off')
    
    im2 = axes[1].imshow(esc_img, cmap='gray', vmin=0, vmax=rss_img.max())
    axes[1].set_title("ESC (Volume Optimized)")
    axes[1].axis('off')
    
    diff = np.abs(rss_img - esc_img)
    im3 = axes[2].imshow(diff, cmap='inferno')
    axes[2].set_title("Difference Map")
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()

def convert_dataset_fast(input_path, output_path, num_viz=0):
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    files = sorted(list(input_path.iterdir()))
    print(f"Converting {len(files)} volumes (Fast + Viz)...")
    
    viz_count = 0
    skipped_count = 0
    
    for fname in tqdm(files):
        out_name = output_path / fname.name
        
        # Check if output file already exists
        if out_name.exists():
            skipped_count += 1
            continue
        
        try:
            with h5py.File(fname, 'r') as source_hf:
                # Load Full Volume
                kspace_all = T.to_tensor(source_hf['kspace'][:])
                
                # Run Optimization
                kspace_esc_vol = emulate_single_coil_fast(kspace_all, subsample_pixels=20000)
                
                # Visualization Trigger
                if viz_count < num_viz:
                    visualize_volume_comparison(kspace_all, kspace_esc_vol, fname.name)
                    viz_count += 1
                
                # Save
                new_kspace_buffer = torch.view_as_complex(kspace_esc_vol).numpy().astype(np.complex64)
                
                with h5py.File(out_name, 'w') as target_hf:
                    target_hf.create_dataset('kspace', data=new_kspace_buffer)
                    for key, val in source_hf.attrs.items():
                        target_hf.attrs[key] = val
                    
                    # Create reconstruction_esc target
                    img_vol = T.ifft2(kspace_esc_vol)
                    recons_esc = T.complex_abs(img_vol).numpy()
                    target_hf.create_dataset('reconstruction_esc', data=recons_esc)

        except Exception as e:
            print(f"Error processing {fname.name}: {e}")
    
    print(f"\nSkipped {skipped_count} already processed files")

if __name__ == "__main__":
    convert_dataset_fast(
        input_path='fastmri/brain/multicoil_train', 
        output_path='fastmri/brain/singlecoil_emulated_train',
        num_viz=0
    )