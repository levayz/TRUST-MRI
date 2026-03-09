import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import h5py
import pathlib
import random
import pandas as pd
from data import transforms
from data.subsample import create_mask_for_mask_type

# Labels for each pathology task
ACL_LABELS = {"Ligament - ACL High Grade Sprain", "Ligament - ACL Low Grade sprain"}
CARTILAGE_LABELS = {"Cartilage - Partial Thickness loss/defect", "Cartilage - Full Thickness loss/defect"}


def load_slice_labels(label_csv: str) -> dict:
    """
    Parse knee.csv into a per-(file_stem, slice) binary label dict.
    Returns: {(file_stem, slice_idx): [acl_label, cartilage_label]}
    """
    df = pd.read_csv(label_csv)
    label_dict = {}
    for _, row in df.iterrows():
        key = (str(row["file"]).strip(), int(row["slice"]))
        if key not in label_dict:
            label_dict[key] = [0, 0]  # [acl, cartilage]
        lbl = str(row["label"]).strip()
        if lbl in ACL_LABELS:
            label_dict[key][0] = 1
        if lbl in CARTILAGE_LABELS:
            label_dict[key][1] = 1
    return label_dict

def clip_complex_by_magnitude(x, low, high, eps=1e-8):
    """
    Clip a complex tensor (last dim=2 real/imag) by clamping its magnitude,
    while preserving phase: x <- x * (|x|_clipped / (|x| + eps)).
    """
    mag = transforms.complex_abs(x)  # (..., H, W)
    low = low.to(device=mag.device, dtype=mag.dtype)
    high = high.to(device=mag.device, dtype=mag.dtype)
    mag_clipped = mag.clamp(low, high)
    scale = (mag_clipped / (mag + eps)).unsqueeze(-1)  # (..., H, W, 1)
    return x * scale


def get_invertible_weights(H, W, center_width=0.08, noise_suppression=0.2, min_floor=0.1):
    """
    Creates 'Soft' Band-Pass weights that are mathematically invertible.
    Instead of 0.0, the suppressed regions (DC and Noise) are set to 'min_floor'.
    
    Args:
        min_floor (float): The minimum weight allowed. 
                           MUST be > 0.0 (e.g., 0.1) for stability.
    """
    # 1. Create Grid
    ky = torch.arange(H, dtype=torch.float32) - H // 2
    kx = torch.arange(W, dtype=torch.float32) - W // 2
    ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing="ij")
    
    # 2. Compute Radius
    radius = torch.sqrt(kx_grid**2 + ky_grid**2)
    radius = radius / radius.max()
    
    # 3. Create Difference of Gaussians (DoG)
    g_signal = torch.exp(-(radius**2) / (2 * noise_suppression**2))
    g_background = torch.exp(-(radius**2) / (2 * center_width**2))
    
    weights = g_signal - g_background
    
    # 4. ENFORCE INVERTIBILITY: Lift the floor
    # We normalize 0..1 first, then squeeze into min_floor..1.0
    weights = torch.clamp(weights, min=0.0) 
    weights = weights / (weights.max() + 1e-8) # Normalize to 0..1
    
    # Linear scaling: 0 -> min_floor, 1 -> 1.0
    weights = weights * (1.0 - min_floor) + min_floor
    
    return weights

def radial_weighting_transform(kspace, weights=None):
    """
    Applies the radial weights (Forward Pass).
    kspace: (..., H, W, 2) or complex
    """
    if weights is None:
        # Default to the invertible weights generator
        weights = get_invertible_weights(kspace.shape[-3], kspace.shape[-2]).to(kspace.device)
        
    return kspace * weights.unsqueeze(-1)

def radial_weighting_transform_image(image, weights=None):
    """
    Applies the radial weights (Forward Pass).
    kspace: (..., H, W, 2) or complex
    """
    kspace = transforms.fft2(image)
    kspace = radial_weighting_transform(kspace, weights)
        
    return transforms.ifft2(kspace)

def inverse_radial_weighting_transform_kspace(kspace, weights=None):
    """
    Inverts the radial weights (Backward Pass).
    kspace: (..., H, W, 2)
    """
    if weights is None:
        weights = get_invertible_weights(kspace.shape[-3], kspace.shape[-2]).to(kspace.device)
    
    # Division is safe because get_invertible_weights guarantees min_floor > 0
    return kspace / weights.unsqueeze(-1)

def inverse_radial_weighting_transform_image(image, weights=None):
    """
    Takes an image, applies FFT, removes weights, and returns image.
    (Note: This function name implies 'Undo the weights from an image')
    """
    # 1. FFT (Image -> Kspace)
    kspace = transforms.fft2(image)
    
    # 2. Remove Weights
    kspace_restored = inverse_radial_weighting_transform_kspace(kspace, weights)
    
    # 3. iFFT (Kspace -> Image)
    return transforms.ifft2(kspace_restored)


class SliceMRICodeDataset(Dataset):
    def __init__(self, root, code_dir, mask_func, resolution, challenge="singlecoil",
                 use_seed=True, num_samples=None, acquisition_type=None, seed=42,
                 clip_percentile: bool = False, radial_weighting=False, custom_split:str=None,
                 end_slice_skip: int = 0, label_csv: str = None):
        self.root = pathlib.Path(root)
        if code_dir is not None:
            self.code_dir = pathlib.Path(code_dir)
        else:
            self.code_dir = None
        self.mask_func = mask_func
        self.resolution = resolution
        self.challenge = challenge
        self.use_seed = use_seed
        self.acquisition_type = acquisition_type
        self.clip_percentile = clip_percentile
        self.radial_weighting = radial_weighting
        self.end_slice_skip = end_slice_skip
        self.radial_weights = None
        self.percentile_cache = {}
        random.seed(seed)

        # Optional pathology labels: {(file_stem, slice_idx): [acl, cartilage]}
        self.slice_labels = load_slice_labels(label_csv) if label_csv is not None else None
        if self.slice_labels is not None:
            print(f"Loaded labels for {len(self.slice_labels)} annotated slices from {label_csv}")

        # discover slices (like SliceData)
        self.examples = []
        
        if custom_split is not None:
            with open(custom_split, 'r') as f:
                split_filenames = {line.strip() for line in f}
            files = [self.root / fname for fname in split_filenames if (self.root / fname).exists()]
            print(f"using custom split with {len(files)} files")
        else:
            files = list(self.root.iterdir())
            if num_samples is not None:
                random.shuffle(files)
                files = files[:num_samples]
        
        for fname in sorted(files):
            with h5py.File(fname, "r") as data:
                num_slices = data["kspace"].shape[0]
                attrs = dict(data.attrs)
                if acquisition_type == 'PDFS' and attrs['acquisition'] != 'CORPDFS_FBK':
                    continue
                if acquisition_type == 'PD' and attrs['acquisition'] != 'CORPD_FBK':
                    continue
                num_slices -= self.end_slice_skip
                for s in range(num_slices):
                    self.examples.append((fname, s))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, s = self.examples[i]
        
        if self.clip_percentile and fname not in self.percentile_cache:
            with h5py.File(fname, "r") as data:
                kspace_np = data["kspace"][()]
                kspace_tensor = transforms.to_tensor(kspace_np)
                img = transforms.ifft2(kspace_tensor)
                img = transforms.complex_pad(img, self.resolution, self.resolution)
                img = transforms.complex_center_crop(img, (self.resolution, self.resolution))
                img_abs = transforms.complex_abs(img)
                percentile_005th = torch.quantile(img_abs, 0.005)
                percentile_995th = torch.quantile(img_abs, 0.995)
                self.percentile_cache[fname] = (percentile_005th, percentile_995th)
        
        with h5py.File(fname, "r") as data:
            kspace = transforms.to_tensor(data["kspace"][s]).clone()  # (..., H, W, 2)
            # reconstruct image at target resolution
            image = transforms.ifft2(kspace)
            # image = F.interpolate(image.permute(2, 0, 1).unsqueeze(0), size=(self.resolution, self.resolution), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
            # Pad image if needed to reach target resolution
            H, W, _ = image.shape
            if H < self.resolution or W < self.resolution:
                image = transforms.complex_pad(image, self.resolution, self.resolution)
            image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
            if self.clip_percentile:
                percentile_005th, percentile_995th = self.percentile_cache[fname]
                image = clip_complex_by_magnitude(image, percentile_005th, percentile_995th)
            kspace_resized = transforms.fft2(image)

            if self.radial_weighting:
                if self.radial_weights is None:
                    self.radial_weights = get_invertible_weights(self.resolution, self.resolution)
                kspace_resized = radial_weighting_transform(kspace_resized, self.radial_weights)
                image = transforms.ifft2(kspace_resized)

            if self.mask_func:
                seed = None if not self.use_seed else tuple(map(ord, pathlib.Path(fname).name))
                masked_kspace, mask = transforms.apply_mask(kspace_resized, self.mask_func, seed)
                mask = mask.repeat(self.resolution,1,1)
            else:
                masked_kspace, mask = kspace_resized, None

            # zero-filled reconstruction of masked image
            masked_image = transforms.ifft2(masked_kspace)
            gt_image = image

            masked_mag = transforms.complex_abs(masked_image).unsqueeze(-1)  # (H, W, 1)
            gt_mag = transforms.complex_abs(gt_image).unsqueeze(-1)

            # normalize abs value image to [-1, 1] using per-slice max
            max_val = gt_mag.max()
            masked_in = (masked_mag / max_val).clamp(0, 1) * 2 - 1
            gt_in = (gt_mag / max_val).clamp(0, 1) * 2 - 1

            # normalize complex image, using max_val
            masked_image = masked_image / max_val
            gt_image = gt_image / max_val

            # to CHW for encoder
            masked_in = masked_in.repeat(1, 1, 3).permute(2, 0, 1)  # [3,H,W]
            gt_in = gt_in.repeat(1, 1, 3).permute(2, 0, 1)  # [3,H,W]

            # load GT codes from disk (precomputed)
            if self.code_dir is not None:
                code_path = self.code_dir / f"{pathlib.Path(fname).stem}_codes.pt"
                codes_file = torch.load(code_path)
                gt_tokens = codes_file["codes"][s]  # [num_codebooks, 400]
            else:
                gt_tokens = torch.zeros([2, 8, 256])

            # Pathology labels: [acl_sprain, cartilage_loss] — 0=healthy, 1=diseased
            if self.slice_labels is not None:
                key = (pathlib.Path(fname).stem, s)
                lbl = self.slice_labels.get(key, [0, 0])
                labels = torch.tensor(lbl, dtype=torch.float32)
            else:
                labels = None

            return {
                "masked_in": masked_in,
                "gt_in": gt_in, # normed [-1,1] this is the magnitude image
                "gt_tokens": gt_tokens,
                "masked_c_image": masked_image,
                "gt_c_image": gt_image, # complex image,  normed [-1, 1]
                "max_val": float(max_val),
                "fname": pathlib.Path(fname).name,
                "slice": s,
                "mask": mask,
                # "labels": labels,  # [acl_sprain, cartilage_loss] or None
            }

def build_mri_code(args, **kwargs):
    mask = create_mask_for_mask_type('random', [getattr(args, "center_fractions", 0.08)], [getattr(args, "accelerations", 4)])
    acquisition_type = getattr(args, "acquisition_type", None)
    use_seed = getattr(args, "use_seed", False)
    if acquisition_type is not None:
        print(f"[DEBUG] using acquisition: {acquisition_type}")
    return SliceMRICodeDataset(
        root=args.data_root,
        code_dir=args.code_dir,
        mask_func=mask,
        resolution=args.image_size,
        challenge="singlecoil",
        use_seed=use_seed,
        num_samples=getattr(args, 'num_samples', None),
        seed=getattr(args, 'global_seed', 42),
        acquisition_type=acquisition_type,
        clip_percentile=getattr(args, "clip_mri", False),
        radial_weighting=getattr(args, "radial_weighting", False),
        custom_split=getattr(args, "data_split", None),
        label_csv=getattr(args, "label_csv", None),
    )