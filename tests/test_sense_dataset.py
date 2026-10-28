"""SenseSliceDataset vs the eval_sense_block.py reference computation (skipped without the data)."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "fastmri/brain/multicoil_val"
MAPS = "fastmri/brain/multicoil_val_sens_maps_gpu,fastmri/brain/multicoil_val_sens_maps"
SPLIT = ROOT / "fastmri/splits/suno_brain_test.txt"

pytestmark = pytest.mark.skipif(not (DATA.exists() and SPLIT.exists()), reason="fastMRI brain multicoil val data not available")


def _reference(name, s, cs=256, cf=0.04, R=8):
    """eval_sense_block.py per-slice computation, verbatim conventions."""
    import h5py
    from dataset.data.subsample import create_mask_for_mask_type
    from models.sense_ops import fft2c, ifft2c, center_crop_or_pad
    with h5py.File(DATA / name) as f:
        ksp = torch.from_numpy(f["kspace"][s])
    sens = None
    for d in MAPS.split(","):
        if (ROOT / d / name).exists():
            with h5py.File(ROOT / d / name) as f:
                sens = torch.from_numpy(f["sens_maps"][s])
            break
    m, _ = create_mask_for_mask_type("random", [cf], [R])((1, cs, cs, 2), seed=tuple(map(ord, name)))
    mask_cols = (m.reshape(-1) > 0)
    mask_hw = mask_cols[None, :].expand(cs, cs)
    x = ifft2c(ksp)
    coils = center_crop_or_pad(center_crop_or_pad(x, sens.shape[-2:]), (cs, cs))[None]
    maps = center_crop_or_pad(sens, (cs, cs))[None]
    ss = (maps.abs() ** 2).sum(1); support = ss > 1e-6

    def combine(c):
        num = (maps.conj() * c).sum(1)
        return torch.where(support, num / ss.clamp_min(1e-6), torch.zeros_like(num))
    x0 = combine(coils)
    max_val = x0.abs().flatten(1).max(1).values.clamp_min(1e-12)
    x0n, coils_n = x0 / max_val[:, None, None], coils / max_val[:, None, None, None]
    k_meas = fft2c(coils_n) * mask_hw
    x_zf = combine(ifft2c(k_meas))
    return dict(mask_cols=mask_cols, x0=x0n[0], x_zf=x_zf[0], max_val=float(max_val[0]), coils=coils_n[0],
                maps=maps[0], support=support[0])


def test_dataset_matches_reference():
    from dataset.data.subsample import create_mask_for_mask_type
    from dataset.sense_data import SenseSliceDataset
    from models.sense_ops import r2c
    mf = create_mask_for_mask_type("random", [0.04], [8])
    ds = SenseSliceDataset(DATA, MAPS, mf, resolution=256, max_coils=20, use_seed=True, custom_split=str(SPLIT),
                           return_oracle_maps=True)
    assert len(ds) > 0
    names = sorted({f.name for f, _ in ds.examples})[:2]
    checked = 0
    t0 = time.time()
    for idx, (fname, s) in enumerate(ds.examples):
        if fname.name not in names or s not in (3, 9):
            continue
        item = ds[idx]
        ref = _reference(fname.name, s)
        C = item["num_coils"]
        assert item["coils"].shape == (20, 256, 256, 2) and item["coils"].dtype == torch.float32
        assert item["mask"].shape == (256, 256, 1) and item["gt_c_image"].shape == (256, 256, 2)
        assert item["support"].dtype == torch.bool and item["support"].shape == (256, 256)
        assert torch.equal(item["mask"][0, :, 0].bool(), ref["mask_cols"])
        assert torch.equal(item["mask"][0], item["mask"][100])
        assert abs(item["max_val"] - ref["max_val"]) <= 1e-6 * ref["max_val"]
        assert torch.allclose(r2c(item["gt_c_image"]), ref["x0"], atol=1e-6)
        assert torch.allclose(r2c(item["masked_c_image"]), ref["x_zf"], atol=1e-6)
        assert torch.allclose(r2c(item["coils"][:C]), ref["coils"], atol=1e-6)
        assert torch.all(item["coils"][C:] == 0)
        assert torch.allclose(r2c(item["maps_oracle"][:C]), ref["maps"], atol=1e-6)
        assert torch.equal(item["support"], ref["support"])
        assert r2c(item["gt_c_image"]).abs().max().item() == pytest.approx(1.0, abs=1e-5)
        assert item["fname"] == fname.name and item["slice"] == s
        checked += 1
    assert checked == 4
    dt = (time.time() - t0) / checked
    print(f"checked {checked} slices, {dt:.2f}s per item incl. reference")
