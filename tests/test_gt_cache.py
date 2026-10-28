"""Ground-truth cache and the single-pass GT encoding (needs a GPU and the real tokenizer; skipped otherwise).
Checks: encode_features-based tokens / latent equal the old encode() / latent_target() outputs; reconstruct_far never touches the
token path of the input for a v2.1 model; a cache written by precompute_gt_cache.py round-trips through SenseSliceDataset.
Run: CUDA_VISIBLE_DEVICES=5 /opt/miniconda3/envs/meditok/bin/python -m pytest tests/test_gt_cache.py -q
"""
import os, shutil, subprocess, sys
from pathlib import Path
import pytest, torch

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "results/brain_sense/cart_1d/run11_v21_latent/cf_0.04/acc_8_maps_net_tok512_v2.1asym_latent/checkpoints/best_final.pt"
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not CKPT.exists(), reason="needs a GPU and run11")
PY = sys.executable
VAL_ROOT, VAL_SENS = "fastmri/brain/multicoil_val", "fastmri/brain/multicoil_val_sens_maps_gpu,fastmri/brain/multicoil_val_sens_maps"


@pytest.fixture(scope="module")
def tokenizer():
    from autoregressive.sample.reconstruct_far import load_meditok
    return load_meditok(str(ROOT / "weights/meditok/config_512.json"), str(ROOT / "weights/meditok/brain_mc_sense_cplxpair_res512_ep5_frozen.pth"), torch.device("cuda"))


@pytest.fixture(scope="module")
def one_volume():
    from dataset.sense_data import SenseSliceDataset
    from dataset.data.subsample import create_mask_for_mask_type
    split = ROOT / "results/sense/splits/one_volume.txt"
    ds = SenseSliceDataset(str(ROOT / VAL_ROOT), [str(ROOT / d) for d in VAL_SENS.split(",")], create_mask_for_mask_type("random", [0.04], [8]),
                           resolution=256, custom_split=str(split))
    return ds, split


def test_single_pass_matches_encode_and_latent_target(tokenizer, one_volume):
    from autoregressive.sample.reconstruct_far import encode
    from models.sense_ops import latent_target, tok_upsample, encode_features, features_to_tokens, features_to_latent
    ds, _ = one_volume
    gt = torch.stack([ds[i]["gt_c_image"] for i in (3, 8)]).cuda()
    gt = tok_upsample(gt, 512, "bicubic")
    with torch.no_grad():
        for c in range(2):
            f = encode_features(tokenizer, gt[..., c])
            assert torch.equal(features_to_tokens(tokenizer, f), encode(tokenizer, gt[..., c]))
            assert torch.allclose(features_to_latent(f), latent_target(tokenizer, gt[..., c]), atol=1e-5)


def test_v21_reconstruction_never_encodes_input_tokens(tokenizer, one_volume):
    from types import SimpleNamespace
    from autoregressive.sample.reconstruct_far import load_gpt
    from autoregressive.models.gpt_kar import infer_model_args_from_state_dict
    from autoregressive.models.generate import reconstruct_far
    from models.sense_ops import tok_upsample
    ga = SimpleNamespace(gpt_model="GPT-L", vocab_size=32768, image_size=512, downsample_size=16, num_classes=8, cls_token_num=1, cond_token_num=1024,
                         gpt_type="c2i", num_codebooks=8, num_output_layer=1, vq_embed_path=None, probe_model=False, gpt_ckpt=str(CKPT), compile=False)
    sd = torch.load(CKPT, map_location="cpu"); sd = sd.get("model", sd)
    for k, v in infer_model_args_from_state_dict(sd, 256).items(): setattr(ga, k, v)
    del sd
    gpt = load_gpt(ga, torch.device("cuda"), torch.bfloat16)
    ds, _ = one_volume
    item = ds[5]
    masked = tok_upsample(item["masked_c_image"][None].cuda(), 512, "bicubic")
    mask = item["mask"][None].cuda().expand(1, 256, 256, 1).contiguous()
    calls = {"n": 0}
    orig = tokenizer.quantizer.f_to_idx
    def counting(f):
        calls["n"] += 1; return orig(f)
    tokenizer.quantizer.f_to_idx = counting
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            x, aux = reconstruct_far(gpt, tokenizer, masked, mask, sampling_kwargs={"temperature": 1.0, "top_k": 0, "top_p": 1.0}, n_steps=1,
                                     far_mask_mode="radial", center_fractions=0.04, timestep_offset=0, dc_fn=lambda img, m: img)   # identity DC: only the tokenisation count matters here
    finally:
        tokenizer.quantizer.f_to_idx = orig
    assert calls["n"] == 0, "a v2.1 model must not tokenise its input"
    assert x.shape == (1, 512, 512, 2) and aux["codes"].shape == (1, 2, 8, 1024)


def test_cache_round_trip(tokenizer, one_volume):
    from dataset.sense_data import SenseSliceDataset
    from dataset.data.subsample import create_mask_for_mask_type
    from autoregressive.sample.reconstruct_far import encode
    from models.sense_ops import latent_target, tok_upsample
    ds, split = one_volume
    out = ROOT / "results/gt_cache_test"
    shutil.rmtree(out, ignore_errors=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    subprocess.run([PY, str(ROOT / "precompute_gt_cache.py"), "--data-root", VAL_ROOT, "--data-split", str(split), "--sens-dirs", VAL_SENS,
                    "--out-dir", str(out), "--batch-size", "4", "--num-workers", "2"], check=True, cwd=ROOT, env=env)
    ds_c = SenseSliceDataset(str(ROOT / VAL_ROOT), [str(ROOT / d) for d in VAL_SENS.split(",")], create_mask_for_mask_type("random", [0.04], [8]),
                             resolution=256, custom_split=str(split), gt_cache_dir=str(out), gt_cache_token_size=512)
    assert len(ds_c) == len(ds)
    for i in (0, 7, len(ds) - 1):
        it = ds_c[i]
        assert it["gt_cached"] and it["gt_tokens"].shape == (2, 8, 1024) and it["gt_latent"].shape == (2, 1024, 64)
        gt = tok_upsample(it["gt_c_image"][None].cuda(), 512, "bicubic")
        with torch.no_grad():
            for c in range(2):
                # the cache was encoded in batches of 4; a single-image encode flips about 0.05 percent of near-tie codes (the same flips
                # occur between batch sizes on the fly); against the same batch composition the cache is bit-exact
                assert (it["gt_tokens"][c].cuda() != encode(tokenizer, gt[..., c])[0]).float().mean() < 0.005
                assert torch.allclose(it["gt_latent"][c].cuda(), latent_target(tokenizer, gt[..., c])[0], atol=2e-3)
    shutil.rmtree(out, ignore_errors=True)
