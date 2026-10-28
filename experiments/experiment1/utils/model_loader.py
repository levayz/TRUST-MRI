"""Model loading utilities for Experiment 1 (singlecoil).

Reuses patterns from autoregressive/sample/reconstruct_far.py:load_gpt and load_meditok.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Tuple

import torch

# Ensure TRUST-MRI root is on path
_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def load_singlecoil_model(
    checkpoint_path: str | pathlib.Path,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.nn.Module, argparse.Namespace]:
    """Load a singlecoil GPT model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint saved by train.py.
        device:          Target device.
        dtype:           Model dtype (default float32).

    Returns:
        (model, ckpt_args) where ckpt_args is the training argparse.Namespace.
    """
    from autoregressive.models.gpt_kar import KAR_GPT_models as GPT_models

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", argparse.Namespace())

    # Defaults that older checkpoints may be missing
    _defaults = {
        "gpt_model": "GPT-L",
        "vocab_size": 32768,
        "image_size": 256,
        "downsample_size": 16,
        "num_classes": 1,
        "cls_token_num": 1,
        "cond_token_num": 0,
        "gpt_type": "c2i",
        "num_codebooks": 8,
        "num_output_layer": 2,
        "vq_embed_path": None,
        "probe_model": False,
        "compile": False,
    }
    for k, v in _defaults.items():
        if not hasattr(ckpt_args, k):
            setattr(ckpt_args, k, v)

    block_size = (ckpt_args.image_size // ckpt_args.downsample_size) ** 2
    model = GPT_models[ckpt_args.gpt_model](
        vocab_size=ckpt_args.vocab_size,
        block_size=block_size,
        num_classes=ckpt_args.num_classes,
        cls_token_num=ckpt_args.cls_token_num,
        cond_token_num=ckpt_args.cond_token_num,
        model_type=ckpt_args.gpt_type,
        num_codebooks=ckpt_args.num_codebooks,
        n_output_layer=ckpt_args.num_output_layer,
        vq_embed_path=ckpt_args.vq_embed_path,
    )
    model = model.to(device, dtype=dtype)

    state = ckpt.get("model", ckpt.get("module", ckpt))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt_args


def load_meditok(
    config_path: str | pathlib.Path,
    ckpt_path: str | pathlib.Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load frozen MedITok encoder/decoder.

    Replicates reconstruct_far.py:load_meditok exactly.

    Args:
        config_path: Path to config.json (e.g. weights/meditok/config.json).
        ckpt_path:   Path to .pth weights (e.g. weights/meditok/meditok_simple_v1.pth).
        device:      Target device.

    Returns:
        MedITok model in eval mode with frozen parameters.
    """
    from models.meditok import MedITok, DotDict

    with open(str(config_path), "r") as f:
        enc_cfg = json.load(f)
    vt = MedITok(DotDict(enc_cfg)).to(device).eval()
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    vt.load_state_dict(state, strict=True)
    for p in vt.parameters():
        p.requires_grad_(False)
    return vt
