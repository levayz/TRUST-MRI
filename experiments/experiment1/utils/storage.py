"""Storage utilities for Experiment 1.

All arrays are saved as float16 (configurable) to save disk space.
Complex arrays are stored as two float arrays (real/imag suffix).
"""

from __future__ import annotations

import json
import pathlib
import subprocess
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# NPZ helpers
# ---------------------------------------------------------------------------

def _to_numpy(v, dtype=None):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    if isinstance(v, np.ndarray):
        if dtype is not None:
            return v.astype(dtype)
        return v
    # scalar / python type — leave as-is (saved as 0-d array)
    return np.array(v)


def save_step_npz(
    path: str | pathlib.Path,
    data: dict,
    dtype: type = np.float16,
    keep_float32: tuple[str, ...] = ("rec_rss",),
) -> None:
    """Save a step's data dict as a compressed .npz file.

    Complex ndarrays are automatically split into _re / _im halves.
    Integer arrays (masks, token indices) keep their dtype.
    Keys listed in keep_float32 are stored as float32 regardless of dtype.
    Scalar values are stored as 0-d arrays.

    Args:
        path:          Destination .npz path (parent directory must exist).
        data:          Dict of (name, value) pairs.
        dtype:         Float dtype for real-valued arrays (default float16).
        keep_float32:  Keys to store as float32 (e.g. reconstruction images
                       used for offline metric computation).
    """
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for k, v in data.items():
        if v is None:
            continue
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        if isinstance(v, np.ndarray):
            if np.iscomplexobj(v):
                arrays[f"{k}_re"] = v.real.astype(dtype)
                arrays[f"{k}_im"] = v.imag.astype(dtype)
            elif v.dtype.kind in ("i", "u", "b"):
                # Integer/bool arrays — preserve dtype
                arrays[k] = v
            else:
                out_dtype = np.float32 if k in keep_float32 else dtype
                arrays[k] = v.astype(out_dtype)
        else:
            arrays[k] = np.array(v)
    np.savez_compressed(str(path), **arrays)


def load_step_npz(path: str | pathlib.Path) -> dict:
    """Load a step .npz and reassemble complex arrays.

    Pairs of _re / _im keys are merged back into a single complex array.

    Returns:
        Dict mapping key → numpy array (or scalar).
    """
    data = dict(np.load(str(path), allow_pickle=True))
    out = {}
    re_keys = {k[:-3] for k in data if k.endswith("_re")}
    for base in re_keys:
        if f"{base}_im" in data:
            out[base] = data[f"{base}_re"].astype(np.float32) + 1j * data[f"{base}_im"].astype(np.float32)
        else:
            out[base] = data[f"{base}_re"]
    for k, v in data.items():
        if k.endswith("_re") or k.endswith("_im"):
            continue
        arr = v
        out[k] = arr.item() if arr.ndim == 0 else arr
    return out


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------

def save_parquet(path: str | pathlib.Path, rows: list[dict]) -> None:
    """Save a list of dicts as a Parquet file.

    Arrays stored as bytes via JSON are acceptable for small arrays. For
    large arrays, callers should store them separately and reference by path.
    """
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(str(path), index=False)


def load_parquet(path: str | pathlib.Path) -> pd.DataFrame:
    return pd.read_parquet(str(path))


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def write_manifest(
    run_dir: str | pathlib.Path,
    config: dict,
    extra: dict | None = None,
) -> None:
    """Write manifest.json to run_dir.

    Args:
        run_dir: Run output directory.
        config:  Loaded YAML config dict.
        extra:   Additional keys to merge in (e.g. paths, slice counts).
    """
    run_dir = pathlib.Path(run_dir)
    manifest: dict[str, Any] = {
        "run_id": run_dir.name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "git_commit": _git_commit(),
        "config": config,
    }
    if extra:
        manifest.update(extra)
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
