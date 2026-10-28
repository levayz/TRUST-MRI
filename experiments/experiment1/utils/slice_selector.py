"""Center-out slice selection for Experiment 1.

Produces a deterministic, reproducible list of (scan, slice) pairs for the
heavy subset used in candidate-line and oracle experiments.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import h5py


def make_sample_id(dataset: str, scan_id: str, slice_index: int) -> str:
    """Canonical sample identifier used across all output files."""
    stem = pathlib.Path(scan_id).stem  # strip .h5 if present
    return f"{dataset}_{stem}_slice{slice_index:04d}"


def _center_out_order(num_slices: int) -> list[int]:
    """Return slice indices in center-out order for a volume of num_slices slices."""
    center = num_slices // 2
    order = [center]
    for offset in range(1, num_slices):
        if center - offset >= 0:
            order.append(center - offset)
        if center + offset < num_slices:
            order.append(center + offset)
    return order


def select_slices_fixed_fractions(
    data_root: str | pathlib.Path,
    split_file: str | pathlib.Path,
    dataset: str,
    acceleration: int,
    num_scans: int,
    fractions: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> list[dict]:
    """Select slices at fixed volume fractions (e.g. 25%, 50%, 75%) from the first N scans.

    Args:
        data_root:    Directory containing H5 files.
        split_file:   Text file with one H5 filename per line.
        dataset:      "brain" or "knee".
        acceleration: Integer acceleration factor (for metadata only).
        num_scans:    Use only the first N scans from the split file.
        fractions:    Volume fractions at which to take a slice from each scan.

    Returns:
        List of dicts with keys:
            global_index, scan_id, volume_id, slice_index, volume_fraction,
            n_slices, dataset, contrast, acceleration.
    """
    data_root = pathlib.Path(data_root)
    split_file = pathlib.Path(split_file)

    filenames = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    h5_paths = [data_root / fn for fn in filenames if (data_root / fn).exists()]
    h5_paths = sorted(h5_paths)[:num_scans]

    records: list[dict] = []
    global_idx = 0

    for path in h5_paths:
        with h5py.File(path, "r") as f:
            n_slices = int(f["kspace"].shape[0])
            attrs = dict(f.attrs)
        acq = attrs.get("acquisition", "")
        if "PDFS" in acq:
            contrast = "PDFS"
        elif "PD" in acq:
            contrast = "PD"
        else:
            contrast = None
        for frac in fractions:
            slice_idx = min(int(n_slices * frac), n_slices - 1)
            records.append({
                "global_index": global_idx,
                "scan_id": path.name,
                "volume_id": path.name,
                "slice_index": slice_idx,
                "volume_fraction": frac,
                "n_slices": n_slices,
                "dataset": dataset,
                "contrast": contrast,
                "acceleration": acceleration,
            })
            global_idx += 1

    return records


def select_slices_center_out(
    data_root: str | pathlib.Path,
    split_file: str | pathlib.Path,
    dataset: str,
    acceleration: int,
    num_slices: Optional[int],
    seed: int = 42,
) -> list[dict]:
    """Build a deterministic center-out slice list from a split file.

    Args:
        data_root:    Directory containing H5 files.
        split_file:   Text file with one H5 filename per line.
        dataset:      "brain" or "knee".
        acceleration: Integer acceleration factor (for metadata only).
        num_slices:   Maximum number of slices to select (None = all).
        seed:         Unused (selection is fully deterministic), kept for API consistency.

    Returns:
        List of dicts, one per selected slice, with keys:
            global_index, scan_id, volume_id, slice_index,
            center_slice_index, offset_from_center,
            dataset, contrast, acceleration.
    """
    data_root = pathlib.Path(data_root)
    split_file = pathlib.Path(split_file)

    # Read filenames from split file
    filenames = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]

    # Filter to files that actually exist
    h5_paths = []
    for fn in filenames:
        p = data_root / fn
        if p.exists():
            h5_paths.append(p)

    # Sort deterministically
    h5_paths = sorted(h5_paths)

    # Build (path, slice_order) pairs per scan
    scan_info: list[tuple[pathlib.Path, list[int], int, str]] = []  # (path, slice_order, num_slices, contrast)
    for path in h5_paths:
        with h5py.File(path, "r") as f:
            n = int(f["kspace"].shape[0])
            attrs = dict(f.attrs)
        acq = attrs.get("acquisition", "")
        if "PDFS" in acq:
            contrast = "PDFS"
        elif "PD" in acq:
            contrast = "PD"
        else:
            contrast = None
        order = _center_out_order(n)
        scan_info.append((path, order, n, contrast))

    # Interleave scans in center-out order until quota reached
    records: list[dict] = []
    global_idx = 0

    # Round-robin across scans, advancing each scan's offset simultaneously
    max_depth = max((len(info[1]) for info in scan_info), default=0)
    done = False
    for depth in range(max_depth):
        for path, order, n_slices, contrast in scan_info:
            if depth >= len(order):
                continue
            slice_idx = order[depth]
            center_idx = n_slices // 2
            offset = slice_idx - center_idx
            rec = {
                "global_index": global_idx,
                "scan_id": path.name,
                "volume_id": path.name,
                "slice_index": slice_idx,
                "center_slice_index": center_idx,
                "offset_from_center": offset,
                "dataset": dataset,
                "contrast": contrast,
                "acceleration": acceleration,
            }
            records.append(rec)
            global_idx += 1
            if num_slices is not None and global_idx >= num_slices:
                done = True
                break
        if done:
            break

    return records


def select_stratified_center_slices(
    data_root: str | pathlib.Path,
    split_file: str | pathlib.Path,
    dataset: str,
    acceleration: int,
    scans_per_contrast: int = 8,
    n_center_slices: int = 15,
    seed: int = 42,
) -> list[dict]:
    """Randomly select scans stratified by contrast, then take the N center slices.

    Reads all scans from the split file, groups them by contrast (PD / PDFS),
    randomly samples `scans_per_contrast` from each group, and for each selected
    scan returns the `n_center_slices` slices closest to the volume center.

    Args:
        data_root:           Directory containing H5 files.
        split_file:          Text file with one H5 filename per line.
        dataset:             "brain" or "knee".
        acceleration:        Acceleration factor (for metadata only).
        scans_per_contrast:  How many scans to pick from each contrast group.
        n_center_slices:     How many center slices to take per scan.
        seed:                Random seed for reproducible scan selection.

    Returns:
        List of dicts with keys: scan_id, slice_index, n_slices, contrast,
        dataset, acceleration, volume_fraction.
    """
    import random

    data_root = pathlib.Path(data_root)
    split_file = pathlib.Path(split_file)

    filenames = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]

    # Collect (filename, n_slices, contrast) for all available scans
    by_contrast: dict[str, list[tuple[str, int]]] = {}
    for fn in filenames:
        h5_path = data_root / fn
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, "r") as f:
            n_slices = int(f["kspace"].shape[0])
            attrs = dict(f.attrs)
        acq = attrs.get("acquisition", "")
        if "PDFS" in acq:
            contrast = "PDFS"
        elif "PD" in acq:
            contrast = "PD"
        else:
            contrast = "unknown"
        by_contrast.setdefault(contrast, []).append((fn, n_slices))

    # Sample scans_per_contrast from each contrast group
    rng = random.Random(seed)
    selected: list[tuple[str, int, str]] = []  # (filename, n_slices, contrast)
    for contrast, entries in sorted(by_contrast.items()):
        n = min(scans_per_contrast, len(entries))
        chosen = rng.sample(entries, n)
        selected.extend((fn, ns, contrast) for fn, ns in chosen)
        if len(entries) < scans_per_contrast:
            print(
                f"  Warning: only {len(entries)} {contrast} scans available "
                f"in split (requested {scans_per_contrast})"
            )

    # Build slice records: n_center_slices around the volume midpoint
    records: list[dict] = []
    for fn, n_slices, contrast in selected:
        center = n_slices // 2
        half = n_center_slices // 2
        start = max(0, center - half)
        end = min(n_slices, start + n_center_slices)
        start = max(0, end - n_center_slices)  # re-anchor if clamped at top
        for idx in range(start, end):
            records.append({
                "scan_id": fn,
                "slice_index": idx,
                "n_slices": n_slices,
                "dataset": dataset,
                "contrast": contrast,
                "acceleration": acceleration,
                "volume_fraction": idx / max(n_slices - 1, 1),
            })

    return records


def select_all_slices_from_split(
    data_root: str | pathlib.Path,
    split_file: str | pathlib.Path,
    dataset: str,
    acceleration: int,
    num_scans: int | None = None,
    scan_offset: int = 0,
) -> list[dict]:
    """Return one record per slice for every scan in the split file.

    Args:
        data_root:    Directory containing H5 files.
        split_file:   Text file with one H5 filename per line.
        dataset:      "brain" or "knee".
        acceleration: Integer acceleration factor (for metadata only).
        num_scans:    Use only this many scans after offset (None = all).
        scan_offset:  Skip this many scans from the beginning of the file.

    Returns:
        List of dicts with keys: scan_id, slice_index, n_slices, dataset, acceleration.
    """
    data_root = pathlib.Path(data_root)
    split_file = pathlib.Path(split_file)

    filenames = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
    filenames = filenames[scan_offset:]
    if num_scans is not None:
        filenames = filenames[:num_scans]

    records: list[dict] = []
    for fn in filenames:
        h5_path = data_root / fn
        if not h5_path.exists():
            print(f"  SKIP (not found): {h5_path}")
            continue
        with h5py.File(h5_path, "r") as f:
            n_slices = int(f["kspace"].shape[0])
        for idx in range(n_slices):
            records.append({
                "scan_id": fn,
                "slice_index": idx,
                "n_slices": n_slices,
                "dataset": dataset,
                "acceleration": acceleration,
            })
    return records
