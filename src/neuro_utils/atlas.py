from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from .paths import resolve_atlas_path


def load_dlabel(atlas_path: Path) -> tuple[np.ndarray, dict[int, str]]:
    """Load a CIFTI dlabel parcellation.

    Returns:
        labels: per-grayordinate integer label key.
        names: label key -> label name.
    """
    img = nib.load(str(atlas_path))
    labels = img.get_fdata().astype(int).ravel()
    label_table = img.header.get_axis(0).label[0]
    names = {key: value[0] for key, value in label_table.items()}
    return labels, names


def load_atlas(name: str) -> tuple[np.ndarray, dict[int, str]]:
    """Load a preset atlas by name ('mmp360' | 'glasser360' | 'cole_anticevic' | 'yeo7')."""
    return load_dlabel(resolve_atlas_path(name))


def mmp_base_name(label_name: str) -> str:
    """Strip the L_/R_ hemisphere prefix and _ROI suffix from an MMP label name.

    'L_IPS1_ROI' -> 'ips1', 'R_4_ROI' -> '4', 'R_Somatomotor' -> 'somatomotor'.
    Returned lower-case for case-insensitive exact matching.
    """
    core = label_name
    for prefix in ("L_", "R_", "l_", "r_"):
        if core.startswith(prefix):
            core = core[len(prefix):]
            break
    for suffix in ("_ROI", "_roi"):
        if core.endswith(suffix):
            core = core[:-len(suffix)]
            break
    return core.lower()


def parcel_name_to_id(name: str, id_to_name: dict[int, str]) -> int:
    """Exact, hemisphere-agnostic lookup: base name -> label key."""
    wanted = mmp_base_name(name)
    matches = [key for key, label in id_to_name.items() if mmp_base_name(label) == wanted]
    if not matches:
        near = sorted({mmp_base_name(v) for v in id_to_name.values()})
        raise KeyError(f"No parcel named {name!r} (normalized {wanted!r}). Known base names: {near}")
    return matches[0]


def parcel_id_to_name(pid: int, id_to_name: dict[int, str]) -> str:
    return id_to_name[pid]
