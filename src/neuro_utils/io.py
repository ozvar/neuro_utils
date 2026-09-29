from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.cifti2 import Cifti2Header, Cifti2Image
from nibabel.cifti2.cifti2_axes import BrainModelAxis, ScalarAxis

from .constants import N_CORTEX_GRAYORDINATES, N_GRAYORDINATES_DENSE


def load_cifti(path: Path) -> tuple[np.ndarray, tuple]:
    """Load a CIFTI file.

    Returns:
        data: raw get_fdata() output, no implicit squeeze/ravel.
        axes: (row_axis, column_axis) via header.get_axis(0)/(1).
    """
    img = nib.load(str(path))
    data = img.get_fdata()
    axes = (img.header.get_axis(0), img.header.get_axis(1))
    return data, axes


def cifti_data(path: Path) -> np.ndarray:
    """Data-only convenience wrapper around load_cifti."""
    data, _ = load_cifti(path)
    return data


def get_brain_model_axis(img: Cifti2Image) -> BrainModelAxis:
    return img.header.get_axis(1)


def cortex_only(data: np.ndarray) -> np.ndarray:
    """Slice the leading cortical block from full dense (91282) data."""
    assert data.shape[-1] == N_GRAYORDINATES_DENSE, (
        f"expected {N_GRAYORDINATES_DENSE} grayordinates, got {data.shape[-1]}"
    )
    return data[..., :N_CORTEX_GRAYORDINATES]


def to_full_grayordinates(cortex_data: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Pad cortex-only data back to full dense (91282) grayordinates."""
    assert cortex_data.shape[-1] == N_CORTEX_GRAYORDINATES, (
        f"expected {N_CORTEX_GRAYORDINATES} grayordinates, got {cortex_data.shape[-1]}"
    )
    pad_width = N_GRAYORDINATES_DENSE - N_CORTEX_GRAYORDINATES
    pad = [(0, 0)] * (cortex_data.ndim - 1) + [(0, pad_width)]
    return np.pad(cortex_data, pad, constant_values=fill)


def save_cifti_like(data: np.ndarray, template_path: Path, out_path: Path) -> None:
    """Rebuild a Cifti2Image using template_path's header axes and save to out_path."""
    template = nib.load(str(template_path))
    header = template.header
    brain_model_axis = header.get_axis(1)
    assert data.shape[-1] == len(brain_model_axis), (
        f"data has {data.shape[-1]} grayordinates, template expects {len(brain_model_axis)}"
    )
    new_img = Cifti2Image(data, header=Cifti2Header.from_axes((header.get_axis(0), brain_model_axis)))
    new_img.to_filename(str(out_path))


def new_dscalar(
    data: np.ndarray,
    template_path: Path,
    map_names: list[str] | None = None,
    out_path: Path | None = None,
) -> Cifti2Image:
    """Build a dscalar Cifti2Image matching template_path's BrainModelAxis.

    data is (n_grayordinates,) or (n_maps, n_grayordinates). Writes to
    out_path if given, and always returns the constructed image.
    """
    template = nib.load(str(template_path))
    brain_model_axis = template.header.get_axis(1)

    if data.ndim == 1:
        data = data[np.newaxis, :]
    assert data.shape[-1] == len(brain_model_axis), (
        f"data has {data.shape[-1]} grayordinates, template expects {len(brain_model_axis)}"
    )
    n_maps = data.shape[0]
    if map_names is None:
        map_names = [f"map_{i}" for i in range(n_maps)]

    scalar_axis = ScalarAxis(map_names)
    header = Cifti2Header.from_axes((scalar_axis, brain_model_axis))
    img = Cifti2Image(data, header=header)
    if out_path is not None:
        img.to_filename(str(out_path))
    return img
