from __future__ import annotations

import numpy as np
from nibabel.cifti2 import Cifti2Header, Cifti2Image, cifti2_axes
import pytest


@pytest.fixture
def synthetic_dlabel(tmp_path):
    """10-vertex dlabel CIFTI written to tmp_path; no real atlas needed."""
    n_vertices = 10
    bm_axis = cifti2_axes.BrainModelAxis.from_mask(
        np.ones(n_vertices, dtype=bool), name="cortex_left"
    )
    label_table = {
        0: ("???", (1.0, 1.0, 1.0, 0.0)),
        1: ("L_IPS1_ROI", (0.0, 0.0, 1.0, 1.0)),
        2: ("L_4_ROI", (0.0, 1.0, 0.0, 1.0)),
    }
    label_axis = cifti2_axes.LabelAxis(name=["parcels"], label=label_table)
    data = np.array([[0, 0, 1, 1, 1, 2, 2, 2, 2, 2]], dtype=np.float32)

    header = Cifti2Header.from_axes((label_axis, bm_axis))
    path = tmp_path / "synthetic.dlabel.nii"
    Cifti2Image(data, header=header).to_filename(str(path))
    return path


@pytest.fixture
def synthetic_dscalar_template(tmp_path):
    """10-vertex dscalar CIFTI template written to tmp_path; no real data needed."""
    n_vertices = 10
    bm_axis = cifti2_axes.BrainModelAxis.from_mask(
        np.ones(n_vertices, dtype=bool), name="cortex_left"
    )
    scalar_axis = cifti2_axes.ScalarAxis(["template_map"])
    data = np.arange(n_vertices, dtype=np.float32)[np.newaxis, :]

    header = Cifti2Header.from_axes((scalar_axis, bm_axis))
    path = tmp_path / "synthetic.dscalar.nii"
    Cifti2Image(data, header=header).to_filename(str(path))
    return path
