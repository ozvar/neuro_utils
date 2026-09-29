from __future__ import annotations

import numpy as np
import pytest

from neuro_utils.constants import N_CORTEX_GRAYORDINATES, N_GRAYORDINATES_DENSE
from neuro_utils.io import cortex_only, load_cifti, new_dscalar, save_cifti_like, to_full_grayordinates


def test_cortex_only_shape():
    data = np.zeros(N_GRAYORDINATES_DENSE)
    assert cortex_only(data).shape == (N_CORTEX_GRAYORDINATES,)


def test_cortex_only_wrong_shape_raises():
    with pytest.raises(AssertionError, match=str(N_GRAYORDINATES_DENSE)):
        cortex_only(np.zeros(100))


def test_to_full_grayordinates_wrong_shape_raises():
    with pytest.raises(AssertionError, match=str(N_CORTEX_GRAYORDINATES)):
        to_full_grayordinates(np.zeros(100))


def test_to_full_grayordinates_shape_and_fill():
    cortex_data = np.ones(N_CORTEX_GRAYORDINATES)
    full = to_full_grayordinates(cortex_data, fill=-1.0)
    assert full.shape == (N_GRAYORDINATES_DENSE,)
    np.testing.assert_array_equal(full[:N_CORTEX_GRAYORDINATES], 1.0)
    np.testing.assert_array_equal(full[N_CORTEX_GRAYORDINATES:], -1.0)


def test_cortex_to_full_round_trip():
    rng = np.random.default_rng(0)
    original = rng.normal(size=N_CORTEX_GRAYORDINATES)
    round_tripped = cortex_only(to_full_grayordinates(original))
    np.testing.assert_allclose(round_tripped, original)


def test_new_dscalar_write_reload_round_trip(synthetic_dscalar_template, tmp_path):
    data = np.arange(10, dtype=np.float32) * 2
    out_path = tmp_path / "out.dscalar.nii"

    new_dscalar(data, synthetic_dscalar_template, map_names=["doubled"], out_path=out_path)

    reloaded_data, (row_axis, col_axis) = load_cifti(out_path)
    np.testing.assert_allclose(reloaded_data.squeeze(), data)
    assert row_axis.name.tolist() == ["doubled"]
    _, (_, template_col_axis) = load_cifti(synthetic_dscalar_template)
    np.testing.assert_array_equal(col_axis.name, template_col_axis.name)


def test_new_dscalar_wrong_grayordinate_count_raises(synthetic_dscalar_template):
    wrong_shape_data = np.zeros(5)  # template has 10 grayordinates
    with pytest.raises(AssertionError, match="grayordinates"):
        new_dscalar(wrong_shape_data, synthetic_dscalar_template)


def test_save_cifti_like_wrong_grayordinate_count_raises(synthetic_dscalar_template, tmp_path):
    wrong_shape_data = np.zeros((1, 5))  # template has 10 grayordinates
    with pytest.raises(AssertionError, match="grayordinates"):
        save_cifti_like(wrong_shape_data, synthetic_dscalar_template, tmp_path / "out.dscalar.nii")
