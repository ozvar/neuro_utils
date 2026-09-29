from __future__ import annotations

import numpy as np
import pytest

from neuro_utils.atlas import (
    load_atlas,
    load_dlabel,
    mmp_base_name,
    parcel_id_to_name,
    parcel_name_to_id,
)
from neuro_utils.paths import AtlasNotFoundError, resolve_atlas_path


def test_load_dlabel_known_values(synthetic_dlabel):
    labels, names = load_dlabel(synthetic_dlabel)
    np.testing.assert_array_equal(labels, [0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    assert names == {0: "???", 1: "L_IPS1_ROI", 2: "L_4_ROI"}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("L_IPS1_ROI", "ips1"),
        ("R_4_ROI", "4"),
        ("R_Somatomotor", "somatomotor"),
        ("l_v1_roi", "v1"),
    ],
)
def test_mmp_base_name(raw, expected):
    assert mmp_base_name(raw) == expected


def test_parcel_name_to_id_and_back_round_trip(synthetic_dlabel):
    _, names = load_dlabel(synthetic_dlabel)
    pid = parcel_name_to_id("IPS1", names)
    assert pid == 1
    assert parcel_id_to_name(pid, names) == "L_IPS1_ROI"


def test_parcel_name_to_id_unknown_raises_with_candidates(synthetic_dlabel):
    _, names = load_dlabel(synthetic_dlabel)
    with pytest.raises(KeyError, match="ips1"):
        parcel_name_to_id("NOT_A_PARCEL", names)


def test_resolve_atlas_path_unknown_preset_lists_valid_names():
    with pytest.raises(KeyError, match="mmp360"):
        resolve_atlas_path("not_a_real_preset")


def test_resolve_atlas_path_missing_file_names_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("NEURO_UTILS_ATLAS_DIR", str(tmp_path / "nonexistent"))
    with pytest.raises(AtlasNotFoundError, match="NEURO_UTILS_ATLAS_DIR"):
        resolve_atlas_path("mmp360")


def test_resolve_atlas_path_unset_env_var_raises_and_names_it(monkeypatch):
    monkeypatch.delenv("NEURO_UTILS_ATLAS_DIR", raising=False)
    with pytest.raises(AtlasNotFoundError, match="NEURO_UTILS_ATLAS_DIR is not set"):
        resolve_atlas_path("mmp360")


def test_load_atlas_reads_from_env_var_dir(monkeypatch, synthetic_dlabel):
    monkeypatch.setenv("NEURO_UTILS_ATLAS_DIR", str(synthetic_dlabel.parent))
    monkeypatch.setattr(
        "neuro_utils.paths._ATLAS_FILENAMES",
        {**{"mmp360": synthetic_dlabel.name}},
    )
    labels, names = load_atlas("mmp360")
    np.testing.assert_array_equal(labels, [0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    assert names[1] == "L_IPS1_ROI"
