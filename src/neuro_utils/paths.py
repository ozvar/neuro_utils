from __future__ import annotations

import os
from pathlib import Path


class AtlasNotFoundError(FileNotFoundError):
    pass


_ENV_VAR = "NEURO_UTILS_ATLAS_DIR"

_ATLAS_FILENAMES: dict[str, str] = {
    "mmp360": "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii",
    "glasser360": "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii",
    "cole_anticevic": "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_netassignments_LR.dlabel.nii",
    "yeo7": "Yeo7_Yeo-Buckner-Choi-Raut.dlabel.nii",
}


def atlas_dir() -> Path:
    """Atlas root from the NEURO_UTILS_ATLAS_DIR environment variable, re-read on every call."""
    if _ENV_VAR not in os.environ:
        raise AtlasNotFoundError(
            f"{_ENV_VAR} is not set. Set it to the directory holding the HCP atlas files."
        )
    return Path(os.environ[_ENV_VAR])


def resolve_atlas_path(name: str) -> Path:
    """Resolve a preset atlas name to its file path."""
    if name not in _ATLAS_FILENAMES:
        raise KeyError(f"Unknown atlas preset {name!r}. Valid presets: {sorted(_ATLAS_FILENAMES)}")
    path = atlas_dir() / _ATLAS_FILENAMES[name]
    if not path.exists():
        raise AtlasNotFoundError(
            f"Atlas file not found: {path}\n"
            f"Check that {_ENV_VAR} points to the directory holding the atlas files."
        )
    return path
