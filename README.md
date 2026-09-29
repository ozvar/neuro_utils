# neuro_utils

Installable package for CIFTI I/O, atlas parcellation loading and map statistics. Covers loading, parsing and template-based writing of CIFTI dscalar/dlabel files, and Pearson correlation between two maps (`map_corr`).

Supported data: HCP-style CIFTI files on the 32k fsLR surface, in the standard 91,282 grayordinate layout (59,412 cortical vertices after medial wall exclusion). Other grayordinate layouts are not supported, and several functions assert these counts.

## Install

```bash
pip install -e /path/to/neuro_utils
```

## Atlas files

Atlas files (Glasser MMP360, Cole-Anticevic CAB-NP, Yeo7) are not bundled, because they carry an HCP Data Use Agreement. There is no default location: you must set `NEURO_UTILS_ATLAS_DIR` to the directory holding them (for example your `HCP_S1200_Atlas` folder) before calling `load_atlas` or `resolve_atlas_path`:

Linux and macOS (bash):

```bash
export NEURO_UTILS_ATLAS_DIR=/path/to/HCP_S1200_Atlas
```

Windows (PowerShell, current session only):

```powershell
$env:NEURO_UTILS_ATLAS_DIR = "C:\path\to\HCP_S1200_Atlas"
```

Windows (cmd, persists for new sessions):

```bat
setx NEURO_UTILS_ATLAS_DIR "C:\path\to\HCP_S1200_Atlas"
```

If it is unset, or the atlas file is not in that directory, an `AtlasNotFoundError` names the variable. On Linux and macOS, put the export in your shell profile and in any batch script that loads atlases.

## Usage

```python
from neuro_utils import load_cifti, load_atlas, cortex_only, map_corr

data, axes = load_cifti("group_R2_full_mean.dscalar.nii")
labels, names = load_atlas("mmp360")
r, n = map_corr(map_a, map_b)  # Pearson r over shared finite vertices, and the vertex count
```

## Tests

```bash
pytest
```
