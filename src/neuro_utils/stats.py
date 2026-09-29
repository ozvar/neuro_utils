from __future__ import annotations

import warnings

import numpy as np


def map_corr(
    a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None
) -> tuple[float, int]:
    """Pearson correlation between two maps over shared valid vertices.

    A vertex is dropped if it is non-finite in either map or False in `mask`.
    Zeros are kept (not treated as missing). The maps must share the same
    grayordinate ordering.

    Args:
        a, b: 1-D maps of identical length, or (1, N) arrays such as a dscalar.
        mask: optional boolean array of the same length; True = use this vertex.

    Returns:
        r: Pearson r, or NaN if either map is constant over the valid vertices.
        n: number of vertices used.
    """
    a = np.squeeze(np.asarray(a, dtype=float))
    b = np.squeeze(np.asarray(b, dtype=float))
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError(f"expected two 1-D maps of equal length, got {a.shape} and {b.shape}")

    keep = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != a.shape:
            raise ValueError(f"mask shape {mask.shape} does not match maps {a.shape}")
        # keep a vertex only if it is finite in both maps AND selected by the mask
        keep &= mask
    n = int(keep.sum())

    a_centred = a[keep] - a[keep].mean()
    b_centred = b[keep] - b[keep].mean()
    denom = np.sqrt((a_centred @ a_centred) * (b_centred @ b_centred))
    if denom == 0:
        warnings.warn("map_corr: a map is constant over valid vertices, returning NaN")
        return float("nan"), n
    return float((a_centred @ b_centred) / denom), n
