from __future__ import annotations

import numpy as np
import pytest

from neuro_utils.stats import map_corr

N_VERTICES = 1000


@pytest.fixture
def maps():
    """Two independent random 1000-vertex maps; no CIFTI files needed."""
    rng = np.random.default_rng(0)
    return rng.normal(size=N_VERTICES), rng.normal(size=N_VERTICES)


def test_matches_numpy_corrcoef(maps):
    a, b = maps
    r, n = map_corr(a, b)
    np.testing.assert_allclose(r, np.corrcoef(a, b)[0, 1])
    assert n == N_VERTICES


def test_known_values(maps):
    a, b = maps
    np.testing.assert_allclose(map_corr(a, a)[0], 1.0)
    np.testing.assert_allclose(map_corr(a, -a)[0], -1.0)
    assert abs(map_corr(a, b)[0]) < 0.1  # independent noise, 1000 vertices


def test_invariant_to_scale_and_offset(maps):
    a, b = maps
    np.testing.assert_allclose(map_corr(a * 5 + 3, b)[0], map_corr(a, b)[0])


def test_nan_vertices_dropped_from_either_map(maps):
    a, b = maps
    a, b = a.copy(), b.copy()
    a[:100] = np.nan
    b[50:200] = np.nan
    valid = np.isfinite(a) & np.isfinite(b)
    r, n = map_corr(a, b)
    np.testing.assert_allclose(r, np.corrcoef(a[valid], b[valid])[0, 1])
    assert n == valid.sum()


def test_inf_vertices_dropped(maps):
    a, b = maps
    a = a.copy()
    a[10] = np.inf
    valid = np.isfinite(a)
    r, n = map_corr(a, b)
    np.testing.assert_allclose(r, np.corrcoef(a[valid], b[valid])[0, 1])
    assert n == valid.sum()


def test_mask_selects_vertices(maps):
    a, b = maps
    mask = np.zeros(N_VERTICES, dtype=bool)
    mask[:100] = True
    r, n = map_corr(a, b, mask=mask)
    np.testing.assert_allclose(r, np.corrcoef(a[mask], b[mask])[0, 1])
    assert n == 100


def test_nan_inside_mask_dropped_and_outside_mask_ignored(maps):
    a, b = maps
    a = a.copy()
    a[5] = np.nan     # inside mask
    a[500] = np.nan   # outside mask
    mask = np.zeros(N_VERTICES, dtype=bool)
    mask[:100] = True
    r, n = map_corr(a, b, mask=mask)
    keep = mask & np.isfinite(a)
    np.testing.assert_allclose(r, np.corrcoef(a[keep], b[keep])[0, 1])
    assert n == 99


def test_wrong_mask_length_raises(maps):
    a, b = maps
    with pytest.raises(ValueError, match="mask shape"):
        map_corr(a, b, mask=np.ones(10, dtype=bool))


def test_dscalar_shape_matches_1d(maps):
    a, b = maps
    r_2d, n_2d = map_corr(a[None, :], b[None, :])
    r_1d, n_1d = map_corr(a, b)
    np.testing.assert_allclose(r_2d, r_1d)
    assert n_2d == n_1d


def test_timeseries_shape_raises(maps):
    a, b = maps
    stack = np.vstack([a, a])  # (2, N), as a dtseries would be
    with pytest.raises(ValueError, match="1-D"):
        map_corr(stack, stack)


def test_length_mismatch_raises(maps):
    a, b = maps
    with pytest.raises(ValueError, match="equal length"):
        map_corr(a, b[:-1])


def test_constant_map_returns_nan_with_warning(maps):
    a, _ = maps
    with pytest.warns(UserWarning, match="constant"):
        r, n = map_corr(a, np.zeros_like(a))
    assert np.isnan(r)
    assert n == N_VERTICES


def test_zeros_are_kept(maps):
    a, b = maps
    a = a.copy()
    a[:500] = 0.0
    r, n = map_corr(a, b)
    np.testing.assert_allclose(r, np.corrcoef(a, b)[0, 1])
    assert n == N_VERTICES
