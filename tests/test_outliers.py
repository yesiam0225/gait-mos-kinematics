"""Tests for Tier-1 marker spike rejection and Tier-2 Mahalanobis rejection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gait_mos_kinematics.gait_kinematics import reject_marker_spikes
from gait_mos_kinematics.outlier_rejection import mahalanobis_reject


def _make_marker_df(n: int = 50, marker: str = 'RASI') -> pd.DataFrame:
    t = np.arange(n, dtype=float)
    return pd.DataFrame({
        f'{marker}_x': 100.0 + t,
        f'{marker}_y': 200.0,
        f'{marker}_z': 900.0,
    })


def test_reject_marker_spikes_removes_large_jump():
    df = _make_marker_df(50)
    # Insert a 5000 mm spike at frame 20→21
    df.loc[21, 'RASI_x'] = df.loc[20, 'RASI_x'] + 5000.0

    cleaned, report = reject_marker_spikes(df, ['RASI'], threshold_mm_per_frame=100.0)

    assert report['RASI'] >= 1
    assert np.isnan(cleaned.loc[20, 'RASI_x'])
    assert np.isnan(cleaned.loc[21, 'RASI_x'])


def test_mahalanobis_reject_flags_obvious_outlier_curve():
    rng = np.random.default_rng(0)
    n_strides, n_pts = 25, 101
    curves = np.zeros((n_strides, n_pts))
    curves[:-1] = rng.normal(0.0, 0.01, (n_strides - 1, n_pts))
    curves[-1] = 1e6  # clearly outside the tight cluster

    keep = mahalanobis_reject(curves, alpha=0.001, n_components=5, min_curves=4)

    assert keep.sum() == n_strides - 1
    assert not keep[-1]


def test_mahalanobis_reject_returns_all_true_when_few_curves():
    curves = np.random.randn(3, 101)
    keep = mahalanobis_reject(curves, min_curves=4)
    assert keep.all()
