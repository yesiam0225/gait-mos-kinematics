"""Tier-2 Mahalanobis stride rejection for ensemble averaging."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2
from sklearn.decomposition import PCA


def mahalanobis_reject(curves: np.ndarray, alpha: float = 0.001,
                       n_components: int = 5, min_curves: int = 4) -> np.ndarray:
    """
    Flag outlier stride curves via PCA + Mahalanobis distance in reduced space.

    Parameters
    ----------
    curves : (N, T) array — one time-normalized curve per stride.

    Returns
    -------
    keep_mask : length-N bool, True = inlier (keep).
    """
    curves = np.asarray(curves, dtype=float)
    n, _t = curves.shape
    if n < min_curves:
        return np.ones(n, dtype=bool)

    n_comp = min(n_components, n - 1)
    if n_comp < 1:
        return np.ones(n, dtype=bool)

    filled = curves.copy()
    for j in range(filled.shape[1]):
        col = filled[:, j]
        m = np.nanmean(col)
        col[np.isnan(col)] = m
        filled[:, j] = col

    reduced = PCA(n_components=n_comp).fit_transform(filled)
    mu = np.mean(reduced, axis=0)
    cov = np.cov(reduced, rowvar=False)
    if reduced.shape[1] == 1:
        cov = np.array([[float(cov)]])

    inv_cov = np.linalg.pinv(cov)
    dists = np.array([
        np.sqrt((x - mu) @ inv_cov @ (x - mu)) for x in reduced
    ])
    threshold = np.sqrt(chi2.ppf(1.0 - alpha, df=n_comp))
    return dists <= threshold
