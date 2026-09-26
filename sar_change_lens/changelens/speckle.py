"""Speckle reduction that respects no-data pixels.

Speckle is the grainy, multiplicative noise every coherent radar image has.
Without filtering, pixel-by-pixel differences produce "salt and pepper" false
changes. The Lee filter smooths flat areas but keeps edges (shorelines).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def _masked_local_stats(x: np.ndarray, size: int):
    """Local mean and variance over valid (finite) pixels only.

    NaNs are given zero weight, so a coastline or swath edge does not smear
    no-data into its neighbours (normalised convolution).
    """
    valid = np.isfinite(x)
    w = valid.astype(np.float64)
    xv = np.where(valid, x, 0.0).astype(np.float64)
    wsum = uniform_filter(w, size=size, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = uniform_filter(xv, size=size, mode="constant") / wsum
        mean_sq = uniform_filter(xv * xv, size=size, mode="constant") / wsum
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return mean, var, wsum


def lee_filter(power: np.ndarray, size: int = 5, enl: float = 4.0, min_valid_fraction: float = 0.3) -> np.ndarray:
    """Classic Lee filter on *linear power* backscatter.

    size   odd window width in pixels (5-7 is typical for 10-30 m products).
    enl    equivalent number of looks of the input. NISAR GCOV and OPERA
           RTC-S1 are already multi-looked, so ENL > 1; estimate it with
           estimate_enl() over a uniform area or keep the default.
    Pixels that were NaN stay NaN. Pixels whose window is mostly no-data are
    set to NaN rather than filtered from too few neighbours.
    """
    if size % 2 == 0 or size < 3:
        raise ValueError("size must be an odd integer >= 3")
    x = np.asarray(power, dtype=np.float64)
    mean, var, wsum = _masked_local_stats(x, size)
    cu2 = 1.0 / float(enl)  # squared coefficient of variation of pure speckle
    with np.errstate(invalid="ignore", divide="ignore"):
        signal_var = (var - mean * mean * cu2) / (1.0 + cu2)
        k = np.clip(signal_var / var, 0.0, 1.0)
    k = np.where(var > 0, k, 0.0)
    out = mean + k * (x - mean)
    out[~np.isfinite(x) | (wsum < min_valid_fraction)] = np.nan
    return out.astype(np.float32)


def estimate_enl(power: np.ndarray, size: int = 9) -> float:
    """Rough equivalent-number-of-looks estimate: median of mean^2/var in local windows.

    Heterogeneous windows (edges, texture) lower the ratio, so this errs on the
    side of *more* smoothing. Prefer measuring over a hand-picked uniform field.
    """
    x = np.asarray(power, dtype=np.float64)
    mean, var, wsum = _masked_local_stats(x, size)
    ok = np.isfinite(x) & (var > 0) & (wsum > 0.9)
    if ok.sum() < 100:
        return 1.0
    return float(np.median(mean[ok] ** 2 / var[ok]))
