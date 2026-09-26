"""SYNTHETIC scenes for --dry-run and tests. NOT observations, NOT NASA data.

They exist so the pipeline, dashboard and tests can be built before real data
has downloaded, and so every run of the tests checks that a known, injected
change is recovered. Every synthetic scene is labelled as such in its
provenance and on the dashboard.
"""

from __future__ import annotations

import numpy as np
from affine import Affine
from scipy.ndimage import gaussian_filter

from .scene import SarScene

SYNTHETIC_SOURCE = {
    "dataset": "SYNTHETIC speckled backscatter (changelens.synthetic)",
    "label": "SYNTHETIC test data (not an observation)",
    "url": None,
    "maturity": "n/a",
}


def _speckle(rng, mean_db: np.ndarray, looks: float) -> np.ndarray:
    """Gamma-distributed multiplicative speckle around a mean power (in dB)."""
    mean = 10 ** (mean_db / 10)
    return (mean * rng.gamma(looks, 1.0 / looks, size=mean.shape)).astype(np.float32)


def make_synthetic_case(shape=(240, 240), pixel_m: float = 20.0, epsg: int = 32631,
                        origin=(500_000.0, 110_000.0), flood: bool = True, seed: int = 0,
                        looks: float = 4.0):
    """Return (pre, post, hand, truth) for a toy river valley.

    The grid is placed in the open Gulf of Guinea (UTM 31N, ~1 deg N, 3 deg E) on
    purpose: a "river valley" drawn over the ocean cannot be mistaken for a
    real observation of a real place.

    Layout (rows x cols): a river along the middle, a low floodplain that floods
    in `post` if flood=True, a hill (high HAND) on the right where a dark
    patch appears in `post` (e.g. wet snow or shadow: a known false-positive
    trap that HAND masking must suppress), and a no-data stripe on the left.
    """
    rng = np.random.default_rng(seed)
    rows, cols = shape
    rr, cc = np.mgrid[0:rows, 0:cols]

    land_db, water_db = -8.0, -22.0
    # Smooth-ish land texture (fields of different roughness) plus speckle below.
    texture = gaussian_filter(rng.normal(0, 1.0, shape), 3)
    texture = 2.0 * texture / (texture.std() + 1e-9)
    base = (land_db + texture).astype(np.float32)
    river = np.abs(rr - rows // 2) < 6
    floodplain = (np.abs(rr - rows // 2) < 40) & (cc > 40) & (cc < 150) & ~river
    hill = cc >= 180
    dark_patch = hill & (np.abs(rr - rows // 4) < 15) & (cc < 215)

    pre_mean = np.where(river, water_db, base)
    post_mean = pre_mean.copy()
    if flood:
        post_mean = np.where(floodplain, water_db, post_mean)
    post_mean = np.where(dark_patch, water_db, post_mean)

    pre = _speckle(rng, pre_mean, looks)
    post = _speckle(rng, post_mean, looks)
    nodata = cc < 6
    pre[nodata] = np.nan
    post[nodata] = np.nan

    hand = np.clip(np.abs(rr - rows // 2) * 0.3, 0, None).astype(np.float32)
    hand[hill] = 60.0

    transform = Affine(pixel_m, 0, origin[0], 0, -pixel_m, origin[1])
    src_pre = {**SYNTHETIC_SOURCE, "granule_id": f"SYNTH-PRE-seed{seed}"}
    src_post = {**SYNTHETIC_SOURCE, "granule_id": f"SYNTH-POST-seed{seed}"}
    pre_s = SarScene(pre, f"EPSG:{epsg}", transform, "HH", "2026-07-01T00:00:00Z", src_pre)
    post_s = SarScene(post, f"EPSG:{epsg}", transform, "HH", "2026-07-13T00:00:00Z", src_post)
    truth = {
        "new_water": (floodplain if flood else np.zeros(shape, bool)) & ~nodata,
        "river": river & ~nodata,
        "false_positive_trap": dark_patch,
    }
    return pre_s, post_s, hand, truth
