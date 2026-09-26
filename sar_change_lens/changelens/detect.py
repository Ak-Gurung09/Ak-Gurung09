"""The change engine: backscatter statistics -> per-pixel change classes.

Science in one paragraph: calm open water reflects radar energy away from the
satellite (specular reflection), so it looks dark. When land floods, the
backscatter at that pixel drops sharply. We therefore flag a pixel as
"possible flood-related change" only if BOTH
  1. its post-event backscatter falls in the dark "water-like" population
     learned from this very scene (an unsupervised 2-component Gaussian
     mixture in dB), and
  2. it got darker by at least `delta_db` compared with the pre-event image,
  3. and it was not already water-like before the event.
Everything else that changed strongly is reported as *unclassified* change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .scene import InsufficientDataError

CLASS_CODES = {
    0: ("nodata", "No data / outside swath / masked input"),
    1: ("no_change", "No significant change detected"),
    2: ("new_water", "Possible flood-related change (new open-water-like surface)"),
    3: ("persistent_water", "Water-like in both images (river, lake, reservoir)"),
    4: ("receded_water", "Water-like before, not after (possible recession)"),
    5: ("other_decrease", "Strong backscatter decrease not reported as flood (not water-like, "
                          "or smaller than the minimum mapping unit)"),
    6: ("other_increase", "Strong backscatter increase (unclassified: could be inundated "
                          "vegetation, wetter soil, crops, structures)"),
    7: ("masked_terrain", "Masked: terrain too high above drainage to flood (HAND)"),
}
NEW_WATER = 2


@dataclass
class DetectionParams:
    """All tunable numbers in one place; every value is written to provenance.json."""

    speckle_size: int = 5
    enl: float = 4.0
    delta_db: float = 3.0            # minimum darkening for "new water"
    hand_max_m: float = 15.0         # above this, water-like pixels are masked
    min_area_m2: float = 20_000.0    # minimum mapping unit (2 ha)
    max_water_mean_db: float = -15.0  # plausibility check on the learned water class
    fallback_threshold_db: float = -18.0  # used only if the mixture model is implausible
    gmm_max_samples: int = 200_000
    min_valid_pixels: int = 500
    seed: int = 0
    # Experimental forest-disturbance mode (cross-pol, e.g. NISAR HV)
    forest_min_hv_db: float = -16.0  # placeholder: check against your scene's histogram
    forest_delta_db: float = 3.0


def to_db(power: np.ndarray) -> np.ndarray:
    """Linear power -> decibels. Non-positive / NaN stay NaN."""
    p = np.asarray(power, dtype=np.float64)
    out = np.full(p.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(p) & (p > 0)
    out[ok] = (10.0 * np.log10(p[ok])).astype(np.float32)
    return out


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold for a 1-D sample (maximises between-class variance)."""
    v = values[np.isfinite(values)]
    hist, edges = np.histogram(v, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu0 = m0 / w0
        mu1 = (m0[-1] - m0) / w1
        between = w0 * w1 * (mu0 - mu1) ** 2
    between[~np.isfinite(between)] = -1
    return float(centers[int(np.argmax(between))])


def _normal_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


@dataclass
class WaterModel:
    """Two-population model of backscatter (dB): dark 'water-like' vs 'land'."""

    method: str                 # "gmm" or "fallback_threshold"
    water_mean: float
    water_std: float
    water_weight: float
    land_mean: float
    land_std: float
    land_weight: float
    threshold_db: float         # where P(water) = 0.5
    otsu_db: float
    ashman_d: float             # > 2 means the two populations separate cleanly
    flags: list = field(default_factory=list)

    def _raw_posterior(self, x: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore", under="ignore"):
            lw = self.water_weight * _normal_pdf(x, self.water_mean, self.water_std)
            ll = self.land_weight * _normal_pdf(x, self.land_mean, self.land_std)
            p = lw / (lw + ll)
        # Both densities underflow far from the data: dark side water, bright side land.
        mid = 0.5 * (self.water_mean + self.land_mean)
        return np.where(np.isfinite(p), p, np.where(x < mid, 1.0, 0.0))

    def posterior_water(self, db: np.ndarray) -> np.ndarray:
        """P(water-like | backscatter in dB), NaN where input is NaN.

        With unequal variances a raw Gaussian-mixture posterior can turn back
        up in the bright tail (calling very bright pixels "water") or down in
        the dark tail. Physically, darker must never mean less water-like, so
        we evaluate the posterior on a fine dB grid, clamp the tails, force it
        to be non-increasing, and interpolate.
        """
        x = np.asarray(db, dtype=np.float64)
        if self.method == "fallback_threshold":
            with np.errstate(over="ignore"):
                p = 1.0 / (1.0 + np.exp((x - self.threshold_db) / 1.0))
        else:
            lo = min(-60.0, self.water_mean - 10 * self.water_std)
            hi = max(30.0, self.land_mean + 10 * self.land_std)
            grid = np.linspace(lo, hi, 4001)
            pg = self._raw_posterior(grid)
            p_w = self._raw_posterior(np.array([self.water_mean]))[0]
            p_l = self._raw_posterior(np.array([self.land_mean]))[0]
            pg = np.where(grid <= self.water_mean, np.fmax(pg, p_w), pg)
            pg = np.where(grid >= self.land_mean, np.fmin(pg, p_l), pg)
            pg = np.minimum.accumulate(pg)
            p = np.interp(x, grid, pg)
        p = np.where(np.isfinite(x), p, np.nan)
        return p.astype(np.float32)

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def fit_water_model(pre_db: np.ndarray, post_db: np.ndarray, params: DetectionParams) -> WaterModel:
    """Learn water/land backscatter populations from the pooled pre+post pixels.

    Unsupervised: no labels are used. Pooling both dates puts them on one
    scale, so a pixel is judged "water-like" by the same rule before and after.
    """
    from sklearn.mixture import GaussianMixture

    sample = np.concatenate([pre_db[np.isfinite(pre_db)], post_db[np.isfinite(post_db)]])
    if sample.size < params.min_valid_pixels:
        raise InsufficientDataError(
            f"Only {sample.size} valid pixels in the AOI; need {params.min_valid_pixels}.")
    rng = np.random.default_rng(params.seed)
    if sample.size > params.gmm_max_samples:
        sample = rng.choice(sample, params.gmm_max_samples, replace=False)

    gmm = GaussianMixture(n_components=2, covariance_type="full", n_init=3,
                          random_state=params.seed).fit(sample.reshape(-1, 1))
    means = gmm.means_.ravel()
    stds = np.sqrt(gmm.covariances_.ravel())
    weights = gmm.weights_.ravel()
    w, l = int(np.argmin(means)), int(np.argmax(means))
    ashman = float(np.sqrt(2) * abs(means[l] - means[w]) / np.sqrt(stds[w] ** 2 + stds[l] ** 2))
    otsu = otsu_threshold(sample)

    model = WaterModel("gmm", float(means[w]), float(stds[w]), float(weights[w]),
                       float(means[l]), float(stds[l]), float(weights[l]),
                       threshold_db=float("nan"), otsu_db=otsu, ashman_d=ashman)
    grid = np.linspace(means[w], means[l], 2001)
    post = model.posterior_water(grid)
    model.threshold_db = float(grid[np.argmin(np.abs(post - 0.5))])

    if ashman < 2.0:
        model.flags.append("weak_bimodality")
    if abs(model.threshold_db - otsu) > 3.0:
        model.flags.append("threshold_methods_disagree")
    if model.water_mean > params.max_water_mean_db or model.water_weight < 0.005:
        # The "dark" population is not plausibly open water (e.g. a scene with
        # no water at all splits land into two groups). Fall back to a fixed,
        # documented threshold rather than inventing water.
        model.flags.append("implausible_water_class_used_fallback_threshold")
        model.method = "fallback_threshold"
        model.threshold_db = params.fallback_threshold_db
    return model


def remove_small_objects(mask: np.ndarray, min_pixels: int) -> tuple[np.ndarray, int]:
    """Drop 8-connected blobs smaller than min_pixels. Returns (clean mask, n removed)."""
    if min_pixels <= 1 or not mask.any():
        return mask, 0
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    small = np.nonzero(sizes < min_pixels)[0] + 1
    return mask & ~np.isin(labels, small), int(small.size)


@dataclass
class ChangeResult:
    classes: np.ndarray        # uint8 codes, see CLASS_CODES
    delta_db: np.ndarray       # post - pre (dB), NaN where invalid
    p_water_pre: np.ndarray
    p_water_post: np.ndarray
    noise_sigma_db: float      # robust spread of delta over stable land
    stats: dict


def classify_flood_change(pre_db: np.ndarray, post_db: np.ndarray, model: WaterModel,
                          params: DetectionParams, pixel_area_m2: float,
                          hand: np.ndarray | None = None) -> ChangeResult:
    valid = np.isfinite(pre_db) & np.isfinite(post_db)
    p_pre = model.posterior_water(pre_db)
    p_post = model.posterior_water(post_db)
    d = np.where(valid, post_db - pre_db, np.nan).astype(np.float32)

    wpre = valid & (p_pre >= 0.5)
    wpost = valid & (p_post >= 0.5)
    with np.errstate(invalid="ignore"):
        dec = valid & (d <= -params.delta_db)
        inc = valid & (d >= params.delta_db)
    new = wpost & ~wpre & dec
    persist = wpre & wpost
    receded = wpre & ~wpost & inc

    min_pixels = int(np.ceil(params.min_area_m2 / pixel_area_m2))
    new, n_small = remove_small_objects(new, min_pixels)

    classes = np.ones(pre_db.shape, dtype=np.uint8)
    classes[~valid] = 0
    classes[dec & ~persist] = 5
    classes[inc & ~receded] = 6
    classes[persist] = 3
    classes[receded] = 4
    classes[new] = 2

    hand_checked = hand is not None
    if hand_checked:
        high = valid & np.isfinite(hand) & (hand > params.hand_max_m) & np.isin(classes, (2, 3, 4))
        classes[high] = 7

    stable_land = valid & ~wpre & ~wpost & np.isfinite(d)
    if stable_land.sum() >= 50:
        ds = d[stable_land]
        sigma = float(1.4826 * np.median(np.abs(ds - np.median(ds))))
    else:
        sigma = float("nan")

    km2 = pixel_area_m2 / 1e6
    stats = {
        "valid_pixels": int(valid.sum()),
        "valid_area_km2": round(float(valid.sum() * km2), 3),
        "area_km2_by_class": {CLASS_CODES[c][0]: round(float((classes == c).sum() * km2), 3)
                              for c in CLASS_CODES},
        "small_new_water_objects_removed": n_small,
        "min_mapping_unit_pixels": min_pixels,
        "hand_check_applied": hand_checked,
        "noise_sigma_db": round(sigma, 3) if np.isfinite(sigma) else None,
    }
    return ChangeResult(classes, d, p_pre, p_post, sigma, stats)


def detect_forest_disturbance(pre_hv_db: np.ndarray, post_hv_db: np.ndarray,
                              p_water_post: np.ndarray, params: DetectionParams,
                              pixel_area_m2: float) -> tuple[np.ndarray, np.ndarray]:
    """EXPERIMENTAL: candidate forest disturbance from a cross-pol (HV) decrease.

    Forest canopies scatter strongly into cross-pol (volume scattering); losing
    the canopy (fire, clear-cut, landslide) lowers HV. Soil moisture, rain,
    freeze/thaw and phenology also change HV, so a single pair gives
    *candidates* only. Returns (candidate mask, delta_hv_db).
    """
    valid = np.isfinite(pre_hv_db) & np.isfinite(post_hv_db)
    d = np.where(valid, post_hv_db - pre_hv_db, np.nan).astype(np.float32)
    with np.errstate(invalid="ignore"):
        cand = (valid & (pre_hv_db >= params.forest_min_hv_db) & (d <= -params.forest_delta_db)
                & ~(p_water_post >= 0.5))  # new water is handled by the flood mode
    cand, _ = remove_small_objects(cand, int(np.ceil(params.min_area_m2 / pixel_area_m2)))
    return cand, d
