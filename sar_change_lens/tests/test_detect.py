"""Science core: does the detector find injected change and ignore non-change?"""

import numpy as np
import pytest
from scipy.ndimage import binary_erosion

from changelens.detect import (NEW_WATER, DetectionParams, WaterModel, classify_flood_change,
                               fit_water_model, otsu_threshold, remove_small_objects, to_db)
from changelens.pipeline import analyze_pair
from changelens.scene import InsufficientDataError
from changelens.synthetic import make_synthetic_case


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def test_to_db_handles_invalid_values():
    out = to_db(np.array([1.0, 0.1, 0.0, -1.0, np.nan]))
    assert out[0] == pytest.approx(0.0) and out[1] == pytest.approx(-10.0)
    assert np.isnan(out[2:]).all()


def test_otsu_splits_two_populations():
    rng = np.random.default_rng(1)
    v = np.concatenate([rng.normal(-22, 1.5, 5000), rng.normal(-8, 1.5, 20000)])
    assert -18 < otsu_threshold(v) < -12


def test_posterior_is_monotonic_darker_means_more_water():
    rng = np.random.default_rng(2)
    a = np.concatenate([rng.normal(-22, 3.0, 3000), rng.normal(-8, 1.0, 20000)])
    m = fit_water_model(a, a, DetectionParams())
    p = m.posterior_water(np.linspace(-60, 30, 900))
    assert np.all(np.diff(p) <= 1e-6)
    assert p[0] > 0.99 and p[-1] < 0.01  # very dark = water-like, very bright = land-like
    assert m.method == "gmm" and m.ashman_d > 2


def test_injected_flood_is_recovered_and_trap_is_masked():
    pre, post, hand, truth = make_synthetic_case(flood=True, seed=3)
    an = analyze_pair(pre, post, DetectionParams(), hand=hand)
    found = an.result.classes == NEW_WATER
    assert iou(found, truth["new_water"]) > 0.9
    # The dark patch on the high-HAND hill must not be reported as flooding.
    assert not (found & truth["false_positive_trap"]).any()
    assert (an.result.classes[truth["false_positive_trap"]] == 7).mean() > 0.9
    # The river core is permanent water, not new water. (Its edges are blurred by
    # the speckle filter in the pre image, so they may legitimately widen.)
    core = binary_erosion(truth["river"], iterations=2)
    assert (an.result.classes[core] == NEW_WATER).mean() < 0.02
    lowland = core.copy()
    lowland[:, 180:] = False  # the synthetic hill (HAND = 60 m) starts at column 180
    assert (an.result.classes[lowland] == 3).mean() > 0.9
    assert (an.result.classes[core & ~lowland] == 7).mean() > 0.9  # masked by HAND, by design
    assert all(f["properties"]["confidence_label"] in ("High", "Medium") for f in an.features)


def test_unchanged_pair_gives_no_detections():
    pre, post, hand, _ = make_synthetic_case(flood=False, seed=4)
    an = analyze_pair(pre, post, DetectionParams(), hand=hand)
    assert an.result.stats["area_km2_by_class"]["new_water"] < 0.01
    assert an.features == []


def test_without_hand_the_trap_is_flagged_but_confidence_is_reduced():
    pre, post, _, truth = make_synthetic_case(flood=True, seed=5)
    an = analyze_pair(pre, post, DetectionParams(), hand=None)
    assert (an.result.classes[truth["false_positive_trap"]] == NEW_WATER).mean() > 0.9
    assert np.nanmax(an.confidence) <= 0.75 ** (1 / 3) + 1e-6
    assert "no HAND layer" in " ".join(an.confidence_info["notes"])


def test_nodata_stays_nodata():
    pre, post, hand, _ = make_synthetic_case(seed=6)
    an = analyze_pair(pre, post, DetectionParams(), hand=hand)
    assert (an.result.classes[:, :6] == 0).all()


def test_scene_without_water_uses_fallback_threshold_not_invented_water():
    rng = np.random.default_rng(7)
    land = rng.normal(-8, 1.0, (200, 200)).astype(np.float32)
    land2 = land + rng.normal(0, 0.5, land.shape).astype(np.float32)
    model = fit_water_model(land, land2, DetectionParams())
    assert model.method == "fallback_threshold"
    res = classify_flood_change(land, land2, model, DetectionParams(), pixel_area_m2=400.0)
    assert (res.classes == NEW_WATER).sum() == 0


def test_too_few_pixels_raises():
    tiny = np.full((10, 10), -10.0, dtype=np.float32)
    with pytest.raises(InsufficientDataError):
        fit_water_model(tiny, tiny, DetectionParams())


def test_remove_small_objects():
    m = np.zeros((20, 20), bool)
    m[1:3, 1:3] = True      # 4 px
    m[10:18, 10:18] = True  # 64 px
    clean, n = remove_small_objects(m, 10)
    assert n == 1 and clean.sum() == 64
