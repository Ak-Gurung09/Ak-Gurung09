import numpy as np
import pytest

from changelens.speckle import estimate_enl, lee_filter
from changelens.validate import agreement_by_confidence, binary_agreement, dswx_water, reference_new_water


def test_binary_agreement_counts():
    pred = np.array([1, 1, 0, 0, 1], bool)
    ref = np.array([1, 0, 1, 0, 1], bool)
    valid = np.array([1, 1, 1, 1, 0], bool)
    m = binary_agreement(pred, ref, valid)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["iou"] == pytest.approx(0.333, abs=1e-3)
    assert "accuracy" not in m


def test_dswx_codes():
    wtr = np.array([0, 1, 3, 250, 251, 255], np.uint8)
    water, valid = dswx_water(wtr)
    assert water.tolist() == [False, True, False, False, False, False]
    assert valid.tolist() == [True, True, True, False, False, False]
    water_iv, _ = dswx_water(wtr, include_inundated_vegetation=True)
    assert water_iv[2]
    new, v = reference_new_water(np.array([0, 1, 0, 251]), np.array([1, 1, 0, 1]))
    assert new.tolist() == [True, False, False, False] and v.tolist() == [True, True, True, False]


def test_lee_filter_reduces_speckle_and_keeps_nodata():
    rng = np.random.default_rng(0)
    x = (0.1 * rng.gamma(4, 1 / 4, (100, 100))).astype(np.float32)
    x[:, :5] = np.nan
    y = lee_filter(x, size=7, enl=4)
    assert np.isnan(y[:, :5]).all()
    assert np.nanstd(y) < 0.6 * np.nanstd(x)
    assert np.nanmean(y) == pytest.approx(np.nanmean(x), rel=0.05)


def test_estimate_enl_on_homogeneous_field():
    rng = np.random.default_rng(1)
    x = rng.gamma(6, 1 / 6, (200, 200)).astype(np.float32)
    assert 4.0 < estimate_enl(x) < 7.0


def test_agreement_by_confidence_bands():
    conf = np.array([0.2, 0.2, 0.5, 0.9, 0.9, np.nan])
    pred = np.array([1, 1, 1, 1, 1, 0], bool)
    ref = np.array([0, 1, 1, 1, 1, 0], bool)
    rows = {r["confidence"]: r for r in agreement_by_confidence(conf, pred, ref, np.ones(6, bool))}
    assert rows["Low"]["reference_agrees"] == 0.5 and rows["High"]["reference_agrees"] == 1.0
    assert rows["High"]["flagged_pixels"] == 2 and rows["High"]["too_few_pixels"]
