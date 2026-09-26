"""Compare our detections with an independent reference layer, honestly.

The usual reference is NASA JPL OPERA DSWx-S1 (surface water from Sentinel-1,
C-band) for the dates closest to our pre- and post-event images. It is a
validated *product*, not ground truth: it has its own errors, a different
wavelength, and usually a different acquisition time. Report "agreement with
DSWx-S1", never "accuracy".
"""

from __future__ import annotations

import numpy as np

# OPERA DSWx-S1 WTR (B01) layer codes, from the DSWx-S1 Product Specification
# (JPL D-108761). Re-check against the spec version of the granules you use.
DSWX_S1_WTR = {
    0: "not water",
    1: "open water",
    3: "inundated vegetation",
    250: "HAND masked",
    251: "layover/shadow masked",
    255: "fill / no data",
}
DSWX_S1_INVALID = (250, 251, 255)


def dswx_water(wtr: np.ndarray, include_inundated_vegetation: bool = False):
    """Return (water mask, valid mask) from a DSWx-S1 WTR array."""
    water_codes = (1, 3) if include_inundated_vegetation else (1,)
    valid = np.isin(wtr, (0, 1, 3))
    return np.isin(wtr, water_codes) & valid, valid


def reference_new_water(pre_wtr: np.ndarray, post_wtr: np.ndarray, include_inundated_vegetation=False):
    """Reference 'new water' = water in the post-event DSWx-S1 but not in the pre-event one."""
    pre_w, pre_v = dswx_water(pre_wtr, include_inundated_vegetation)
    post_w, post_v = dswx_water(post_wtr, include_inundated_vegetation)
    valid = pre_v & post_v
    return post_w & ~pre_w & valid, valid


def binary_agreement(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray) -> dict:
    """Confusion counts and scores over pixels valid in BOTH maps.

    Overall accuracy is deliberately not reported: when 98 % of pixels are
    "no change", a detector that finds nothing scores 98 %.
    """
    p = pred & valid
    r = ref & valid
    tp = int((p & r).sum())
    fp = int((p & ~r).sum())
    fn = int((~p & r & valid).sum())
    tn = int((~p & ~r & valid).sum())

    def ratio(a, b):
        return round(a / b, 3) if b else None

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = (round(2 * precision * recall / (precision + recall), 3)
          if precision is not None and recall is not None and (precision + recall) > 0 else None)
    return {
        "pixels_compared": int(valid.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,   # of what we flagged, share the reference also flags
        "recall": recall,         # of what the reference flags, share we also flagged
        "f1": f1,
        "iou": ratio(tp, tp + fp + fn),
        "reference_positive_share": ratio(tp + fn, int(valid.sum())),
    }


def agreement_by_confidence(confidence: np.ndarray, pred: np.ndarray, ref: np.ndarray,
                            valid: np.ndarray, cuts=(0.0, 0.4, 0.7, 1.01)) -> list[dict]:
    """Share of flagged pixels the reference agrees with, per confidence band.

    This is how a confidence score earns trust: if 'High' pixels agree with the
    reference much more often than 'Low' ones, the score is informative for
    this case. With few pixels in a band, the share is shown but flagged.
    """
    rows = []
    names = ("Low", "Medium", "High")
    for name, lo, hi in zip(names, cuts[:-1], cuts[1:]):
        band = pred & valid & np.isfinite(confidence) & (confidence >= lo) & (confidence < hi)
        n = int(band.sum())
        agree = int((band & ref).sum())
        rows.append({"confidence": name, "flagged_pixels": n,
                     "reference_agrees": round(agree / n, 3) if n else None,
                     "too_few_pixels": n < 100})
    return rows
