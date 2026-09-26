"""Turn several independent pieces of evidence into one 0-1 confidence score.

IMPORTANT: this is a transparent *evidence score*, not a calibrated
probability. It says "how many independent checks agree, and how strongly".
Only after comparing against a reference layer (validate.py) can you say how
often "High" detections are actually right for a given case.
"""

from __future__ import annotations

import numpy as np

from .detect import NEW_WATER, ChangeResult, DetectionParams, WaterModel

LEVELS = ((0.7, "High"), (0.4, "Medium"), (0.0, "Low"))

# Plain-language versions of the water-model flags (shown to dashboard users).
FLAG_TEXT = {
    "weak_bimodality": "water-like and land-like backscatter overlap in this scene",
    "threshold_methods_disagree": "two independent thresholding methods disagree by more than 3 dB",
    "implausible_water_class_used_fallback_threshold":
        "no clear water population was found, so a fixed fallback threshold was used",
}


def confidence_label(score: float) -> str:
    if score is None or not np.isfinite(score):
        return "Unknown"
    for cut, name in LEVELS:
        if score >= cut:
            return name
    return "Low"


def new_water_confidence(result: ChangeResult, model: WaterModel, params: DetectionParams,
                         hand: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Per-pixel confidence for 'possible flood-related change' pixels (NaN elsewhere).

    Evidence combined (geometric mean, so one weak link pulls the score down):
      class_evidence    P(water after) x P(not water before) from the mixture model
      change_evidence   how far the darkening exceeds delta_db, relative to the
                        scene's own noise level (robust sigma over stable land)
      terrain_evidence  1.0 on low ground (HAND <= 5 m) falling to 0.5 at hand_max_m;
                        0.75 if no HAND layer was supplied (unknown, so not full credit)
    Scene-level caps: weak bimodality or a fallback threshold limit the score.
    """
    mask = result.classes == NEW_WATER
    conf = np.full(result.classes.shape, np.nan, dtype=np.float32)
    notes = []
    if not mask.any():
        return conf, {"notes": ["no new-water pixels"], "scene_cap": None}

    class_ev = result.p_water_post[mask] * (1.0 - result.p_water_pre[mask])

    sigma = result.noise_sigma_db if np.isfinite(result.noise_sigma_db) else 1.5
    sigma = max(sigma, 0.5)
    excess = (-result.delta_db[mask] - params.delta_db) / sigma
    change_ev = 1.0 / (1.0 + np.exp(-excess))  # 0.5 exactly at the threshold

    if hand is not None:
        h = hand[mask]
        terrain_ev = np.clip(1.0 - 0.5 * (h - 5.0) / max(params.hand_max_m - 5.0, 1e-6), 0.5, 1.0)
        terrain_ev = np.where(np.isfinite(h), terrain_ev, 0.75)
    else:
        terrain_ev = np.full(class_ev.shape, 0.75)
        notes.append("no HAND layer: terrain plausibility not checked")

    score = np.cbrt(np.clip(class_ev, 0, 1) * np.clip(change_ev, 0, 1) * terrain_ev)

    cap = 1.0
    if "weak_bimodality" in model.flags:
        cap = min(cap, 0.6)
        notes.append(FLAG_TEXT["weak_bimodality"])
    if model.method == "fallback_threshold":
        cap = min(cap, 0.5)
        notes.append(FLAG_TEXT["implausible_water_class_used_fallback_threshold"])
    if "threshold_methods_disagree" in model.flags:
        cap = min(cap, 0.7)
        notes.append(FLAG_TEXT["threshold_methods_disagree"])
    conf[mask] = np.minimum(score, cap).astype(np.float32)
    return conf, {"notes": notes, "scene_cap": cap}


def scene_quality(model: WaterModel, result: ChangeResult, days_between: float | None,
                  masked_fraction: float) -> dict:
    """Plain-language scene-level uncertainty summary shown in the dashboard."""
    issues = [FLAG_TEXT.get(f, f) for f in model.flags]
    if days_between is not None and days_between > 24:
        issues.append(f"images are {days_between:.0f} days apart; the flood peak may be missed "
                      "and unrelated changes accumulate")
    if masked_fraction > 0.3:
        issues.append(f"{masked_fraction:.0%} of the AOI is no-data or masked")
    if result.noise_sigma_db and np.isfinite(result.noise_sigma_db) and result.noise_sigma_db > 2.0:
        issues.append(f"high background change noise ({result.noise_sigma_db:.1f} dB) over stable land")
    level = "good" if not issues else ("fair" if len(issues) <= 2 else "poor")
    return {"overall": level, "issues": issues}
