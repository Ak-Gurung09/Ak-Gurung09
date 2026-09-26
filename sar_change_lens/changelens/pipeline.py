"""End-to-end pipeline: case config -> maps, detections, explanations, provenance.

    python -m changelens.pipeline --dry-run                 # synthetic, no downloads
    python -m changelens.pipeline --dry-run --no-change     # synthetic negative case
    python -m changelens.pipeline --case config/cases/my_case.yaml

Outputs land in outputs/<case_id>/ and are what the dashboard reads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import rasterio

from . import render
from .align import align_to_reference, reproject_to_grid
from .confidence import new_water_confidence, scene_quality
from .detect import (CLASS_CODES, NEW_WATER, ChangeResult, DetectionParams, WaterModel,
                     classify_flood_change, detect_forest_disturbance, fit_water_model, to_db)
from .explain import evidence_for, llm_explanation, template_explanation
from .io_readers import read_categorical, read_geotiff_scene, read_nisar_gcov_scene
from .provenance import Provenance
from .scene import AlignmentError, ChangeLensError, SarScene
from .speckle import lee_filter
from .synthetic import make_synthetic_case
from .validate import agreement_by_confidence, binary_agreement, reference_new_water
from .vectorize import DISCLAIMER, change_objects, write_geojson

HAZARD_LABELS = {
    "flood": "Possible flood-related change",
    "forest_disturbance": "Candidate forest disturbance (experimental)",
}


@dataclass
class Analysis:
    pre_db: np.ndarray
    post_db: np.ndarray
    model: WaterModel
    result: ChangeResult
    confidence: np.ndarray
    confidence_info: dict
    features: list
    quality: dict
    hazard: str
    detection_mask: np.ndarray


def _date(iso: str) -> str:
    return (iso or "unknown")[:10]


def _days_between(a: str, b: str) -> float | None:
    try:
        return abs((datetime.fromisoformat(b[:19]) - datetime.fromisoformat(a[:19])).total_seconds()) / 86400
    except (TypeError, ValueError):
        return None


def pixel_area_m2(scene: SarScene) -> float:
    if not scene.crs.is_projected:
        raise AlignmentError("Scenes must be on a projected grid in metres (e.g. UTM) so areas are real.")
    w, h = scene.pixel_size
    return w * h


def analyze_pair(pre: SarScene, post: SarScene, params: DetectionParams, hand: np.ndarray | None = None,
                 hazard: str = "flood") -> Analysis:
    """Core science on two already-loaded scenes. No file I/O, so it is easy to test."""
    post = align_to_reference(pre, post)
    area = pixel_area_m2(pre)
    pre_db = to_db(lee_filter(pre.data, params.speckle_size, params.enl))
    post_db = to_db(lee_filter(post.data, params.speckle_size, params.enl))
    both = np.isfinite(pre_db) & np.isfinite(post_db)
    pre_db[~both] = np.nan
    post_db[~both] = np.nan

    model = fit_water_model(pre_db, post_db, params)
    result = classify_flood_change(pre_db, post_db, model, params, area, hand=hand)

    layers = {"delta_db": result.delta_db, "pre_db": pre_db, "post_db": post_db}
    if hand is not None:
        layers["hand_m"] = hand
    if hazard == "flood":
        mask = result.classes == NEW_WATER
        conf, info = new_water_confidence(result, model, params, hand)
    elif hazard == "forest_disturbance":
        mask, _ = detect_forest_disturbance(pre_db, post_db, result.p_water_post, params, area)
        # Experimental mode: one flat, explicitly low confidence rather than a score
        # we cannot justify without a disturbance-specific evidence model.
        conf = np.where(mask, 0.35, np.nan).astype(np.float32)
        info = {"notes": ["experimental: fixed low confidence; single-pair cross-pol decrease only"],
                "scene_cap": 0.35}
    else:
        raise ChangeLensError(f"Hazard {hazard!r} is not supported by this MVP (see docs, section 18).")

    features = change_objects(mask, pre.transform, pre.crs, area, layers, conf, hazard, HAZARD_LABELS[hazard])
    masked_fraction = 1.0 - both.sum() / both.size
    quality = scene_quality(model, result, _days_between(pre.acquired, post.acquired), masked_fraction)
    return Analysis(pre_db, post_db, model, result, conf, info, features, quality, hazard, mask)


def _write_tif(path: str, arr: np.ndarray, ref: SarScene, nodata, descriptions: dict | None = None) -> None:
    profile = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=arr.dtype,
                   crs=ref.crs, transform=ref.transform, nodata=nodata, compress="deflate",
                   tiled=arr.shape[0] >= 256 and arr.shape[1] >= 256)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
        if descriptions:
            dst.update_tags(1, **{f"class_{k}": f"{v[0]}: {v[1]}" for k, v in descriptions.items()})


def write_outputs(an: Analysis, pre: SarScene, post: SarScene, params: DetectionParams, out_dir: str,
                  case: dict, prov: Provenance, validation: dict | None = None, use_llm: bool = False) -> dict:
    os.makedirs(os.path.join(out_dir, "previews"), exist_ok=True)
    files = []

    def out(name):
        p = os.path.join(out_dir, name)
        files.append(p)
        return p

    _write_tif(out("change_classes.tif"), an.result.classes, pre, 0, CLASS_CODES)
    _write_tif(out("confidence.tif"), an.confidence.astype(np.float32), pre, np.nan)
    _write_tif(out("delta_db.tif"), an.result.delta_db.astype(np.float32), pre, np.nan)

    bounds = None
    for name, arr, kind in (("pre_db", an.pre_db, "db"), ("post_db", an.post_db, "db"),
                            ("classes", an.result.classes, "classes"),
                            ("confidence", an.confidence, "confidence"),
                            ("delta_db", an.result.delta_db, "delta")):
        merc, bounds = render.to_web_mercator(arr, pre.crs, pre.transform, categorical=(kind == "classes"))
        if kind == "db":
            rgba = render.db_to_rgba(merc, *np.nanpercentile(an.pre_db, [2, 98]))
        elif kind == "classes":
            rgba = render.classes_to_rgba(merc)
        elif kind == "confidence":
            rgba = render.confidence_to_rgba(merc)
        else:
            rgba = render.delta_to_rgba(merc)
        render.save_png(rgba, out(f"previews/{name}.png"))

    scene_info = {
        "pre_date": _date(pre.acquired), "post_date": _date(post.acquired),
        "polarization": pre.polarization, "dataset": pre.source.get("dataset"),
        "threshold_db": round(an.model.threshold_db, 2),
        "delta_db_required": params.forest_delta_db if an.hazard == "forest_disturbance" else params.delta_db,
        "hand_max_m": params.hand_max_m,
        "scene_issues": an.quality["issues"],
    }
    explanations = {}
    for f in an.features:
        ev = evidence_for(f["properties"], scene_info)
        if use_llm:
            text, info = llm_explanation(ev)
        else:
            text, info = template_explanation(ev), {"source": "template"}
        f["properties"]["explanation"] = text
        explanations[f["properties"]["id"]] = {"text": text, **info}

    write_geojson(an.features, out("detections.geojson"),
                  {"case_id": case["case_id"], "hazard": an.hazard, "mode": prov.record["mode"]})

    summary = {
        "status": "ok" if an.quality["overall"] != "poor" else "degraded",
        "case_id": case["case_id"],
        "title": case.get("title", case["case_id"]),
        "mode": prov.record["mode"],
        "hazard": an.hazard,
        "hazard_label": HAZARD_LABELS[an.hazard],
        "disclaimer": DISCLAIMER,
        "aoi_bbox_wgs84": case.get("aoi", {}).get("bbox"),
        "map_bounds_latlon": bounds,
        "scene": scene_info,
        "datasets": prov.record["datasets"],
        "water_model": an.model.to_dict(),
        "parameters": asdict(params),
        "quality": an.quality,
        "confidence_notes": an.confidence_info,
        "stats": an.result.stats,
        "detections": len(an.features),
        "detected_area_km2": round(sum(f["properties"]["area_km2"] for f in an.features), 3),
        "validation": validation,
        "explanations": explanations,
        "legend": {k: {"color": v[0], "label": v[2]} for k, v in render.MAP_CLASSES.items()},
    }
    with open(out("summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    _write_report(summary, an.features, out("report.md"))

    for p in files:
        prov.add_output(p)
    prov.write(os.path.join(out_dir, "provenance.json"))
    return summary


def _write_report(summary: dict, features: list, path: str) -> None:
    s = summary
    lines = [f"# {s['title']}", "", f"> {s['disclaimer']}", "",
             f"*Mode:* {s['mode']}  ", f"*Hazard view:* {s['hazard_label']}  ",
             f"*Images:* {s['scene']['pre_date']} -> {s['scene']['post_date']} "
             f"({s['scene']['polarization']})  ", f"*Scene quality:* {s['quality']['overall']}", ""]
    lines += ["## Data sources", "", "| Role | Dataset | Granule | Label |", "|---|---|---|---|"]
    for d in s["datasets"]:
        lines.append(f"| {d.get('role')} | {d.get('dataset')} | {d.get('granule_id', '')} | {d.get('label')} |")
    lines += ["", "## Detections", "",
              f"{s['detections']} objects, {s['detected_area_km2']} km2 in total.", "",
              "| ID | Area (km2) | Confidence | Mean change (dB) | Lat, Lon |", "|---|---|---|---|---|"]
    for f in features:
        p = f["properties"]
        lines.append(f"| {p['id']} | {p['area_km2']} | {p.get('confidence_label')} ({p.get('confidence')}) | "
                     f"{p.get('mean_delta_db')} | {p['centroid_lat']}, {p['centroid_lon']} |")
    if s["quality"]["issues"]:
        lines += ["", "## Scene caveats", ""] + [f"- {i}" for i in s["quality"]["issues"]]
    if s.get("validation"):
        v = s["validation"]
        lines += ["", "## Agreement with reference", "", f"Reference: {v.get('reference')}", "",
                  "| Metric | Value |", "|---|---|"]
        lines += [f"| {k} | {v['metrics'][k]} |" for k in ("precision", "recall", "f1", "iou", "pixels_compared")]
    lines += ["", "## Method", "",
              "Lee speckle filter -> dB -> unsupervised two-population (Gaussian mixture) model of water-like vs "
              "land backscatter -> new-water rule (water-like after, not before, darker by at least "
              f"{s['parameters']['delta_db']} dB) -> terrain (HAND) mask -> minimum mapping unit -> "
              "evidence-based confidence. Full parameters and file checksums are in provenance.json."]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- entry points

def run_dry_run(out_root: str, flood: bool = True, seed: int = 0) -> dict:
    case_id = "dryrun_synthetic_flood" if flood else "dryrun_synthetic_no_change"
    pre, post, hand, _ = make_synthetic_case(flood=flood, seed=seed)
    params = DetectionParams()
    prov = Provenance(case_id, "dry-run (synthetic)")
    prov.add_dataset("pre-event", pre.source)
    prov.add_dataset("post-event", post.source)
    prov.add_step("synthetic_generation", seed=seed, flood=flood)
    an = analyze_pair(pre, post, params, hand=hand)
    prov.add_step("analyze_pair", **asdict(params))
    case = {"case_id": case_id, "title": "SYNTHETIC dry run (not real data)" + ("" if flood else ": no change")}
    return write_outputs(an, pre, post, params, os.path.join(out_root, case_id), case, prov)


def _load_scene(spec: dict, bbox) -> SarScene:
    fmt = spec.get("format")
    if fmt == "nisar_gcov":
        return read_nisar_gcov_scene(spec["path"], spec.get("polarization", "HH"), spec.get("frequency", "A"),
                                     source=spec.get("source"), bbox=bbox)
    if fmt == "geotiff":
        return read_geotiff_scene(spec["path"], spec["polarization"], spec["acquired"], spec.get("source"),
                                  bbox=bbox, invalid_mask_path=spec.get("invalid_mask_path"),
                                  invalid_mask_values=spec.get("invalid_mask_values", ()))
    raise ChangeLensError(f"Unknown scene format {fmt!r}; use 'nisar_gcov' or 'geotiff'.")


def _load_to_grid(path: str, ref: SarScene, bbox, categorical: bool) -> np.ndarray:
    arr, crs, transform, nodata = read_categorical(path, bbox=bbox)
    if not categorical:
        arr = arr.astype(np.float32)
        if nodata is not None:
            arr[arr == nodata] = np.nan
    return reproject_to_grid(arr, crs, transform, ref, categorical=categorical,
                             src_nodata=nodata if categorical else None)


def run_case(cfg: dict, out_root: str, use_llm: bool = False) -> dict:
    case_id = cfg["case_id"]
    bbox = cfg.get("aoi", {}).get("bbox")
    params = DetectionParams(**cfg.get("params", {}))
    hazard = cfg.get("hazard", "flood")
    prov = Provenance(case_id, "real-data")

    pre = _load_scene(cfg["pre"], bbox)
    post = _load_scene(cfg["post"], bbox)
    prov.add_dataset("pre-event", pre.source)
    prov.add_dataset("post-event", post.source)

    hand = None
    if cfg.get("ancillary", {}).get("hand"):
        spec = cfg["ancillary"]["hand"]
        hand = _load_to_grid(spec["path"], pre, bbox, categorical=False)
        prov.add_dataset("terrain (HAND)", spec["source"])

    an = analyze_pair(pre, post, params, hand=hand, hazard=hazard)
    prov.add_step("analyze_pair", hazard=hazard, **asdict(params))

    validation = None
    ref = cfg.get("reference")
    if ref and ref.get("type") == "dswx_s1" and hazard == "flood":
        pre_wtr = _load_to_grid(ref["pre_wtr"], pre, bbox, categorical=True)
        post_wtr = _load_to_grid(ref["post_wtr"], pre, bbox, categorical=True)
        incl = bool(ref.get("include_inundated_vegetation", False))
        ref_new, ref_valid = reference_new_water(pre_wtr, post_wtr, incl)
        valid = ref_valid & np.isfinite(an.result.delta_db) & (an.result.classes != 7)
        validation = {"reference": ref.get("description", "OPERA DSWx-S1 new water"),
                      "include_inundated_vegetation": incl,
                      "metrics": binary_agreement(an.detection_mask, ref_new, valid),
                      "by_confidence": agreement_by_confidence(an.confidence, an.detection_mask, ref_new, valid),
                      "note": "Agreement with another product, not accuracy against ground truth."}
        for key in ("pre_source", "post_source"):
            if ref.get(key):
                prov.add_dataset(f"reference ({key[:-7]})", ref[key])
        prov.add_step("validate_against_dswx_s1", include_inundated_vegetation=incl)

    return write_outputs(an, pre, post, params, os.path.join(out_root, case_id), cfg, prov,
                         validation=validation, use_llm=use_llm)


def _fail(out_root: str, case_id: str, exc: Exception) -> dict:
    """Record a failure without writing any map, so nothing fake is ever displayed."""
    summary = {"status": "failed", "case_id": case_id, "error": f"{exc.__class__.__name__}: {exc}"}
    d = os.path.join(out_root, case_id)
    os.makedirs(d, exist_ok=True)
    for stale in ("detections.geojson", "summary.json"):
        if os.path.exists(os.path.join(d, stale)):
            os.remove(os.path.join(d, stale))
    with open(os.path.join(d, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", help="YAML case file (see config/cases/)")
    ap.add_argument("--dry-run", action="store_true", help="run on SYNTHETIC data, no downloads")
    ap.add_argument("--no-change", action="store_true", help="with --dry-run: synthetic pair with no flood")
    ap.add_argument("--out", default="outputs", help="output root folder")
    ap.add_argument("--llm", action="store_true", help="optional Claude rewrite of explanations (grounded)")
    args = ap.parse_args(argv)

    if args.dry_run:
        s = run_dry_run(args.out, flood=not args.no_change)
    elif args.case:
        import yaml
        with open(args.case) as fh:
            cfg = yaml.safe_load(fh)
        if cfg.get("status") == "template":
            print("This case file is a template: fill in real granule paths and set status: ready.")
            return 2
        try:
            s = run_case(cfg, args.out, use_llm=args.llm)
        except (ChangeLensError, KeyError, TypeError, ValueError, OSError) as exc:
            # Bad data *or* a malformed case file: record the failure, show no map.
            s = _fail(args.out, cfg.get("case_id", "unknown_case"), exc)
            print(f"FAILED safely: {s['error']}", file=sys.stderr)
            return 2
    else:
        ap.print_help()
        return 1
    print(f"[{s['status']}] {s['case_id']}: {s['detections']} detections, "
          f"{s['detected_area_km2']} km2, scene quality {s['quality']['overall']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
