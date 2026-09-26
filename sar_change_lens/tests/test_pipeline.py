"""End to end: outputs exist, are georeferenced, labelled, and failures are safe."""

import json
import os

import numpy as np
import pytest
import rasterio

from changelens.detect import DetectionParams
from changelens.pipeline import _fail, analyze_pair, main, run_dry_run
from changelens.scene import ChangeLensError
from changelens.synthetic import make_synthetic_case

EXPECTED = ["change_classes.tif", "confidence.tif", "delta_db.tif", "detections.geojson",
            "summary.json", "provenance.json", "report.md", "previews/classes.png",
            "previews/pre_db.png", "previews/post_db.png", "previews/confidence.png"]


def test_dry_run_writes_all_outputs(tmp_path):
    s = run_dry_run(str(tmp_path))
    d = tmp_path / "dryrun_synthetic_flood"
    for name in EXPECTED:
        assert (d / name).exists(), name
    assert s["status"] == "ok" and s["detections"] >= 1
    assert "SYNTHETIC" in s["mode"] or "synthetic" in s["mode"]
    prov = json.loads((d / "provenance.json").read_text())
    assert all(ds["label"].startswith("SYNTHETIC") for ds in prov["datasets"])
    assert len(prov["outputs"]) >= 8 and all(len(o["sha256"]) == 64 for o in prov["outputs"])


def test_geojson_is_lon_lat_inside_map_bounds(tmp_path):
    s = run_dry_run(str(tmp_path))
    gj = json.loads((tmp_path / "dryrun_synthetic_flood" / "detections.geojson").read_text())
    (south, west), (north, east) = s["map_bounds_latlon"]
    assert "Not an official" in gj["disclaimer"]
    for f in gj["features"]:
        ring = f["geometry"]["coordinates"][0]
        lons = [c[0] for c in ring]
        lats = [c[1] for c in ring]
        assert west - 1e-6 <= min(lons) and max(lons) <= east + 1e-6
        assert south - 1e-6 <= min(lats) and max(lats) <= north + 1e-6
        assert "not a confirmed flood" in f["properties"]["explanation"]


def test_output_rasters_keep_crs_and_grid(tmp_path):
    run_dry_run(str(tmp_path))
    pre, _, _, _ = make_synthetic_case()
    with rasterio.open(tmp_path / "dryrun_synthetic_flood" / "change_classes.tif") as ds:
        assert ds.crs.to_epsg() == pre.crs.to_epsg()
        assert ds.transform == pre.transform
        assert ds.nodata == 0
        assert "new_water" in ds.tags(1)["class_2"]


def test_negative_dry_run(tmp_path):
    s = run_dry_run(str(tmp_path), flood=False)
    assert s["detections"] == 0


def test_forest_mode_is_experimental_and_low_confidence(tmp_path):
    pre, post, hand, _ = make_synthetic_case(seed=9)
    an = analyze_pair(pre, post, DetectionParams(), hand=hand, hazard="forest_disturbance")
    for f in an.features:
        assert "experimental" in f["properties"]["label"]
        assert f["properties"]["confidence_label"] == "Low"


def test_unsupported_hazard_is_refused():
    pre, post, hand, _ = make_synthetic_case()
    with pytest.raises(ChangeLensError):
        analyze_pair(pre, post, DetectionParams(), hand=hand, hazard="landslide")


def test_failure_removes_stale_maps(tmp_path):
    run_dry_run(str(tmp_path))
    s = _fail(str(tmp_path), "dryrun_synthetic_flood", ChangeLensError("granule missing"))
    d = tmp_path / "dryrun_synthetic_flood"
    assert s["status"] == "failed"
    assert not (d / "detections.geojson").exists()
    assert json.loads((d / "summary.json").read_text())["status"] == "failed"


def test_cli_refuses_template_case(tmp_path):
    p = tmp_path / "case.yaml"
    p.write_text("case_id: x\nstatus: template\n")
    assert main(["--case", str(p), "--out", str(tmp_path)]) == 2


def test_cli_real_case_with_missing_file_fails_safely(tmp_path):
    p = tmp_path / "case.yaml"
    p.write_text(
        "case_id: broken\nstatus: ready\n"
        "pre: {format: geotiff, path: nope_pre.tif, polarization: VV, acquired: '2025-12-01', "
        "source: {label: NASA-derived SAR product}}\n"
        "post: {format: geotiff, path: nope_post.tif, polarization: VV, acquired: '2025-12-13', "
        "source: {label: NASA-derived SAR product}}\n")
    assert main(["--case", str(p), "--out", str(tmp_path)]) == 2
    s = json.loads((tmp_path / "broken" / "summary.json").read_text())
    assert s["status"] == "failed" and "not found" in s["error"]


def test_cli_malformed_case_fails_safely(tmp_path):
    p = tmp_path / "case.yaml"
    p.write_text("case_id: malformed\nstatus: ready\npre: {format: geotiff}\n")  # missing keys
    assert main(["--case", str(p), "--out", str(tmp_path)]) == 2
    s = json.loads((tmp_path / "malformed" / "summary.json").read_text())
    assert s["status"] == "failed"
