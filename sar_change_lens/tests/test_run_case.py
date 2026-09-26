"""Exercise the real-data code path (run_case) with GeoTIFF inputs on mismatched grids.

Inputs are the SYNTHETIC scenes written to disk, so this checks file handling,
reprojection of HAND and a 30 m DSWx-style reference onto the 20 m analysis
grid, and the validation block - not scientific accuracy.
"""

import json

import numpy as np
import rasterio
from affine import Affine

from changelens.pipeline import run_case
from changelens.synthetic import SYNTHETIC_SOURCE, make_synthetic_case


def write(path, arr, transform, crs, dtype, nodata):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=dtype, crs=crs, transform=transform, nodata=nodata) as dst:
        dst.write(arr.astype(dtype), 1)


def test_run_case_with_geotiffs_hand_and_reference(tmp_path):
    pre, post, hand, truth = make_synthetic_case(seed=11)
    crs = pre.crs
    write(tmp_path / "pre.tif", pre.data, pre.transform, crs, "float32", np.nan)
    write(tmp_path / "post.tif", post.data, post.transform, crs, "float32", np.nan)
    write(tmp_path / "hand.tif", hand, pre.transform, crs, "float32", -9999)

    # Reference on a coarser 30 m grid, DSWx-S1 WTR codes: 0 not water, 1 open water, 255 fill.
    t30 = Affine(30.0, 0, pre.transform.c, 0, -30.0, pre.transform.f)
    rows, cols = 160, 160
    rr, cc = np.mgrid[0:rows, 0:cols]
    # map 30 m pixel centres back to the 20 m truth grid
    r20 = np.clip(((rr + 0.5) * 30 / 20).astype(int), 0, 239)
    c20 = np.clip(((cc + 0.5) * 30 / 20).astype(int), 0, 239)
    river = truth["river"][r20, c20]
    flood = truth["new_water"][r20, c20]
    pre_wtr = np.where(river, 1, 0).astype(np.uint8)
    post_wtr = np.where(river | flood, 1, 0).astype(np.uint8)
    write(tmp_path / "pre_wtr.tif", pre_wtr, t30, crs, "uint8", 255)
    write(tmp_path / "post_wtr.tif", post_wtr, t30, crs, "uint8", 255)

    src = dict(SYNTHETIC_SOURCE)
    cfg = {
        "case_id": "geotiff_case", "title": "SYNTHETIC GeoTIFF case", "status": "ready", "hazard": "flood",
        "pre": {"format": "geotiff", "path": str(tmp_path / "pre.tif"), "polarization": "HH",
                "acquired": "2026-07-01T00:00:00", "source": src},
        "post": {"format": "geotiff", "path": str(tmp_path / "post.tif"), "polarization": "HH",
                 "acquired": "2026-07-13T00:00:00", "source": src},
        "ancillary": {"hand": {"path": str(tmp_path / "hand.tif"), "source": src}},
        "reference": {"type": "dswx_s1", "pre_wtr": str(tmp_path / "pre_wtr.tif"),
                      "post_wtr": str(tmp_path / "post_wtr.tif"), "pre_source": src, "post_source": src},
    }
    s = run_case(cfg, str(tmp_path / "out"))
    assert s["status"] in ("ok", "degraded") and s["detections"] >= 1
    m = s["validation"]["metrics"]
    assert m["precision"] > 0.85 and m["recall"] > 0.85
    assert "not accuracy" in s["validation"]["note"]
    assert [r["confidence"] for r in s["validation"]["by_confidence"]] == ["Low", "Medium", "High"]
    prov = json.loads((tmp_path / "out" / "geotiff_case" / "provenance.json").read_text())
    assert {d["role"] for d in prov["datasets"]} >= {"pre-event", "post-event", "terrain (HAND)", "reference (pre)"}
