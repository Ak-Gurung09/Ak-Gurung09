"""Georeferencing: CRS, transforms, no-data, subsetting, alignment."""

import numpy as np
import pytest
import rasterio
from affine import Affine

from changelens.align import align_to_reference, same_grid
from changelens.io_readers import read_geotiff_scene, read_nisar_gcov_scene
from changelens.scene import AlignmentError, DataLoadError, SarScene

T = Affine(20.0, 0, 500_000.0, 0, -20.0, 110_000.0)  # UTM 31N, ~1 deg N 3 deg E


def write_tif(path, arr, transform=T, crs="EPSG:32631", nodata=np.nan):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs=crs, transform=transform, nodata=nodata) as dst:
        dst.write(arr.astype(np.float32), 1)


def test_geotiff_reader_keeps_crs_and_masks_nodata(tmp_path):
    arr = np.full((50, 60), 0.1, np.float32)
    arr[0, 0] = np.nan
    arr[1, 1] = 0.0          # zero power is invalid
    arr[2, 2] = -9999
    p = tmp_path / "rtc.tif"
    write_tif(p, arr, nodata=-9999)
    s = read_geotiff_scene(str(p), "VV", "2025-12-01", {"label": "NASA-derived SAR product"})
    assert s.crs.to_epsg() == 32631 and s.transform == T
    assert np.isnan(s.data[0, 0]) and np.isnan(s.data[1, 1]) and np.isnan(s.data[2, 2])
    assert s.valid.sum() == 50 * 60 - 3


def test_geotiff_bbox_subset_is_georeferenced(tmp_path):
    p = tmp_path / "rtc.tif"
    write_tif(p, np.full((100, 100), 0.1, np.float32))
    import pyproj
    to_ll = pyproj.Transformer.from_crs(32631, 4326, always_xy=True)
    w, s_ = to_ll.transform(500_400, 110_000 - 1_400)
    e, n = to_ll.transform(500_800, 110_000 - 1_000)
    sc = read_geotiff_scene(str(p), "VV", "x", {}, bbox=(w, s_, e, n))
    left, bottom, right, top = sc.bounds
    assert 500_380 <= left <= 500_420 and 108_980 <= top <= 109_020
    assert sc.shape[0] < 100 and sc.shape[1] < 100


def test_geotiff_aoi_outside_raises(tmp_path):
    p = tmp_path / "rtc.tif"
    write_tif(p, np.full((10, 10), 0.1, np.float32))
    with pytest.raises(DataLoadError):
        read_geotiff_scene(str(p), "VV", "x", {}, bbox=(100.0, 40.0, 100.1, 40.1))


def test_missing_file_raises():
    with pytest.raises(DataLoadError):
        read_geotiff_scene("does_not_exist.tif", "VV", "x", {})


def make_mock_gcov(path, nrows=40, ncols=50):
    """Mock file following the documented GCOV layout (verify against a real granule)."""
    import h5py
    x = 500_000.0 + 10.0 + 20.0 * np.arange(ncols)      # pixel centres
    y = 110_000.0 - 10.0 - 20.0 * np.arange(nrows)
    hh = np.full((nrows, ncols), 0.05, np.float32)
    hh[0, :] = np.nan
    with h5py.File(path, "w") as f:
        g = f.create_group("/science/LSAR/GCOV/grids/frequencyA")
        g["xCoordinates"] = x
        g["yCoordinates"] = y
        g["HHHH"] = hh
        g["HVHV"] = hh / 10
        proj = g.create_dataset("projection", data=np.uint32(32631))
        proj.attrs["epsg_code"] = 32631
        ident = f.create_group("/science/LSAR/identification")
        ident["zeroDopplerStartTime"] = np.bytes_("2026-07-14T01:02:03.000000000")
        ident["trackNumber"] = 77


def test_nisar_gcov_reader_uses_pixel_corner_transform(tmp_path):
    p = tmp_path / "gcov.h5"
    make_mock_gcov(p)
    s = read_nisar_gcov_scene(str(p), "HH", source={"label": "Primary NASA SAR data"})
    assert s.crs.to_epsg() == 32631
    assert s.transform.c == pytest.approx(500_000.0) and s.transform.f == pytest.approx(110_000.0)
    assert s.acquired.startswith("2026-07-14T01:02:03")
    assert np.isnan(s.data[0]).all() and s.source["covariance_term"] == "HHHH"
    assert s.source["identification"]["trackNumber"] == "77"


def test_nisar_gcov_bbox_subset(tmp_path):
    p = tmp_path / "gcov.h5"
    make_mock_gcov(p)
    s = read_nisar_gcov_scene(str(p), "HV", bbox=(500_100, 109_500, 500_300, 109_900), bbox_crs="EPSG:32631")
    assert s.shape == (20, 10)
    assert s.transform.c == pytest.approx(500_100.0)


def test_nisar_gcov_missing_polarization(tmp_path):
    p = tmp_path / "gcov.h5"
    make_mock_gcov(p)
    with pytest.raises(DataLoadError, match="available terms"):
        read_nisar_gcov_scene(str(p), "VV")


def test_alignment_resamples_to_reference_grid():
    ref = SarScene(np.full((50, 50), 0.1, np.float32), "EPSG:32631", T, "HH", "a")
    shifted = Affine(20.0, 0, 500_010.0, 0, -20.0, 110_000.0)
    other = SarScene(np.full((50, 50), 0.2, np.float32), "EPSG:32631", shifted, "HH", "b")
    assert not same_grid(ref, other)
    out = align_to_reference(ref, other)
    assert same_grid(ref, out)
    assert np.nanmean(out.data) == pytest.approx(0.2, rel=1e-4)
    assert out.source["resampled_to_reference_grid"]


def test_alignment_refuses_when_no_overlap():
    ref = SarScene(np.full((50, 50), 0.1, np.float32), "EPSG:32631", T, "HH", "a")
    far = Affine(20.0, 0, 600_000.0, 0, -20.0, 110_000.0)
    other = SarScene(np.full((50, 50), 0.2, np.float32), "EPSG:32631", far, "HH", "b")
    with pytest.raises(AlignmentError):
        align_to_reference(ref, other)
