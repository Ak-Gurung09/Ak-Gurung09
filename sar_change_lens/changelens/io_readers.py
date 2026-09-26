"""Readers that turn NASA SAR products into SarScene objects.

Two input pathways share one output type, so everything downstream is identical:

* NISAR L2 GCOV (HDF5)       -> read_nisar_gcov_scene   (primary NASA SAR data)
* OPERA RTC-S1 / any GeoTIFF -> read_geotiff_scene      (NASA-derived product / fallback)

Both return *linear power* backscatter with NaN for invalid pixels.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np
import rasterio
import rasterio.errors
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from .scene import DataLoadError, SarScene

# Layout of a NISAR L2 GCOV granule, from the NISAR Data User Guide
# (https://nisar-docs.asf.alaska.edu/gcov/). Verify against a real granule on
# day one: `h5ls -r file.h5 | grep frequencyA` or open it in Panoply/HDFView.
NISAR_GCOV_GRID = "/science/LSAR/GCOV/grids/frequency{freq}"
NISAR_IDENTIFICATION = "/science/LSAR/identification"


def _bbox_to_crs(bbox: Sequence[float], bbox_crs: str | CRS, dst_crs: CRS) -> tuple:
    """Reproject a (minx, miny, maxx, maxy) box. Edges are densified by rasterio."""
    return transform_bounds(CRS.from_user_input(bbox_crs), dst_crs, *bbox, densify_pts=21)


def _clean_power(arr: np.ndarray, fill_values: Iterable[float] = ()) -> np.ndarray:
    """float32 copy with fill values, non-finite and non-positive power set to NaN."""
    out = np.array(arr, dtype=np.float32, copy=True)
    for fv in fill_values:
        if fv is not None and np.isfinite(fv):
            out[out == np.float32(fv)] = np.nan
    out[~np.isfinite(out) | (out <= 0)] = np.nan
    return out


def read_geotiff_scene(
    path: str,
    polarization: str,
    acquired: str,
    source: dict | None = None,
    bbox: Sequence[float] | None = None,
    bbox_crs: str = "EPSG:4326",
    invalid_mask_path: str | None = None,
    invalid_mask_values: Sequence[int] = (),
) -> SarScene:
    """Read one band of linear-power backscatter from a GeoTIFF / COG.

    Works for OPERA RTC-S1 (one COG per polarization, gamma-0 linear power) and
    for ASF HyP3 RTC products saved in power scale. If the file is in dB or
    amplitude, convert it before calling this function.

    invalid_mask_path / invalid_mask_values: optional companion layer (for
    example an RTC layover/shadow mask); pixels whose mask value is listed are
    set to NaN. Check the product specification for the meaning of each value.
    """
    if not os.path.exists(path):
        raise DataLoadError(f"File not found: {path}")
    try:
        with rasterio.open(path) as ds:
            if ds.crs is None:
                raise DataLoadError(f"{path} has no CRS; refusing to guess its location.")
            window = None
            if bbox is not None:
                b = _bbox_to_crs(bbox, bbox_crs, ds.crs)
                window = from_bounds(*b, transform=ds.transform).round_offsets().round_lengths()
                window = window.intersection(Window(0, 0, ds.width, ds.height))
                if window.width < 1 or window.height < 1:
                    raise DataLoadError(f"AOI does not overlap {os.path.basename(path)}")
            arr = ds.read(1, window=window)
            transform = ds.window_transform(window) if window is not None else ds.transform
            crs = ds.crs
            nodata = ds.nodata
    except rasterio.errors.RasterioIOError as exc:
        raise DataLoadError(f"Cannot open {path}: {exc}") from exc
    except (rasterio.errors.WindowError, ValueError) as exc:  # AOI outside the raster
        raise DataLoadError(f"AOI does not overlap {os.path.basename(path)}: {exc}") from exc

    data = _clean_power(arr, [nodata])
    if invalid_mask_path:
        mask_scene = read_categorical(invalid_mask_path, bbox=bbox, bbox_crs=bbox_crs)
        if mask_scene[0].shape != data.shape:
            raise DataLoadError("Invalid-pixel mask is not on the same grid as the backscatter layer.")
        data[np.isin(mask_scene[0], list(invalid_mask_values))] = np.nan

    src = dict(source or {})
    src.setdefault("file", os.path.basename(path))
    return SarScene(data=data, crs=crs, transform=transform, polarization=polarization,
                    acquired=acquired, source=src)


def read_categorical(path: str, bbox=None, bbox_crs: str = "EPSG:4326"):
    """Read a categorical raster (e.g. OPERA DSWx-S1 WTR, a mask) without changing values.

    Returns (array, crs, transform, nodata).
    """
    if not os.path.exists(path):
        raise DataLoadError(f"File not found: {path}")
    try:
        with rasterio.open(path) as ds:
            window = None
            if bbox is not None:
                b = _bbox_to_crs(bbox, bbox_crs, ds.crs)
                window = from_bounds(*b, transform=ds.transform).round_offsets().round_lengths()
                window = window.intersection(Window(0, 0, ds.width, ds.height))
            arr = ds.read(1, window=window)
            transform = ds.window_transform(window) if window is not None else ds.transform
            return arr, ds.crs, transform, ds.nodata
    except (rasterio.errors.RasterioIOError, rasterio.errors.WindowError, ValueError) as exc:
        raise DataLoadError(f"Cannot read {os.path.basename(path)} for this AOI: {exc}") from exc


def _h5_attr_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _h5_attr_str(value[()])
    return str(value)


def _read_identification(h5) -> dict:
    """Pull a few identification fields if present. Missing fields are simply skipped."""
    meta = {}
    if NISAR_IDENTIFICATION not in h5:
        return meta
    ident = h5[NISAR_IDENTIFICATION]
    for key in ("zeroDopplerStartTime", "zeroDopplerEndTime", "trackNumber", "frameNumber",
                "orbitPassDirection", "lookDirection", "granuleId", "productVersion",
                "productType", "processingType"):
        if key in ident:
            meta[key] = _h5_attr_str(ident[key][()])
    return meta


def transform_from_center_coords(x: np.ndarray, y: np.ndarray) -> Affine:
    """Affine transform from 1-D pixel-*centre* coordinate vectors.

    NISAR GCOV stores xCoordinates / yCoordinates as CF-style coordinate
    variables, i.e. the centre of each pixel. A GDAL/rasterio transform needs
    the upper-left *corner*, so we shift by half a pixel. If you ever find the
    NISAR coordinates are corner-based, this is the one function to change.
    """
    if x.size < 2 or y.size < 2:
        raise DataLoadError("Need at least 2x2 pixels to derive a grid spacing.")
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])  # negative for north-up grids
    return Affine(dx, 0.0, float(x[0]) - dx / 2.0, 0.0, dy, float(y[0]) - dy / 2.0)


def read_nisar_gcov_scene(
    path: str,
    polarization: str = "HH",
    frequency: str = "A",
    source: dict | None = None,
    bbox: Sequence[float] | None = None,
    bbox_crs: str = "EPSG:4326",
) -> SarScene:
    """Read one diagonal covariance term (e.g. HHHH = HH backscatter power) from a GCOV granule.

    Only the AOI window is read from disk, so full-frame files (hundreds of MB
    to several GB) stay manageable on a laptop.
    """
    import h5py  # imported here so GeoTIFF-only users do not need it

    if not os.path.exists(path):
        raise DataLoadError(f"File not found: {path}")
    term = polarization.upper() * 2  # "HH" -> "HHHH"
    grid_path = NISAR_GCOV_GRID.format(freq=frequency.upper())
    try:
        with h5py.File(path, "r") as h5:
            if grid_path not in h5:
                raise DataLoadError(f"{grid_path} not found in {os.path.basename(path)}; is this a GCOV file?")
            grid = h5[grid_path]
            if term not in grid:
                available = sorted(k for k in grid.keys() if k.isupper() and len(k) == 4)
                raise DataLoadError(f"{term} not in {os.path.basename(path)}; available terms: {available}")
            x = grid["xCoordinates"][()]
            y = grid["yCoordinates"][()]
            if "projection" not in grid or "epsg_code" not in grid["projection"].attrs:
                raise DataLoadError("GCOV grid has no projection/epsg_code; refusing to guess the CRS.")
            crs = CRS.from_epsg(int(grid["projection"].attrs["epsg_code"]))

            r0, r1, c0, c1 = 0, y.size, 0, x.size
            if bbox is not None:
                minx, miny, maxx, maxy = _bbox_to_crs(bbox, bbox_crs, crs)
                cols = np.nonzero((x >= minx) & (x <= maxx))[0]
                rows = np.nonzero((y >= miny) & (y <= maxy))[0]
                if cols.size < 2 or rows.size < 2:
                    raise DataLoadError(f"AOI does not overlap {os.path.basename(path)}")
                r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1

            dset = grid[term]
            fill = dset.attrs.get("_FillValue", None)
            arr = dset[r0:r1, c0:c1]
            transform = transform_from_center_coords(x[c0:c1], y[r0:r1])
            meta = _read_identification(h5)
    except OSError as exc:
        raise DataLoadError(f"Cannot open {path} as HDF5: {exc}") from exc

    fill_values = [float(np.asarray(fill).ravel()[0])] if fill is not None else []
    data = _clean_power(arr, fill_values)
    src = dict(source or {})
    src.setdefault("file", os.path.basename(path))
    src.setdefault("frequency", frequency.upper())
    src.setdefault("covariance_term", term)
    src["identification"] = meta
    acquired = src.get("acquired") or meta.get("zeroDopplerStartTime", "unknown")
    return SarScene(data=data, crs=crs, transform=transform, polarization=polarization.upper(),
                    acquired=acquired, source=src)
