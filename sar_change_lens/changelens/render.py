"""Web-map previews: reproject to Web Mercator and write transparent PNG overlays.

Leaflet/Folium image overlays are stretched between lat/lon corners on a Web
Mercator map, so the image itself must be in EPSG:3857 or it will be subtly
misplaced. The GeoTIFFs in outputs/ stay in the native CRS for analysis.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

# Map colours (validated for colour-vision deficiency with the three-slot rule;
# aqua is < 3:1 contrast, so the legend and detections table always accompany it).
MAP_CLASSES = {
    # display group: (hex, alpha 0-255, legend text)
    "new_water": ("#2a78d6", 230, "Possible flood-related change"),
    "other_change": ("#eb6834", 170, "Other strong change (unclassified)"),
    "persistent_water": ("#1baf7a", 190, "Water-like in both images"),
    "masked": ("#898781", 110, "Masked (terrain / no data)"),
}
# Detector class code -> display group (receded and unclassified changes are
# folded into one colour to keep the map readable).
CODE_TO_GROUP = {2: "new_water", 3: "persistent_water", 4: "other_change", 5: "other_change",
                 6: "other_change", 7: "masked"}
# Sequential single-hue ramp (light -> dark) for confidence 0..1.
CONFIDENCE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# Diverging pair for change intensity: blue = darker after, red = brighter after.
DIVERGING_NEG, DIVERGING_POS = "#2a78d6", "#e34948"


def _hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def to_web_mercator(data: np.ndarray, crs, transform, categorical: bool = False):
    """Reproject a 2-D array to EPSG:3857. Returns (array, [[south, west], [north, east]])."""
    h, w = data.shape
    src_bounds = array_bounds(h, w, transform)  # (west, south, east, north) in the native CRS
    dst_t, dw, dh = calculate_default_transform(crs, "EPSG:3857", w, h, *src_bounds)
    if categorical:
        dst = np.zeros((dh, dw), dtype=np.uint8)
        reproject(data.astype(np.uint8), dst, src_transform=transform, src_crs=crs, src_nodata=0,
                  dst_transform=dst_t, dst_crs="EPSG:3857", dst_nodata=0, resampling=Resampling.nearest)
    else:
        dst = np.full((dh, dw), np.nan, dtype=np.float32)
        reproject(data.astype(np.float32), dst, src_transform=transform, src_crs=crs, src_nodata=np.nan,
                  dst_transform=dst_t, dst_crs="EPSG:3857", dst_nodata=np.nan, resampling=Resampling.bilinear)
    west, south, east, north = transform_bounds("EPSG:3857", "EPSG:4326", *array_bounds(dh, dw, dst_t))
    return dst, [[south, west], [north, east]]


def db_to_rgba(db: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """Greyscale backscatter image (dark = low backscatter), transparent where NaN."""
    ok = np.isfinite(db)
    if vmin is None or vmax is None:
        lo, hi = np.percentile(db[ok], [2, 98]) if ok.any() else (-25.0, 0.0)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    g = np.clip((np.nan_to_num(db, nan=vmin) - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    rgba = np.zeros(db.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = (g * 255).astype(np.uint8)[..., None]
    rgba[..., 3] = np.where(ok, 255, 0)
    return rgba


def classes_to_rgba(codes: np.ndarray) -> np.ndarray:
    rgba = np.zeros(codes.shape + (4,), dtype=np.uint8)
    for code, group in CODE_TO_GROUP.items():
        hexcol, alpha, _ = MAP_CLASSES[group]
        m = codes == code
        rgba[m, :3] = _hex_rgb(hexcol)
        rgba[m, 3] = alpha
    return rgba


def confidence_to_rgba(conf: np.ndarray) -> np.ndarray:
    ok = np.isfinite(conf)
    idx = np.clip((np.nan_to_num(conf) * len(CONFIDENCE_RAMP)).astype(int), 0, len(CONFIDENCE_RAMP) - 1)
    ramp = np.array([_hex_rgb(c) for c in CONFIDENCE_RAMP], dtype=np.uint8)
    rgba = np.zeros(conf.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = ramp[idx]
    rgba[..., 3] = np.where(ok, 230, 0)
    return rgba


def delta_to_rgba(delta_db: np.ndarray, limit: float = 10.0, hide_below: float = 1.0) -> np.ndarray:
    """Diverging change-intensity map; |change| < hide_below dB is left transparent (the neutral midpoint)."""
    ok = np.isfinite(delta_db) & (np.abs(np.nan_to_num(delta_db)) >= hide_below)
    t = np.clip(np.abs(np.nan_to_num(delta_db)) / limit, 0, 1)
    neg, pos = np.array(_hex_rgb(DIVERGING_NEG)), np.array(_hex_rgb(DIVERGING_POS))
    colour = np.where((np.nan_to_num(delta_db) < 0)[..., None], neg, pos)
    rgba = np.zeros(delta_db.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = colour.astype(np.uint8)
    rgba[..., 3] = np.where(ok, (60 + 180 * t).astype(np.uint8), 0)
    return rgba


def save_png(rgba: np.ndarray, path: str) -> None:
    Image.fromarray(rgba, mode="RGBA").save(path, optimize=True)
