"""The one data structure every module passes around: a georeferenced SAR scene.

Keeping pixels, CRS, transform and provenance together in a single object is what
stops "pixel (row, col)" from being confused with "map (x, y)" later on.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import array_bounds


class ChangeLensError(Exception):
    """Base class for errors the pipeline reports to the user instead of crashing."""


class DataLoadError(ChangeLensError):
    """A file is missing, unreadable, or does not contain the expected layer."""


class AlignmentError(ChangeLensError):
    """Two scenes cannot be put on the same grid (no overlap, wrong CRS, ...)."""


class InsufficientDataError(ChangeLensError):
    """Too few valid pixels to estimate statistics honestly."""


@dataclass
class SarScene:
    """A single-polarization SAR backscatter image on a map grid.

    data        2-D float32 array of *linear power* backscatter (e.g. gamma-0).
                Invalid pixels (fill, layover/shadow, outside swath) are NaN.
    crs         Coordinate reference system of the grid (e.g. EPSG:32646).
    transform   Affine transform mapping (col, row) of a pixel's upper-left
                corner to (x, y) in CRS units. Use it; never treat row/col as
                coordinates.
    polarization  e.g. "HH", "HV", "VV", "VH".
    acquired    ISO-8601 UTC acquisition time (start of the observation).
    source      Provenance: dataset name, granule id, URL, maturity, data label.
    """

    data: np.ndarray
    crs: CRS
    transform: Affine
    polarization: str
    acquired: str
    source: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(f"SarScene.data must be 2-D, got shape {self.data.shape}")
        self.data = np.asarray(self.data, dtype=np.float32)
        if not isinstance(self.crs, CRS):
            self.crs = CRS.from_user_input(self.crs)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def valid(self) -> np.ndarray:
        """Boolean mask: True where the backscatter value can be used."""
        return np.isfinite(self.data) & (self.data > 0)

    @property
    def pixel_size(self) -> tuple[float, float]:
        """(width, height) of one pixel in CRS units (metres for UTM)."""
        return abs(self.transform.a), abs(self.transform.e)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(left, bottom, right, top) in CRS units."""
        rows, cols = self.shape
        return array_bounds(rows, cols, self.transform)

    def with_data(self, data: np.ndarray, **changes) -> "SarScene":
        """Copy of this scene with new pixel values (same grid and provenance)."""
        return replace(self, data=np.asarray(data, dtype=np.float32), **changes)
