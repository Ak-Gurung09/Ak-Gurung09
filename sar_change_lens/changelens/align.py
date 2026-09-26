"""Put two rasters on exactly the same map grid before comparing them.

Comparing pixel [i, j] of two images is only meaningful if both pixels cover
the same patch of ground. Same-track, same-frame NISAR GCOV pairs usually share
a grid already; OPERA DSWx-S1 tiles (MGRS) and HAND tiles never do.
"""

from __future__ import annotations

import numpy as np
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject

from .scene import AlignmentError, SarScene


def same_grid(a: SarScene, b: SarScene, tol: float = 1e-6) -> bool:
    return (
        a.shape == b.shape
        and CRS.from_user_input(a.crs) == CRS.from_user_input(b.crs)
        and a.transform.almost_equals(b.transform, precision=tol)
    )


def reproject_to_grid(data, src_crs, src_transform, ref: SarScene, categorical: bool = False,
                      src_nodata=None) -> np.ndarray:
    """Resample any 2-D array onto ref's grid.

    Continuous data (backscatter power, HAND) uses bilinear resampling on
    *linear* values; categorical data (class codes, masks) uses nearest
    neighbour so no invented class values appear.
    """
    if categorical:
        fill = 255 if src_nodata is None else src_nodata
        dst = np.full(ref.shape, fill, dtype=np.asarray(data).dtype)
        resampling = Resampling.nearest
        nodata = fill
    else:
        data = np.asarray(data, dtype=np.float32)
        dst = np.full(ref.shape, np.nan, dtype=np.float32)
        resampling = Resampling.bilinear
        nodata = np.nan if src_nodata is None else src_nodata
    reproject(
        source=data, destination=dst,
        src_transform=src_transform, src_crs=src_crs, src_nodata=nodata,
        dst_transform=ref.transform, dst_crs=ref.crs, dst_nodata=nodata,
        resampling=resampling,
    )
    return dst


def align_to_reference(ref: SarScene, other: SarScene, min_overlap: float = 0.2) -> SarScene:
    """Return `other` resampled onto `ref`'s grid (unchanged if already aligned).

    Raises AlignmentError if less than `min_overlap` of ref's valid pixels are
    also valid in the aligned scene, because change statistics over a sliver
    of overlap are not trustworthy.
    """
    if same_grid(ref, other):
        aligned = other
    else:
        data = reproject_to_grid(other.data, other.crs, other.transform, ref)
        src = dict(other.source)
        src["resampled_to_reference_grid"] = True
        aligned = SarScene(data=data, crs=ref.crs, transform=ref.transform,
                           polarization=other.polarization, acquired=other.acquired, source=src)

    ref_valid = ref.valid.sum()
    if ref_valid == 0:
        raise AlignmentError("Reference scene has no valid pixels in the AOI.")
    overlap = (ref.valid & aligned.valid).sum() / ref_valid
    if overlap < min_overlap:
        raise AlignmentError(
            f"Only {overlap:.0%} of the reference scene overlaps the other scene "
            f"(minimum {min_overlap:.0%}). Pick acquisitions from the same track/frame."
        )
    return aligned
