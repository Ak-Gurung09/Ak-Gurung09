"""Record where every number came from: datasets, parameters, software, outputs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone

# The five labels used everywhere in the project (docs, dashboard, manifests).
DATA_LABELS = (
    "Primary NASA SAR data",
    "NASA-derived SAR product",
    "NASA-affiliated or partner data",
    "Supporting non-SAR data",
    "Optional future data",
    "SYNTHETIC test data (not an observation)",
)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _software_versions() -> dict:
    versions = {"python": platform.python_version()}
    for mod in ("numpy", "scipy", "rasterio", "h5py", "sklearn", "shapely", "pyproj"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # optional dependency not installed
            pass
    try:
        import rasterio
        versions["gdal"] = rasterio.__gdal_version__
    except Exception:
        pass
    return versions


class Provenance:
    def __init__(self, case_id: str, mode: str):
        self.record = {
            "case_id": case_id,
            "mode": mode,  # "real-data" | "dry-run (synthetic)"
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "software": _software_versions(),
            "datasets": [],
            "steps": [],
            "outputs": [],
        }

    def add_dataset(self, role: str, source: dict) -> None:
        label = source.get("label", "UNLABELLED")
        if label not in DATA_LABELS:
            raise ValueError(f"Dataset label {label!r} must be one of {DATA_LABELS}")
        self.record["datasets"].append({"role": role, **source})

    def add_step(self, name: str, **params) -> None:
        self.record["steps"].append({"step": name, **params})

    def add_output(self, path: str) -> None:
        self.record["outputs"].append({"file": os.path.basename(path), "sha256": _sha256(path)})

    def write(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.record, f, indent=2, default=str)
