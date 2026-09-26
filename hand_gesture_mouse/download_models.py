"""
Download the two MediaPipe model files into ./models (run once, about 8 MB).

    python download_models.py           download missing models
    python download_models.py --force   download again even if present

The URLs point at a fixed model version and each file is checked against a
SHA-256 hash, so everyone gets exactly the files this project was tested with.
Only the models are downloaded; this program never uploads anything.
"""

from __future__ import annotations

import argparse
import hashlib
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"
BASE_URL = "https://storage.googleapis.com/mediapipe-models"
TIMEOUT_S = 60
CHUNK_BYTES = 1 << 16

MODELS = [
    {
        "file": "hand_landmarker.task",
        "url": f"{BASE_URL}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "sha256": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    },
    {
        "file": "blaze_face_short_range.tflite",
        "url": f"{BASE_URL}/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        "sha256": "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f",
    },
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ssl_context() -> ssl.SSLContext:
    # certifi (installed with MediaPipe) avoids CERTIFICATE_VERIFY_FAILED errors on
    # python.org macOS builds that never ran "Install Certificates.command".
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=TIMEOUT_S, context=ssl_context()) as response, open(partial, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        received = 0
        shown = -1
        for chunk in iter(lambda: response.read(CHUNK_BYTES), b""):
            out.write(chunk)
            received += len(chunk)
            percent = 100 * received // total if total else -1
            if percent != shown and percent % 10 == 0:
                print(f"\r  {destination.name}: {percent:3d}%", end="", flush=True)
                shown = percent
    print()
    partial.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the MediaPipe models used by this project.")
    parser.add_argument("--force", action="store_true", help="download even if the file already exists")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    for model in MODELS:
        destination = MODELS_DIR / model["file"]
        if destination.is_file() and not args.force and sha256_of(destination) == model["sha256"]:
            print(f"OK (already present): {destination}")
            continue
        print(f"Downloading {model['url']}")
        try:
            download(model["url"], destination)
        except (urllib.error.URLError, OSError) as exc:
            failures += 1
            print(f"  FAILED: {exc}")
            print(f"  Download it manually in a browser and save it as: {destination}")
            continue
        actual = sha256_of(destination)
        if actual != model["sha256"]:
            failures += 1
            destination.unlink()
            print(f"  FAILED: checksum mismatch (expected {model['sha256']}, got {actual}). File removed.")
            continue
        print(f"OK: {destination}")

    if failures:
        print(f"\n{failures} model(s) could not be downloaded. Check your internet connection or proxy.")
        return 1
    print("\nAll models ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
