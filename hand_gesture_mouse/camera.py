"""
Webcam (or video file) capture with clear error messages.

Frames stay in memory only; nothing is written to disk or sent anywhere.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

# Map the names used in config.py to OpenCV "API preference" constants.
_BACKENDS = {
    "auto": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "avfoundation": cv2.CAP_AVFOUNDATION,
    "v4l2": cv2.CAP_V4L2,
}

# Some webcams return a few empty frames while they warm up (or while macOS
# shows the camera-permission prompt), so we retry for up to this long.
_OPEN_TIMEOUT_S = 5.0
_OPEN_RETRY_DELAY_S = 0.1

# Video files with no readable frame rate are assumed to be 30 FPS.
_DEFAULT_FILE_FPS = 30.0


class CameraError(RuntimeError):
    """The camera could not be opened or stopped delivering frames."""


def camera_help() -> str:
    """Platform-specific hints shown when the camera cannot be opened."""
    hints = [
        "Close other apps that may be using the camera (Zoom, Teams, browser tabs).",
        "Try another CAMERA_INDEX in config.py (0, 1, 2 ...) or --camera 1.",
    ]
    if sys.platform == "darwin":
        hints.append(
            "macOS: System Settings > Privacy & Security > Camera > enable your "
            "terminal/IDE (Terminal, iTerm, VS Code), then restart it."
        )
    elif sys.platform == "win32":
        hints.append(
            "Windows: Settings > Privacy & security > Camera > allow desktop apps. "
            'If opening is slow or fails, set CAMERA_BACKEND = "dshow".'
        )
    else:
        hints.append(
            "Linux: check that /dev/video0 exists (ls /dev/video*) and that your user "
            "is in the 'video' group (sudo usermod -aG video $USER, then log out/in)."
        )
    return "\n  - " + "\n  - ".join(hints)


class Camera:
    """Reads frames from a webcam index or a video file.

    ``read()`` returns ``(frame, timestamp_s)``. For a live camera the timestamp
    is ``time.monotonic()``; for a video file it is the frame's position in the
    video, so replays behave the same every time.
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        backend: str = "auto",
        mirror: bool = True,
        max_failed_reads: int = 30,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend
        self.mirror = mirror
        self.max_failed_reads = max_failed_reads
        self.is_file = isinstance(source, str)
        self._cap: cv2.VideoCapture | None = None
        self._failed_reads = 0
        self._file_frame_index = 0
        self._file_fps = _DEFAULT_FILE_FPS
        self.frame_size: tuple[int, int] | None = None  # (width, height) actually delivered

    def open(self) -> None:
        if self.is_file:
            if not Path(self.source).is_file():
                raise CameraError(f"Video file not found: {self.source}")
            self._cap = cv2.VideoCapture(self.source)
        else:
            if self.backend not in _BACKENDS:
                raise CameraError(
                    f"Unknown CAMERA_BACKEND '{self.backend}'. Use one of: {', '.join(_BACKENDS)}"
                )
            self._cap = cv2.VideoCapture(self.source, _BACKENDS[self.backend])

        if not self._cap.isOpened():
            self.release()
            raise CameraError(f"Could not open camera/video source {self.source!r}." + camera_help())

        if self.is_file:
            file_fps = self._cap.get(cv2.CAP_PROP_FPS)
            if file_fps and file_fps > 0:
                self._file_fps = file_fps
        else:
            # Requests only: the driver may ignore them. We read back the real size below.
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            # A 1-frame buffer reduces lag on backends that support it (ignored elsewhere).
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Wait for the first real frame so errors surface now, not mid-session.
        deadline = time.monotonic() + _OPEN_TIMEOUT_S
        while True:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                break
            if self.is_file or time.monotonic() > deadline:
                self.release()
                raise CameraError(
                    f"Source {self.source!r} opened but returned no frames "
                    "(often a permission problem or a camera in use)." + camera_help()
                )
            time.sleep(_OPEN_RETRY_DELAY_S)

        self.frame_size = (frame.shape[1], frame.shape[0])
        if self.is_file:
            # Rewind so the first frame is not skipped.
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def read(self):
        """Return ``(frame_bgr, timestamp_s)`` or ``(None, None)`` at the end of a video.

        Raises CameraError if a live camera stops delivering frames.
        """
        if self._cap is None:
            raise CameraError("Camera.read() called before open().")

        while True:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                self._failed_reads = 0
                break
            if self.is_file:
                return None, None
            self._failed_reads += 1
            if self._failed_reads >= self.max_failed_reads:
                raise CameraError("The camera stopped delivering frames (unplugged or taken by another app?).")
            time.sleep(1.0 / max(self.fps, 1))  # wait about one frame before retrying

        if self.mirror:
            frame = cv2.flip(frame, 1)

        if self.is_file:
            timestamp = self._file_frame_index / self._file_fps
            self._file_frame_index += 1
        else:
            timestamp = time.monotonic()
        return frame, timestamp

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def parse_source(value: str | int) -> int | str:
    """'0' -> 0 (camera index); anything else is treated as a video file path."""
    if isinstance(value, int):
        return value
    return int(value) if value.strip().isdigit() else value
