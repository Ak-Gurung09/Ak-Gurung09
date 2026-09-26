"""
Hand landmark detection with the MediaPipe Tasks ``HandLandmarker``.

Why the Tasks API: MediaPipe removed the old ``mp.solutions.hands`` module in
version 0.10.30+, so tutorials that call ``mp.solutions.hands.Hands()`` fail
with "module 'mediapipe' has no attribute 'solutions'". The Tasks API used here
works on both the older 0.10.x releases and the current 1.x releases.

MediaPipe is imported inside ``HandTracker`` (not at the top of the file) so the
landmark constants and ``HandData`` can be used, and unit-tested, without
loading MediaPipe.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# The 21 hand landmarks (same numbering in every MediaPipe version)
# ---------------------------------------------------------------------------
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

NUM_LANDMARKS = 21

# MCP = knuckle, PIP = middle joint, DIP = joint below the tip, TIP = fingertip.
LANDMARK_NAMES = {
    WRIST: "wrist",
    THUMB_TIP: "thumb tip",
    INDEX_MCP: "index knuckle",
    INDEX_TIP: "index tip",
    MIDDLE_TIP: "middle tip",
    RING_TIP: "ring tip",
    PINKY_TIP: "pinky tip",
}

# Lines drawn between landmarks (defined here so drawing never depends on
# MediaPipe's drawing helpers, which moved between versions).
HAND_CONNECTIONS = (
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (WRIST, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
    (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, RING_MCP), (RING_MCP, PINKY_MCP),
)


class TrackerError(RuntimeError):
    """A MediaPipe model could not be loaded or failed while processing."""


def mediapipe_load_help() -> str:
    hints = ["Reinstall with:  pip install -r requirements.txt"]
    if sys.platform.startswith("linux"):
        hints.append(
            "Linux: MediaPipe 1.x needs the EGL/GLES system libraries:  "
            "sudo apt install libegl1 libgles2   (Fedora: sudo dnf install mesa-libEGL mesa-libGLES)"
        )
    if sys.platform == "darwin":
        hints.append(
            "Intel Mac: MediaPipe 0.10.30+ ships Apple-Silicon wheels only. Use Python 3.12 "
            "or older with:  pip install mediapipe==0.10.21"
        )
    return "\n  - " + "\n  - ".join(hints)


def check_model_file(path: Path) -> None:
    if not Path(path).is_file():
        raise TrackerError(
            f"Model file not found: {path}\n  Download the models first:  python download_models.py"
        )


def import_mediapipe():
    """Import MediaPipe's Tasks API, converting failures into a helpful TrackerError."""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise TrackerError(f"Could not import MediaPipe ({exc})." + mediapipe_load_help()) from exc
    return mp, mp_python, vision


@dataclass
class HandData:
    """One detected hand.

    ``landmarks`` holds 21 ``(x, y, z)`` tuples. ``x`` and ``y`` are normalized to
    the frame (0..1); ``z`` is depth relative to the wrist (smaller = closer).
    """

    landmarks: list[tuple[float, float, float]]
    handedness: str          # "Left" or "Right", as the person sees their own hand
    handedness_score: float  # how sure the model is about Left vs Right (0.5..1.0)
    image_width: int
    image_height: int

    def point(self, index: int) -> tuple[float, float]:
        """Normalized (x, y) of one landmark."""
        x, y, _ = self.landmarks[index]
        return x, y

    def points_px(self) -> list[tuple[float, float]]:
        """All landmarks in pixel units. Use these for distances: normalized x and y
        have different scales when the frame is not square (640x480)."""
        return [(x * self.image_width, y * self.image_height) for x, y, _ in self.landmarks]


class HandTracker:
    """Runs MediaPipe HandLandmarker in VIDEO mode (synchronous, one frame at a time)."""

    def __init__(
        self,
        model_path: Path,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_presence_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        swap_handedness: bool = True,
    ) -> None:
        check_model_file(model_path)
        self._mp, mp_python, vision = import_mediapipe()
        self.swap_handedness = swap_handedness
        self._last_timestamp_ms = -1
        self._landmarker = None

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        try:
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except (OSError, RuntimeError, ValueError) as exc:
            raise TrackerError(f"Could not create the hand landmarker: {exc}" + mediapipe_load_help()) from exc

    def process(self, rgb_frame, timestamp_ms: int) -> list[HandData]:
        """Detect hands in an RGB frame. ``timestamp_ms`` must increase every call."""
        # VIDEO mode raises ValueError if a timestamp repeats or goes backwards,
        # which can happen when two frames arrive within the same millisecond.
        timestamp_ms = max(int(timestamp_ms), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        try:
            result = self._landmarker.detect_for_video(image, timestamp_ms)
        except (RuntimeError, ValueError) as exc:
            raise TrackerError(f"Hand tracking failed: {exc}") from exc

        height, width = rgb_frame.shape[:2]
        hands = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            if len(landmarks) != NUM_LANDMARKS or not handedness:
                continue  # malformed result: skip rather than guess
            category = handedness[0]
            label = category.category_name
            if self.swap_handedness:
                label = {"Left": "Right", "Right": "Left"}.get(label, label)
            hands.append(
                HandData(
                    landmarks=[(lm.x, lm.y, lm.z) for lm in landmarks],
                    handedness=label,
                    handedness_score=float(category.score),
                    image_width=width,
                    image_height=height,
                )
            )
        return hands

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
