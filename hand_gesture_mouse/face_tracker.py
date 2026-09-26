"""
Face *detection* (is there a face, and where?) with MediaPipe's FaceDetector.

This is not face recognition: nothing identifies who you are, and nothing is
stored. The face is used only to
  * suspend cursor control when nobody is in front of the camera, and
  * give "move closer / move back" positioning hints.
Set ENABLE_FACE_DETECTION = False (or run with --no-face) to remove it entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hand_tracker import TrackerError, check_model_file, import_mediapipe, mediapipe_load_help


@dataclass
class FaceData:
    """One detected face. The box is in pixels and may extend past the frame edge."""

    x: int
    y: int
    width: int
    height: int
    score: float                          # detection confidence, 0..1
    keypoints: list[tuple[float, float]]  # 6 normalized points: eyes, nose, mouth, ears

    @property
    def area(self) -> int:
        return self.width * self.height

    def touches_edge(self, frame_width: int, frame_height: int) -> bool:
        return (
            self.x <= 0
            or self.y <= 0
            or self.x + self.width >= frame_width
            or self.y + self.height >= frame_height
        )


class FaceTracker:
    """Runs the short-range BlazeFace detector (designed for faces within ~2 m)."""

    def __init__(self, model_path: Path, min_detection_confidence: float = 0.6) -> None:
        check_model_file(model_path)
        self._mp, mp_python, vision = import_mediapipe()
        self._last_timestamp_ms = -1
        self._detector = None

        options = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=min_detection_confidence,
        )
        try:
            self._detector = vision.FaceDetector.create_from_options(options)
        except (OSError, RuntimeError, ValueError) as exc:
            raise TrackerError(f"Could not create the face detector: {exc}" + mediapipe_load_help()) from exc

    def process(self, rgb_frame, timestamp_ms: int) -> list[FaceData]:
        """Detect faces in an RGB frame. ``timestamp_ms`` must increase every call."""
        timestamp_ms = max(int(timestamp_ms), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        try:
            result = self._detector.detect_for_video(image, timestamp_ms)
        except (RuntimeError, ValueError) as exc:
            raise TrackerError(f"Face detection failed: {exc}") from exc

        faces = []
        for detection in result.detections:
            box = detection.bounding_box
            score = detection.categories[0].score if detection.categories else 0.0
            faces.append(
                FaceData(
                    x=int(box.origin_x),
                    y=int(box.origin_y),
                    width=int(box.width),
                    height=int(box.height),
                    score=float(score or 0.0),
                    keypoints=[(kp.x, kp.y) for kp in detection.keypoints],
                )
            )
        return faces

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None


def main_face(faces: list[FaceData]) -> FaceData | None:
    """The user is normally the largest (closest) face in view."""
    return max(faces, key=lambda face: face.area) if faces else None


def positioning_hint(
    face: FaceData | None,
    frame_width: int,
    frame_height: int,
    min_width: float,
    max_width: float,
) -> str | None:
    """Return a short instruction if the user is poorly positioned, else None."""
    if face is None:
        return "No face detected - sit in front of the camera"
    relative_width = face.width / frame_width
    if relative_width < min_width:
        return "Move a little closer to the camera"
    if relative_width > max_width:
        return "Move back a little"
    if face.touches_edge(frame_width, frame_height):
        return "Center yourself in the camera view"
    return None
