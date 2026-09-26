"""
Stage 2 - Face detection with MediaPipe's FaceDetector (Tasks API).

Draws a box, the 6 face keypoints and the detection confidence.
Press F to switch face detection on/off (the face part is optional),
Q or Esc to quit.

    python download_models.py                     # once
    python stages/stage2_face_detection.py [camera index or video file]

Is face detection needed for cursor control? No. The cursor is driven by the
hand alone. The face is only used later (Stage 7) as a safety check ("is
someone actually sitting here?") and for positioning hints. This is face
*detection* only: nothing identifies who you are.
"""

import sys
import time
from pathlib import Path

import cv2

# ---- Settings ----------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MIRROR = True
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "blaze_face_short_range.tflite"
MIN_DETECTION_CONFIDENCE = 0.6   # hide faces the model is less sure about
ENABLE_FACE = True               # start with detection on (toggle with F)
WINDOW = "Stage 2 - face detection (F = toggle, Q / Esc = quit)"
KEY_ESC = 27
GREEN = (80, 220, 80)


def create_face_detector():
    # Imported here so a missing/broken MediaPipe gives a clear message.
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"{MODEL_PATH} is missing. Run:  python download_models.py")
    options = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,  # frames arrive one by one, with timestamps
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    )
    return mp, vision.FaceDetector.create_from_options(options)


def draw_detection(frame, detection) -> None:
    height, width = frame.shape[:2]
    box = detection.bounding_box                     # pixels: origin_x, origin_y, width, height
    score = detection.categories[0].score if detection.categories else 0.0
    x, y, w, h = int(box.origin_x), int(box.origin_y), int(box.width), int(box.height)
    cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
    for keypoint in detection.keypoints:             # normalized 0..1 -> pixels
        cv2.circle(frame, (int(keypoint.x * width), int(keypoint.y * height)), 3, GREEN, cv2.FILLED)
    cv2.putText(frame, f"face {score:.2f}", (max(0, x), max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv2.LINE_AA)


def main() -> int:
    source = sys.argv[1] if len(sys.argv) > 1 else CAMERA_INDEX
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    try:
        mp, detector = create_face_detector()
    except (ImportError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: could not start face detection: {exc}")
        return 1

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        detector.close()
        print(f"ERROR: could not open camera/video {source!r} (see README > Troubleshooting).")
        return 1
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    enabled = ENABLE_FACE
    start = time.monotonic()
    last_timestamp_ms = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No more frames (end of video, or the camera stopped).")
                break
            if MIRROR:
                frame = cv2.flip(frame, 1)

            if enabled:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe wants RGB; OpenCV gives BGR
                # VIDEO mode requires strictly increasing timestamps.
                timestamp_ms = max(int((time.monotonic() - start) * 1000), last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms
                result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
                for detection in result.detections:
                    draw_detection(frame, detection)
                status = f"faces: {len(result.detections)}"
            else:
                status = "face detection OFF"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2, cv2.LINE_AA)

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), KEY_ESC):
                break
            if key in (ord("f"), ord("F")):
                enabled = not enabled
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
