"""
Stage 3 - Hand tracking with MediaPipe's HandLandmarker (Tasks API).

Draws the 21 hand landmarks and their connections, labels the important ones,
and shows handedness (Left/Right) with its score.

    python download_models.py                    # once
    python stages/stage3_hand_tracking.py [camera index or video file]

Keys: Q / Esc = quit.

Two hands: set NUM_HANDS = 2 and both hands are tracked and drawn. A mouse
then needs a rule for which hand is in control; main.py uses a preferred hand
(CONTROL_HAND) and otherwise the hand nearest to where the controlling hand was
in the previous frame.
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
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
NUM_HANDS = 1
MIN_HAND_DETECTION_CONFIDENCE = 0.7  # palm detector score needed to start tracking
MIN_HAND_PRESENCE_CONFIDENCE = 0.6   # "a hand is really here" score of the landmark model
MIN_TRACKING_CONFIDENCE = 0.6        # below this, the palm detector runs again
WINDOW = "Stage 3 - hand tracking (Q / Esc to quit)"
KEY_ESC = 27

# Landmark numbering (identical in every MediaPipe version).
LABELS = {0: "wrist", 4: "thumb tip", 5: "index knuckle", 8: "index tip",
          12: "middle tip", 16: "ring tip", 20: "pinky tip"}
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (9, 10), (10, 11), (11, 12),               # middle
    (13, 14), (14, 15), (15, 16),              # ring
    (0, 17), (17, 18), (18, 19), (19, 20),     # pinky
    (5, 9), (9, 13), (13, 17),                 # palm
)
WHITE, CYAN, YELLOW, GREEN, BLACK = (255, 255, 255), (230, 200, 0), (0, 230, 230), (80, 220, 80), (0, 0, 0)


def create_hand_landmarker():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"{MODEL_PATH} is missing. Run:  python download_models.py")
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    return mp, vision.HandLandmarker.create_from_options(options)


def label(frame, text, origin, color, scale=0.45) -> None:
    """Text on a black box (readable on skin and on any background)."""
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = origin
    cv2.rectangle(frame, (x - 2, y - h - 2), (x + w + 2, y + baseline), BLACK, cv2.FILLED)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_hand(frame, landmarks, handedness_label, handedness_score) -> None:
    height, width = frame.shape[:2]
    # Landmarks are normalized (0..1); multiply by the frame size to get pixels.
    points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], WHITE, 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        cv2.circle(frame, point, 4, CYAN, cv2.FILLED, cv2.LINE_AA)
        if index in LABELS:
            label(frame, f"{index} {LABELS[index]}", (point[0] + 6, point[1] - 6), YELLOW)
    wrist = points[0]
    label(frame, f"{handedness_label} hand  score {handedness_score:.2f}",
          (max(2, wrist[0] - 60), min(height - 10, wrist[1] + 25)), GREEN, 0.55)


def main() -> int:
    source = sys.argv[1] if len(sys.argv) > 1 else CAMERA_INDEX
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    try:
        mp, landmarker = create_hand_landmarker()
    except (ImportError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: could not start hand tracking: {exc}")
        return 1

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        landmarker.close()
        print(f"ERROR: could not open camera/video {source!r} (see README > Troubleshooting).")
        return 1
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

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

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = max(int((time.monotonic() - start) * 1000), last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)

            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                category = handedness[0]
                name = category.category_name
                if MIRROR:
                    # MediaPipe labels hands as they look in an un-mirrored image,
                    # so the label must be swapped for a mirrored preview.
                    name = {"Left": "Right", "Right": "Left"}.get(name, name)
                draw_hand(frame, landmarks, name, category.score)

            label(frame, f"hands: {len(result.hand_landmarks)}", (10, 30), GREEN, 0.7)
            cv2.imshow(WINDOW, frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), KEY_ESC):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
