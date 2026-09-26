"""
Stage 4 - Move the real cursor with your index fingertip (no clicking yet).

New compared with Stage 3:
  * a control rectangle in the camera image is mapped to the whole screen,
  * exponential smoothing removes jitter,
  * the cursor only moves after the hand has been tracked for a few frames,
  * a per-frame step limit stops jumps when the hand disappears and reappears,
  * the cursor never touches a screen corner (PyAutoGUI fail-safe),
  * moving your real mouse pauses gesture control.

    python stages/stage4_cursor_movement.py              # starts PAUSED: press P to start
    python stages/stage4_cursor_movement.py --dry-run    # never touches the real cursor
    python stages/stage4_cursor_movement.py --source 1   # camera 1 (or a video file)

Keys: P = pause/resume, Q / Esc = quit (the preview window must be focused).
Clicking, dragging and scrolling are added in main.py (Stages 5-7).
"""

import argparse
import math
import sys
import time
from pathlib import Path

import cv2

# ---- Settings ----------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MIRROR = True                          # mirror view: hand right -> cursor right
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
MIN_HAND_DETECTION_CONFIDENCE = 0.7
MIN_HAND_PRESENCE_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

CURSOR_LANDMARK = 8                    # index fingertip
CONTROL_REGION = (0.20, 0.20, 0.80, 0.65)  # (left, top, right, bottom), fractions of the frame
SMOOTHING_ALPHA = 0.35                 # 1 = raw/jittery, lower = smoother but laggier
STABLE_FRAMES_REQUIRED = 5             # frames a hand must be tracked before it may move the cursor
MAX_STEP_FRACTION = 0.08               # max cursor step per frame, fraction of the screen diagonal
MIN_MOVE_PX = 2                        # ignore smaller moves (hand held still)
SCREEN_EDGE_MARGIN_PX = 2              # never an exact corner (PyAutoGUI fail-safe)
MANUAL_OVERRIDE_PX = 40                # real mouse moved this far -> pause
FALLBACK_SCREEN_SIZE = (1920, 1080)    # dry run on a machine without a desktop
WINDOW = "Stage 4 - cursor movement (P = pause, Q / Esc = quit)"
KEY_ESC = 27
GREEN, ORANGE, WHITE, CYAN = (80, 220, 80), (0, 165, 255), (255, 255, 255), (230, 200, 0)


def load_pyautogui():
    try:
        import pyautogui
    except SystemExit as exc:  # PyAutoGUI exits on Linux when tkinter is missing
        raise RuntimeError(f"{exc}") from None
    except Exception as exc:   # ImportError, missing DISPLAY, ...
        raise RuntimeError(f"PyAutoGUI could not start: {exc}") from exc
    pyautogui.PAUSE = 0        # default 0.1 s sleep after every call would cap us at ~10 FPS
    pyautogui.FAILSAFE = True  # cursor pushed into a corner -> FailSafeException
    return pyautogui


def map_to_screen(x, y, screen_w, screen_h):
    """Normalized point inside CONTROL_REGION -> screen pixels (clamped)."""
    left, top, right, bottom = CONTROL_REGION
    u = min(max((x - left) / (right - left), 0.0), 1.0)
    v = min(max((y - top) / (bottom - top), 0.0), 1.0)
    return u * (screen_w - 1), v * (screen_h - 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="never move the real cursor")
    parser.add_argument("--source", default=str(CAMERA_INDEX), help="camera index or video file")
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    gui = None
    try:
        gui = load_pyautogui()
        screen_w, screen_h = gui.size()
    except RuntimeError as exc:
        if not args.dry_run:
            print(f"ERROR: {exc}\nSee README > Troubleshooting, or run with --dry-run.")
            return 1
        screen_w, screen_h = FALLBACK_SCREEN_SIZE
    if screen_w <= 0 or screen_h <= 0:
        print(f"ERROR: invalid screen size {screen_w}x{screen_h}")
        return 1
    if args.dry_run:
        gui_for_moves = None  # read the screen size only; never move
    else:
        gui_for_moves = gui

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"{MODEL_PATH} is missing. Run:  python download_models.py")
        landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        ))
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

    max_step = MAX_STEP_FRACTION * math.hypot(screen_w, screen_h)
    paused = True             # safety: start paused
    stable_frames = 0
    active = False            # True while the hand is actually steering the cursor
    smoothed = None           # smoothed cursor position (floats)
    last_sent = None          # last integer position given to moveTo
    start = time.monotonic()
    last_timestamp_ms = -1
    print(f"Screen {screen_w}x{screen_h}. {'DRY RUN' if args.dry_run else 'LIVE'}. Press P to start.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No more frames (end of video, or the camera stopped).")
                break
            if MIRROR:
                frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = max(int((time.monotonic() - start) * 1000), last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)

            tip = None
            if result.hand_landmarks:
                landmark = result.hand_landmarks[0][CURSOR_LANDMARK]
                tip = (landmark.x, landmark.y)
                stable_frames += 1
            else:
                stable_frames = 0  # hand lost: it must be stable again before moving

            should_move = tip is not None and stable_frames >= STABLE_FRAMES_REQUIRED and not paused
            if should_move and not active:
                # (Re)start from where the cursor really is, then glide - never teleport.
                smoothed = tuple(map(float, gui_for_moves.position())) if gui_for_moves else (screen_w / 2, screen_h / 2)
                last_sent = None
            active = should_move

            if active and gui_for_moves and last_sent is not None:
                real = gui_for_moves.position()
                if math.hypot(real[0] - last_sent[0], real[1] - last_sent[1]) > MANUAL_OVERRIDE_PX:
                    paused, active = True, False
                    print("PAUSED: you moved the real mouse.")

            if active:
                target = map_to_screen(tip[0], tip[1], screen_w, screen_h)
                # Exponential smoothing: move a fraction of the way toward the target.
                sx = smoothed[0] + SMOOTHING_ALPHA * (target[0] - smoothed[0])
                sy = smoothed[1] + SMOOTHING_ALPHA * (target[1] - smoothed[1])
                # Step limit: at most max_step pixels per frame.
                dx, dy = sx - smoothed[0], sy - smoothed[1]
                step = math.hypot(dx, dy)
                if step > max_step:
                    sx, sy = smoothed[0] + dx * max_step / step, smoothed[1] + dy * max_step / step
                # Keep a margin from the edges so we never trigger the corner fail-safe ourselves.
                sx = min(max(sx, SCREEN_EDGE_MARGIN_PX), screen_w - 1 - SCREEN_EDGE_MARGIN_PX)
                sy = min(max(sy, SCREEN_EDGE_MARGIN_PX), screen_h - 1 - SCREEN_EDGE_MARGIN_PX)
                smoothed = (sx, sy)
                ix, iy = int(round(sx)), int(round(sy))
                if last_sent is None or max(abs(ix - last_sent[0]), abs(iy - last_sent[1])) >= MIN_MOVE_PX:
                    if gui_for_moves:
                        gui_for_moves.moveTo(ix, iy)
                    last_sent = (ix, iy)

            # ---- drawing ----
            left, top, right, bottom = CONTROL_REGION
            color = GREEN if active else ORANGE
            top_left = (int(left * width), int(top * height))
            cv2.rectangle(frame, top_left, (int(right * width), int(bottom * height)), color, 1)
            if tip is not None:
                cv2.circle(frame, (int(tip[0] * width), int(tip[1] * height)), 8, color, 2, cv2.LINE_AA)
            state = "PAUSED (press P)" if paused else ("ACTIVE" if active else "waiting for a steady hand")
            mode = "DRY RUN" if args.dry_run else "LIVE"
            cv2.putText(frame, f"{mode} | {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            if smoothed is not None:
                cv2.putText(frame, f"cursor {smoothed[0]:.0f}, {smoothed[1]:.0f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), KEY_ESC):
                break
            if key in (ord("p"), ord("P")):
                paused = not paused
                active = False if paused else active
                print("PAUSED" if paused else "RESUMED")
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if gui is not None and isinstance(exc, gui.FailSafeException):
            print("Fail-safe triggered (cursor in a screen corner): stopping.")
        else:
            raise
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
