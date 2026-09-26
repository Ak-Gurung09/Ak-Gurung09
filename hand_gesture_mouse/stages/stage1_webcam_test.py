"""
Stage 1 - Webcam test.

Opens the webcam, shows the mirrored video with an FPS counter and exits
cleanly on Q, Esc, Ctrl+C or the window's close button.

    python stages/stage1_webcam_test.py            # camera 0
    python stages/stage1_webcam_test.py 1          # camera 1
    python stages/stage1_webcam_test.py clip.mp4   # a video file instead of a camera

Only OpenCV is needed for this stage.
"""

import sys
import time

import cv2

# ---- Settings ----------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 640         # requested; the camera may choose the nearest size it supports
CAMERA_HEIGHT = 480
MIRROR = True              # flip horizontally so the preview behaves like a mirror
MAX_FAILED_READS = 30      # consecutive failed reads before giving up (~1 s)
FPS_SMOOTHING = 0.1        # weight of the newest frame in the displayed FPS
WINDOW = "Stage 1 - webcam test (Q / Esc to quit)"
KEY_ESC = 27


def open_camera(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera/video {source!r}. Close other apps using the camera, "
            "try another index (0, 1, 2), and check camera permissions (see README)."
        )
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    return cap


def window_was_closed(window_seen: bool) -> tuple[bool, bool]:
    """Return (closed, window_seen). Only trust 'not visible' after it was visible once."""
    try:
        visible = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return window_seen, window_seen
    return (window_seen and not visible), (window_seen or visible)


def main() -> int:
    source = sys.argv[1] if len(sys.argv) > 1 else CAMERA_INDEX
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    try:
        cap = open_camera(source)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    fps = 0.0
    last_time = time.perf_counter()
    failed_reads = 0
    window_seen = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                failed_reads += 1
                if not isinstance(source, int) or failed_reads >= MAX_FAILED_READS:
                    print("No more frames (end of video, or the camera stopped).")
                    break
                time.sleep(0.03)  # about one frame at 30 FPS before retrying
                continue
            failed_reads = 0

            if MIRROR:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt, last_time = now - last_time, now
            if dt > 0:
                fps = 1 / dt if fps == 0 else fps + FPS_SMOOTHING * (1 / dt - fps)

            height, width = frame.shape[:2]
            cv2.putText(frame, f"FPS: {fps:5.1f}   {width}x{height}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF  # also lets the window redraw
            if key in (ord("q"), ord("Q"), KEY_ESC):
                break
            closed, window_seen = window_was_closed(window_seen)
            if closed:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()            # free the camera for other apps
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
