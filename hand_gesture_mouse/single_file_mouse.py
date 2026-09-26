"""
Hand-gesture mouse in ONE file, for quick testing.

Simpler than main.py: no face detection, no drag, and confirmation is counted
in frames instead of seconds. The safety rules are the same: starts paused,
dry-run mode, confidence gates, smoothing, hysteresis, one click per pinch,
cooldowns, auto-pause when you touch the real mouse, and the corner fail-safe.

    python download_models.py                 # once
    python single_file_mouse.py --dry-run     # safe: never touches the real cursor
    python single_file_mouse.py               # live, starts PAUSED

Gestures
    index finger up, middle down ...... move the cursor (follows the index knuckle)
    thumb tip touches index tip ....... left click (on release, once per pinch)
    thumb touches middle tip, index up  right click
    index + middle up, move hand ...... scroll
    open palm for 1 second ............ pause / resume
Keys (preview window focused): P = pause/resume, Q or Esc = quit
"""

import argparse
import math
import sys
import time
from pathlib import Path

import cv2

# =============================================================================
# SETTINGS (see config.py and README.md for what each one does)
# =============================================================================
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MIRROR = True
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"

MIN_HAND_DETECTION_CONFIDENCE = 0.7
MIN_HAND_PRESENCE_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6
MIN_PALM_SIZE = 0.08              # smaller hands (far away / background people) are ignored
STABLE_FRAMES_REQUIRED = 5        # frames before a newly seen hand may act

CURSOR_LANDMARK = 5               # index knuckle (use 8 for the fingertip)
CONTROL_REGION = (0.20, 0.20, 0.80, 0.65)
SMOOTHING_ALPHA = 0.35
MIN_MOVE_PX = 2
MAX_STEP_FRACTION = 0.08
SCREEN_EDGE_MARGIN_PX = 2
MANUAL_OVERRIDE_PX = 40

FINGER_EXTENDED_RATIO = 1.15
FINGER_FOLDED_RATIO = 1.0         # tip closer to the wrist than its knuckle -> folded (fist)
THUMB_OPEN_RATIO = 0.55
PINCH_PRESS_RATIO = 0.25
PINCH_RELEASE_RATIO = 0.40
CONFIRM_FRAMES = 2                # a pinch/release must be seen this many frames in a row
LEFT_CLICK_COOLDOWN_S = 0.35
RIGHT_CLICK_COOLDOWN_S = 0.8
MAX_CLICK_PINCH_S = 0.5           # a left pinch held longer than this is ignored (no drag here)

SCROLL_ENTER_FRAMES = 6           # hold the two-finger pose this many frames first
SCROLL_STEP = 0.03                # hand travel (fraction of frame height) per wheel notch
SCROLL_UNITS_PER_NOTCH = {"win32": 120, "darwin": 3}.get(sys.platform, 1)

PALM_HOLD_S = 1.0
PAUSE_TOGGLE_COOLDOWN_S = 1.5
FALLBACK_SCREEN_SIZE = (1920, 1080)
WINDOW = "Single-file gesture mouse (P = pause, Q / Esc = quit)"
KEY_ESC = 27

# Landmark indices
WRIST, THUMB_TIP, INDEX_MCP, INDEX_PIP, INDEX_TIP = 0, 4, 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
FINGERS = {"index": (5, 6, 8), "middle": (9, 10, 12), "ring": (13, 14, 16), "pinky": (17, 18, 20)}  # MCP, PIP, TIP
CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11),
               (11, 12), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20),
               (5, 9), (9, 13), (13, 17))
GREEN, ORANGE, WHITE, CYAN, BLACK = (80, 220, 80), (0, 165, 255), (255, 255, 255), (230, 200, 0), (0, 0, 0)


# =============================================================================
# Geometry
# =============================================================================
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hand_features(landmarks, width, height):
    """Palm-relative measurements (independent of distance to the camera)."""
    pts = [(lm.x * width, lm.y * height) for lm in landmarks]  # pixels: correct aspect ratio
    palm = dist(pts[WRIST], pts[MIDDLE_MCP])
    if palm < 1:
        return None
    extended = {name: dist(pts[WRIST], pts[tip]) > FINGER_EXTENDED_RATIO * dist(pts[WRIST], pts[pip])
                for name, (mcp, pip, tip) in FINGERS.items()}
    folded = [dist(pts[WRIST], pts[tip]) < FINGER_FOLDED_RATIO * dist(pts[WRIST], pts[mcp])
              for mcp, pip, tip in FINGERS.values()]
    return {
        "palm_size": palm / min(width, height),
        "index_pinch": dist(pts[THUMB_TIP], pts[INDEX_TIP]) / palm,
        "middle_pinch": dist(pts[THUMB_TIP], pts[MIDDLE_TIP]) / palm,
        "extended": extended,
        "fist": all(folded),  # a fist can look like a pinch: never click from it
        "thumb_open": dist(pts[THUMB_TIP], pts[INDEX_MCP]) / palm > THUMB_OPEN_RATIO,
        "anchor": (landmarks[CURSOR_LANDMARK].x, landmarks[CURSOR_LANDMARK].y),
        "points": pts,
    }


# =============================================================================
# Gestures
# =============================================================================
class PinchButton:
    """One event per pinch, with hysteresis, frame confirmation and a cooldown.

    on_release=True fires when the fingers open again (and only for a short
    pinch), so a pinch that turns into a fist can still be cancelled.
    """

    def __init__(self, cooldown_s, on_release):
        self.cooldown_s = cooldown_s
        self.on_release = on_release
        self.last_fired = -math.inf
        self.pressed_at = 0.0
        self.reset()

    def reset(self):
        self.pressed = False
        self.armed = False  # becomes True once the fingers are seen apart
        self.count = 0

    @property
    def busy(self):
        return self.pressed or self.count > 0

    def _fire(self, now):
        if now - self.last_fired < self.cooldown_s:
            return False
        self.last_fired = now
        return True

    def update(self, ratio, allowed, now):
        """Return True at most once per pinch."""
        if not self.pressed:
            if ratio > PINCH_RELEASE_RATIO:
                self.armed = True
            closing = allowed and self.armed and ratio < PINCH_PRESS_RATIO
            self.count = self.count + 1 if closing else 0
            if self.count >= CONFIRM_FRAMES:
                self.pressed, self.armed, self.count = True, False, 0
                self.pressed_at = now
                if not self.on_release:
                    return self._fire(now)
        elif ratio > PINCH_RELEASE_RATIO:          # between the thresholds nothing changes
            self.count += 1
            if self.count >= CONFIRM_FRAMES:
                self.pressed, self.count, self.armed = False, 0, True
                if self.on_release and now - self.pressed_at <= MAX_CLICK_PINCH_S:
                    return self._fire(now)
        else:
            self.count = 0
        return False


# =============================================================================
# Cursor
# =============================================================================
def load_pyautogui():
    try:
        import pyautogui
    except SystemExit as exc:  # MouseInfo exits on Linux when tkinter is missing
        raise RuntimeError(f"{exc}") from None
    except Exception as exc:
        raise RuntimeError(f"PyAutoGUI could not start: {exc}") from exc
    pyautogui.PAUSE = 0        # remove the default 0.1 s delay after every call
    pyautogui.FAILSAFE = True  # cursor in a screen corner -> FailSafeException
    return pyautogui


class Cursor:
    def __init__(self, gui, width, height, dry_run):
        self.gui = None if dry_run else gui
        self.w, self.h = width, height
        self.max_step = MAX_STEP_FRACTION * math.hypot(width, height)
        self.pos = (width / 2, height / 2)
        self.last_sent = None
        self.last_action = ""
        self.dry_run = dry_run

    def begin(self):
        if self.gui:
            self.pos = tuple(map(float, self.gui.position()))
        self.last_sent = None

    def user_moved_mouse(self):
        if not self.gui or self.last_sent is None:
            return False
        x, y = self.gui.position()
        return dist((x, y), self.last_sent) > MANUAL_OVERRIDE_PX

    def move(self, anchor):
        left, top, right, bottom = CONTROL_REGION
        u = min(max((anchor[0] - left) / (right - left), 0.0), 1.0)
        v = min(max((anchor[1] - top) / (bottom - top), 0.0), 1.0)
        tx, ty = u * (self.w - 1), v * (self.h - 1)
        x = self.pos[0] + SMOOTHING_ALPHA * (tx - self.pos[0])
        y = self.pos[1] + SMOOTHING_ALPHA * (ty - self.pos[1])
        step = dist((x, y), self.pos)
        if step > self.max_step:
            x = self.pos[0] + (x - self.pos[0]) * self.max_step / step
            y = self.pos[1] + (y - self.pos[1]) * self.max_step / step
        x = min(max(x, SCREEN_EDGE_MARGIN_PX), self.w - 1 - SCREEN_EDGE_MARGIN_PX)
        y = min(max(y, SCREEN_EDGE_MARGIN_PX), self.h - 1 - SCREEN_EDGE_MARGIN_PX)
        self.pos = (x, y)
        ix, iy = int(round(x)), int(round(y))
        if self.last_sent is None or max(abs(ix - self.last_sent[0]), abs(iy - self.last_sent[1])) >= MIN_MOVE_PX:
            if self.gui:
                self.gui.moveTo(ix, iy)
            self.last_sent = (ix, iy)

    def act(self, name, button=None, scroll=0):
        self.last_action = name + (" (dry run)" if self.dry_run else "")
        print(f"[{time.strftime('%H:%M:%S')}] {self.last_action}")
        if self.gui:
            if button:
                self.gui.click(button=button)
            else:
                self.gui.scroll(scroll * SCROLL_UNITS_PER_NOTCH)


# =============================================================================
# Main loop
# =============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Single-file hand-gesture mouse")
    parser.add_argument("--dry-run", action="store_true", help="never move or click the real cursor")
    parser.add_argument("--source", default=str(CAMERA_INDEX), help="camera index or video file")
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    print("WARNING: virtual mouse control can interfere with normal computer use.\n"
          "Starts PAUSED. P or a 1 s open palm resumes. Q/Esc quits. Moving your real mouse pauses.\n"
          "Video stays in memory on this computer; nothing is saved or uploaded.")

    gui = None
    try:
        gui = load_pyautogui()
        screen_w, screen_h = gui.size()
    except RuntimeError as exc:
        if not args.dry_run:
            print(f"ERROR: {exc}\nRun with --dry-run, or see README > Troubleshooting.")
            return 1
        screen_w, screen_h = FALLBACK_SCREEN_SIZE
    if screen_w <= 0 or screen_h <= 0:
        print(f"ERROR: invalid screen size {screen_w}x{screen_h}")
        return 1

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

    cursor = Cursor(gui, screen_w, screen_h, args.dry_run)
    left = PinchButton(LEFT_CLICK_COOLDOWN_S, on_release=True)    # click when the pinch opens
    right = PinchButton(RIGHT_CLICK_COOLDOWN_S, on_release=False)  # index must be up: fire at once
    paused, active = True, False
    stable_frames = 0
    scroll_frames, scroll_ref = 0, None
    palm_since, last_toggle, palm_armed = None, -math.inf, True
    start, last_ts = time.monotonic(), -1

    def set_paused(value, reason):
        nonlocal paused, active
        paused = value
        active = False
        print(f"{'PAUSED' if value else 'RESUMED'} ({reason})")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No more frames (end of video, or the camera stopped).")
                break
            if MIRROR:
                frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            now = time.monotonic()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_ts = max(int((now - start) * 1000), last_ts + 1)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), last_ts)

            raw = None
            if result.hand_landmarks:
                raw = hand_features(result.hand_landmarks[0], width, height)
                if raw is not None and raw["palm_size"] < MIN_PALM_SIZE:
                    raw = None
            stable_frames = stable_frames + 1 if raw else 0
            # Only a confidently tracked, stable hand may do anything.
            f = raw if stable_frames >= STABLE_FRAMES_REQUIRED else None
            if f is None:
                left.reset()
                right.reset()
                scroll_frames, scroll_ref, palm_since = 0, None, None

            label = "acquiring hand..." if raw else "no hand"
            if f is not None:
                ext = f["extended"]
                label = "point with your index finger"
                # --- open palm: toggle pause (works while paused), once per hold ---
                palm = f["thumb_open"] and all(ext.values())
                if not palm:
                    palm_since, palm_armed = None, True
                elif palm_armed and now - last_toggle >= PAUSE_TOGGLE_COOLDOWN_S:
                    if palm_since is None:
                        palm_since = now
                    label = f"open palm {min((now - palm_since) / PALM_HOLD_S, 1):.0%}"
                    if now - palm_since >= PALM_HOLD_S:
                        set_paused(not paused, "open palm")
                        last_toggle, palm_since, palm_armed = now, None, False

                if paused:
                    left.reset()
                    right.reset()
                else:
                    if not active:
                        cursor.begin()
                        active = True
                    if cursor.user_moved_mouse():
                        set_paused(True, "real mouse moved, or the OS ignored our move - see README")
                    else:
                        # --- clicks ---
                        if f["fist"]:
                            left.reset()  # a pinch that became a fist is cancelled, never clicked
                        left_ok = f["index_pinch"] <= f["middle_pinch"] and not f["fist"] and not right.busy
                        right_ok = ext["index"] and f["index_pinch"] > PINCH_RELEASE_RATIO and not left.busy
                        if left.update(f["index_pinch"], left_ok, now):
                            cursor.act("LEFT CLICK", button="left")
                        if right.update(f["middle_pinch"], right_ok, now):
                            cursor.act("RIGHT CLICK", button="right")

                        # --- scroll: index + middle up, ring + pinky down, thumb away ---
                        two_up = (ext["index"] and ext["middle"] and not ext["ring"] and not ext["pinky"]
                                  and f["index_pinch"] > PINCH_RELEASE_RATIO and f["middle_pinch"] > PINCH_RELEASE_RATIO)
                        scroll_frames = scroll_frames + 1 if two_up else 0
                        if scroll_frames == SCROLL_ENTER_FRAMES:
                            scroll_ref = f["anchor"][1]
                        elif scroll_frames > SCROLL_ENTER_FRAMES:
                            steps = int((scroll_ref - f["anchor"][1]) / SCROLL_STEP)  # hand up -> positive
                            if steps:
                                scroll_ref -= steps * SCROLL_STEP
                                cursor.act(f"SCROLL {steps:+d}", scroll=steps)
                        else:
                            scroll_ref = None if scroll_frames == 0 else scroll_ref

                        # --- move: only in the pointer pose and when nothing else is happening ---
                        pointer = ext["index"] and not ext["middle"]
                        if left.busy or right.busy:
                            label = "pinch"
                        elif scroll_frames:
                            label = "scroll"
                        elif pointer:
                            label = "pointer"
                            cursor.move(f["anchor"])
            elif active:
                active = False  # hand lost: stop controlling until it is stable again

            # --- drawing ---
            color = ORANGE if paused else (GREEN if active else CYAN)
            x0, y0, x1, y1 = CONTROL_REGION
            cv2.rectangle(frame, (int(x0 * width), int(y0 * height)), (int(x1 * width), int(y1 * height)), color, 1)
            if raw is not None:
                pts = [(int(x), int(y)) for x, y in raw["points"]]
                for a, c in CONNECTIONS:
                    cv2.line(frame, pts[a], pts[c], WHITE, 2, cv2.LINE_AA)
                for p in pts:
                    cv2.circle(frame, p, 3, CYAN, cv2.FILLED)
                cv2.circle(frame, pts[CURSOR_LANDMARK], 8, GREEN, 2)
                label += f" | pinch {raw['index_pinch']:.2f} / {raw['middle_pinch']:.2f}"
            status = f"{'DRY RUN' if args.dry_run else 'LIVE'} | {'PAUSED' if paused else 'ACTIVE'} | {label}"
            for text, y in ((status, 25), (f"last: {cursor.last_action}", 50)):
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (6, y - th - 6), (14 + tw, y + 6), BLACK, cv2.FILLED)
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), KEY_ESC):
                break
            if key in (ord("p"), ord("P")):
                set_paused(not paused, "P key")
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if gui is not None and isinstance(exc, gui.FailSafeException):
            print("Fail-safe triggered (cursor pushed into a screen corner): stopping.")
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
