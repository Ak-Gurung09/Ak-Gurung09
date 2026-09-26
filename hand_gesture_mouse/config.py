"""
All tunable settings for the hand-gesture mouse, grouped in one place.

Edit a value, save the file and restart ``python main.py``. No other module
contains tuning numbers, so this is the only file you need while calibrating.

Units used below
----------------
* normalized         - a fraction of the camera frame: 0.0 = left/top edge, 1.0 = right/bottom edge.
* palm-relative      - a distance divided by your palm length (wrist -> middle-finger knuckle).
                       It stays the same whether your hand is near or far from the camera.
* ``_S`` suffix      - seconds.
* ``_PX`` suffix     - screen pixels.
"""

from pathlib import Path

# =============================================================================
# 1. Safety and run mode
# =============================================================================

# True = detect and display gestures, but never move or click the real cursor.
# Use this first, every time you change thresholds. (CLI: --dry-run / --live)
DRY_RUN = False

# Start with cursor control OFF. Resume with the P key or by holding an open palm.
START_PAUSED = True

# If you physically move the mouse/touchpad this far (pixels) away from where the
# program last put the cursor, gesture control pauses immediately ("you took over").
MANUAL_OVERRIDE_PX = 40

# =============================================================================
# 2. Camera
# =============================================================================

CAMERA_INDEX = 0          # 0 = built-in webcam; try 1, 2 ... for external cameras.
CAMERA_WIDTH = 640        # Requested size. The driver may choose the nearest size it supports.
CAMERA_HEIGHT = 480       # 640x480 keeps hand tracking fast on ordinary laptops.
CAMERA_FPS = 30           # Requested frame rate (also not guaranteed by the driver).

# "auto" lets OpenCV choose. Other options: "dshow" or "msmf" (Windows),
# "avfoundation" (macOS), "v4l2" (Linux). Try "dshow" if a Windows camera opens slowly.
CAMERA_BACKEND = "auto"

# Flip frames horizontally so the preview behaves like a mirror. With a normal
# front-facing laptop webcam, keep this True: moving your hand right then moves
# the cursor right.
MIRROR_FRAME = True

# Give up after this many consecutive failed frame reads (about 1 s at 30 FPS).
CAMERA_MAX_FAILED_READS = 30

# =============================================================================
# 3. Model files (download them with:  python download_models.py)
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
FACE_MODEL_PATH = MODELS_DIR / "blaze_face_short_range.tflite"

# =============================================================================
# 4. Hand tracking
# =============================================================================

MAX_NUM_HANDS = 1                   # 1 is faster and avoids confusion; 2 is supported.
MIN_HAND_DETECTION_CONFIDENCE = 0.7  # Palm detector score needed to start tracking a hand.
MIN_HAND_PRESENCE_CONFIDENCE = 0.6   # Landmark model's "a hand is really here" score.
MIN_TRACKING_CONFIDENCE = 0.6        # Frame-to-frame tracking quality before re-detecting.

# Which hand may control the cursor: "any", "Right" or "Left".
CONTROL_HAND = "any"

# Handedness labels are corrected automatically for the mirrored preview. If the
# preview still says "Left" when you raise your right hand, set this to True.
INVERT_HANDEDNESS = False

# Ignore hands whose palm length is smaller than this fraction of the frame's
# shorter side (about 1.3 m away from a laptop webcam). Rejects people in the background.
MIN_PALM_SIZE = 0.08

# Consecutive good frames required after a hand appears before it may control
# anything (about 0.17 s at 30 FPS). Prevents actions from half-detected hands.
STABLE_FRAMES_REQUIRED = 5

# If the cursor anchor jumps farther than this (normalized) between two frames,
# treat it as a tracking glitch and re-acquire the hand. 0.2 of the frame in one
# frame is ~4x faster than a quick deliberate hand movement.
MAX_ANCHOR_JUMP = 0.20

# Short tracking dropouts (motion blur) shorter than this keep the gesture state;
# longer ones reset everything and release any held mouse button.
LOST_HAND_GRACE_S = 0.15

# =============================================================================
# 5. Face detection (optional, used only for presence and positioning hints)
# =============================================================================

ENABLE_FACE_DETECTION = True         # CLI: --no-face
MIN_FACE_DETECTION_CONFIDENCE = 0.6
FACE_DETECTION_INTERVAL = 2          # Run face detection on every Nth frame to save CPU.

REQUIRE_FACE_FOR_CONTROL = True      # Suspend control while no face is visible.
FACE_ABSENT_TIMEOUT_S = 1.0          # Face may be missing this long before suspending.
FACE_ABSENT_PAUSE_S = 10.0           # Absent this long -> fully PAUSED (needs manual resume).

FACE_MIN_WIDTH = 0.12                # Face narrower than this fraction of the frame -> "move closer".
FACE_MAX_WIDTH = 0.45                # Wider than this -> "move back".

# =============================================================================
# 6. Cursor mapping and smoothing
# =============================================================================

# Landmark that drives the cursor. 5 = index-finger knuckle (recommended: it barely
# moves when you pinch, so clicks land where you aimed). 8 = index fingertip.
CURSOR_LANDMARK = 5

# Part of the camera frame that maps to the whole screen: (left, top, right, bottom),
# normalized. The margins mean you never have to reach the frame edge. For equal
# horizontal/vertical speed, make its pixel width:height roughly match your screen
# (0.6*640 : 0.45*480 = 384 : 216 = 16:9 for the defaults).
CONTROL_REGION = (0.20, 0.20, 0.80, 0.65)

# Exponential smoothing: 1.0 = raw (jittery, instant); lower = smoother but laggier.
SMOOTHING_ALPHA = 0.35

# Do not send moves smaller than this (removes 1-pixel shimmer when your hand is still).
MIN_MOVE_PX = 2

# Largest cursor step per frame, as a fraction of the screen diagonal. Stops the
# cursor teleporting when a hand reappears somewhere else.
MAX_STEP_FRACTION = 0.08

# Keep the cursor this many pixels away from the screen edges. PyAutoGUI's
# fail-safe fires when the cursor sits exactly in a screen corner, so the program
# must never put it there itself.
SCREEN_EDGE_MARGIN_PX = 2

# =============================================================================
# 7. Gesture recognition
# =============================================================================

# A finger is "extended" when its tip is at least this many times farther from
# the wrist than its middle joint (PIP). Straight finger ~1.3, bent finger < 1.1.
FINGER_EXTENDED_RATIO = 1.15

# A finger is "folded" (fist) when its tip is closer to the wrist than this many
# times its knuckle (MCP) distance, i.e. the tip is tucked back into the palm.
FINGER_FOLDED_RATIO = 1.0

# The thumb is "open" when its tip is farther than this (palm-relative) from the
# index knuckle. Spread thumb ~0.6-0.9, thumb resting against the hand ~0.3-0.45.
THUMB_OPEN_RATIO = 0.55

# A pinch must be seen for this long before it counts, and released for this long
# before the release counts (2-3 consecutive frames; 3 at 30 FPS). Filters
# single-frame landmark noise.
GESTURE_CONFIRM_S = 0.06

# --- Left click and drag: thumb tip touches index tip ------------------------
PINCH_PRESS_RATIO = 0.25     # Palm-relative distance below which the pinch closes.
PINCH_RELEASE_RATIO = 0.40   # ...and above which it opens again (hysteresis gap).
LEFT_CLICK_COOLDOWN_S = 0.35 # Minimum time between two left clicks.

ENABLE_DRAG = True           # CLI: --no-drag
DRAG_HOLD_S = 0.5            # Pinch held this long starts a drag. With drag disabled,
                             # pinches longer than this are ignored (never clicked).

# --- Right click: thumb tip touches middle tip, index finger stays up --------
ENABLE_RIGHT_CLICK = True    # CLI: --no-right-click
RIGHT_CLICK_COOLDOWN_S = 0.8

# --- Scroll: index + middle fingers up (ring + pinky down), move hand up/down -
ENABLE_SCROLL = True         # CLI: --no-scroll
SCROLL_ENTER_S = 0.20        # Hold the pose this long before scrolling starts.
SCROLL_EXIT_S = 0.15         # Pose may flicker off this long without leaving scroll mode.
SCROLL_STEP = 0.03           # Vertical hand travel (fraction of frame height) per scroll step.
SCROLL_NOTCHES_PER_STEP = 1  # Mouse-wheel notches sent per step (raise to scroll faster).
SCROLL_MAX_NOTCHES_PER_FRAME = 3
SCROLL_INVERT = False        # True = "natural" direction (hand up -> content moves up).

# Scroll units that equal one physical wheel notch. None = automatic per OS:
# Windows 120 (WHEEL_DELTA), macOS 3 (lines), Linux 1 (one button-4/5 click).
SCROLL_UNITS_PER_NOTCH = None

# --- Pause / resume: open palm (all five fingers spread) --------------------
ENABLE_PALM_PAUSE = True
PALM_HOLD_S = 1.0            # Hold the open palm this long to toggle pause.
PAUSE_TOGGLE_COOLDOWN_S = 1.5

# =============================================================================
# 8. Display
# =============================================================================

WINDOW_NAME = "Hand Gesture Mouse  |  P = pause  |  Q / Esc = quit"
PREVIEW_ALWAYS_ON_TOP = True   # Keeps the status visible (ignored where unsupported).
SHOW_LANDMARK_LABELS = True
SHOW_MINIMAP = True            # Small screen map showing where the cursor is/would be.
PRINT_EVENTS = True            # Log clicks/drags/scrolls/pauses to the terminal.
BANNER_DURATION_S = 3.0        # How long on-screen notices stay visible.

# Screen size assumed in dry-run mode when the real size cannot be read
# (for example on a machine without a desktop session).
FALLBACK_SCREEN_SIZE = (1920, 1080)


# =============================================================================
# 9. Sanity checks (called once at start-up by main.py)
# =============================================================================


def validate(s) -> None:
    """Raise ValueError listing every impossible setting (``s`` is this module)."""
    problems = []
    left, top, right, bottom = s.CONTROL_REGION
    if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
        problems.append("CONTROL_REGION must be (left, top, right, bottom) with 0 <= left < right <= 1 "
                        "and 0 <= top < bottom <= 1")
    if not 0.0 < s.SMOOTHING_ALPHA <= 1.0:
        problems.append("SMOOTHING_ALPHA must be greater than 0 and at most 1")
    if not 0.0 < s.PINCH_PRESS_RATIO < s.PINCH_RELEASE_RATIO:
        problems.append("PINCH_PRESS_RATIO must be positive and smaller than PINCH_RELEASE_RATIO")
    if not 0 <= s.CURSOR_LANDMARK <= 20:
        problems.append("CURSOR_LANDMARK must be a landmark index from 0 to 20")
    if s.MAX_NUM_HANDS < 1:
        problems.append("MAX_NUM_HANDS must be at least 1")
    if str(s.CONTROL_HAND).lower() not in ("any", "left", "right"):
        problems.append('CONTROL_HAND must be "any", "Left" or "Right"')
    for name in ("MIN_HAND_DETECTION_CONFIDENCE", "MIN_HAND_PRESENCE_CONFIDENCE",
                 "MIN_TRACKING_CONFIDENCE", "MIN_FACE_DETECTION_CONFIDENCE"):
        if not 0.0 <= getattr(s, name) <= 1.0:
            problems.append(f"{name} must be between 0 and 1")
    for name in ("FACE_DETECTION_INTERVAL", "STABLE_FRAMES_REQUIRED", "SCREEN_EDGE_MARGIN_PX",
                 "SCROLL_NOTCHES_PER_STEP", "SCROLL_MAX_NOTCHES_PER_FRAME", "CAMERA_MAX_FAILED_READS"):
        if getattr(s, name) < 1:
            problems.append(f"{name} must be at least 1")
    for name in ("SCROLL_STEP", "PALM_HOLD_S", "DRAG_HOLD_S", "MAX_STEP_FRACTION", "MIN_PALM_SIZE"):
        if getattr(s, name) <= 0:
            problems.append(f"{name} must be greater than 0")
    if s.SCROLL_UNITS_PER_NOTCH is not None and s.SCROLL_UNITS_PER_NOTCH <= 0:
        problems.append("SCROLL_UNITS_PER_NOTCH must be None or a positive number")
    if problems:
        raise ValueError("Invalid settings in config.py:\n  - " + "\n  - ".join(problems))
