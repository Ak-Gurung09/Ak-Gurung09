# Hand Gesture Mouse

Control your laptop cursor with hand gestures through the built-in webcam, using
Python, OpenCV, MediaPipe and PyAutoGUI. Everything runs locally and nothing is
recorded.

```bash
pip install -r requirements.txt       # see section 5 for per-OS steps
python download_models.py             # once: two MediaPipe model files (~8 MB)
python main.py --dry-run              # practise safely: the real cursor is never touched
python main.py                        # live mode, starts PAUSED (press P or hold an open palm)
```

> **Defaults assumed.** No clarifying answers were given, so this project targets
> one built-in webcam, one hand, a single 1920x1080-class display, and Python
> 3.10-3.12 on Windows, macOS or Linux. Everything is adjustable in `config.py`.

---

## Contents

1. [Project overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Technology and compatibility](#3-technology-and-compatibility)
4. [Folder structure](#4-folder-structure)
5. [Installation](#5-installation)
6. [Stage 1: webcam test](#6-stage-1-webcam-test)
7. [Stage 2: face detection](#7-stage-2-face-detection)
8. [Stage 3: hand tracking](#8-stage-3-hand-tracking)
9. [Stage 4: cursor movement](#9-stage-4-cursor-movement)
10. [Complete MVP (Stages 5-7)](#10-complete-mvp-implementation-stages-5-7)
11. [Gesture logic](#11-gesture-logic)
12. [Configuration guide](#12-configuration-guide)
13. [Testing plan](#13-testing-plan)
14. [Troubleshooting](#14-troubleshooting)
15. [Future improvements](#15-future-improvements)
16. [Safety and privacy checklist](#16-safety-and-privacy-checklist)

---

## 1. Project overview

The program watches you through the webcam and turns hand poses into mouse actions:

| You do | The computer does |
|---|---|
| Point with your index finger (middle finger down) | Cursor follows your hand |
| Touch thumb tip to index tip briefly | Left click |
| Pinch thumb and index and hold for 0.5 s | Drag (button held until you open the pinch) |
| Touch thumb tip to middle tip, index finger up | Right click |
| Index and middle fingers up, move the hand up or down | Scroll |
| Hold an open palm for 1 s | Pause / resume |
| Make a fist | Rest: nothing moves, and any pinch is cancelled |
| Press `Q` or `Esc` (preview window focused) | Quit |

**Essential (the MVP):** webcam capture, hand tracking, cursor movement,
left click, pause and exit, dry-run mode, and safety gates.
**Included but optional (each can be switched off):** right click, drag, scroll,
open-palm pause, face presence check and positioning hints.
**Not included, on purpose:** face *recognition*, video recording, keyboard
control, anything unrelated to the cursor.

---

## 2. Architecture

```
 +--------+   BGR frame   +--------+   RGB    +--------------+  box, score   +------------------+
 | Camera |-------------->| mirror |--------->| FaceTracker  |-------------->| presence gate +  |
 +--------+ camera.py     +--------+          | (every 2nd   |               | positioning hint |
                                   |          |  frame)      |               +------------------+
                                   |          +--------------+                        |
                                   |   RGB    +--------------+  21 landmarks          v
                                   +--------->| HandTracker  |---------+    +--------------------+
                                              +--------------+         +--->| GestureController  |
                                              hand_tracker.py               | (state machines)   |
                                                                            +--------------------+
                                                                  intentions: move? click? drag?
                                                                    scroll? toggle pause?
                                                                                     |
            +-----------------------+      draw       +------------------+           v
            | preview window + keys |<----------------|   overlay.py     |   +------------------+
            | (P pause, Q/Esc quit) |                 +------------------+   | CursorController |
            +-----------------------+                                        | map, smooth,     |
                                                                             | PyAutoGUI or     |
                                                                             | dry run          |
                                                                             +------------------+
```

`main.py` runs this loop once per frame. The key design choice is that
**`GestureController` only decides and `CursorController` only acts**. The
decision logic never touches the mouse, so it can be unit-tested with synthetic
hands and replayed in dry-run mode.

### 2.1 How webcam frames are captured

`cv2.VideoCapture(index, backend)` opens the camera. `camera.py` requests
640x480 at 30 FPS (the driver may choose something else, so the real size is
read back from the first frame) and sets a 1-frame buffer to reduce lag where
the backend supports it. It waits up to 5 s for the first real frame, because
some cameras return empty frames while warming up or while macOS shows its
permission dialog. Each frame is flipped horizontally (`cv2.flip(frame, 1)`) so
the preview behaves like a mirror, then converted from OpenCV's **BGR** to the
**RGB** that MediaPipe expects. Frames live only in memory.

### 2.2 How the face is detected

MediaPipe's **FaceDetector** with the *BlazeFace short-range* model is a small
single-shot detector designed for faces within about 2 m of a front camera. For
each face it returns a pixel bounding box, a confidence score and 6 keypoints
(eyes, nose tip, mouth centre, ears). This is *detection* ("there is a face
here"), not recognition ("this is Alex"). See section 7 for whether it is needed.

### 2.3 How hand landmarks are detected

MediaPipe's **HandLandmarker** is a two-step pipeline:

1. A **palm detector** scans the whole image for hands (score must pass
   `MIN_HAND_DETECTION_CONFIDENCE`).
2. A **landmark model** looks at the cropped hand and predicts 21 3D points, a
   "hand presence" score and the handedness (Left/Right).

In `VIDEO` mode the landmarks from one frame define the crop for the next frame
(tracking), so the palm detector only runs again when tracking fails
(`MIN_TRACKING_CONFIDENCE`) or the presence score drops below
`MIN_HAND_PRESENCE_CONFIDENCE`. Hands that fail these thresholds are not
returned at all. MediaPipe does **not** give a per-frame confidence for the
landmarks themselves: `visibility`/`presence` are not filled in for hands (None or 0.0). The
handedness score only says how sure the model is about Left vs Right. So this
project adds its own gates on top: a minimum hand size, several stable frames
before acting, and a jump detector (section 2.6).

### 2.4 Which landmarks are which

```
        8   12  16  20        TIP  (fingertip)
        |   |   |   |
        7   11  15  19        DIP
        |   |   |   |
   4    6   10  14  18        PIP  (middle joint)
   |    |   |   |   |
   3    5---9---13--17        MCP  (knuckle)
   |   /             /
   2  /             /
    1              /
     \           /
         0                    WRIST
```

| Index | Name | Used for |
|---|---|---|
| 0 | wrist | reference point for finger extension; palm length |
| 4 | thumb tip | pinches |
| 5 | index MCP (knuckle) | **cursor anchor** (default), thumb-open test |
| 6, 8 | index PIP, index tip | index extended?; left-click pinch |
| 9 | middle MCP | palm length = distance(0, 9) |
| 10, 12 | middle PIP, middle tip | middle extended?; right-click pinch |
| 13-16, 17-20 | ring, pinky | extension tests (fist, palm, scroll pose) |

Coordinates: `x` and `y` are **normalized** (0 = left/top edge, 1 = right/bottom
edge of the image). `z` is depth relative to the wrist, roughly on the same scale
as `x`, with smaller values closer to the camera. Because a 640x480 image is not
square, a normalized distance of 0.1 horizontally is not the same length as 0.1
vertically. That is why all distances are computed in pixels.

### 2.5 Mapping camera coordinates to the screen

Only a rectangle in the middle of the camera image, `CONTROL_REGION =
(left, top, right, bottom)`, maps to the whole screen. That leaves margins, so
the hand never has to reach the frame edge where tracking is weakest:

```
u = clamp((x - left) / (right - left), 0, 1)
v = clamp((y - top)  / (bottom - top), 0, 1)
screen_x = u * (screen_width  - 1)
screen_y = v * (screen_height - 1)
```

The screen size is read at run time with `pyautogui.size()` and never hardcoded.
The frame is already mirrored, so moving your hand to your right moves the
cursor right. The default region is 384x216 camera pixels, which has the same
16:9 shape as a laptop screen. Horizontal and vertical speed therefore match
(about 5 screen pixels per camera pixel on a 1920x1080 screen). The final
position is clamped to 2 px inside the screen edges, for the reason in
section 2.9.

### 2.6 How smoothing (and jump protection) reduces jitter

Landmarks shimmer by 1-2 camera pixels even when your hand is still, which is
5-10 screen pixels after mapping. Exponential smoothing fixes that:

```
smoothed = smoothed + alpha * (target - smoothed)       # alpha = SMOOTHING_ALPHA = 0.35
```

Each frame the cursor moves 35 % of the remaining way, so it reaches 95 % of a
new position in about 7 frames (0.23 s at 30 FPS). Random jitter is reduced to
about 46 %. Three more rules stop sudden jumps:

* **Step limit:** at most `MAX_STEP_FRACTION` (8 %) of the screen diagonal per
  frame, so a hand that reappears elsewhere glides instead of teleporting.
* **Dead zone:** moves smaller than `MIN_MOVE_PX` (2 px) are not sent.
* **Jump detector:** if the anchor jumps more than 20 % of the frame between two
  frames (a tracking glitch or a different hand), everything resets and the hand
  must be re-acquired. After re-acquisition the cursor continues from where it
  really is.

### 2.7 How gestures are classified

Every frame, `compute_features()` measures the hand in a way that does not
depend on distance to the camera. Each distance is divided by the palm length
(wrist to middle knuckle):

* **Finger extended:** the tip is more than 1.15x as far from the wrist as the
  finger's middle joint (`dist(0, tip) > 1.15 * dist(0, pip)`). This works at
  any hand rotation, unlike the common `tip.y < pip.y` test, which only works
  for an upright hand.
* **Finger folded (fist):** the tip is closer to the wrist than the knuckle is.
* **Thumb open:** the thumb tip is more than 0.55 palm lengths from the index knuckle.
* **Pinch ratios:** thumb tip to index tip, and thumb tip to middle tip, in palm lengths.

Poses are combinations of these (pointer, two-finger, open palm, fist). Each
gesture is then a small **state machine** in `gesture_controller.py` with
explicit states, confirmation times, hysteresis and cooldowns (section 11).
Only one mouse gesture may be active at a time, and while it is active the
others are held in their reset state.

### 2.8 How click debouncing prevents repeated clicks

Four layers, all in `LeftClickDragGesture`:

1. **Hysteresis:** the pinch *closes* below 0.25 and *opens* above 0.40. Jitter
   between the two thresholds changes nothing, so the state cannot flicker.
2. **Confirmation:** closing and opening must each last `GESTURE_CONFIRM_S`
   (60 ms, 3 frames at 30 FPS). A single noisy frame is ignored.
3. **One event per cycle:** the click is emitted on the transition
   PINCHED -> OPEN, which can only happen once per pinch. After tracking loss,
   the state machine starts in `WAIT_OPEN` and must see open fingers before a
   pinch can count.
4. **Cooldown:** at most one left click every 0.35 s.

### 2.9 How the program pauses or exits safely

| Mechanism | Works when | Effect |
|---|---|---|
| Starts paused (`START_PAUSED`) | always | nothing happens until you resume |
| `P` / `Space` | preview window focused | pause / resume |
| Open palm held 1 s | anywhere, hand visible | pause / resume |
| Move your real mouse/touchpad | anywhere | instant pause ("you took over") |
| Cursor pushed into a screen corner | anywhere | PyAutoGUI fail-safe, pause |
| No face for 1 s / 10 s | face check on | suspended / paused |
| Hand lost > 0.15 s | always | control stops, held button released |
| `Q`, `Esc`, window close button, `Ctrl+C` | window / terminal focused | clean exit |

On every exit path, `close()` releases a held mouse button first, then the camera
and the MediaPipe models, then the windows. **Keyboard keys only reach an OpenCV
window when it has focus.** After you click into another app, use the open palm
or your real mouse to pause, or click the preview window before pressing `Q`.

---

## 3. Technology and compatibility

All of the following was checked by installing the packages in September 2026,
not taken from older tutorials.

| Package | Role | Tested versions | Notes |
|---|---|---|---|
| Python | language | 3.10, 3.11, 3.12, 3.13 | 3.11/3.12 recommended. Intel Mac: 3.12 or older |
| `mediapipe` | hand + face models | 1.0.1 and 0.10.21 | Tasks API only (see below) |
| `opencv-contrib-python` | camera, drawing, window | 5.0.0.93 and 4.11.0.86 | installed automatically by MediaPipe |
| `numpy` | image arrays | 2.4.6 and 1.26.4 | |
| `pyautogui` | move/click/scroll | 0.9.54 | chosen over `pynput` (below) |

**MediaPipe changed its API.** Starting with **0.10.30**, the `mediapipe.solutions`
package (`mp.solutions.hands`, `mp.solutions.face_detection`,
`mp.solutions.drawing_utils`) is **no longer included** (verified: present in
0.10.21, absent in 0.10.30, 0.10.31 and 1.0.1). Most online tutorials use it and
fail with `AttributeError: module 'mediapipe' has no attribute 'solutions'`.
This project uses the **MediaPipe Tasks API** (`HandLandmarker`, `FaceDetector`),
which works on both 0.10.21 and 1.0.1. Consequences:

* The Tasks API needs **model files** (`.task` / `.tflite`). `download_models.py`
  fetches fixed versions and checks their SHA-256 hashes.
* `VIDEO` mode needs **strictly increasing timestamps**. A repeated timestamp
  raises `ValueError: Input timestamp must be monotonically increasing`, so the
  trackers enforce `timestamp = max(timestamp, last + 1)`.
* Hand drawing is done with our own `HAND_CONNECTIONS` list, not MediaPipe's
  drawing helpers, which moved between versions.
* **Handedness:** with the current hand model, "Left"/"Right" matches the
  person's real hand in an **un-mirrored** image (verified on MediaPipe's own
  sample photo with both 0.10.21 and 1.0.1). This differs from the old
  documentation. Because we mirror the frame, the label is swapped
  automatically; `INVERT_HANDEDNESS` flips it if your setup disagrees.
  Handedness is only used for display and the optional `CONTROL_HAND` filter.

**Platform limits of MediaPipe 1.x:**

* **Intel Macs:** 0.10.30 and later publish Apple-Silicon (`arm64`) wheels
  only. On an Intel Mac use Python 3.12 or older and pip installs 0.10.21
  automatically. Python 3.13 on an Intel Mac has no MediaPipe wheel at all.
* **Linux:** 1.x loads `libEGL.so.1` and `libGLESv2.so.2`. Minimal installs,
  WSL and containers may need `sudo apt install libegl1 libgles2`.
* Wheels exist for Windows x64/ARM64, macOS arm64 and Linux x86_64/aarch64
  (glibc 2.28+).

**OpenCV notes:**

* Install **only** `opencv-contrib-python`. MediaPipe depends on it. Also
  installing `opencv-python` puts two different `cv2` packages in the same
  environment, a well-known source of broken imports.
* **OpenCV 5.0** (installed automatically with MediaPipe 1.0.1) draws
  `FONT_HERSHEY_SIMPLEX` with a new proportional font that gets wider when it
  gets bolder. The old trick of drawing a thicker black copy under the text
  therefore leaves "ghost" letters, so the overlay draws text on a filled box
  instead. That works on OpenCV 4 and 5.

**Why PyAutoGUI (and its traps):** it is cross-platform, can read the screen
size and cursor position, and has a built-in corner fail-safe. `pynput` is a
good alternative, but it cannot report the screen size (you would need another
package), so PyAutoGUI keeps the dependency list shorter. Things the code
handles:

* `pyautogui.PAUSE` defaults to **0.1 s of sleep after every call**, which would
  cap the program at about 10 FPS. It is set to 0.
* **Linux:** `import pyautogui` calls `sys.exit()` (via its MouseInfo dependency)
  when `tkinter` is missing, which would silently kill the program. The import
  is wrapped, and the fix is `sudo apt install python3-tk python3-dev`. It also
  needs an **X11** session: under **Wayland**, synthetic events do not reach
  native Wayland apps (see Troubleshooting).
* **macOS:** mouse control needs the **Accessibility** permission for your
  terminal/IDE. Without it macOS silently ignores the events. The program
  checks this with `AXIsProcessTrusted()` and warns. Drags must be sent as
  "dragged" events, so the code uses `dragTo(..., mouseDownUp=False)` while the
  button is held.
* **Windows:** PyAutoGUI marks the process DPI-aware, so `size()` and `moveTo()`
  both use physical pixels on scaled (125 % / 150 %) displays. A normal program
  cannot click into windows running as Administrator (Windows UIPI).
* **Scroll units differ by OS:** `pyautogui.scroll(1)` is 1/120 of a wheel notch
  on Windows, one line on macOS and one notch on Linux. `SCROLL_UNITS_PER_NOTCH`
  converts automatically (120 / 3 / 1).
* **Fail-safe corners:** PyAutoGUI raises `FailSafeException` when the cursor is
  on any of the four exact corner pixels. The program keeps its own cursor at
  least 2 px from every edge, so only *you* can trigger it.
* Only the **primary monitor** is supported by PyAutoGUI.

---

## 4. Folder structure

```
hand_gesture_mouse/
├── README.md                  this guide
├── requirements.txt           runtime dependencies
├── requirements-dev.txt       + pytest
├── download_models.py         fetches + verifies the two MediaPipe models
├── models/                    model files are downloaded here (git-ignored)
├── config.py                  ALL tunable settings + validation
├── camera.py                  webcam / video-file capture, errors with OS hints
├── hand_tracker.py            MediaPipe HandLandmarker, landmark constants, HandData
├── face_tracker.py            MediaPipe FaceDetector, positioning hints
├── gesture_controller.py      features + gesture state machines (no mouse access)
├── cursor_controller.py       mapping, smoothing, PyAutoGUI / dry run, fail-safe
├── overlay.py                 all drawing on the preview
├── main.py                    the complete MVP (run this)
├── single_file_mouse.py       simpler all-in-one version for quick tests
├── practice_pad.py            harmless practice window (counts clicks/drags/scrolls)
├── stages/
│   ├── stage1_webcam_test.py
│   ├── stage2_face_detection.py
│   ├── stage3_hand_tracking.py
│   └── stage4_cursor_movement.py
├── pytest.ini                 tells pytest to look only in tests/
└── tests/                     52 unit tests (pytest), no camera needed
```

---

## 5. Installation

Use a virtual environment so these packages do not clash with other projects.

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_models.py
python stages\stage1_webcam_test.py
```

Also allow camera access: *Settings > Privacy & security > Camera > Let desktop
apps access your camera*.

### macOS (Terminal)

```bash
python3 -m venv .venv               # Intel Mac: use Python 3.12 or older (see section 3)
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_models.py
python stages/stage1_webcam_test.py
```

The first run asks for **Camera** permission for your terminal app. For live
cursor control, also enable your terminal (Terminal, iTerm2 or VS Code) under
*System Settings > Privacy & Security > Accessibility*, then **restart that
app**. With Homebrew Python, `practice_pad.py` also needs `brew install python-tk`.

### Linux (Ubuntu/Debian)

```bash
sudo apt install python3-venv python3-tk python3-dev libegl1 libgles2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_models.py
python stages/stage1_webcam_test.py
```

Fedora: `sudo dnf install python3-tkinter python3-devel mesa-libEGL mesa-libGLES`.
Cursor control needs an **Xorg** session (see Troubleshooting for Wayland). If the
camera cannot be opened: `sudo usermod -aG video $USER`, then log out and in.

**Exact tested versions** (optional, for reproducibility):
`pip install mediapipe==1.0.1 opencv-contrib-python==5.0.0.93 numpy==2.4.6 pyautogui==0.9.54`

**Run the automated tests:** `pip install -r requirements-dev.txt` then `python -m pytest`.

---

## 6. Stage 1: webcam test

File: [`stages/stage1_webcam_test.py`](stages/stage1_webcam_test.py). Needs OpenCV only.

```bash
python stages/stage1_webcam_test.py          # camera 0
python stages/stage1_webcam_test.py 1        # another camera
python stages/stage1_webcam_test.py clip.mp4 # a video file
```

What it shows: the core loop that every later stage keeps.

```python
cap = cv2.VideoCapture(source)
while True:
    ok, frame = cap.read()               # BGR image, shape (height, width, 3)
    frame = cv2.flip(frame, 1)           # mirror view
    ...                                  # FPS = smoothed 1 / (time between frames)
    cv2.imshow(WINDOW, frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):   # waitKey also lets the window redraw
        break
cap.release(); cv2.destroyAllWindows()   # in a finally: block, so it always runs
```

Expected: a mirrored video at 15-30 FPS (many laptop webcams deliver at most
30 FPS, and fewer in dim light) that closes on `Q`, `Esc`, `Ctrl+C` or the window's X.

---

## 7. Stage 2: face detection

File: [`stages/stage2_face_detection.py`](stages/stage2_face_detection.py). Press `F`
to switch detection on and off.

```python
options = vision.FaceDetectorOptions(
    base_options=mp_python.BaseOptions(model_asset_path="models/blaze_face_short_range.tflite"),
    running_mode=vision.RunningMode.VIDEO,
    min_detection_confidence=0.6)
detector = vision.FaceDetector.create_from_options(options)
result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
for d in result.detections:
    d.bounding_box            # origin_x, origin_y, width, height in PIXELS
    d.categories[0].score     # detection confidence 0..1
    d.keypoints               # 6 NORMALIZED points (eyes, nose, mouth, ears)
```

**Is face detection needed for cursor control?** No: the cursor is driven by
the hand alone, and the app works with `--no-face`. It is kept because it
provides two real safety benefits (Stage 7): control is suspended when nobody is
in front of the laptop, and you get "move closer / move back" hints. It is cheap
(about 2-3 ms per 640x480 frame on a 4-core CPU, versus about 10 ms for hand
tracking) and runs only on every 2nd frame. In the MVP it lives in its own
module (`face_tracker.py`), and if the model is missing the app warns and
continues without it.

---

## 8. Stage 3: hand tracking

File: [`stages/stage3_hand_tracking.py`](stages/stage3_hand_tracking.py).

```python
options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6)
result = landmarker.detect_for_video(mp_image, timestamp_ms)
for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
    landmarks[8].x, landmarks[8].y       # index fingertip, normalized
    handedness[0].category_name          # "Left"/"Right" (swap it for a mirrored image)
    handedness[0].score                  # certainty of Left vs Right
```

It draws all 21 landmarks and the 21 connections, labels the important points
(wrist, fingertips, index knuckle), and prints handedness with its score.

**Two hands:** set `NUM_HANDS = 2` (or `python main.py --hands 2`). Both hands
are drawn. For control, the MVP's `select_hand()` picks one hand: the
`CONTROL_HAND` you set ("Right"/"Left"), otherwise the hand nearest to where the
controlling hand was last frame (so control does not jump between hands),
otherwise the largest hand. A two-hand scheme (e.g. one hand moves, the other
clicks) is listed in section 15. Tracking two hands costs roughly one more
landmark-model run per frame.

---

## 9. Stage 4: cursor movement

File: [`stages/stage4_cursor_movement.py`](stages/stage4_cursor_movement.py).
It starts **paused**: press `P` to start. Use `--dry-run` to watch without moving
anything.

As you asked for this stage, it uses the **index fingertip (landmark 8)**. The
steps each frame (full explanation in sections 2.5-2.6):

```python
target   = map_to_screen(tip.x, tip.y, screen_w, screen_h)       # control rectangle -> screen
smoothed = smoothed + ALPHA * (target - smoothed)                 # exponential smoothing
smoothed = limit_step(previous, smoothed, max_step)               # no teleporting
smoothed = clamp(smoothed, 2, size - 3)                           # never a fail-safe corner
if moved >= MIN_MOVE_PX: pyautogui.moveTo(round(x), round(y))
```

Conditions for moving: the hand has been tracked for 5 consecutive frames, you
are not paused, and you have not touched the real mouse (that pauses at once).
When the hand disappears, the counter resets. When it comes back, the cursor
continues from where it really is.

---

## 10. Complete MVP implementation (Stages 5-7)

Run `python main.py --help` for all options:

```bash
python main.py --dry-run                 # simulate everything (always start here)
python main.py                           # live, starts paused
python main.py --no-face --no-drag       # switch features off
python main.py --source recording.mp4    # replay a video (deterministic timing)
python main.py --camera 1 --hands 2
```

The preview shows: the mode (LIVE/DRY RUN) and state (ACTIVE/PAUSED/SUSPENDED/
WAITING), FPS and processing time, the recognised gesture, the last action,
live **pinch meters** with the press (green) and release (orange) thresholds,
the control rectangle, a mini-map of the screen with the (real or simulated)
cursor, the open-palm progress bar, face box and positioning hints.

**Stage 5, clicking** (`LeftClickDragGesture`): pinch distance normalized by
palm length, hysteresis (0.25/0.40), 60 ms confirmation, one click per pinch
sent **on release**, a 0.35 s cooldown, and the cursor frozen while the pinch is
closing. The pinch meters show the live value.

**Stage 6, right click, drag and scroll:** each is an independent class
(`RightClickGesture`, the drag part of `LeftClickDragGesture`, and
`ScrollGesture`) with its own settings. `GestureController` gives the hand to
one gesture at a time (priority: left/drag, then right, then scroll) and resets
the others. Disable any of them with `ENABLE_RIGHT_CLICK`, `ENABLE_DRAG`,
`ENABLE_SCROLL`, `ENABLE_PALM_PAUSE` or `--no-right-click`, `--no-drag`,
`--no-scroll`.

**Stage 7, face-aware improvements** (all optional, `--no-face`):

* control is **suspended** after 1 s without a face and resumes when you return,
* after 10 s without a face it is fully **PAUSED** (you left: resume by hand),
* "move closer / move back / centre yourself" hints from the face box size and position,
* a positioning check only: no identity, no storage.

**Dry-run mode** (`--dry-run` or `DRY_RUN = True`): everything runs, including
the state machines, mapping and smoothing, but `CursorController` only records
what it *would* do ("LEFT CLICK (dry run)") and moves the mini-map dot. If
PyAutoGUI cannot start (e.g. no desktop session), dry run still works with an
assumed 1920x1080 screen.

**Single-file version:** [`single_file_mouse.py`](single_file_mouse.py) is one
self-contained script (hand model only) with move, left click (on release, fist
cancels), right click, scroll, open-palm / `P` pause, dry run, manual-override
pause and the fail-safe. It leaves out face detection and drag, and counts
confirmation in frames instead of seconds.

**Exception handling:** camera unavailable or permission denied (`CameraError`
with per-OS hints), cursor library unavailable (`CursorBackendError`, including
the Linux `tkinter` exit and missing `DISPLAY`), invalid screen dimensions
(validated positive integers), tracking failure or missing models
(`TrackerError`, with the `libEGL`/Intel-Mac hints), `PermissionError`, invalid
settings (`config.validate()` lists every problem before anything starts), the
PyAutoGUI fail-safe (turned into a pause), and `Ctrl+C`. Cleanup runs in
`finally:` on every path.

---

## 11. Gesture logic

**Recommended change from your default: the cursor anchor.** You suggested
the index fingertip. When you pinch, the fingertip itself travels a few
centimetres toward the thumb. With the default mapping (about 5 screen pixels
per camera pixel), that easily adds up to 100+ screen pixels, so the click lands
away from where you aimed. The MVP therefore steers with the
**index knuckle (landmark 5)**, which barely moves during a pinch. You still
point with your index finger and move your hand. Positioning and gesturing are
decoupled, and clicks land on target. Stage 4 uses the fingertip as requested,
and `CURSOR_LANDMARK = 8` restores it in the MVP.

Timings below assume 30 FPS. "Palm" = palm length (wrist to middle knuckle).
Common protections for **all** gestures: nothing happens until the hand has been
tracked for 5 frames. A hand whose palm is shorter than 8 % of the frame height is ignored. While
paused or suspended only the palm toggle works. After any reset the pinch state
machines start in `WAIT_OPEN`.

| Gesture | Landmarks | Rule | Thresholds | Min. duration | Cooldown | False-activation guards | If tracking is lost |
|---|---|---|---|---|---|---|---|
| **Move** | 5 (anchor), 0, 6, 8, 10, 12 | index extended **and** middle not extended; no other gesture active | extended: `d(0,tip) > 1.15 d(0,pip)` | 5 stable frames | none | fist/palm/pinch/scroll poses do not move; jump > 0.2 frame resets; step limit; 2 px dead zone | movement stops at once; resumes from the real cursor position after 5 stable frames |
| **Left click** | 4, 8, 12 | `d(4,8)/palm < 0.25` to close, `> 0.40` to open; click on open | 0.25 / 0.40 | closed and opened 60 ms each; held < 0.5 s | 0.35 s | must see open fingers first; `d(4,8) <= d(4,12)` (not a right-click pinch); fist cancels; cursor frozen while pinching | pinch cancelled, no click |
| **Drag** | 4, 8 | same pinch held >= `DRAG_HOLD_S` -> button down; opens -> button up | 0.25 / 0.40 | 0.5 s hold; 60 ms release confirmation | - | fist ends the drag; only one gesture at a time | button held through dropouts < 0.15 s, then **released** |
| **Right click** | 4, 12, 6, 8 | `d(4,12)/palm < 0.25` **and** index extended **and** `d(4,8)/palm > 0.40`; fires once confirmed | 0.25 / 0.40 | 60 ms | 0.8 s | index-up requirement rules out fists and left pinches; must open (> 0.40) before the next one | cancelled; needs open fingers first |
| **Scroll** | 5 (height), 6-20 | index + middle extended, ring + pinky not, thumb away from both; each `SCROLL_STEP` (3 % of frame height) of vertical travel = 1 notch | step 0.03, max 3 notches/frame | pose held 0.2 s before scrolling | - | cursor frozen; flicker < 0.15 s tolerated; reference re-anchors after a flicker | scroll mode ends |
| **Pause / resume** | all tips, 4, 5 | all four fingers extended **and** thumb open (`d(4,5)/palm > 0.55`) | 0.55 | held 1.0 s (progress bar) | 1.5 s, and the palm must be lowered before it can toggle again | cursor never moves in this pose | hold timer resets |
| **Fist (rest)** | tips + MCPs | all four fingertips closer to the wrist than their knuckles | ratio 1.0 | immediate | - | no movement, cancels pinches | - |

**Why left click fires on release:** it is the only way to let one pinch mean
either "click" (short) or "drag" (held). It also lets a pinch that turns into a
fist be cancelled before anything is sent. The cost is about 60 ms more latency.
**Double click:** two quick pinches, less than the OS double-click time
(~0.5 s) apart, and at least 0.35 s apart because of the cooldown. Lower
`LEFT_CLICK_COOLDOWN_S` to about 0.25 if double clicks are hard to make.

---

## 12. Configuration guide

All settings are in [`config.py`](config.py), grouped by topic. `config.validate()`
rejects impossible values at start-up.

| Setting | Default | Raise it to... | Lower it to... |
|---|---|---|---|
| `CAMERA_INDEX` | 0 | pick another camera (1, 2 ...) | - |
| `CAMERA_WIDTH/HEIGHT` | 640x480 | get more detail (slower, rarely needed) | get more FPS on weak laptops (e.g. 480x360) |
| `CAMERA_BACKEND` | "auto" | - | use `"dshow"` if a Windows camera opens slowly |
| `MIRROR_FRAME` | True | - | False only for a camera that does not face you (e.g. top-down) |
| `MIN_HAND_DETECTION_CONFIDENCE` | 0.7 | ignore more false hands (patterned sleeves, posters) | find hands in poor light (more false detections) |
| `MIN_HAND_PRESENCE_CONFIDENCE` / `MIN_TRACKING_CONFIDENCE` | 0.6 | drop hands sooner when uncertain | keep tracking through blur/partial hands |
| `MAX_NUM_HANDS` | 1 | track 2 hands (slower) | - |
| `CONTROL_HAND` | "any" | - | "Right"/"Left" to ignore the other hand |
| `CONTROL_REGION` | (0.2, 0.2, 0.8, 0.65) | a larger rectangle = slower, more precise cursor, but more arm travel | a smaller rectangle = faster cursor, less travel, more jitter |
| `CURSOR_LANDMARK` | 5 (knuckle) | - | 8 = fingertip (more direct, but pinches drag the cursor) |
| `SMOOTHING_ALPHA` | 0.35 | ~0.5-0.6: snappier, more jitter | ~0.2: steadier, laggier |
| `MIN_MOVE_PX` | 2 | a still hand holds the cursor steadier | finer tiny movements |
| `MAX_STEP_FRACTION` | 0.08 | allow faster flicks | stronger anti-teleport |
| `STABLE_FRAMES_REQUIRED` | 5 | fewer accidental actions when the hand appears | faster start |
| `PINCH_PRESS_RATIO` | 0.25 | easier pinch (also more accidental clicks) | pinch must be tighter |
| `PINCH_RELEASE_RATIO` | 0.40 | a pinch must open wider to release (more robust) | releases sooner (keep >= press + 0.1) |
| `GESTURE_CONFIRM_S` | 0.06 | fewer false triggers, more latency | more responsive, noisier |
| `LEFT_CLICK_COOLDOWN_S` | 0.35 | block accidental repeats | allow faster double clicks |
| `DRAG_HOLD_S` | 0.5 | fewer accidental drags; long clicks still click | start drags sooner |
| `RIGHT_CLICK_COOLDOWN_S` | 0.8 | fewer repeated context menus | faster repeats |
| `SCROLL_STEP` | 0.03 | less sensitive scrolling | more sensitive |
| `SCROLL_NOTCHES_PER_STEP` | 1 | faster scrolling | - |
| `SCROLL_INVERT` | False | True = touch-screen ("natural") direction | - |
| `PALM_HOLD_S` | 1.0 | fewer accidental pauses | quicker pause |
| `START_PAUSED` | True | - | False (or `--start-active`) to start in control |
| `DRY_RUN` | False | True (or `--dry-run`) for simulation only | - |
| `ENABLE_FACE_DETECTION` / `REQUIRE_FACE_FOR_CONTROL` | True / True | - | False to drop the face check (saves ~1-2 ms/frame on average) |
| `MANUAL_OVERRIDE_PX` | 40 | fewer auto-pauses on a shaky desk | pause on smaller touches |

**Calibrating the pinch:** run `python main.py --dry-run`, watch the
`thumb-index` meter, and note the value when your fingers touch (typically
0.05-0.2) and when your hand is relaxed (typically 0.5-1.0). Put
`PINCH_PRESS_RATIO` a little above the touching value and `PINCH_RELEASE_RATIO`
about 0.1-0.15 higher.

---

## 13. Testing plan

**Automated (no camera):** `python -m pytest` runs 52 unit tests. They cover
every gesture state machine with synthetic hands (one click per pinch,
hysteresis, noise, cooldown, fist, drag and drag release on tracking loss,
right click, scroll direction, palm toggle, pause), the coordinate mapping (no
fail-safe corners, clamping, smoothing, step limit), the cursor controller
against a fake PyAutoGUI (manual override, fail-safe release, OS scroll units,
dry run), settings validation and face hints.

**Manual:** do tests 1-12 in `--dry-run` first, then repeat 4-12 live with
[`practice_pad.py`](practice_pad.py) open, so every click lands in a harmless window.

| # | Test | Procedure | Expected result | If it fails |
|---|---|---|---|---|
| 1 | Webcam access | `python stages/stage1_webcam_test.py` | mirrored video, 15-30 FPS, closes on Q/Esc/X | see "Camera" in section 14; try index 1; close Zoom/Teams |
| 2 | Face detection in different light | Stage 2 in daylight, lamp only, backlit (window behind you), and dim | box and score > 0.8 in good light; may drop out when backlit or dim | face a light source; avoid a window behind you; lower `MIN_FACE_DETECTION_CONFIDENCE` to 0.5 |
| 3 | Hand detection at different distances | Stage 3 at 30, 50, 80, 120 cm, palm facing the camera | 21 landmarks, stable at roughly 30-100 cm; the MVP ignores very distant hands (roughly beyond 1.3 m, `MIN_PALM_SIZE`) | better light; plain background; lower confidences to 0.5 |
| 4 | Cursor movement, no gestures | MVP, pointer pose, trace the screen edges slowly | cursor reaches all four edges (within 2 px); the mini-map matches; a still hand keeps the cursor still | adjust `CONTROL_REGION`; raise `MIN_MOVE_PX` if it shimmers |
| 5 | Slow and fast movement | slow circles, then quick side-to-side swipes | smooth path; slight lag on fast moves; no jumps | raise `SMOOTHING_ALPHA` for less lag, lower it for less jitter |
| 6 | Left-click accuracy | 20 pinches on the practice-pad button | >= 18/20 counted, exactly one click per pinch, cursor does not drift during a pinch | calibrate the pinch (section 12); keep `CURSOR_LANDMARK = 5` |
| 7 | Right-click accuracy | 10 thumb-middle pinches with the index up, on the orange area | >= 9/10, one per pinch, never while pinching the index | keep the index clearly straight; check the `thumb-middle` meter |
| 8 | Dragging | pinch and hold 0.5 s on the blue square, move, open | square follows and drops; "Drags" +1; no extra click | hold still until "dragging" appears; raise `PINCH_RELEASE_RATIO` if drops happen early |
| 9 | Scrolling | two fingers up, hold 0.2 s, move the hand up then down over the list | list scrolls up then down; cursor does not move | fold ring and pinky clearly; adjust `SCROLL_STEP`; set `SCROLL_INVERT` to taste |
| 10 | Loss of tracking | while moving and while dragging, drop your hand out of view | movement stops at once; a drag is released within ~0.15 s; on return the cursor glides, no jump | expected behaviour; check "hand lost" in the panel |
| 11 | Accidental gestures | type, drink, scratch your face, wave, make fists for 1 minute with control active | no clicks (check the practice-pad counters and the terminal log); pointing may move the cursor | raise `STABLE_FRAMES_REQUIRED` or `GESTURE_CONFIRM_S`; pause when not using it |
| 12 | Pause and emergency exit | P; open palm 1 s; move the real mouse; push the cursor into a corner with the touchpad; Q; Esc; window X; Ctrl+C | each pauses (banner shows why) or exits; after exit no mouse button stays held | click the preview window before pressing keys (focus) |
| 13 | CPU usage and FPS | watch the panel's FPS and ms/frame; Task Manager / Activity Monitor / `top` | >= 20 FPS and < 40 ms/frame on a typical laptop; the exit summary prints averages | use 480x360, `--no-face` or `FACE_DETECTION_INTERVAL = 3`, one hand, plug in the charger |
| 14 | Safe live testing | only after 1-13 pass in dry run: open `practice_pad.py`, keep other apps minimised and work saved, start paused, keep a hand on the real mouse | all clicks land in the practice pad | stay in dry run until the counters look right |

---

## 14. Troubleshooting

**Camera**
* *"Could not open camera"*: another app has the camera; wrong index; or
  permissions. **Windows:** Privacy > Camera > desktop apps; try
  `CAMERA_BACKEND = "dshow"`. **macOS:** allow your terminal under Privacy &
  Security > Camera, then restart it. **Linux:** `ls /dev/video*`, add yourself
  to the `video` group.
* *Opens but black / "returned no frames"*: usually permissions (macOS) or a
  privacy shutter/hotkey on the laptop.
* *Very slow start on Windows*: set `CAMERA_BACKEND = "dshow"`.

**MediaPipe**
* `module 'mediapipe' has no attribute 'solutions'`: you are running code from
  an old tutorial. This project does not use `solutions` (section 3).
* `libEGL.so.1: cannot open shared object file` (Linux): `sudo apt install libegl1 libgles2`.
* `No matching distribution found for mediapipe`: Python version or CPU not
  supported (Intel Mac with Python 3.13, 32-bit Python). Use 64-bit Python 3.11 or 3.12.
* `Model file not found`: run `python download_models.py`. Behind a proxy,
  download the two URLs in that file with a browser into `models/`.
* `Input timestamp must be monotonically increasing`: only if you write your own
  loop; always pass a larger timestamp than last time.
* Hand label says Left for your right hand: set `INVERT_HANDEDNESS = True`.

**Cursor control**
* *Nothing moves on macOS*: Accessibility permission for your terminal/IDE,
  then restart it. The app warns at start-up when it can tell, and pauses with
  "the OS ignored the cursor moves" if its first moves have no effect.
* *"PAUSED (fail-safe ...)" right after resuming*: the cursor is still in a
  screen corner. Move it out of the corner with the real mouse, then resume.
* *Nothing moves on Linux*: Wayland session. `echo $XDG_SESSION_TYPE` prints
  `wayland`. Log out, pick "Ubuntu on Xorg" (gear icon), log in. Remaining
  Wayland-only options are outside this project (e.g. `ydotool` with uinput).
* *Program quits at start with "You must install tkinter"*: `sudo apt install python3-tk python3-dev`.
* *Cannot click some Windows windows*: they run as Administrator. Run the
  terminal as Administrator only if you really need to.
* *Only part of the screen reachable / wrong position on a second monitor*:
  PyAutoGUI controls the primary monitor only.
* *Scroll too fast or too slow*: `SCROLL_STEP`, `SCROLL_NOTCHES_PER_STEP`, `SCROLL_UNITS_PER_NOTCH`.
* *Keys do nothing*: the OpenCV window must be focused. Click the preview first,
  or use the open palm.
* *Unexpected "PAUSED (you moved the real mouse)"*: your touchpad or desk is
  being bumped. Raise `MANUAL_OVERRIDE_PX`.

**Detection quality**
* Jittery cursor: lower `SMOOTHING_ALPHA`, raise `MIN_MOVE_PX`, use more light.
* Clicks not detected: calibrate the pinch (section 12); face the palm to the camera.
* Accidental clicks: lower `PINCH_PRESS_RATIO`, raise `GESTURE_CONFIRM_S`.
* Scroll/palm not recognised: spread or fold the fingers clearly; the palm
  should face the camera. Check the panel's "Gesture:" line.
* Low FPS: see test 13. Harmless Qt messages on Linux such as
  `QFontDatabase: Cannot find font directory` can be ignored.

---

## 15. Future improvements

The MVP above is deliberately simple. Ideas, roughly in order of value:

| Improvement | Why | Effort |
|---|---|---|
| **One Euro filter** instead of plain exponential smoothing | little smoothing when moving fast, strong smoothing when still: less lag *and* less jitter | small |
| **Calibration mode** | record your touching/relaxed pinch values and the reach of your hand, then write the thresholds and `CONTROL_REGION` for you | medium |
| **User profiles / sensitivity presets** | save calibrated settings per person (JSON) | small |
| **Adaptive thresholds by hand pose** | thresholds that also account for hand tilt (use `z` or several palm measurements) | medium |
| **Kalman filter** | smoothing plus short-term prediction to hide camera latency | medium |
| **Explicit finite-state machine for the whole app** | one diagram of ACTIVE/PAUSED/SUSPENDED/DRAGGING transitions (the gestures are already FSMs) | small |
| **Two-hand controls** | e.g. right hand moves, left hand clicks: no pinch drift at all | medium |
| **Global hotkey** (e.g. with `pynput`) | pause/quit even when the preview is not focused (needs Input-Monitoring permission on macOS; X11 only on Linux) | small |
| **Multi-monitor** | map to the virtual desktop with a screen-layout library; PyAutoGUI alone cannot | medium |
| **Accessibility UI** | large high-contrast status, audio/haptic click feedback, dwell-click (hover to click) for users who cannot pinch | medium |
| **Metrics log (no video)** | CSV of FPS, latency, gesture counts, false-click notes for experiments | small |
| **Lightweight ML gesture classifier** | train a small classifier (e.g. scikit-learn on the 21 normalized landmarks, or MediaPipe's `GestureRecognizer` customised with Model Maker) for more gestures | large |

---

## 16. Safety and privacy checklist

- [x] Frames are processed **locally**, in memory. The program makes no network
      connections (only `download_models.py` downloads the two models, once).
- [x] **No footage is stored**: no frames or video are written to disk, in any mode.
- [x] Face **detection** only, with no recognition and no identity data.
- [x] A **warning** about interference with normal computer use is printed at every start.
- [x] **Starts paused**, with **dry-run mode** for practice.
- [x] **Pause**: `P`/`Space`, open palm (1 s), auto-pause on real mouse movement,
      face absence, and the corner fail-safe.
- [x] **Emergency exit**: `Q`, `Esc`, window close button, `Ctrl+C`.
- [x] **No action without a confident hand**: MediaPipe thresholds, minimum hand
      size, 5 stable frames, jump detection.
- [x] **Smoothing + dead zone + step limit** prevent erratic movement and jumps.
- [x] **Debounced clicks**: hysteresis, confirmation time, one click per pinch, cooldowns.
- [x] A **held mouse button is always released** on pause, tracking loss, fail-safe, error or exit.
- [x] The cursor **never enters a screen corner** on its own, so the fail-safe stays in your hands.
- [x] **Only cursor actions** are sent (move, click, drag, scroll): no key presses or other OS actions.
- [ ] Before live use: save your work, close sensitive apps, practise with `practice_pad.py`,
      keep a hand near the real mouse.

**Limitations to keep in mind:** keyboard shortcuts need the preview window to be
focused. Linux Wayland sessions are not supported for cursor control. PyAutoGUI
controls the primary monitor only. The thresholds are sensible defaults, not
guaranteed values: lighting, camera quality and hand shape differ, so calibrate
before relying on it. This code has been tested on Linux (X11) with automated
tests and synthetic input; Windows and macOS behaviour follows the documented
PyAutoGUI/MediaPipe behaviour and should be verified with the testing plan above.
