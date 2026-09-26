"""
Hand-gesture mouse: the complete MVP (Stages 1-7 combined).

    python main.py               live mode, starts PAUSED
    python main.py --dry-run     detect and display gestures, never touch the real cursor
    python main.py --help        all options

Data flow for every frame:
    Camera -> (mirror) -> FaceTracker (every Nth frame) -> HandTracker
           -> GestureController (what do you intend?) -> CursorController (do it)
           -> overlay (draw) -> preview window
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

import config
import overlay
from camera import Camera, CameraError, parse_source
from cursor_controller import CursorBackendError, CursorController, FailSafeTriggered
from face_tracker import FaceTracker, main_face, positioning_hint
from gesture_controller import DRAG_END, DRAG_START, LEFT_CLICK, RIGHT_CLICK, GestureController, select_hand
from hand_tracker import HandTracker, TrackerError

KEY_ESC = 27
QUIT_KEYS = {ord("q"), ord("Q"), KEY_ESC}
PAUSE_KEYS = {ord("p"), ord("P"), ord(" ")}
PROCESSING_TIME_SMOOTHING = 0.1  # weight of the newest frame in the on-screen ms/frame value

SAFETY_NOTICE = """
======================================================================
 HAND GESTURE MOUSE - please read
 * {mode}
 * Virtual mouse control can interfere with normal computer use and
   click things you did not intend. Save your work first.
 * Video is processed locally, in memory. Nothing is recorded or uploaded.
 * It starts {start}. Resume/pause: press P (preview window focused)
   or hold an open palm for {palm:.0f} s.
 * Emergency stop: Q or Esc (preview window focused), move your real
   mouse or touchpad (auto-pause), push the cursor into a screen
   corner (fail-safe), or press Ctrl+C in this terminal.
======================================================================
"""


class GestureMouseApp:
    def __init__(self, settings, source) -> None:
        self.s = settings
        self.source = source
        self.camera: Camera | None = None
        self.hands: HandTracker | None = None
        self.faces: FaceTracker | None = None
        self.cursor: CursorController | None = None
        self.gestures = GestureController(settings)
        self.paused = settings.START_PAUSED
        self.control_active = False

        self._frame_index = 0
        self._start_time: float | None = None
        self._last_face = None
        self._last_face_time: float | None = None
        self._banner_text = ""
        self._banner_until = 0.0
        self._window_seen = False
        self.fps = overlay.FpsCounter()
        self._processing_ms = 0.0
        self._total_processing_s = 0.0
        self._wall_start: float | None = None

    # ------------------------------------------------------------------ setup

    def setup(self) -> None:
        s = self.s
        # Cursor first: if the cursor library is missing we fail before the camera turns on.
        self.cursor = CursorController(s, dry_run=s.DRY_RUN)
        for warning in self.cursor.warnings:
            print(f"WARNING: {warning}")

        # MediaPipe's handedness matches an un-mirrored image, so swap it when mirroring.
        self.hands = HandTracker(
            s.HAND_MODEL_PATH,
            max_num_hands=s.MAX_NUM_HANDS,
            min_detection_confidence=s.MIN_HAND_DETECTION_CONFIDENCE,
            min_presence_confidence=s.MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=s.MIN_TRACKING_CONFIDENCE,
            swap_handedness=s.MIRROR_FRAME != s.INVERT_HANDEDNESS,
        )
        if s.ENABLE_FACE_DETECTION:
            try:
                self.faces = FaceTracker(s.FACE_MODEL_PATH, s.MIN_FACE_DETECTION_CONFIDENCE)
            except TrackerError as exc:
                print(f"WARNING: continuing WITHOUT face detection (presence check disabled): {exc}")

        self.camera = Camera(
            self.source, s.CAMERA_WIDTH, s.CAMERA_HEIGHT, s.CAMERA_FPS,
            s.CAMERA_BACKEND, s.MIRROR_FRAME, s.CAMERA_MAX_FAILED_READS,
        )
        self.camera.open()

        width, height = self.camera.frame_size
        print(f"Camera {width}x{height} | screen {self.cursor.screen_w}x{self.cursor.screen_h} | "
              f"{'DRY RUN' if self.cursor.dry_run else 'LIVE'} | face detection {'on' if self.faces else 'off'}")

        cv2.namedWindow(s.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        if s.PREVIEW_ALWAYS_ON_TOP:
            try:
                cv2.setWindowProperty(s.WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
            except (cv2.error, AttributeError):
                pass  # not supported by every OpenCV GUI backend

    # --------------------------------------------------------------- main loop

    def run(self) -> None:
        s = self.s
        self._wall_start = time.perf_counter()
        while True:
            frame, timestamp = self.camera.read()
            if frame is None:
                print("End of video file.")
                break

            started = time.perf_counter()
            self._process(frame, timestamp)
            elapsed = time.perf_counter() - started
            self._total_processing_s += elapsed
            self._processing_ms += PROCESSING_TIME_SMOOTHING * (elapsed * 1000 - self._processing_ms)
            self.fps.tick()

            cv2.imshow(s.WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in QUIT_KEYS:
                print("Quit key pressed.")
                break
            if key in PAUSE_KEYS:
                self.set_paused(not self.paused, "P key", timestamp)
            if self._window_closed():
                print("Preview window closed.")
                break

    def _process(self, frame, now: float) -> None:
        s = self.s
        if self._start_time is None:
            self._start_time = now
        timestamp_ms = int(now * 1000)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe expects RGB, OpenCV gives BGR

        # --- Face presence (Stage 7) -------------------------------------------
        if self.faces is not None and self._frame_index % s.FACE_DETECTION_INTERVAL == 0:
            self._last_face = main_face(self.faces.process(rgb, timestamp_ms))
            if self._last_face is not None:
                self._last_face_time = now
        self._frame_index += 1

        suspended = False
        if self.faces is not None and s.REQUIRE_FACE_FOR_CONTROL:
            seen = self._last_face_time if self._last_face_time is not None else self._start_time
            absent_for = now - seen
            suspended = absent_for > s.FACE_ABSENT_TIMEOUT_S
            if absent_for > s.FACE_ABSENT_PAUSE_S and not self.paused:
                self.set_paused(True, "nobody in front of the camera", now)

        # --- Hands and gestures --------------------------------------------------
        hands = self.hands.process(rgb, timestamp_ms)
        hand = select_hand(hands, s.CONTROL_HAND, self.gestures.last_anchor, s.CURSOR_LANDMARK)
        out = self.gestures.update(None if suspended else hand, now, self.paused)
        if out.toggle_pause:
            self.set_paused(not self.paused, "open palm", now)

        active = out.tracking_ready and not self.paused and not suspended
        try:
            self._apply(out, active, now)
        except FailSafeTriggered:
            self.set_paused(True, "fail-safe: cursor hit a screen corner", now)

        self._draw(frame, hands, hand if not suspended else None, out, suspended, now)

    def _apply(self, out, active: bool, now: float) -> None:
        """Carry out the gesture output on the cursor (only while control is active)."""
        if active and not self.control_active:
            self.cursor.begin()
        elif not active and self.control_active:
            self.cursor.end()  # releases a held button
        self.control_active = active
        if not active:
            return

        override = self.cursor.check_override()
        if override == "user":
            self.set_paused(True, "you moved the real mouse", now)
            return
        if override == "ignored":
            self.set_paused(True, "the OS ignored the cursor moves - see README > Troubleshooting", now)
            print("  The cursor did not move. macOS: allow your terminal/IDE under Privacy & Security > "
                  "Accessibility and restart it. Linux: use an Xorg session, not Wayland.")
            return

        actions = {
            LEFT_CLICK: self.cursor.left_click,
            RIGHT_CLICK: self.cursor.right_click,
            DRAG_START: self.cursor.drag_start,
            DRAG_END: self.cursor.drag_end,
        }
        for event in out.events:
            actions[event]()
        if out.scroll_notches:
            self.cursor.scroll(out.scroll_notches)
        if out.move_cursor and out.anchor is not None:
            self.cursor.move_to_anchor(out.anchor)

    def set_paused(self, paused: bool, reason: str, now: float) -> None:
        if paused == self.paused:
            return
        self.paused = paused
        if paused and self.control_active:
            self.cursor.end()
            self.control_active = False
        message = f"{'PAUSED' if paused else 'RESUMED'} ({reason})"
        self._banner_text = message
        self._banner_until = now + self.s.BANNER_DURATION_S
        print(message)

    # ---------------------------------------------------------------- drawing

    def _draw(self, frame, hands, controlling_hand, out, suspended: bool, now: float) -> None:
        s = self.s
        height, width = frame.shape[:2]

        if suspended:
            state, color = "SUSPENDED - no face", overlay.RED
        elif self.paused:
            state, color = "PAUSED", overlay.ORANGE
        elif self.control_active:
            state, color = "ACTIVE", overlay.GREEN
        else:
            state, color = "WAITING FOR HAND", overlay.YELLOW

        overlay.draw_control_region(frame, s.CONTROL_REGION, color)
        if self._last_face is not None:
            overlay.draw_face(frame, self._last_face)
        for hand in hands:
            anchor = s.CURSOR_LANDMARK if hand is controlling_hand else None
            overlay.draw_hand(frame, hand, anchor, s.SHOW_LANDMARK_LABELS)

        lines = [
            (f"{'DRY RUN' if self.cursor.dry_run else 'LIVE'} | {state}", color),
            (f"FPS {self.fps.fps:4.1f} | {self._processing_ms:4.1f} ms/frame", overlay.WHITE),
            (f"Gesture: {out.label}", overlay.WHITE),
        ]
        if self.cursor.last_action:
            lines.append((f"Last action: {self.cursor.last_action}", overlay.CYAN))
        note = self.gestures.left.note or self.gestures.right.note
        if note:
            lines.append((f"Note: {note}", overlay.GREY))
        lines.append(("P pause/resume | Q/Esc quit", overlay.GREY))
        overlay.draw_panel(frame, lines)

        if out.features is not None:
            meter_y = 10 + len(lines) * overlay.LINE_HEIGHT + 2 * overlay.PANEL_PADDING + 10
            overlay.draw_meter(frame, (10, meter_y), "thumb-index", out.features.index_pinch,
                               s.PINCH_PRESS_RATIO, s.PINCH_RELEASE_RATIO)
            overlay.draw_meter(frame, (10, meter_y + overlay.LINE_HEIGHT), "thumb-middle",
                               out.features.middle_pinch, s.PINCH_PRESS_RATIO, s.PINCH_RELEASE_RATIO)

        if s.SHOW_MINIMAP:
            overlay.draw_minimap(frame, self.cursor.position, (self.cursor.screen_w, self.cursor.screen_h), color)
        if out.palm_progress > 0:
            overlay.draw_progress(frame, out.palm_progress, "Open palm: hold to " + ("resume" if self.paused else "pause"))
        if self.faces is not None:
            hint = positioning_hint(self._last_face, width, height, s.FACE_MIN_WIDTH, s.FACE_MAX_WIDTH)
            if hint:
                overlay.put_text(frame, hint, (10, height - 50), overlay.YELLOW)
        if now < self._banner_until:
            overlay.draw_banner(frame, self._banner_text)

    def _window_closed(self) -> bool:
        """True once the user closed the preview with the window's X button."""
        try:
            visible = cv2.getWindowProperty(self.s.WINDOW_NAME, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            return self._window_seen
        if visible >= 1:
            self._window_seen = True  # only trust "not visible" after it was visible once
            return False
        return self._window_seen

    # ---------------------------------------------------------------- cleanup

    def close(self) -> None:
        """Release everything. Safe to call even if setup() failed half-way."""
        if self.cursor is not None:
            try:
                self.cursor.end()  # never leave a mouse button held down
            except Exception as exc:
                print(f"WARNING: could not release the mouse button: {exc}")
        if self.camera is not None:
            self.camera.release()
        for tracker in (self.hands, self.faces):
            if tracker is not None:
                try:
                    tracker.close()
                except Exception as exc:
                    print(f"WARNING: could not close a MediaPipe model cleanly: {exc}")
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # lets some GUI backends finish closing the window

        if self._frame_index and self._wall_start is not None:
            wall = time.perf_counter() - self._wall_start
            print(f"Processed {self._frame_index} frames | average {self._frame_index / wall:.1f} FPS | "
                  f"average {1000 * self._total_processing_s / self._frame_index:.1f} ms of processing per frame")
        if self.cursor is not None and self.cursor.action_counts:
            counts = sorted(self.cursor.action_counts.items())
            print("Actions: " + ", ".join(f"{name}={count}" for name, count in counts))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Control the mouse cursor with hand gestures (webcam).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="detect gestures but never move or click the real cursor")
    mode.add_argument("--live", action="store_true", help="control the real cursor even if DRY_RUN = True in config.py")
    parser.add_argument("--source", help="camera index (0, 1, ...) or a video file to replay (default: CAMERA_INDEX)")
    parser.add_argument("--camera", type=int, help="shortcut for --source <index>")
    parser.add_argument("--hands", type=int, choices=(1, 2), help="maximum number of hands to track")
    parser.add_argument("--start-active", action="store_true", help="start with control ON instead of paused")
    parser.add_argument("--no-face", action="store_true", help="disable face detection and the face presence check")
    parser.add_argument("--no-drag", action="store_true", help="disable drag (long pinches are then ignored)")
    parser.add_argument("--no-right-click", action="store_true", help="disable right click")
    parser.add_argument("--no-scroll", action="store_true", help="disable scrolling")
    return parser.parse_args(argv)


def apply_args(args, settings) -> None:
    if args.dry_run:
        settings.DRY_RUN = True
    if args.live:
        settings.DRY_RUN = False
    if args.hands:
        settings.MAX_NUM_HANDS = args.hands
    if args.start_active:
        settings.START_PAUSED = False
    if args.no_face:
        settings.ENABLE_FACE_DETECTION = False
    if args.no_drag:
        settings.ENABLE_DRAG = False
    if args.no_right_click:
        settings.ENABLE_RIGHT_CLICK = False
    if args.no_scroll:
        settings.ENABLE_SCROLL = False


def main(argv=None) -> int:
    args = parse_args(argv)
    apply_args(args, config)
    try:
        config.validate(config)
    except ValueError as exc:
        print(exc)
        return 2

    if args.source is not None:
        source = parse_source(args.source)
    elif args.camera is not None:
        source = args.camera
    else:
        source = config.CAMERA_INDEX

    print(SAFETY_NOTICE.format(
        mode=("DRY RUN: gestures are only simulated; the real cursor is never touched."
              if config.DRY_RUN else
              "LIVE: this program moves and clicks your REAL cursor. Try --dry-run first."),
        start="PAUSED" if config.START_PAUSED else "ACTIVE",
        palm=config.PALM_HOLD_S,
    ))

    app = GestureMouseApp(config, source)
    try:
        app.setup()
        app.run()
    except (CameraError, TrackerError, CursorBackendError) as exc:
        print(f"\nERROR: {exc}")
        return 1
    except PermissionError as exc:
        print(f"\nERROR: permission denied ({exc}). See README > Troubleshooting > Permissions.")
        return 1
    except KeyboardInterrupt:
        print("\nStopped with Ctrl+C.")
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
