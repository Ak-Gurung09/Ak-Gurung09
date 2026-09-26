"""
Moves, clicks and scrolls the real cursor with PyAutoGUI, or only simulates it
in dry-run mode. It also owns the coordinate mapping and smoothing.

Coordinate pipeline for one frame:
  anchor (normalized camera coords, already mirrored)
    -> map_to_screen()     control rectangle -> full screen, clamped
    -> ExponentialSmoother removes jitter
    -> limit_step()        no teleporting
    -> clamp_to_screen()   keeps a small margin (never an exact screen corner)
    -> pyautogui.moveTo()  only if it moved at least MIN_MOVE_PX
"""

from __future__ import annotations

import math
import os
import sys
import time

import config

# Scroll units equal to one physical wheel notch. PyAutoGUI passes its argument
# straight to the OS, and each OS uses a different unit.
_SCROLL_UNITS_PER_NOTCH = {"win32": 120, "darwin": 3}  # Windows: WHEEL_DELTA; macOS: lines
_DEFAULT_SCROLL_UNITS = 1                              # Linux/X11: one button-4/5 click


class CursorBackendError(RuntimeError):
    """The cursor-control library is unavailable or the screen size is unusable."""


class FailSafeTriggered(RuntimeError):
    """PyAutoGUI's fail-safe fired: the cursor was pushed into a screen corner."""


def default_scroll_units() -> int:
    return _SCROLL_UNITS_PER_NOTCH.get(sys.platform, _DEFAULT_SCROLL_UNITS)


def running_on_wayland() -> bool:
    return sys.platform.startswith("linux") and os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"


def cursor_backend_help() -> str:
    hints = ["Install it with:  pip install -r requirements.txt",
             "Or test without cursor control:  python main.py --dry-run"]
    if sys.platform.startswith("linux"):
        hints += [
            "Linux: PyAutoGUI needs tkinter:  sudo apt install python3-tk python3-dev",
            "Linux: PyAutoGUI needs an X11 desktop session (DISPLAY must be set). On Wayland, "
            "log out and choose 'Ubuntu on Xorg' (gear icon on the login screen).",
        ]
    elif sys.platform == "darwin":
        hints.append(
            "macOS: allow your terminal/IDE in System Settings > Privacy & Security > Accessibility."
        )
    return "\n  - " + "\n  - ".join(hints)


def load_pyautogui():
    """Import PyAutoGUI and switch off its built-in delay."""
    try:
        import pyautogui
    except SystemExit as exc:
        # PyAutoGUI's MouseInfo dependency calls sys.exit() on Linux when tkinter
        # is missing. Without this, the whole program would silently quit.
        raise CursorBackendError(f"PyAutoGUI could not start: {exc}" + cursor_backend_help()) from None
    except Exception as exc:  # ImportError, KeyError('DISPLAY'), Xlib connection errors, ...
        raise CursorBackendError(
            f"PyAutoGUI could not start ({type(exc).__name__}: {exc})." + cursor_backend_help()
        ) from exc

    # PyAutoGUI sleeps 0.1 s after EVERY call by default, which would cap the
    # program at about 10 FPS. We pace ourselves by the camera instead.
    pyautogui.PAUSE = 0
    # Keep the fail-safe on: pushing the cursor into a screen corner raises
    # FailSafeException, which this program treats as an emergency stop.
    pyautogui.FAILSAFE = True
    return pyautogui


def macos_accessibility_trusted() -> bool | None:
    """On macOS, return whether this process may control the mouse (None = unknown).

    Without the Accessibility permission macOS silently ignores synthetic mouse
    events, so warning early saves a lot of confusion.
    """
    if sys.platform != "darwin":
        return None
    try:
        import ctypes
        import ctypes.util

        path = ctypes.util.find_library("ApplicationServices")
        if not path:
            return None
        library = ctypes.cdll.LoadLibrary(path)
        library.AXIsProcessTrusted.restype = ctypes.c_bool
        library.AXIsProcessTrusted.argtypes = []
        return bool(library.AXIsProcessTrusted())
    except (OSError, AttributeError):
        return None


def validate_screen_size(width, height) -> tuple[int, int]:
    try:
        w, h = int(width), int(height)
    except (TypeError, ValueError):
        raise CursorBackendError(f"Invalid screen dimensions: {width!r} x {height!r}") from None
    if w <= 0 or h <= 0:
        raise CursorBackendError(f"Invalid screen dimensions: {w} x {h}")
    return w, h


# =============================================================================
# Pure helper functions (unit-tested in tests/test_cursor_mapping.py)
# =============================================================================


def map_to_screen(
    anchor: tuple[float, float],
    region: tuple[float, float, float, float],
    screen_width: int,
    screen_height: int,
) -> tuple[float, float]:
    """Map a normalized camera point inside ``region`` to screen pixels.

    Points outside the region are clamped to the screen edge, so you never need
    to reach the edge of the camera frame to reach the edge of the screen.
    """
    left, top, right, bottom = region
    u = (anchor[0] - left) / (right - left)
    v = (anchor[1] - top) / (bottom - top)
    u = min(max(u, 0.0), 1.0)
    v = min(max(v, 0.0), 1.0)
    return u * (screen_width - 1), v * (screen_height - 1)


def clamp_to_screen(x: float, y: float, screen_width: int, screen_height: int, margin: int) -> tuple[float, float]:
    return (
        min(max(x, margin), screen_width - 1 - margin),
        min(max(y, margin), screen_height - 1 - margin),
    )


def limit_step(current: tuple[float, float], target: tuple[float, float], max_step: float) -> tuple[float, float]:
    """Move from ``current`` toward ``target`` by at most ``max_step`` pixels."""
    dx, dy = target[0] - current[0], target[1] - current[1]
    length = math.hypot(dx, dy)
    if length <= max_step:
        return target
    scale = max_step / length
    return current[0] + dx * scale, current[1] + dy * scale


class ExponentialSmoother:
    """new = old + alpha * (target - old).

    alpha = 1 follows the target exactly; alpha = 0.35 moves 35 % of the way
    each frame, which averages out landmark jitter at the cost of a little lag.
    """

    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"SMOOTHING_ALPHA must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.value: tuple[float, float] | None = None

    def reset(self, value: tuple[float, float] | None = None) -> None:
        self.value = value

    def update(self, target: tuple[float, float]) -> tuple[float, float]:
        if self.value is None:
            self.value = target
        else:
            x, y = self.value
            self.value = (x + self.alpha * (target[0] - x), y + self.alpha * (target[1] - y))
        return self.value


# =============================================================================
# The controller
# =============================================================================


class CursorController:
    def __init__(self, settings=config, dry_run: bool = False) -> None:
        self.s = settings
        self.dry_run = dry_run
        self.warnings: list[str] = []
        self._gui = None

        if dry_run:
            size = self._screen_size_for_dry_run()
        else:
            self._gui = load_pyautogui()
            try:
                size = self._gui.size()
            except Exception as exc:
                raise CursorBackendError(f"Could not read the screen size: {exc}") from exc
            if running_on_wayland():
                self.warnings.append(
                    "Wayland session detected: the cursor may not move in native Wayland apps. "
                    "Use an Xorg session (see README > Troubleshooting)."
                )
            if macos_accessibility_trusted() is False:
                self.warnings.append(
                    "macOS Accessibility permission is missing: the cursor will not move. Allow your "
                    "terminal/IDE in System Settings > Privacy & Security > Accessibility, then restart it."
                )

        self.screen_w, self.screen_h = validate_screen_size(*size)
        self._smoother = ExponentialSmoother(settings.SMOOTHING_ALPHA)
        self._max_step = settings.MAX_STEP_FRACTION * math.hypot(self.screen_w, self.screen_h)
        self.scroll_units = settings.SCROLL_UNITS_PER_NOTCH or default_scroll_units()
        self.position = (self.screen_w / 2, self.screen_h / 2)  # where the cursor is (or would be)
        self.button_down = False
        self.last_action = ""
        self.action_counts: dict[str, int] = {}
        self._last_sent: tuple[int, int] | None = None
        self._begin_os_position: tuple[int, int] | None = None
        self._moves_confirmed = False  # has the OS applied at least one of our moves?

    def _screen_size_for_dry_run(self):
        try:
            return load_pyautogui().size()  # read-only: nothing is moved or clicked
        except Exception:
            width, height = self.s.FALLBACK_SCREEN_SIZE
            self.warnings.append(f"Could not read the screen size; dry run assumes {width}x{height}.")
            return width, height

    # --- control sessions ---------------------------------------------------

    def begin(self) -> None:
        """Control is starting: continue smoothly from where the cursor really is."""
        if self._gui is not None:
            x, y = self._gui.position()
            self.position = (float(x), float(y))
            self._begin_os_position = (int(x), int(y))
        self._smoother.reset(self.position)
        self._last_sent = None
        self._moves_confirmed = False

    def end(self) -> None:
        """Control is stopping: never leave a mouse button held down."""
        self.release_buttons()
        self._last_sent = None

    def check_override(self) -> str | None:
        """Compare the real cursor with where we last put it.

        Returns None if they match, "user" if the physical mouse/touchpad moved it,
        or "ignored" if the OS never applied our moves (typically the missing macOS
        Accessibility permission, or a Wayland session).
        """
        if self._gui is None or self._last_sent is None:
            return None
        x, y = self._gui.position()
        if math.hypot(x - self._last_sent[0], y - self._last_sent[1]) <= self.s.MANUAL_OVERRIDE_PX:
            self._moves_confirmed = True
            return None
        if not self._moves_confirmed and (int(x), int(y)) == self._begin_os_position:
            return "ignored"
        return "user"

    # --- movement -----------------------------------------------------------

    def move_to_anchor(self, anchor: tuple[float, float]) -> None:
        target = map_to_screen(anchor, self.s.CONTROL_REGION, self.screen_w, self.screen_h)
        smoothed = self._smoother.update(target)
        x, y = limit_step(self.position, smoothed, self._max_step)
        x, y = clamp_to_screen(x, y, self.screen_w, self.screen_h, self.s.SCREEN_EDGE_MARGIN_PX)
        self._smoother.reset((x, y))
        self.position = (x, y)

        ix, iy = int(round(x)), int(round(y))
        if self._last_sent is not None:
            moved = max(abs(ix - self._last_sent[0]), abs(iy - self._last_sent[1]))
            if moved < self.s.MIN_MOVE_PX:
                return
        if self._gui is not None:
            if self.button_down:
                # macOS only recognises a drag from "dragged" events, which dragTo sends.
                self._call(self._gui.dragTo, ix, iy, button="left", mouseDownUp=False)
            else:
                self._call(self._gui.moveTo, ix, iy)
        self._last_sent = (ix, iy)

    # --- buttons and wheel --------------------------------------------------

    def left_click(self) -> None:
        self._act("LEFT CLICK", lambda gui: gui.click(button="left"))

    def right_click(self) -> None:
        self._act("RIGHT CLICK", lambda gui: gui.click(button="right"))

    def drag_start(self) -> None:
        if self.button_down:
            return
        self._act("DRAG START", lambda gui: gui.mouseDown(button="left"))
        self.button_down = True

    def drag_end(self) -> None:
        if not self.button_down:
            return
        self._act("DRAG END", lambda gui: gui.mouseUp(button="left"))
        self.button_down = False

    def scroll(self, notches: int) -> None:
        if notches == 0:
            return
        direction = "UP" if notches > 0 else "DOWN"
        self._act(f"SCROLL {direction} x{abs(notches)}", lambda gui: gui.scroll(notches * self.scroll_units))

    def release_buttons(self) -> None:
        """Release a held button even if the fail-safe corner is active."""
        if not self.button_down:
            return
        self.button_down = False
        if self._gui is None:
            return
        previous = self._gui.FAILSAFE
        self._gui.FAILSAFE = False  # the fail-safe check would otherwise block this release
        try:
            self._gui.mouseUp(button="left")
        finally:
            self._gui.FAILSAFE = previous

    def _act(self, name: str, action) -> None:
        if self._gui is not None:
            self._call(action, self._gui)
        self.last_action = name + (" (dry run)" if self.dry_run else "")
        kind = name.split()[0] if name.startswith("SCROLL") else name
        self.action_counts[kind] = self.action_counts.get(kind, 0) + 1
        if self.s.PRINT_EVENTS:
            print(f"[{time.strftime('%H:%M:%S')}] {self.last_action}")

    def _call(self, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except self._gui.FailSafeException as exc:
            self.release_buttons()
            raise FailSafeTriggered("PyAutoGUI fail-safe: cursor reached a screen corner.") from exc
