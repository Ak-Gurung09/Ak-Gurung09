"""
Gesture recognition: turns hand landmarks into mouse *intentions*.

This module never touches the real mouse. Each frame it answers:
  * May the cursor follow the hand right now?
  * Did a click, drag, scroll or pause-toggle just happen?
``cursor_controller.py`` carries those intentions out (or only displays them in
dry-run mode). Keeping the two apart makes this file easy to unit-test.

No gesture is decided from a single frame. Each one is a small state machine
with explicit states, confirmation times, hysteresis and cooldowns.

Gesture summary (details in README.md, "Gesture logic"):
  pointer     index finger up, middle finger down  -> cursor follows the index knuckle
  left click  thumb tip touches index tip briefly   -> click when the fingers open again
  drag        thumb-index pinch held DRAG_HOLD_S    -> button held until the pinch opens
  right click thumb tip touches middle tip, index up
  scroll      index + middle up, ring + pinky down  -> move the hand up/down
  pause       open palm held PALM_HOLD_S            -> toggles pause/resume
  fist        all fingers folded                    -> "resting": no movement, cancels pinches
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import config
from hand_tracker import (
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_TIP,
    WRIST,
    HandData,
)

# Events sent to the cursor controller.
LEFT_CLICK = "left_click"
RIGHT_CLICK = "right_click"
DRAG_START = "drag_start"
DRAG_END = "drag_end"

# (knuckle, middle joint, tip) for the four fingers. The thumb is handled separately.
FINGERS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_TIP),
}

# A palm shorter than one pixel means a broken detection (and would divide by ~0).
_MIN_VALID_PALM_PX = 1.0


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def palm_length_px(hand: HandData) -> float:
    points = hand.points_px()
    return distance(points[WRIST], points[MIDDLE_MCP])


# =============================================================================
# Per-frame geometry
# =============================================================================


@dataclass(frozen=True)
class HandFeatures:
    """Scale-independent measurements of one hand in one frame."""

    palm_size: float          # palm length / shorter frame side (how big the hand looks)
    index_pinch: float        # thumb tip <-> index tip, palm-relative
    middle_pinch: float       # thumb tip <-> middle tip, palm-relative
    extended: dict            # finger name -> True if clearly straight
    folded: dict              # finger name -> True if tucked into the palm
    thumb_open: bool          # thumb spread away from the hand
    anchor: tuple[float, float]  # normalized position that drives the cursor

    @property
    def is_fist(self) -> bool:
        return all(self.folded.values())

    @property
    def is_open_palm(self) -> bool:
        return self.thumb_open and all(self.extended.values())

    @property
    def is_pointer(self) -> bool:
        return self.extended["index"] and not self.extended["middle"]

    @property
    def is_two_finger(self) -> bool:
        e = self.extended
        return e["index"] and e["middle"] and not e["ring"] and not e["pinky"]


def compute_features(hand: HandData, settings=config) -> HandFeatures | None:
    """Measure the hand. All distances use pixel coordinates (correct aspect ratio)
    and are divided by the palm length, so they do not change with hand distance."""
    points = hand.points_px()
    palm_px = distance(points[WRIST], points[MIDDLE_MCP])
    shorter_side = min(hand.image_width, hand.image_height)
    if palm_px < _MIN_VALID_PALM_PX or shorter_side <= 0:
        return None

    wrist = points[WRIST]
    extended, folded = {}, {}
    for name, (mcp, pip, tip) in FINGERS.items():
        tip_to_wrist = distance(wrist, points[tip])
        extended[name] = tip_to_wrist > settings.FINGER_EXTENDED_RATIO * distance(wrist, points[pip])
        folded[name] = tip_to_wrist < settings.FINGER_FOLDED_RATIO * distance(wrist, points[mcp])

    return HandFeatures(
        palm_size=palm_px / shorter_side,
        index_pinch=distance(points[THUMB_TIP], points[INDEX_TIP]) / palm_px,
        middle_pinch=distance(points[THUMB_TIP], points[MIDDLE_TIP]) / palm_px,
        extended=extended,
        folded=folded,
        thumb_open=distance(points[THUMB_TIP], points[INDEX_MCP]) / palm_px > settings.THUMB_OPEN_RATIO,
        anchor=hand.point(settings.CURSOR_LANDMARK),
    )


def select_hand(
    hands: list[HandData],
    preferred: str = "any",
    previous_anchor: tuple[float, float] | None = None,
    anchor_index: int = INDEX_MCP,
) -> HandData | None:
    """Choose the controlling hand: the allowed hand nearest to the one used last
    frame (continuity), otherwise the biggest one (closest to the camera)."""
    if preferred.lower() != "any":
        hands = [h for h in hands if h.handedness.lower() == preferred.lower()]
    if not hands:
        return None
    if previous_anchor is not None:
        return min(hands, key=lambda h: distance(h.point(anchor_index), previous_anchor))
    return max(hands, key=palm_length_px)


class HeldTimer:
    """Measures how long a condition has been continuously true."""

    def __init__(self) -> None:
        self.since: float | None = None

    def update(self, active: bool, now: float) -> float:
        if not active:
            self.since = None
            return 0.0
        if self.since is None:
            self.since = now
        return now - self.since

    def reset(self) -> None:
        self.since = None


# =============================================================================
# Left click + drag (thumb-index pinch)
# =============================================================================


class PinchState(Enum):
    WAIT_OPEN = "waiting for open fingers"
    OPEN = "open"
    PINCHED = "pinched"
    DRAGGING = "dragging"


class LeftClickDragGesture:
    """Short pinch -> one left click, sent when the fingers open again.
    Pinch held for DRAG_HOLD_S -> mouse button down (drag) until the pinch opens.

    Clicking on *release* is what lets one gesture mean both "click" and "drag".
    """

    def __init__(self, settings=config) -> None:
        self.s = settings
        self.state = PinchState.WAIT_OPEN
        self._closed = HeldTimer()
        self._opened = HeldTimer()
        self._pinch_start = 0.0
        self._last_click = -math.inf
        self.note = ""  # why the last pinch did not click (shown on screen)

    @property
    def engaged(self) -> bool:
        """True while a pinch is closing, held or dragging."""
        return self.state in (PinchState.PINCHED, PinchState.DRAGGING) or self._closed.since is not None

    def reset(self) -> list[str]:
        events = [DRAG_END] if self.state is PinchState.DRAGGING else []
        self.state = PinchState.WAIT_OPEN
        self._closed.reset()
        self._opened.reset()
        return events

    def update(self, f: HandFeatures, now: float) -> list[str]:
        s = self.s
        # "closed" also requires the thumb to be nearer the index than the middle
        # finger (so it is not a right-click pinch) and the hand not to be a fist.
        closed = f.index_pinch < s.PINCH_PRESS_RATIO and f.index_pinch <= f.middle_pinch and not f.is_fist
        opened = f.index_pinch > s.PINCH_RELEASE_RATIO
        # Between the two thresholds nothing changes: that gap is the hysteresis.

        if self.state is PinchState.WAIT_OPEN:
            # After (re)acquiring the hand we must first see the fingers apart, so a
            # hand that appears already pinched can never click or drag.
            if opened:
                self.state = PinchState.OPEN
            return []

        if self.state is PinchState.OPEN:
            if closed and self._closed.update(closed, now) >= s.GESTURE_CONFIRM_S:
                self.state = PinchState.PINCHED
                self._pinch_start = self._closed.since
                self._closed.reset()
                self._opened.reset()
            elif not closed:
                self._closed.reset()
            return []

        if f.is_fist:
            self.note = "fist: pinch cancelled"
            return self.reset()

        released_for = self._opened.update(opened, now)

        if self.state is PinchState.PINCHED:
            if opened and released_for >= s.GESTURE_CONFIRM_S:
                held_for = self._opened.since - self._pinch_start
                self.state = PinchState.OPEN
                self._opened.reset()
                if held_for >= s.DRAG_HOLD_S:
                    self.note = "long pinch ignored"
                    return []
                if now - self._last_click < s.LEFT_CLICK_COOLDOWN_S:
                    self.note = "click blocked by cooldown"
                    return []
                self._last_click = now
                self.note = ""
                return [LEFT_CLICK]
            if s.ENABLE_DRAG and not opened and now - self._pinch_start >= s.DRAG_HOLD_S:
                self.state = PinchState.DRAGGING
                return [DRAG_START]
            return []

        # DRAGGING: keep the button down until the pinch has clearly opened.
        if opened and released_for >= s.GESTURE_CONFIRM_S:
            self.state = PinchState.OPEN
            self._opened.reset()
            return [DRAG_END]
        return []


# =============================================================================
# Right click (thumb-middle pinch with the index finger up)
# =============================================================================


class RightClickState(Enum):
    WAIT_OPEN = "waiting for open fingers"
    OPEN = "open"
    PRESSED = "pressed"


class RightClickGesture:
    """One right click per thumb-middle pinch, sent as soon as the pinch is confirmed."""

    def __init__(self, settings=config) -> None:
        self.s = settings
        self.state = RightClickState.WAIT_OPEN
        self._closed = HeldTimer()
        self._opened = HeldTimer()
        self._last_click = -math.inf
        self.note = ""

    @property
    def engaged(self) -> bool:
        return self.state is RightClickState.PRESSED or self._closed.since is not None

    def reset(self) -> list[str]:
        self.state = RightClickState.WAIT_OPEN
        self._closed.reset()
        self._opened.reset()
        return []

    def update(self, f: HandFeatures, now: float) -> list[str]:
        s = self.s
        # Index finger up and clearly away from the thumb: this cannot be confused
        # with the left-click pinch or with a fist.
        closed = (
            f.middle_pinch < s.PINCH_PRESS_RATIO
            and f.extended["index"]
            and f.index_pinch > s.PINCH_RELEASE_RATIO
        )
        opened = f.middle_pinch > s.PINCH_RELEASE_RATIO

        if self.state is RightClickState.WAIT_OPEN:
            if opened:
                self.state = RightClickState.OPEN
            return []

        if self.state is RightClickState.OPEN:
            if closed and self._closed.update(closed, now) >= s.GESTURE_CONFIRM_S:
                self.state = RightClickState.PRESSED
                self._closed.reset()
                self._opened.reset()
                if now - self._last_click < s.RIGHT_CLICK_COOLDOWN_S:
                    self.note = "right click blocked by cooldown"
                    return []
                self._last_click = now
                self.note = ""
                return [RIGHT_CLICK]
            if not closed:
                self._closed.reset()
            return []

        # PRESSED: the fingers must open before another right click is possible.
        if opened and self._opened.update(opened, now) >= s.GESTURE_CONFIRM_S:
            self.state = RightClickState.OPEN
            self._opened.reset()
        elif not opened:
            self._opened.reset()
        return []


# =============================================================================
# Scroll (index + middle up, move the hand vertically)
# =============================================================================


class ScrollState(Enum):
    IDLE = "idle"
    ENTERING = "entering"
    ACTIVE = "active"


class ScrollGesture:
    """While the two-finger pose is held, vertical hand travel becomes wheel notches.

    Travel is measured from a reference height; every SCROLL_STEP of travel sends
    one step and moves the reference, so holding the hand still stops scrolling.
    """

    def __init__(self, settings=config) -> None:
        self.s = settings
        self.state = ScrollState.IDLE
        self._pose = HeldTimer()
        self._lost = HeldTimer()
        self._reference_y: float | None = None

    @property
    def engaged(self) -> bool:
        return self.state is not ScrollState.IDLE

    def reset(self) -> list[str]:
        self.state = ScrollState.IDLE
        self._pose.reset()
        self._lost.reset()
        self._reference_y = None
        return []

    def update(self, f: HandFeatures, now: float) -> int:
        """Return the number of wheel notches to send this frame (+ = up, - = down)."""
        s = self.s
        pose = (
            f.is_two_finger
            and f.index_pinch > s.PINCH_RELEASE_RATIO
            and f.middle_pinch > s.PINCH_RELEASE_RATIO
        )

        if self.state is not ScrollState.ACTIVE:
            held = self._pose.update(pose, now)
            if not pose:
                self.state = ScrollState.IDLE
                return 0
            self.state = ScrollState.ENTERING
            if held >= s.SCROLL_ENTER_S:
                self.state = ScrollState.ACTIVE
                self._reference_y = f.anchor[1]
                self._lost.reset()
            return 0

        if not pose:
            if self._lost.update(True, now) >= s.SCROLL_EXIT_S:
                self.reset()
            else:
                self._reference_y = None  # re-anchor when the pose comes back
            return 0
        self._lost.reset()

        if self._reference_y is None:
            self._reference_y = f.anchor[1]
            return 0

        travel = self._reference_y - f.anchor[1]  # > 0 when the hand moved up (image y grows downward)
        steps = int(travel / s.SCROLL_STEP)       # int() truncates toward zero, same for up and down
        if steps == 0:
            return 0
        self._reference_y -= steps * s.SCROLL_STEP  # keep the remainder for the next frame
        limit = s.SCROLL_MAX_NOTCHES_PER_FRAME
        notches = max(-limit, min(limit, steps * s.SCROLL_NOTCHES_PER_STEP))
        return -notches if s.SCROLL_INVERT else notches


# =============================================================================
# Pause / resume (open palm held)
# =============================================================================


class PalmState(Enum):
    IDLE = "idle"
    HOLDING = "holding"
    WAIT_RELEASE = "waiting for release"


class PalmPauseGesture:
    """Hold an open palm for PALM_HOLD_S to toggle pause. One toggle per hold."""

    def __init__(self, settings=config) -> None:
        self.s = settings
        self.state = PalmState.IDLE
        self._held = HeldTimer()
        self._last_toggle = -math.inf
        self.progress = 0.0  # 0..1, drawn as a progress bar

    def reset(self) -> None:
        self.state = PalmState.IDLE
        self._held.reset()
        self.progress = 0.0

    def update(self, f: HandFeatures, now: float) -> bool:
        s = self.s
        palm = f.is_open_palm
        if self.state is PalmState.WAIT_RELEASE:
            # The palm must be lowered before it can toggle again.
            if not palm:
                self.state = PalmState.IDLE
            self.progress = 0.0
            return False

        cooled_down = now - self._last_toggle >= s.PAUSE_TOGGLE_COOLDOWN_S
        held = self._held.update(palm and cooled_down, now)
        if self._held.since is None:
            self.state = PalmState.IDLE
            self.progress = 0.0
            return False

        self.state = PalmState.HOLDING
        self.progress = min(held / s.PALM_HOLD_S, 1.0)
        if held >= s.PALM_HOLD_S:
            self.state = PalmState.WAIT_RELEASE
            self._held.reset()
            self._last_toggle = now
            self.progress = 0.0
            return True
        return False


# =============================================================================
# The controller that combines everything
# =============================================================================


@dataclass
class GestureOutput:
    features: HandFeatures | None = None
    anchor: tuple[float, float] | None = None
    tracking_ready: bool = False  # a stable, confidently tracked hand is present
    move_cursor: bool = False     # the cursor may follow ``anchor`` this frame
    events: list[str] = field(default_factory=list)
    scroll_notches: int = 0
    toggle_pause: bool = False
    palm_progress: float = 0.0
    label: str = "no hand"        # what the controller thinks you are doing


class GestureController:
    """Feeds each frame's hand into the gesture state machines and arbitrates
    between them: only one mouse gesture may be in progress at a time."""

    def __init__(self, settings=config) -> None:
        self.s = settings
        self.left = LeftClickDragGesture(settings)
        self.right = RightClickGesture(settings)
        self.scroll = ScrollGesture(settings)
        self.palm = PalmPauseGesture(settings)
        self.last_anchor: tuple[float, float] | None = None
        self._stable_frames = 0
        self._last_seen: float | None = None

    @property
    def dragging(self) -> bool:
        return self.left.state is PinchState.DRAGGING

    def _mouse_gestures(self) -> list:
        gestures = [self.left]  # priority order
        if self.s.ENABLE_RIGHT_CLICK:
            gestures.append(self.right)
        if self.s.ENABLE_SCROLL:
            gestures.append(self.scroll)
        return gestures

    def _reset_mouse_gestures(self) -> list[str]:
        return self.left.reset() + self.right.reset() + self.scroll.reset()

    def reset(self) -> list[str]:
        """Forget the current hand. Returns [DRAG_END] if a drag must be released."""
        events = self._reset_mouse_gestures()
        self.palm.reset()
        self.last_anchor = None
        self._stable_frames = 0
        self._last_seen = None
        return events

    def update(self, hand: HandData | None, now: float, paused: bool) -> GestureOutput:
        s = self.s
        out = GestureOutput()

        features = compute_features(hand, s) if hand is not None else None
        if features is not None and features.palm_size < s.MIN_PALM_SIZE:
            features = None
            out.label = "hand too far away"

        # --- No usable hand --------------------------------------------------
        if features is None:
            if self._last_seen is not None and now - self._last_seen <= s.LOST_HAND_GRACE_S:
                # Short dropout (e.g. motion blur): keep every state, including a held
                # drag, but do nothing this frame.
                out.label = "hand lost - waiting"
                out.tracking_ready = self._stable_frames >= s.STABLE_FRAMES_REQUIRED
            else:
                out.events = self.reset()  # releases a drag if one was active
            return out

        # --- Tracking quality gates -------------------------------------------
        if self.last_anchor is not None and distance(features.anchor, self.last_anchor) > s.MAX_ANCHOR_JUMP:
            out.events = self.reset()  # glitch or a different hand: start over
        self._last_seen = now
        self.last_anchor = features.anchor
        self._stable_frames += 1
        out.features = features
        out.anchor = features.anchor

        if self._stable_frames < s.STABLE_FRAMES_REQUIRED:
            out.label = "acquiring hand..."
            return out
        out.tracking_ready = True

        # --- Pause toggle (works while paused, so it can also resume) ---------
        if s.ENABLE_PALM_PAUSE:
            out.toggle_pause = self.palm.update(features, now)
            out.palm_progress = self.palm.progress

        if paused:
            out.events += self._reset_mouse_gestures()
            out.label = "paused"
            return out

        # --- Mouse gestures: one owner at a time -------------------------------
        gestures = self._mouse_gestures()
        owner = next((g for g in gestures if g.engaged), None)
        for gesture in [owner] if owner else gestures:
            if gesture is self.scroll:
                out.scroll_notches = gesture.update(features, now)
            else:
                out.events += gesture.update(features, now)
            if gesture.engaged:
                for other in gestures:
                    if other is not gesture:
                        out.events += other.reset()
                break

        # --- Cursor movement ---------------------------------------------------
        if self.dragging:
            out.move_cursor = True
        elif not any(g.engaged for g in gestures):
            out.move_cursor = features.is_pointer

        out.label = self._describe(features, out)
        return out

    def _describe(self, f: HandFeatures, out: GestureOutput) -> str:
        if self.dragging:
            return "dragging - open pinch to drop"
        if self.left.state is PinchState.PINCHED:
            return "pinch - release to click" + (", hold to drag" if self.s.ENABLE_DRAG else "")
        if self.left.engaged:
            return "pinch closing"
        if self.right.engaged:
            return "right click - open fingers"
        if self.scroll.state is ScrollState.ENTERING:
            return "scroll pose - hold"
        if self.scroll.state is ScrollState.ACTIVE:
            return "scrolling - move hand up/down"
        if out.palm_progress > 0:
            return f"open palm - hold to pause/resume ({out.palm_progress:.0%})"
        if f.is_fist:
            return "fist - resting"
        if out.move_cursor:
            return "pointer - moving"
        return "point with your index finger"
