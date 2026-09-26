"""Unit tests for the gesture state machines (no camera, no MediaPipe, no mouse)."""

import pytest

from gesture_controller import (
    DRAG_END,
    DRAG_START,
    LEFT_CLICK,
    RIGHT_CLICK,
    GestureController,
    compute_features,
    select_hand,
)
from hand_factory import (
    FRAME_H,
    fist,
    left_pinch,
    make_hand,
    open_palm,
    pointer,
    right_pinch,
    settings,
    two_finger,
)

DT = 1 / 30  # 30 FPS


class Clock:
    def __init__(self):
        self.now = 0.0

    def tick(self):
        self.now += DT
        return self.now


def feed(controller, clock, hand, frames, paused=False):
    """Feed the same hand for ``frames`` frames; return all outputs."""
    return [controller.update(hand, clock.tick(), paused) for _ in range(frames)]


def events(outputs):
    return [event for out in outputs for event in out.events]


def frames_for(seconds):
    return int(round(seconds / DT)) + 1


@pytest.fixture
def s():
    return settings()


@pytest.fixture
def ready(s):
    """A controller that has already acquired a pointing hand."""
    controller, clock = GestureController(s), Clock()
    feed(controller, clock, pointer(), s.STABLE_FRAMES_REQUIRED + 2)
    return controller, clock


# --- geometry ---------------------------------------------------------------


def test_features_classify_basic_poses(s):
    f = compute_features(pointer(), s)
    assert f.is_pointer and not f.is_fist and not f.is_open_palm and not f.is_two_finger
    assert compute_features(open_palm(), s).is_open_palm
    assert compute_features(fist(), s).is_fist
    assert compute_features(two_finger(), s).is_two_finger
    assert compute_features(left_pinch(0.05), s).index_pinch == pytest.approx(0.05, abs=1e-6)
    assert compute_features(right_pinch(0.05), s).middle_pinch == pytest.approx(0.05, abs=1e-6)


def test_pinch_ratio_does_not_depend_on_hand_distance(s):
    near = compute_features(left_pinch(0.3, scale=1.4), s)
    far = compute_features(left_pinch(0.3, scale=0.6), s)
    assert near.index_pinch == pytest.approx(far.index_pinch, abs=1e-6)
    assert near.palm_size > far.palm_size


# --- tracking gates -----------------------------------------------------------


def test_nothing_happens_before_the_hand_is_stable(s):
    controller, clock = GestureController(s), Clock()
    outs = feed(controller, clock, pointer(), s.STABLE_FRAMES_REQUIRED - 1)
    assert not any(o.tracking_ready or o.move_cursor for o in outs)
    assert feed(controller, clock, pointer(), 1)[0].tracking_ready


def test_pointer_moves_but_fist_and_palm_do_not(ready):
    controller, clock = ready
    assert feed(controller, clock, pointer(), 1)[0].move_cursor
    assert not feed(controller, clock, fist(), 1)[0].move_cursor
    assert not feed(controller, clock, open_palm(), 1)[0].move_cursor


def test_distant_hand_is_ignored(s):
    controller, clock = GestureController(s), Clock()
    outs = feed(controller, clock, pointer(scale=0.3), 20)
    assert not any(o.tracking_ready for o in outs)
    assert outs[-1].label == "hand too far away"


def test_anchor_jump_forces_reacquisition(ready, s):
    controller, clock = ready
    out = feed(controller, clock, pointer(offset=(250, 0)), 1)[0]  # ~0.39 of the frame in one frame
    assert not out.tracking_ready and not out.move_cursor


# --- left click ----------------------------------------------------------------


def test_one_click_per_pinch_sent_on_release(ready):
    controller, clock = ready
    pinched = feed(controller, clock, left_pinch(), 6)
    assert events(pinched) == []                      # nothing while held
    assert not any(o.move_cursor for o in pinched)    # cursor frozen while pinching
    released = feed(controller, clock, pointer(), 6)
    assert events(released) == [LEFT_CLICK]


def test_no_click_from_single_noisy_frame(ready):
    controller, clock = ready
    feed(controller, clock, left_pinch(), 1)
    assert events(feed(controller, clock, pointer(), 10)) == []


def test_hand_that_appears_pinched_cannot_click(s):
    controller, clock = GestureController(s), Clock()
    feed(controller, clock, left_pinch(), s.STABLE_FRAMES_REQUIRED + 10)
    assert events(feed(controller, clock, pointer(), 10)) == []
    feed(controller, clock, left_pinch(), 5)  # a fresh pinch after opening does click
    assert events(feed(controller, clock, pointer(), 5)) == [LEFT_CLICK]


def test_hysteresis_band_does_not_release(ready, s):
    controller, clock = ready
    feed(controller, clock, left_pinch(0.05), 5)
    band = (s.PINCH_PRESS_RATIO + s.PINCH_RELEASE_RATIO) / 2
    wobble = []
    for _ in range(5):  # ratio jitters inside the gap: still "pinched"
        wobble += feed(controller, clock, left_pinch(band), 1)
        wobble += feed(controller, clock, left_pinch(s.PINCH_PRESS_RATIO + 0.01), 1)
    assert events(wobble) == []
    assert events(feed(controller, clock, left_pinch(s.PINCH_RELEASE_RATIO + 0.1), 5)) == [LEFT_CLICK]


def test_cooldown_blocks_rapid_second_click(ready, s):
    s.LEFT_CLICK_COOLDOWN_S = 1.0
    controller, clock = ready
    out = []
    for _ in range(2):  # two complete pinches within ~0.6 s
        out += feed(controller, clock, left_pinch(), 4)
        out += feed(controller, clock, pointer(), 4)
    assert events(out) == [LEFT_CLICK]
    assert controller.left.note == "click blocked by cooldown"


def test_fist_never_clicks_or_drags(ready, s):
    controller, clock = ready
    out = feed(controller, clock, fist(), frames_for(s.DRAG_HOLD_S * 2))
    out += feed(controller, clock, pointer(), 10)
    assert events(out) == []


# --- drag ----------------------------------------------------------------------


def test_long_pinch_drags_and_release_drops(ready, s):
    controller, clock = ready
    held = feed(controller, clock, left_pinch(), frames_for(s.DRAG_HOLD_S) + 1)
    assert events(held) == [DRAG_START]
    moving = [controller.update(left_pinch(offset=(i, 0)), clock.tick(), False) for i in range(1, 10)]
    assert all(o.move_cursor for o in moving)  # cursor follows the hand while dragging
    released = feed(controller, clock, pointer(offset=(9, 0)), 5)
    assert events(released) == [DRAG_END]      # and no extra click


def test_drag_survives_short_dropout_but_ends_when_hand_is_lost(ready, s):
    controller, clock = ready
    feed(controller, clock, left_pinch(), frames_for(s.DRAG_HOLD_S) + 1)
    assert controller.dragging
    blip = feed(controller, clock, None, 2)  # 67 ms < LOST_HAND_GRACE_S
    assert events(blip) == [] and all(o.tracking_ready for o in blip)
    feed(controller, clock, left_pinch(), 1)
    lost = feed(controller, clock, None, frames_for(s.LOST_HAND_GRACE_S) + 1)
    assert events(lost) == [DRAG_END]
    assert not controller.dragging


def test_long_pinch_is_ignored_when_drag_disabled(s):
    s.ENABLE_DRAG = False
    controller, clock = GestureController(s), Clock()
    feed(controller, clock, pointer(), s.STABLE_FRAMES_REQUIRED + 2)
    out = feed(controller, clock, left_pinch(), frames_for(s.DRAG_HOLD_S) + 5)
    out += feed(controller, clock, pointer(), 5)
    assert events(out) == []


# --- right click ---------------------------------------------------------------


def test_right_click_once_per_pinch(ready, s):
    controller, clock = ready
    out = feed(controller, clock, right_pinch(), 15)
    assert events(out) == [RIGHT_CLICK]
    assert not any(o.move_cursor for o in out)
    feed(controller, clock, pointer(), frames_for(s.RIGHT_CLICK_COOLDOWN_S))
    assert events(feed(controller, clock, right_pinch(), 5)) == [RIGHT_CLICK]


def test_right_click_can_be_disabled(s):
    s.ENABLE_RIGHT_CLICK = False
    controller, clock = GestureController(s), Clock()
    feed(controller, clock, pointer(), s.STABLE_FRAMES_REQUIRED + 2)
    assert events(feed(controller, clock, right_pinch(), 15)) == []


# --- scroll --------------------------------------------------------------------


def scroll_up_by(controller, clock, pixels, start_offset=0.0):
    outs = []
    for i in range(1, int(pixels) + 1):
        outs.append(controller.update(two_finger(offset=(0, start_offset - i)), clock.tick(), False))
    return outs


def test_scroll_up_and_down(ready, s):
    controller, clock = ready
    enter = feed(controller, clock, two_finger(), frames_for(s.SCROLL_ENTER_S) + 1)
    assert all(o.scroll_notches == 0 and not o.move_cursor for o in enter)

    step_px = s.SCROLL_STEP * FRAME_H
    up = scroll_up_by(controller, clock, 3 * step_px + 2)
    assert sum(o.scroll_notches for o in up) == 3 * s.SCROLL_NOTCHES_PER_STEP
    assert not any(o.move_cursor for o in up)

    # Moving back down 2.5 steps from the new reference gives exactly 2 steps.
    offset = -(int(3 * step_px + 2))
    down = [controller.update(two_finger(offset=(0, offset + i)), clock.tick(), False)
            for i in range(1, int(2.5 * step_px) + 2)]
    assert sum(o.scroll_notches for o in down) == -2 * s.SCROLL_NOTCHES_PER_STEP


def test_scroll_invert(s):
    s.SCROLL_INVERT = True
    controller, clock = GestureController(s), Clock()
    feed(controller, clock, pointer(), s.STABLE_FRAMES_REQUIRED + 2)
    feed(controller, clock, two_finger(), frames_for(s.SCROLL_ENTER_S) + 1)
    up = scroll_up_by(controller, clock, 2 * s.SCROLL_STEP * FRAME_H + 2)
    assert sum(o.scroll_notches for o in up) == -2 * s.SCROLL_NOTCHES_PER_STEP


def test_scroll_needs_the_pose_to_be_held(ready):
    controller, clock = ready
    feed(controller, clock, two_finger(), 2)  # shorter than SCROLL_ENTER_S
    feed(controller, clock, pointer(), 10)
    assert controller.scroll.state.name == "IDLE"


# --- pause gesture -------------------------------------------------------------


def test_open_palm_toggles_once_per_hold(ready, s):
    controller, clock = ready
    out = feed(controller, clock, open_palm(), frames_for(s.PALM_HOLD_S * 3))
    assert sum(o.toggle_pause for o in out) == 1
    feed(controller, clock, pointer(), frames_for(s.PAUSE_TOGGLE_COOLDOWN_S))
    out = feed(controller, clock, open_palm(), frames_for(s.PALM_HOLD_S) + 1)
    assert sum(o.toggle_pause for o in out) == 1


def test_paused_blocks_mouse_gestures_but_palm_still_resumes(ready, s):
    controller, clock = ready
    out = feed(controller, clock, left_pinch(), 5, paused=True)
    out += feed(controller, clock, pointer(), 5, paused=True)
    assert events(out) == [] and not any(o.move_cursor for o in out)
    palm = feed(controller, clock, open_palm(), frames_for(s.PALM_HOLD_S) + 1, paused=True)
    assert sum(o.toggle_pause for o in palm) == 1


# --- hand selection ------------------------------------------------------------


def test_select_hand(s):
    right = make_hand(handedness="Right")
    left = make_hand(handedness="Left", offset=(-150, 0), scale=1.2)
    assert select_hand([], "any") is None
    assert select_hand([right, left], "Right") is right
    assert select_hand([right, left], "left") is left
    assert select_hand([right], "Left") is None
    assert select_hand([right, left], "any") is left  # bigger (closer) hand wins
    assert select_hand([right, left], "any", previous_anchor=right.point(s.CURSOR_LANDMARK)) is right
