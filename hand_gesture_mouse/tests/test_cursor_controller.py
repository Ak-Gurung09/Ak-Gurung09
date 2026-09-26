"""Unit tests for coordinate mapping, smoothing and the cursor controller (fake mouse)."""

import itertools

import pytest

import cursor_controller
from cursor_controller import (
    CursorBackendError,
    CursorController,
    ExponentialSmoother,
    FailSafeTriggered,
    clamp_to_screen,
    limit_step,
    map_to_screen,
    validate_screen_size,
)
from hand_factory import settings

REGION = (0.2, 0.2, 0.8, 0.65)
W, H = 1920, 1080


# --- pure functions --------------------------------------------------------------


def test_region_corners_map_to_screen_corners():
    assert map_to_screen((0.2, 0.2), REGION, W, H) == (0, 0)
    assert map_to_screen((0.8, 0.65), REGION, W, H) == pytest.approx((W - 1, H - 1))
    assert map_to_screen((0.5, 0.425), REGION, W, H) == pytest.approx(((W - 1) / 2, (H - 1) / 2))


def test_points_outside_region_are_clamped():
    assert map_to_screen((0.0, 0.0), REGION, W, H) == (0, 0)
    assert map_to_screen((1.0, 1.0), REGION, W, H) == pytest.approx((W - 1, H - 1))
    assert map_to_screen((-3.0, 7.0), REGION, W, H) == pytest.approx((0, H - 1))


def test_final_position_never_hits_a_failsafe_corner():
    corners = {(0, 0), (0, H - 1), (W - 1, 0), (W - 1, H - 1)}
    grid = [-0.5, 0.0, 0.2, 0.5, 0.8, 1.0, 1.5]
    for x, y in itertools.product(grid, grid):
        sx, sy = clamp_to_screen(*map_to_screen((x, y), REGION, W, H), W, H, margin=2)
        assert (int(round(sx)), int(round(sy))) not in corners
        assert 2 <= sx <= W - 3 and 2 <= sy <= H - 3


def test_limit_step():
    assert limit_step((0, 0), (3, 4), 10) == (3, 4)
    assert limit_step((0, 0), (30, 40), 10) == pytest.approx((6, 8))


def test_smoother():
    with pytest.raises(ValueError):
        ExponentialSmoother(0)
    raw = ExponentialSmoother(1.0)
    raw.update((0, 0))
    assert raw.update((10, 10)) == (10, 10)
    smooth = ExponentialSmoother(0.5)
    smooth.update((0, 0))
    assert smooth.update((10, 0)) == (5, 0)
    for _ in range(30):
        value = smooth.update((10, 0))
    assert value == pytest.approx((10, 0), abs=1e-3)


@pytest.mark.parametrize("size", [(0, 1080), (1920, -1), ("wide", 1080), (None, None)])
def test_invalid_screen_sizes_are_rejected(size):
    with pytest.raises(CursorBackendError):
        validate_screen_size(*size)


# --- controller with a fake PyAutoGUI ---------------------------------------------


class FakeFailSafe(Exception):
    pass


class FakeGui:
    FailSafeException = FakeFailSafe

    def __init__(self):
        self.FAILSAFE = True
        self.pos = (100, 100)
        self.calls = []
        self.raise_failsafe = False

    def size(self):
        return W, H

    def position(self):
        return self.pos

    def _record(self, name, *args, **kwargs):
        if self.raise_failsafe and self.FAILSAFE:
            raise FakeFailSafe()
        self.calls.append((name, args, kwargs))

    def moveTo(self, x, y):
        self._record("moveTo", x, y)
        self.pos = (x, y)

    def dragTo(self, x, y, **kwargs):
        self._record("dragTo", x, y, **kwargs)
        self.pos = (x, y)

    def click(self, **kwargs):
        self._record("click", **kwargs)

    def mouseDown(self, **kwargs):
        self._record("mouseDown", **kwargs)

    def mouseUp(self, **kwargs):
        self._record("mouseUp", **kwargs)

    def scroll(self, amount):
        self._record("scroll", amount)


@pytest.fixture
def gui(monkeypatch):
    fake = FakeGui()
    monkeypatch.setattr(cursor_controller, "load_pyautogui", lambda: fake)
    return fake


def names(fake):
    return [call[0] for call in fake.calls]


def test_live_moves_start_from_real_cursor_and_stay_on_screen(gui):
    cursor = CursorController(settings(PRINT_EVENTS=False))
    cursor.begin()
    assert cursor.position == (100, 100)
    for _ in range(100):
        cursor.move_to_anchor((0.99, 0.99))  # far outside the region: bottom-right
    x, y = gui.pos
    assert (x, y) == (W - 3, H - 3)          # 2 px margin, never the corner
    assert names(gui).count("moveTo") > 1


def test_no_teleport_on_first_move(gui):
    s = settings(PRINT_EVENTS=False, SMOOTHING_ALPHA=1.0)
    cursor = CursorController(s)
    cursor.begin()
    cursor.move_to_anchor((0.8, 0.65))
    max_step = s.MAX_STEP_FRACTION * (W ** 2 + H ** 2) ** 0.5
    assert ((gui.pos[0] - 100) ** 2 + (gui.pos[1] - 100) ** 2) ** 0.5 <= max_step + 1


def test_small_jitter_is_not_sent(gui):
    cursor = CursorController(settings(PRINT_EVENTS=False, SMOOTHING_ALPHA=1.0))
    cursor.begin()
    for _ in range(20):  # let the step limiter reach the target first
        cursor.move_to_anchor((0.5, 0.4))
    sent = len(gui.calls)
    cursor.move_to_anchor((0.5 + 1e-5, 0.4))  # far below MIN_MOVE_PX
    assert len(gui.calls) == sent


def test_manual_override_detection(gui):
    cursor = CursorController(settings(PRINT_EVENTS=False))
    cursor.begin()
    assert cursor.check_override() is None     # nothing sent yet
    cursor.move_to_anchor((0.5, 0.4))
    assert cursor.check_override() is None
    gui.pos = (gui.pos[0] + 200, gui.pos[1])   # the user grabbed the touchpad
    assert cursor.check_override() == "user"


def test_os_ignoring_our_moves_is_reported(gui):
    gui.moveTo = lambda x, y: None             # e.g. macOS without Accessibility permission
    cursor = CursorController(settings(PRINT_EVENTS=False))
    cursor.begin()
    cursor.move_to_anchor((0.7, 0.6))          # far from (100, 100): more than MANUAL_OVERRIDE_PX
    assert cursor.check_override() == "ignored"


def test_drag_uses_drag_events_and_is_always_released(gui):
    cursor = CursorController(settings(PRINT_EVENTS=False))
    cursor.begin()
    cursor.drag_start()
    cursor.move_to_anchor((0.6, 0.5))
    assert "dragTo" in names(gui)
    assert gui.calls[-1][2] == {"button": "left", "mouseDownUp": False}
    cursor.end()
    assert names(gui)[-1] == "mouseUp" and not cursor.button_down


def test_failsafe_becomes_emergency_stop_and_releases_button(gui):
    cursor = CursorController(settings(PRINT_EVENTS=False))
    cursor.begin()
    cursor.drag_start()
    gui.raise_failsafe = True
    with pytest.raises(FailSafeTriggered):
        cursor.move_to_anchor((0.6, 0.5))
    assert names(gui)[-1] == "mouseUp" and not cursor.button_down
    assert gui.FAILSAFE is True  # restored after the forced release


def test_scroll_uses_os_units(gui):
    cursor = CursorController(settings(PRINT_EVENTS=False, SCROLL_UNITS_PER_NOTCH=120))
    cursor.scroll(-2)
    assert gui.calls[-1] == ("scroll", (-240,), {})
    cursor.scroll(0)
    assert len(gui.calls) == 1


def test_dry_run_never_touches_the_mouse(monkeypatch):
    def unavailable():
        raise CursorBackendError("no display")

    monkeypatch.setattr(cursor_controller, "load_pyautogui", unavailable)
    s = settings(PRINT_EVENTS=False)
    cursor = CursorController(s, dry_run=True)
    assert (cursor.screen_w, cursor.screen_h) == s.FALLBACK_SCREEN_SIZE
    assert cursor.warnings
    cursor.begin()
    cursor.move_to_anchor((0.5, 0.4))
    cursor.left_click()
    cursor.drag_start()
    cursor.end()
    assert cursor.last_action == "DRAG START (dry run)"
    assert cursor.action_counts == {"LEFT CLICK": 1, "DRAG START": 1}
    assert not cursor.button_down


def test_live_mode_without_backend_raises(monkeypatch):
    def unavailable():
        raise CursorBackendError("no display")

    monkeypatch.setattr(cursor_controller, "load_pyautogui", unavailable)
    with pytest.raises(CursorBackendError):
        CursorController(settings(), dry_run=False)
