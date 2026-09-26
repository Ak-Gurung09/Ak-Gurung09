"""
Builds synthetic 21-landmark hands for the unit tests, so gesture logic can be
tested without a camera or MediaPipe.

The hand is upright with the palm facing the camera, drawn in pixels on a
640x480 frame (palm length = 105 px), then normalized like MediaPipe output.
"""

from __future__ import annotations

from types import SimpleNamespace

import config
from hand_tracker import (
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PINKY_MCP,
    RING_MCP,
    THUMB_CMC,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
    HandData,
)

FRAME_W, FRAME_H = 640, 480
WRIST_PX = (320.0, 400.0)
PALM_PX = 105.0  # distance wrist -> middle knuckle below

KNUCKLES = {
    "index": (290.0, 300.0),
    "middle": (320.0, 295.0),
    "ring": (348.0, 300.0),
    "pinky": (372.0, 312.0),
}
KNUCKLE_INDEX = {"index": INDEX_MCP, "middle": MIDDLE_MCP, "ring": RING_MCP, "pinky": PINKY_MCP}

# (PIP, DIP, TIP) offsets from the knuckle for each finger state.
FINGER_SHAPES = {
    "extended": ((0, -45), (0, -70), (0, -90)),   # tip/PIP distance ratio ~1.3
    "bent": ((0, -40), (10, -48), (15, -35)),     # relaxed curl: not extended, not folded
    "folded": ((0, -25), (5, -10), (5, 15)),      # tip tucked back into the palm (fist)
}

THUMB_SHAPES = {
    "open": ((300, 385), (272, 365), (252, 345), (232, 325)),    # spread away from the hand
    "tucked": ((300, 385), (290, 365), (287, 348), (285, 330)),  # resting against the hand
    "fist": ((300, 385), (295, 360), (300, 340), (305, 320)),    # wrapped over folded fingers
}


def settings(**overrides) -> SimpleNamespace:
    """A copy of config.py's settings that a test may change freely."""
    values = {name: getattr(config, name) for name in dir(config) if name.isupper()}
    values.update(overrides)
    return SimpleNamespace(**values)


def make_hand(
    index="extended",
    middle="bent",
    ring="bent",
    pinky="bent",
    thumb="tucked",
    thumb_to_index=None,
    thumb_to_middle=None,
    offset=(0.0, 0.0),
    scale=1.0,
    handedness="Right",
) -> HandData:
    """Return a HandData. ``thumb_to_index=r`` puts the thumb tip ``r`` palm lengths
    to the left of the index tip (``thumb_to_middle`` likewise, to the right of the
    middle tip), which sets the pinch ratio exactly."""
    points = [None] * 21
    points[WRIST] = WRIST_PX

    fingers = {"index": index, "middle": middle, "ring": ring, "pinky": pinky}
    for name, state in fingers.items():
        kx, ky = KNUCKLES[name]
        mcp = KNUCKLE_INDEX[name]
        points[mcp] = (kx, ky)
        for i, (dx, dy) in enumerate(FINGER_SHAPES[state], start=1):
            points[mcp + i] = (kx + dx, ky + dy)

    for landmark, xy in zip((THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP), THUMB_SHAPES[thumb]):
        points[landmark] = xy
    if thumb_to_index is not None:
        tx, ty = points[INDEX_TIP]
        points[THUMB_TIP] = (tx - thumb_to_index * PALM_PX, ty)
    if thumb_to_middle is not None:
        tx, ty = points[MIDDLE_TIP]
        points[THUMB_TIP] = (tx + thumb_to_middle * PALM_PX, ty)

    wx, wy = WRIST_PX
    landmarks = []
    for x, y in points:
        x = wx + (x - wx) * scale + offset[0]
        y = wy + (y - wy) * scale + offset[1]
        landmarks.append((x / FRAME_W, y / FRAME_H, 0.0))
    return HandData(landmarks, handedness, 0.95, FRAME_W, FRAME_H)


# Named poses used throughout the tests.
def pointer(**kw):
    return make_hand(index="extended", **kw)


def open_palm(**kw):
    return make_hand(index="extended", middle="extended", ring="extended", pinky="extended", thumb="open", **kw)


def fist(**kw):
    return make_hand(index="folded", middle="folded", ring="folded", pinky="folded", thumb="fist", **kw)


def two_finger(**kw):
    return make_hand(index="extended", middle="extended", ring="folded", pinky="folded", **kw)


def left_pinch(ratio=0.05, **kw):
    return make_hand(index="bent", thumb_to_index=ratio, **kw)


def right_pinch(ratio=0.05, **kw):
    return make_hand(index="extended", middle="bent", thumb_to_middle=ratio, **kw)
