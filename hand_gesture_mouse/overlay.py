"""
Everything drawn on the preview window. Drawing only: no decisions are made here.
"""

from __future__ import annotations

import time

import cv2

from hand_tracker import HAND_CONNECTIONS, LANDMARK_NAMES, HandData

# Colours are BGR (OpenCV's channel order).
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (80, 220, 80)
ORANGE = (0, 165, 255)
RED = (60, 60, 255)
YELLOW = (0, 230, 230)
CYAN = (230, 200, 0)
GREY = (160, 160, 160)

FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.5
TEXT_THICKNESS = 1
LINE_HEIGHT = 20
PANEL_PADDING = 8
PANEL_OPACITY = 0.55  # 0 = invisible background, 1 = solid black

LANDMARK_RADIUS = 4
ANCHOR_RADIUS = 8
MINIMAP_WIDTH = 120   # pixels on the preview; height follows the screen's aspect ratio
METER_WIDTH = 150
METER_HEIGHT = 10
METER_FULL_SCALE = 1.0  # palm-relative pinch distance shown at the right end of a meter


class FpsCounter:
    """Frames per second, smoothed so the number is readable."""

    SMOOTHING = 0.1  # weight of the newest frame

    def __init__(self) -> None:
        self._last = None
        self.fps = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                instant = 1.0 / dt
                self.fps = instant if self.fps == 0 else self.fps + self.SMOOTHING * (instant - self.fps)
        self._last = now
        return self.fps


def put_text(frame, text, origin, color=WHITE, scale=TEXT_SCALE, thickness=TEXT_THICKNESS, background=True) -> None:
    """Draw text, by default on a small black box so it is readable on any background.

    (The common trick of drawing a thicker black copy underneath does not line up
    in OpenCV 5, whose default font gets wider as the stroke gets thicker.)
    """
    if background:
        (text_w, text_h), baseline = cv2.getTextSize(text, FONT, scale, thickness)
        x, y = origin
        cv2.rectangle(frame, (x - 2, y - text_h - 2), (x + text_w + 2, y + baseline), BLACK, cv2.FILLED)
    cv2.putText(frame, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_panel(frame, lines: list[tuple[str, tuple]], origin=(10, 10)) -> None:
    """Semi-transparent box with one coloured line of text per entry."""
    if not lines:
        return
    widths = [cv2.getTextSize(text, FONT, TEXT_SCALE, TEXT_THICKNESS)[0][0] for text, _ in lines]
    x0, y0 = origin
    x1 = min(frame.shape[1] - 1, x0 + max(widths) + 2 * PANEL_PADDING)
    y1 = min(frame.shape[0] - 1, y0 + len(lines) * LINE_HEIGHT + PANEL_PADDING)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), BLACK, cv2.FILLED)
    cv2.addWeighted(overlay, PANEL_OPACITY, frame, 1 - PANEL_OPACITY, 0, dst=frame)
    for i, (text, color) in enumerate(lines):
        put_text(frame, text, (x0 + PANEL_PADDING, y0 + (i + 1) * LINE_HEIGHT), color, background=False)


def draw_hand(frame, hand: HandData, anchor_index: int | None = None, show_labels: bool = True) -> None:
    """Draw the 21 landmarks and their connections. ``anchor_index`` marks the
    landmark that drives the cursor (only for the hand that is in control)."""
    height, width = frame.shape[:2]
    points = [(int(x * width), int(y * height)) for x, y, _ in hand.landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], WHITE, 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        cv2.circle(frame, point, LANDMARK_RADIUS, CYAN, cv2.FILLED, cv2.LINE_AA)
        if show_labels and index in LANDMARK_NAMES:
            put_text(frame, f"{index} {LANDMARK_NAMES[index]}", (point[0] + 6, point[1] - 6), YELLOW, 0.4)
    if anchor_index is not None:
        cv2.circle(frame, points[anchor_index], ANCHOR_RADIUS, GREEN, 2, cv2.LINE_AA)

    wrist = points[0]
    put_text(
        frame,
        f"{hand.handedness} hand ({hand.handedness_score:.2f})",
        (wrist[0] - 60, min(height - 10, wrist[1] + 25)),
        GREEN,
    )


def draw_face(frame, face, color=GREEN) -> None:
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (face.x, face.y), (face.x + face.width, face.y + face.height), color, 2)
    for kx, ky in face.keypoints:
        cv2.circle(frame, (int(kx * width), int(ky * height)), 3, color, cv2.FILLED)
    label_y = min(height - 5, face.y + face.height + 18)  # below the box, away from the status panel
    put_text(frame, f"face {face.score:.2f}", (max(2, face.x), label_y), color)


def draw_control_region(frame, region, color) -> None:
    height, width = frame.shape[:2]
    left, top, right, bottom = region
    cv2.rectangle(frame, (int(left * width), int(top * height)), (int(right * width), int(bottom * height)), color, 1)


def draw_meter(frame, origin, label, value, press, release) -> None:
    """Horizontal bar for a pinch distance, with the press/release thresholds marked."""
    x, y = origin

    def to_px(v):
        return x + int(min(max(v / METER_FULL_SCALE, 0.0), 1.0) * METER_WIDTH)

    color = GREEN if value < press else (YELLOW if value < release else GREY)
    cv2.rectangle(frame, (x, y), (x + METER_WIDTH, y + METER_HEIGHT), GREY, 1)
    cv2.rectangle(frame, (x, y), (to_px(value), y + METER_HEIGHT), color, cv2.FILLED)
    cv2.line(frame, (to_px(press), y - 3), (to_px(press), y + METER_HEIGHT + 3), GREEN, 2)
    cv2.line(frame, (to_px(release), y - 3), (to_px(release), y + METER_HEIGHT + 3), ORANGE, 2)
    put_text(frame, f"{label} {value:.2f}", (x + METER_WIDTH + 8, y + METER_HEIGHT), WHITE, 0.45)


def draw_minimap(frame, cursor_position, screen_size, color) -> None:
    """Top-right rectangle representing the screen, with the cursor as a dot."""
    width = frame.shape[1]
    screen_w, screen_h = screen_size
    map_w = MINIMAP_WIDTH
    map_h = max(1, int(map_w * screen_h / screen_w))
    x0, y0 = width - map_w - 10, 25
    cv2.rectangle(frame, (x0, y0), (x0 + map_w, y0 + map_h), GREY, 1)
    cx = x0 + int(cursor_position[0] / screen_w * map_w)
    cy = y0 + int(cursor_position[1] / screen_h * map_h)
    cv2.circle(frame, (cx, cy), 4, color, cv2.FILLED)
    put_text(frame, "screen", (x0, y0 - 5), GREY, 0.4)


def draw_progress(frame, progress: float, label: str) -> None:
    """Bar across the bottom of the frame (used for the open-palm hold)."""
    height, width = frame.shape[:2]
    bar_y = height - 30
    cv2.rectangle(frame, (10, bar_y), (width // 2, bar_y + 12), GREY, 1)
    cv2.rectangle(frame, (10, bar_y), (10 + int(progress * (width // 2 - 10)), bar_y + 12), ORANGE, cv2.FILLED)
    put_text(frame, label, (10, bar_y - 6), ORANGE)


def draw_banner(frame, text: str, color=ORANGE) -> None:
    """Centered message near the top of the frame."""
    height, width = frame.shape[:2]
    (text_w, text_h), _ = cv2.getTextSize(text, FONT, 0.6, 2)
    x = max(5, (width - text_w) // 2)
    y = int(height * 0.12) + text_h
    cv2.rectangle(frame, (x - 6, y - text_h - 8), (x + text_w + 6, y + 8), BLACK, cv2.FILLED)
    put_text(frame, text, (x, y), color, 0.6, 2, background=False)
