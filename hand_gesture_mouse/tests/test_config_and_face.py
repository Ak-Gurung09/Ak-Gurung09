"""Tests for settings validation and the face helpers."""

import pytest

import config
from face_tracker import FaceData, main_face, positioning_hint
from hand_factory import settings


def test_default_config_is_valid():
    config.validate(config)


@pytest.mark.parametrize(
    "override",
    [
        {"CONTROL_REGION": (0.8, 0.2, 0.2, 0.65)},
        {"SMOOTHING_ALPHA": 0},
        {"PINCH_PRESS_RATIO": 0.5, "PINCH_RELEASE_RATIO": 0.4},
        {"CURSOR_LANDMARK": 21},
        {"CONTROL_HAND": "both"},
        {"FACE_DETECTION_INTERVAL": 0},
        {"SCREEN_EDGE_MARGIN_PX": 0},
    ],
)
def test_invalid_settings_are_reported(override):
    with pytest.raises(ValueError):
        config.validate(settings(**override))


def face(x, y, w, h, score=0.9):
    return FaceData(x, y, w, h, score, keypoints=[])


def test_main_face_is_the_largest():
    small, large = face(10, 10, 50, 50), face(200, 100, 150, 150)
    assert main_face([small, large]) is large
    assert main_face([]) is None


def test_positioning_hints():
    args = (640, 480, 0.12, 0.45)
    assert "No face" in positioning_hint(None, *args)
    assert "closer" in positioning_hint(face(300, 200, 40, 40), *args)
    assert "back" in positioning_hint(face(100, 50, 400, 400), *args)
    assert "Center" in positioning_hint(face(-20, 100, 150, 150), *args)
    assert positioning_hint(face(250, 120, 150, 150), *args) is None
