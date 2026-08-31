from pathlib import Path

import cv2
import numpy as np

from config import BotConfig, Region
from fishing_bot.bobber_detector import BobberDetection
from fishing_bot.hover_validator import HoverValidator, detect_fishing_tooltip


def tooltip_frame(text_width: int = 100) -> np.ndarray:
    frame = np.full((180, 320, 3), 120, dtype=np.uint8)
    cv2.rectangle(frame, (150, 115), (300, 160), (8, 8, 8), -1)
    cv2.rectangle(frame, (150, 115), (300, 160), (150, 150, 150), 2)
    for x in range(164, 164 + text_width, 8):
        cv2.rectangle(frame, (x, 130), (x + 4, 142), (0, 210, 255), -1)
    return frame


def test_detects_new_localization_independent_yellow_tooltip() -> None:
    before = np.full((180, 320, 3), 120, dtype=np.uint8)
    assert detect_fishing_tooltip(before, tooltip_frame(80)).found
    assert detect_fishing_tooltip(before, tooltip_frame(120)).found


def test_rejects_preexisting_tooltip() -> None:
    frame = tooltip_frame()
    assert not detect_fishing_tooltip(frame, frame.copy()).found


class FakeCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = iter(frames)

    def capture(self, _region: Region) -> np.ndarray:
        return next(self.frames)


class FakeInput:
    def __init__(self) -> None:
        self.points: list[tuple[int, int]] = []

    def move_to(self, x: int, y: int) -> bool:
        self.points.append((x, y))
        return True


def test_hover_validator_returns_exact_confirmed_point(tmp_path: Path) -> None:
    before = np.full((180, 320, 3), 120, dtype=np.uint8)
    input_controller = FakeInput()
    validator = HoverValidator(
        BotConfig(hover_tooltip_delay=0, hover_points_per_candidate=1),
        FakeCapture([before, tooltip_frame()]),  # type: ignore[arg-type]
        input_controller,  # type: ignore[arg-type]
    )
    candidate = BobberDetection(True, 50, 60, 0.55, "bobber.png", size=(20, 20))
    result = validator.validate([candidate], Region(0, 0, 320, 180), float("inf"), lambda: True, tmp_path)
    assert result.confirmed
    assert result.detection is not None
    assert (result.detection.x, result.detection.y) == input_controller.points[0]

