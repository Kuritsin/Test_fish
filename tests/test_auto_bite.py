import threading

import numpy as np

from config import BotConfig, Region
from fishing_bot.auto_bite_detector import (
    AutoBiteDetector,
    calibrate_colour,
    colour_mask,
    find_component,
)
from fishing_bot.bobber_detector import BobberDetection


def red_template() -> np.ndarray:
    template = np.zeros((12, 10, 3), dtype=np.uint8)
    template[3:9, 3:7] = (15, 25, 230)
    return template


def red_frame(y: int, x: int = 25) -> np.ndarray:
    frame = np.zeros((60, 60, 3), dtype=np.uint8)
    frame[y - 3:y + 3, x - 2:x + 2] = (15, 25, 230)
    return frame


class FakeCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames, self.index = frames, 0

    def capture(self, region: Region) -> np.ndarray:
        frame = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return frame


def test_colour_profile_and_component_follow_red_feather() -> None:
    config = BotConfig()
    profile = calibrate_colour(red_template(), config)
    assert profile is not None
    mask = colour_mask(red_frame(20), profile)
    component = find_component(mask, (25, 20), config)
    assert component is not None
    assert component[:2] == (24, 20)


def test_auto_detector_calibrates_bobbing_then_detects_drop() -> None:
    frames = [red_frame(20), red_frame(21), red_frame(20), red_frame(27)]
    config = BotConfig(auto_baseline_frames=3, auto_min_bite_drop=4,
                       visual_bite_fps=1000, auto_tracking_padding=25)
    detector = AutoBiteDetector(config, FakeCapture(frames))  # type: ignore[arg-type]
    result = detector.wait_for_bite(
        BobberDetection(True, 125, 220, 0.8, "red.png", size=(10, 12)),
        red_template(), Region(100, 190, 100, 100), 0.1,
        threading.Event(), lambda: True,
    )
    assert result.detected
    assert result.reason == "position_drop"
    assert result.vertical_shift >= 4


def test_auto_detector_does_not_treat_normal_bobbing_as_bite() -> None:
    frames = [red_frame(20), red_frame(21), red_frame(19), red_frame(21)]
    config = BotConfig(auto_baseline_frames=3, auto_min_bite_drop=4,
                       visual_bite_fps=1000, auto_tracking_padding=25)
    detector = AutoBiteDetector(config, FakeCapture(frames))  # type: ignore[arg-type]
    result = detector.wait_for_bite(
        BobberDetection(True, 125, 220, 0.8, "red.png", size=(10, 12)),
        red_template(), Region(100, 190, 100, 100), 0.01,
        threading.Event(), lambda: True,
    )
    assert not result.detected
