import threading
import time

import numpy as np

from config import BotConfig, Region
from fishing_bot.automatic_bobber_finder import (
    AutomaticBobberFinder,
    find_candidates,
    novelty_mask,
)


def frame_with_red_bobber(x: int = 30, y: int = 25) -> np.ndarray:
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    frame[y - 3:y + 3, x - 2:x + 2] = (10, 20, 240)
    return frame


class FakeCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames, self.index = frames, 0

    def capture(self, region: Region) -> np.ndarray:
        frame = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return frame


def test_novel_colour_component_is_found_after_cast() -> None:
    config = BotConfig(auto_find_confirmation_frames=2, bobber_search_interval=0)
    before = np.zeros((60, 80, 3), dtype=np.uint8)
    after = frame_with_red_bobber()
    candidates = find_candidates(novelty_mask(before, after, config), config)
    assert candidates and abs(candidates[0].x - 30) <= 1

    finder = AutomaticBobberFinder(config, FakeCapture([after, after]))  # type: ignore[arg-type]
    result = finder.find(before, Region(100, 200, 80, 60), time.monotonic() + 1,
                         threading.Event(), lambda: True)
    assert result.detection.found
    assert abs(result.detection.x - 130) <= 1
    assert abs(result.detection.y - 225) <= 1
    assert result.template is not None and result.template.size > 0


def test_existing_red_object_is_not_novel() -> None:
    config = BotConfig()
    frame = frame_with_red_bobber()
    assert not find_candidates(novelty_mask(frame, frame, config), config)
