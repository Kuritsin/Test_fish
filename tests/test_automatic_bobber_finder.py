import threading
import time

import numpy as np
import cv2

from config import BotConfig, Region
from fishing_bot.automatic_bobber_finder import (
    AutomaticBobberFinder,
    find_candidates,
    novelty_mask,
)
from fishing_bot.calibration import CalibrationProfile


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


def test_permissive_calibration_returns_candidate_below_runtime_threshold() -> None:
    config = BotConfig(
        auto_find_confirmation_frames=1,
        auto_min_candidate_score=0.99,
        bobber_search_interval=0,
    )
    before = np.zeros((60, 80, 3), dtype=np.uint8)
    after = frame_with_red_bobber()
    finder = AutomaticBobberFinder(config, FakeCapture([after]))  # type: ignore[arg-type]
    finder.templates = []
    result = finder.find(
        before, Region(0, 0, 80, 60), time.monotonic() + 1,
        threading.Event(), lambda: True, permissive=True,
    )
    assert result.detection.found
    assert result.best_score > 0


def test_local_template_rejects_similar_water_candidates(tmp_path) -> None:
    before = np.zeros((80, 120, 3), dtype=np.uint8)
    after = before.copy()
    after[38:47, 72:82] = (60, 145, 220)
    after[32:41, 64:75] = (10, 20, 230)
    after[29:34, 74:78] = (230, 230, 230)
    template = after[27:50, 61:85]
    cv2.imwrite(str(tmp_path / "bobber.png"), template)
    # Distractor has plausible water colours but the wrong layout.
    after[55:64, 15:35] = (70, 130, 205)
    config = BotConfig(
        templates_dir=tmp_path,
        auto_find_confirmation_frames=1,
        auto_min_template_score=0.45,
        bobber_search_interval=0,
    )
    finder = AutomaticBobberFinder(config, FakeCapture([after]))  # type: ignore[arg-type]
    result = finder.find(before, Region(0, 0, 120, 80), time.monotonic() + 1,
                         threading.Event(), lambda: True)
    assert result.detection.found
    assert result.detection.x > 55


def test_nearby_bobber_parts_are_merged_into_one_candidate() -> None:
    mask = np.zeros((50, 80), dtype=np.uint8)
    mask[20:28, 25:34] = 255
    mask[22:32, 38:47] = 255
    config = BotConfig(auto_component_merge_gap=5)
    candidates = find_candidates(mask, config)
    assert len(candidates) == 1
    assert candidates[0].width >= 22


def test_calibration_profile_cannot_expand_candidates_beyond_hard_limit() -> None:
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[10:50, 10:65] = 255
    profile = CalibrationProfile((2,), 35, 30, 2, 10_000, 3)
    config = BotConfig(auto_find_hard_max_area=1800)
    assert not find_candidates(mask, config, profile)