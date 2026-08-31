import threading

import numpy as np

from config import BotConfig, Region
from fishing_bot.bobber_detector import BobberDetection
from fishing_bot.visual_bite_detector import (
    VisualBiteDetector,
    make_bite_region,
    motion_score,
)


def test_bite_region_is_clamped_to_client_area() -> None:
    detection = BobberDetection(True, x=103, y=204, size=(10, 12))
    client = Region(100, 200, 80, 60)
    assert make_bite_region(detection, client, 20) == Region(100, 200, 28, 30)


def test_motion_score_ignores_unchanged_frame_and_detects_change() -> None:
    first = np.zeros((40, 40), dtype=np.uint8)
    second = first.copy()
    second[10:30, 10:30] = 255
    assert motion_score(first, first, 24) == 0.0
    assert motion_score(first, second, 24) > 0.2


class FakeCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.index = 0

    def capture(self, region: Region) -> np.ndarray:
        frame = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return frame


def frame_with_template(template: np.ndarray, y: int) -> np.ndarray:
    frame = np.zeros((60, 54), dtype=np.uint8)
    frame[y:y + template.shape[0], 22:22 + template.shape[1]] = template
    return frame


def test_visual_detector_confirms_vertical_bobber_movement() -> None:
    template = np.random.default_rng(7).integers(20, 255, (12, 10), dtype=np.uint8)
    still = frame_with_template(template, 20)
    frames = [still, still.copy(), still.copy(), frame_with_template(template, 25),
              frame_with_template(template, 30)]
    config = BotConfig(
        visual_bite_fps=1000,
        visual_baseline_frames=2,
        visual_motion_threshold=0.01,
        visual_motion_multiplier=2,
        visual_vertical_shift=4,
        visual_confirmation_frames=2,
    )
    detector = VisualBiteDetector(config, FakeCapture(frames))  # type: ignore[arg-type]
    detection = BobberDetection(True, x=127, y=226, template_name="bobber.png", size=(10, 12))
    result = detector.wait_for_bite(
        detection, template, Region(100, 200, 100, 100), 1,
        threading.Event(), lambda: True,
    )
    assert result.detected
    assert result.vertical_shift >= 4
    assert result.reason == "downward_transient"


def test_visual_detector_stops_when_input_is_no_longer_safe() -> None:
    frame = np.zeros((60, 54), dtype=np.uint8)
    detector = VisualBiteDetector(BotConfig(), FakeCapture([frame]))  # type: ignore[arg-type]
    detection = BobberDetection(True, x=127, y=226, size=(10, 12))
    result = detector.wait_for_bite(
        detection, np.ones((12, 10), dtype=np.uint8), Region(100, 200, 100, 100),
        1, threading.Event(), lambda: False,
    )
    assert not result.detected
    assert result.reason == "interrupted"


def test_disappearance_without_downward_motion_is_not_a_bite() -> None:
    template = np.random.default_rng(9).integers(20, 255, (12, 10), dtype=np.uint8)
    still = frame_with_template(template, 20)
    blank = np.zeros_like(still)
    config = BotConfig(visual_bite_fps=1000, visual_baseline_frames=1,
                       visual_tracker_lost_frames=2, visual_reacquire_radius=1)
    detector = VisualBiteDetector(config, FakeCapture([still, still, blank, blank, blank]))  # type: ignore[arg-type]
    result = detector.wait_for_bite(
        BobberDetection(True, 127, 226, 1.0, "bobber.png", size=(10, 12)),
        template, Region(100, 200, 100, 100), 0.05, threading.Event(), lambda: True,
    )
    assert not result.detected
    assert result.reason == "tracker_lost"


def test_loot_is_confirmed_after_template_disappears() -> None:
    template = np.random.default_rng(11).integers(20, 255, (12, 10), dtype=np.uint8)
    blank = np.zeros((60, 54), dtype=np.uint8)
    config = BotConfig(visual_bite_fps=1000, loot_confirmation_frames=2,
                       loot_disappearance_confidence=0.35)
    detector = VisualBiteDetector(config, FakeCapture([blank, blank]))  # type: ignore[arg-type]
    confirmed = detector.wait_for_disappearance(
        BobberDetection(True, 127, 226, size=(10, 12)), template,
        Region(100, 200, 100, 100), 0.1, threading.Event(), lambda: True,
    )
    assert confirmed
