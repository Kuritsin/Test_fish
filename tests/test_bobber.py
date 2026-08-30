from pathlib import Path

import cv2
import numpy as np

from config import Region
from fishing_bot.bobber_detector import BobberDetector


def test_best_template_and_global_coordinates(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    template = rng.integers(0, 255, (12, 10), dtype=np.uint8)
    weaker = rng.integers(0, 255, (8, 8), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "bobber_good.png"), template)
    cv2.imwrite(str(tmp_path / "bobber_other.png"), weaker)
    frame = np.zeros((60, 70), dtype=np.uint8)
    frame[20:32, 30:40] = template
    detector = BobberDetector(tmp_path, 0.9, grayscale=True)
    result = detector.detect(frame, Region(100, 200, 70, 60))
    assert result.found and result.template_name == "bobber_good.png"
    assert (result.x, result.y) == (135, 226)


def test_confidence_threshold_and_oversized_template(tmp_path: Path) -> None:
    cv2.imwrite(str(tmp_path / "large.png"), np.ones((20, 20), dtype=np.uint8))
    detector = BobberDetector(tmp_path, 1.1)
    assert not detector.detect(np.zeros((5, 5), dtype=np.uint8), Region()).found
