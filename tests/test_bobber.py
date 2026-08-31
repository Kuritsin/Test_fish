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


def test_multiscale_detector_handles_a_different_bobber_size(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    template = rng.integers(0, 255, (20, 16), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "bobber.png"), template)
    scaled = cv2.resize(template, (12, 15), interpolation=cv2.INTER_AREA)
    frame = np.zeros((70, 80), dtype=np.uint8)
    frame[30:45, 40:52] = scaled
    detector = BobberDetector(tmp_path, 0.8, scales=(0.75, 1.0))
    result = detector.detect(frame, Region(0, 0, 80, 70))
    assert result.found
    assert result.size == (12, 15)


def test_candidate_detection_finds_multiple_locations(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    template = rng.integers(0, 255, (10, 10), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "bobber.png"), template)
    frame = np.zeros((80, 100), dtype=np.uint8)
    frame[20:30, 30:40] = template
    frame[50:60, 70:80] = template
    detector = BobberDetector(tmp_path, 0.5)
    candidates = detector.detect_candidates(frame, Region(100, 200, 100, 80), 5, 0.9)
    assert {(item.x, item.y) for item in candidates} == {(135, 225), (175, 255)}


def test_transparent_render_uses_alpha_mask(tmp_path: Path) -> None:
    template = np.zeros((20, 24, 4), dtype=np.uint8)
    template[4:16, 6:18, :3] = np.arange(144, dtype=np.uint8).reshape(12, 12, 1)
    template[4:16, 6:18, 3] = 255
    cv2.imwrite(str(tmp_path / "render.png"), template)
    frame = np.full((50, 60), 80, dtype=np.uint8)
    frame[22:34, 31:43] = cv2.cvtColor(template[4:16, 6:18, :3], cv2.COLOR_BGR2GRAY)
    result = BobberDetector(tmp_path, 0.95).detect(frame, Region(0, 0, 60, 50))
    assert result.found and result.size == (12, 12)