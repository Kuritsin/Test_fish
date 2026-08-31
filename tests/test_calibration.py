import json

import numpy as np
import cv2

from config import Region
from fishing_bot.bobber_detector import BobberDetector
from fishing_bot.calibration import CalibrationProfile, CalibrationSample, build_profile, profile_colour_mask, sample_bobber
from fishing_bot.calibration_wizard import _find_with_templates


def test_profile_round_trip_and_mask(tmp_path):
    profile = build_profile([CalibrationSample(2, 180, 160, 20), CalibrationSample(176, 150, 130, 30)])
    path = tmp_path / "profile.json"
    profile.save(path)
    loaded = CalibrationProfile.load(path)
    assert loaded == profile
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    frame[2:5, 2:5] = (0, 0, 220)
    assert np.count_nonzero(profile_colour_mask(frame, loaded, 12)) == 9


def test_sample_and_profile_ignore_dark_background():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[8:12, 9:11] = (0, 0, 255)
    sample = sample_bobber(image)
    profile = build_profile([sample])
    assert sample.coloured_area == 8
    assert profile.min_saturation >= 35
    assert profile.min_area >= 2


def test_invalid_profile_is_ignored(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"hues": "bad"}), encoding="utf-8")
    assert CalibrationProfile.load(path) is None


def test_red_hue_wrap_does_not_average_to_cyan():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:5] = (5, 0, 255)
    image[5:] = (0, 5, 255)
    sample = sample_bobber(image)
    assert sample.hue <= 16 or sample.hue >= 164


def test_calibration_prefers_a_matching_template(tmp_path):
    template = np.zeros((12, 16, 3), dtype=np.uint8)
    template[2:9, 3:12] = (20, 30, 230)
    template[7:11, 9:15] = (60, 150, 220)
    cv2.imwrite(str(tmp_path / "bobber.png"), template)
    frame = np.zeros((50, 70, 3), dtype=np.uint8)
    frame[20:32, 30:46] = template

    class Capture:
        def capture(self, region):
            return frame

    detector = BobberDetector(tmp_path, confidence=0.8)
    detection, crop = _find_with_templates(detector, Capture(), Region(0, 0, 70, 50), 0.2, lambda: True)
    assert detection.found
    assert (detection.x, detection.y) == (38, 26)
    assert crop is not None and crop.size > template.size