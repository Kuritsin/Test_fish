import json

import numpy as np

from fishing_bot.calibration import CalibrationProfile, CalibrationSample, build_profile, profile_colour_mask, sample_bobber


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
