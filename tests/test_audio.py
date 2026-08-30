import numpy as np

from fishing_bot.audio_detector import calculate_rms, calculate_threshold


def test_rms() -> None:
    assert calculate_rms(np.array([-3, 3, -3, 3], dtype=np.int16)) == 3.0
    assert calculate_rms(np.array([], dtype=np.int16)) == 0.0


def test_threshold_uses_baseline_or_floor() -> None:
    assert calculate_threshold(100, 3, 500) == 500
    assert calculate_threshold(300, 3, 500) == 900
