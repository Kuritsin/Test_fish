from dataclasses import replace

import numpy as np

from config import BotConfig
from fishing_bot.bobber_detector import BobberDetection
from fishing_bot.bot import BotState, FishingBot


def test_states_are_explicit() -> None:
    assert BotState.CASTING is not BotState.LISTENING
    assert BotState.STOPPED.name == "STOPPED"
    assert BotState.VERIFYING_LOOT.name == "VERIFYING_LOOT"
    assert BotState.VALIDATING_BOBBER.name == "VALIDATING_BOBBER"


def test_bait_configuration_is_centralized() -> None:
    config = replace(BotConfig(), use_bait=False, bait_interval_minutes=2)
    assert not config.use_bait
    assert config.bait_interval_minutes == 2


def test_attempt_remaining_time_never_becomes_negative(monkeypatch) -> None:
    monkeypatch.setattr("fishing_bot.bot.time.monotonic", lambda: 12.5)
    assert FishingBot._remaining(20.0) == 7.5
    assert FishingBot._remaining(10.0) == 0.0


def test_template_novelty_rejects_preexisting_static_match() -> None:
    baseline = np.zeros((30, 40, 3), dtype=np.uint8)
    detection = BobberDetection(True, top_left=(10, 10), size=(8, 6))
    assert FishingBot._template_novelty(baseline.copy(), baseline, detection) == 0.0
    changed = baseline.copy()
    changed[10:16, 10:18] = 100
    assert FishingBot._template_novelty(changed, baseline, detection) == 100.0


def test_weaker_template_match_requires_more_confirmations() -> None:
    bot = object.__new__(FishingBot)
    bot.config = BotConfig(
        template_strong_confidence=0.64,
        template_confirmation_frames=2,
        template_weak_confirmation_frames=3,
    )
    assert bot._required_template_confirmations(0.70) == 2
    assert bot._required_template_confirmations(0.57) == 3


def test_hover_fallback_is_disabled_in_dry_run() -> None:
    bot = object.__new__(FishingBot)
    bot.config = BotConfig(hover_validation_enabled=True)
    bot.dry_run = True
    bot.detector = type("Detector", (), {"templates": [("bobber.png", np.zeros((2, 2)))]})()
    assert not bot._find_by_hover(float("inf")).found