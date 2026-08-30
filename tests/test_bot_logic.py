from dataclasses import replace

from config import BotConfig
from fishing_bot.bot import BotState, FishingBot


def test_states_are_explicit() -> None:
    assert BotState.CASTING is not BotState.LISTENING
    assert BotState.STOPPED.name == "STOPPED"
    assert BotState.VERIFYING_LOOT.name == "VERIFYING_LOOT"


def test_bait_configuration_is_centralized() -> None:
    config = replace(BotConfig(), use_bait=False, bait_interval_minutes=2)
    assert not config.use_bait
    assert config.bait_interval_minutes == 2


def test_attempt_remaining_time_never_becomes_negative(monkeypatch) -> None:
    monkeypatch.setattr("fishing_bot.bot.time.monotonic", lambda: 12.5)
    assert FishingBot._remaining(20.0) == 7.5
    assert FishingBot._remaining(10.0) == 0.0
