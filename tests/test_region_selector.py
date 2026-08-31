from pathlib import Path
import warnings

from config import BotConfig, Region, load_config, save_search_region, save_user_settings
from fishing_bot.region_selector import roi_to_screen_region
from fishing_bot.search_region import SearchRegionUnavailable, resolve_search_region


def test_roi_is_converted_to_global_coordinates() -> None:
    window = Region(left=100, top=200, width=1200, height=800)
    assert roi_to_screen_region(window, (25, 40, 500, 300)) == Region(125, 240, 500, 300)


def test_roi_supports_monitor_with_negative_coordinates() -> None:
    window = Region(left=-1920, top=-100, width=1920, height=1080)
    assert roi_to_screen_region(window, (100, 50, 800, 600)) == Region(-1820, -50, 800, 600)


def test_roi_rejects_empty_and_out_of_bounds_selection() -> None:
    window = Region(0, 0, 100, 100)
    for roi in ((0, 0, 0, 10), (90, 90, 20, 20)):
        try:
            roi_to_screen_region(window, roi)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid ROI was accepted: {roi}")


def test_user_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "user_config.json"
    expected = Region(-20, 30, 640, 480)
    save_search_region(expected, path)
    loaded = load_config(path)
    assert loaded.search_mode == "custom"
    assert loaded.search_region == expected


class FakeWindow:
    def __init__(self, region: Region | None) -> None:
        self.region = region

    def foreground_client_region(self) -> Region | None:
        return self.region


def test_window_mode_uses_current_window_position() -> None:
    config = BotConfig(search_mode="window")
    window = FakeWindow(Region(-1000, 25, 1920, 1080))
    assert resolve_search_region(config, window) == Region(-1000, 25, 1920, 1080)
    window.region = Region(20, 30, 1280, 720)
    assert resolve_search_region(config, window) == Region(20, 30, 1280, 720)


def test_custom_mode_does_not_query_window() -> None:
    selected = Region(10, 20, 300, 200)
    config = BotConfig(search_mode="custom", search_region=selected)
    assert resolve_search_region(config, FakeWindow(None)) == selected


def test_window_mode_fails_safely_when_client_area_is_unavailable() -> None:
    try:
        resolve_search_region(BotConfig(search_mode="window"), FakeWindow(None))
    except SearchRegionUnavailable:
        pass
    else:
        raise AssertionError("Missing window client area was accepted")


def test_user_config_can_explicitly_select_window_mode(tmp_path: Path) -> None:
    path = tmp_path / "user_config.json"
    path.write_text('{"search_mode": "window"}', encoding="utf-8")
    assert load_config(path).search_mode == "window"


def test_setup_settings_are_saved_without_editing_python(tmp_path: Path) -> None:
    path = tmp_path / "user_config.json"
    save_user_settings({"cast_key": "5", "bait_key": "6", "use_bait": False,
                        "bite_detection_mode": "auto"}, path)
    loaded = load_config(path)
    assert loaded.cast_key == "5"
    assert loaded.bait_key == "6"
    assert not loaded.use_bait
    assert loaded.bite_detection_mode == "auto"


def test_invalid_user_config_falls_back_to_defaults(tmp_path: Path) -> None:
    path = tmp_path / "user_config.json"
    path.write_text('{"search_region": {"left": 0, "top": 0, "width": -1, "height": 50}}')
    with warnings.catch_warnings(record=True) as caught:
        config = load_config(path)
    assert config.search_region == Region()
    assert caught


def test_malformed_user_config_falls_back_to_defaults(tmp_path: Path) -> None:
    path = tmp_path / "user_config.json"
    path.write_text("not json", encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        config = load_config(path)
    assert config.search_region == Region()
    assert caught
