from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import warnings


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Region:
    left: int = 480
    top: int = 120
    width: int = 960
    height: int = 720

    def as_mss(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


@dataclass
class BotConfig:
    cast_key: str = "0"
    bait_key: str = "9"
    pause_key: str = "f8"
    stop_key: str = "f10"
    search_region: Region = field(default_factory=Region)
    search_mode: str = "window"
    wow_window_titles: tuple[str, ...] = ("World of Warcraft",)
    wow_process_names: tuple[str, ...] = ("wow.exe", "wowclassic.exe")
    bobber_confidence: float = 0.75
    grayscale_matching: bool = True
    bobber_search_timeout: float = 5.0
    bobber_search_interval: float = 0.15
    fishing_attempt_timeout: float = 20.0
    loot_safety_margin: float = 0.5
    bite_detection_mode: str = "auto"
    auto_find_pixel_difference: int = 30
    auto_find_min_area: int = 4
    auto_find_max_area: int = 450
    auto_find_confirmation_frames: int = 3
    auto_background_frames: int = 8
    auto_background_interval: float = 0.025
    auto_background_noise_multiplier: float = 2.5
    auto_find_match_radius: int = 36
    auto_find_template_padding: int = 8
    auto_candidate_padding: int = 14
    auto_min_candidate_score: float = 0.34
    auto_min_score_gap: float = 0.06
    auto_template_scales: tuple[float, ...] = (0.65, 0.8, 1.0, 1.2, 1.5)
    auto_hue_tolerance: int = 12
    auto_min_saturation: int = 90
    auto_min_value: int = 55
    auto_min_component_area: int = 3
    auto_max_component_area_ratio: float = 0.35
    auto_tracking_padding: int = 40
    auto_baseline_frames: int = 10
    auto_baseline_window: int = 25
    auto_min_bite_drop: int = 4
    auto_jitter_multiplier: float = 3.0
    auto_max_frame_shift: int = 14
    auto_lost_frames: int = 4
    calibration_attempts: int = 5
    calibration_profile_path: Path = ROOT / "calibration_profile.json"
    visual_bite_fps: float = 20.0
    visual_bite_roi_padding: int = 24
    visual_baseline_frames: int = 8
    visual_pixel_difference: int = 24
    visual_motion_threshold: float = 0.08
    visual_motion_multiplier: float = 3.0
    visual_vertical_shift: int = 4
    visual_template_confidence: float = 0.45
    visual_confirmation_frames: int = 2
    visual_downward_velocity: float = 35.0
    visual_bite_window: float = 0.45
    visual_tracker_lost_frames: int = 3
    visual_reacquire_radius: int = 180
    loot_confirmation_timeout: float = 1.5
    loot_disappearance_confidence: float = 0.35
    loot_confirmation_frames: int = 2
    cast_delay: float = 1.5
    post_loot_delay: float = 1.0
    retry_delay: float = 1.0
    inactive_window_delay: float = 0.5
    input_delay: float = 0.05
    mouse_move_duration: float = 0.15
    use_bait: bool = True
    bait_interval_minutes: float = 10.0
    bait_application_delay: float = 4.0
    apply_bait_on_start: bool = False
    audio_device_name: str | None = None
    audio_chunk_size: int = 1024
    audio_calibration_chunks: int = 12
    audio_threshold_multiplier: float = 3.0
    minimum_audio_threshold: float = 500.0
    audio_consecutive_spikes: int = 2
    templates_dir: Path = ROOT / "templates"
    debug_dir: Path = ROOT / "debug"
    debug: bool = False


USER_CONFIG_PATH = ROOT / "user_config.json"


def load_config(path: Path = USER_CONFIG_PATH) -> BotConfig:
    """Load optional, generated user settings without modifying source code."""
    config = BotConfig()
    if not path.exists():
        return config
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("configuration root must be an object")
        mode = data.get("search_mode", "custom" if "search_region" in data else "window")
        if mode not in {"window", "custom"}:
            raise ValueError("search_mode must be 'window' or 'custom'")
        config.search_mode = mode
        for name in ("cast_key", "bait_key", "pause_key", "stop_key"):
            value = data.get(name)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{name} must be a non-empty string")
                setattr(config, name, value.strip().lower())
        if "use_bait" in data:
            if type(data["use_bait"]) is not bool:
                raise ValueError("use_bait must be a boolean")
            config.use_bait = data["use_bait"]
        bite_mode = data.get("bite_detection_mode")
        if bite_mode is not None:
            if bite_mode not in {"auto", "visual", "audio"}:
                raise ValueError("bite_detection_mode must be 'auto', 'visual' or 'audio'")
            config.bite_detection_mode = bite_mode
        if mode == "custom":
            raw_region = data.get("search_region", {})
            values = {name: raw_region[name] for name in ("left", "top", "width", "height")}
            if any(type(value) is not int for value in values.values()):
                raise ValueError("search_region values must be integers")
            if values["width"] <= 0 or values["height"] <= 0:
                raise ValueError("search_region width and height must be positive")
            config.search_region = Region(**values)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        warnings.warn(f"Ignoring invalid {path.name}: {error}", stacklevel=2)
    return config


def save_search_region(region: Region, path: Path = USER_CONFIG_PATH) -> None:
    payload = _read_payload(path)
    payload.update({"search_mode": "custom", "search_region": region.as_mss()})
    _write_payload(path, payload)


def save_user_settings(settings: dict[str, object], path: Path = USER_CONFIG_PATH) -> None:
    payload = _read_payload(path)
    payload.update(settings)
    payload["schema_version"] = 1
    _write_payload(path, payload)


def _read_payload(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


CONFIG = load_config()
