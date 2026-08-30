from __future__ import annotations

from collections.abc import Callable
import time

from config import BotConfig, Region
from .automatic_bobber_finder import AutomaticBobberFinder
from .calibration import CalibrationSample, build_profile, sample_bobber
from .capture import ScreenCapture
from .input_controller import InputController
from .search_region import resolve_search_region
from .window_manager import WindowManager


def run_calibration(config: BotConfig, ask: Callable[[str], str] = input) -> None:
    """Run several user-confirmed casts and persist a device/location profile."""
    window = WindowManager(config.wow_window_titles, config.wow_process_names)
    capture = ScreenCapture()
    controller = InputController(window.is_wow_active, False, config.input_delay, config.mouse_move_duration)
    finder = AutomaticBobberFinder(config, capture)
    # Calibration must discover freely instead of filtering through a possibly stale profile.
    finder.profile = None
    samples: list[CalibrationSample] = []
    print("=== Автокалибровка поплавка ===")
    print("Будет выполнено несколько забросов. Не двигайте камеру во время одной попытки.")
    try:
        for number in range(1, config.calibration_attempts + 1):
            ask(f"\nПопытка {number}/{config.calibration_attempts}. Нажмите Enter и переключитесь в WoW...")
            deadline = time.monotonic() + 30.0
            while not window.is_wow_active() and time.monotonic() < deadline:
                time.sleep(0.25)
            if not window.is_wow_active():
                print("WoW не стал активным — попытка пропущена.")
                continue
            region = resolve_search_region(config, window)
            before = []
            for index in range(max(1, config.auto_background_frames)):
                before.append(capture.capture(region))
                if index + 1 < config.auto_background_frames:
                    time.sleep(config.auto_background_interval)
            if not controller.cast(config.cast_key):
                print("Заброс отменён проверкой безопасности.")
                continue
            time.sleep(config.cast_delay)
            result = finder.find(before, region, time.monotonic() + config.bobber_search_timeout,
                                 _NeverSet(), window.is_wow_active, config.debug_dir)
            if not result.detection.found or result.template is None:
                print("Кандидат не найден. Эта попытка не испортит профиль.")
                continue
            controller.move_to(result.detection.x, result.detection.y)
            answer = ask("Курсор у настоящего поплавка? [Y/n]: ").strip().lower()
            if answer in {"n", "no", "нет"}:
                print("Ложный кандидат отклонён.")
                continue
            samples.append(sample_bobber(result.template))
            print(f"Образец принят ({len(samples)}).")
        if not samples:
            raise RuntimeError("Не подтверждено ни одного поплавка; профиль не изменён")
        profile = build_profile(samples)
        profile.save(config.calibration_profile_path)
        print(f"Профиль из {profile.sample_count} образцов сохранён: {config.calibration_profile_path}")
    finally:
        controller.release_modifiers()
        capture.close()


class _NeverSet:
    """Minimal Event-compatible object for the synchronous wizard."""

    @staticmethod
    def is_set() -> bool:
        return False

    @staticmethod
    def wait(seconds: float) -> bool:
        time.sleep(seconds)
        return False
