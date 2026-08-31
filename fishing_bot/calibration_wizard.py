from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import time

import numpy as np

from config import BotConfig, Region
from .automatic_bobber_finder import AutomaticBobberFinder
from .bobber_detector import BobberDetection, BobberDetector
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
    calibration_config = replace(
        config,
        auto_find_pixel_difference=min(config.auto_find_pixel_difference, 20),
        auto_find_max_area=max(config.auto_find_max_area, 1200),
        auto_find_confirmation_frames=min(config.auto_find_confirmation_frames, 2),
    )
    finder = AutomaticBobberFinder(calibration_config, capture)
    # Calibration must discover freely instead of filtering through a possibly stale profile.
    finder.profile = None
    finder.templates = []
    template_detector = BobberDetector(
        config.templates_dir, config.calibration_template_confidence, config.grayscale_matching,
        config.template_global_scales, config.template_edge_weight,
    )
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
            detection, candidate_template = _find_with_templates(
                template_detector, capture, region, config.calibration_template_timeout,
                window.is_wow_active,
            )
            reason, best_score = "template_match", detection.confidence
            if not detection.found:
                result = finder.find(before, region, time.monotonic() + config.bobber_search_timeout,
                                     _NeverSet(), window.is_wow_active, config.debug_dir, permissive=True)
                detection, candidate_template = result.detection, result.template
                reason, best_score = result.reason, result.best_score
            if not detection.found or candidate_template is None:
                print(f"Кандидат не найден ({reason}, лучший score={best_score:.3f}). "
                      "Эта попытка не испортит профиль.")
                continue
            print(f"Кандидат найден через {reason}: confidence={best_score:.3f}")
            controller.move_to(detection.x, detection.y)
            answer = ask("Курсор у настоящего поплавка? [Y/n]: ").strip().lower()
            if answer in {"n", "no", "нет"}:
                print("Ложный кандидат отклонён.")
                continue
            samples.append(sample_bobber(candidate_template))
            print(f"Образец принят ({len(samples)}).")
        if not samples:
            raise RuntimeError("Не подтверждено ни одного поплавка; профиль не изменён")
        profile = build_profile(samples)
        profile.save(config.calibration_profile_path)
        print(f"Профиль из {profile.sample_count} образцов сохранён: {config.calibration_profile_path}")
    finally:
        controller.release_modifiers()
        capture.close()


def _find_with_templates(
    detector: BobberDetector,
    capture: ScreenCapture,
    region: Region,
    timeout: float,
    safe: Callable[[], bool],
) -> tuple[BobberDetection, np.ndarray | None]:
    best = BobberDetection(False)
    deadline = time.monotonic() + timeout
    while detector.templates and time.monotonic() < deadline and safe():
        frame = capture.capture(region)
        detection = detector.detect(frame, region)
        if detection.confidence > best.confidence:
            best = detection
        if detection.found:
            left, top = detection.top_left
            width, height = detection.size
            padding = 6
            crop = frame[max(0, top - padding):min(frame.shape[0], top + height + padding),
                         max(0, left - padding):min(frame.shape[1], left + width + padding)].copy()
            return detection, crop
        time.sleep(0.1)
    return best, None


class _NeverSet:
    """Minimal Event-compatible object for the synchronous wizard."""

    @staticmethod
    def is_set() -> bool:
        return False

    @staticmethod
    def wait(seconds: float) -> bool:
        time.sleep(seconds)
        return False