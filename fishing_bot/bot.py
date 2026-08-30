from __future__ import annotations

from enum import Enum, auto
import logging
import threading
import time

import cv2

from config import BotConfig
from .audio_detector import AudioDetector
from .auto_bite_detector import AutoBiteDetector
from .automatic_bobber_finder import AutomaticBobberFinder
from .bobber_detector import BobberDetector, BobberDetection
from .capture import ScreenCapture
from .input_controller import InputController
from .search_region import SearchRegionUnavailable, resolve_search_region
from .visual_bite_detector import VisualBiteDetector
from .window_manager import WindowManager


class BotState(Enum):
    IDLE = auto(); CHECK_WINDOW = auto(); CASTING = auto(); SEARCHING_BOBBER = auto()
    MOVING_TO_BOBBER = auto(); LISTENING = auto(); LOOTING = auto(); VERIFYING_LOOT = auto(); WAITING = auto()
    PAUSED = auto(); ERROR = auto(); STOPPED = auto()


class FishingBot:
    def __init__(self, config: BotConfig, dry_run: bool = False) -> None:
        self.config = config
        self.dry_run = dry_run
        self.log = logging.getLogger(__name__)
        self.stop_event = threading.Event()
        self.paused = threading.Event()
        self.state = BotState.IDLE
        self.window = WindowManager(config.wow_window_titles, config.wow_process_names)
        self.capture = ScreenCapture()
        self.detector = BobberDetector(config.templates_dir, config.bobber_confidence, config.grayscale_matching)
        self.automatic_finder = AutomaticBobberFinder(config, self.capture)
        self.audio = AudioDetector(config)
        self.auto_bite = AutoBiteDetector(config, self.capture)
        self.visual_bite = VisualBiteDetector(config, self.capture)
        self.input = InputController(self._safe_to_send, dry_run, config.input_delay, config.mouse_move_duration)
        self.last_bait = 0.0 if config.apply_bait_on_start else time.monotonic()
        self._attempt_template = None

    def _safe_to_send(self) -> bool:
        return not self.stop_event.is_set() and not self.paused.is_set() and self.window.is_wow_active()

    def toggle_pause(self) -> None:
        if self.paused.is_set():
            self.paused.clear(); self.log.info("Bot resumed")
        else:
            self.paused.set(); self.state = BotState.PAUSED; self.log.info("Bot paused")

    def stop(self) -> None:
        self.stop_event.set(); self.input.release_modifiers(); self.audio.close()

    def _sleep(self, seconds: float) -> bool:
        return not self.stop_event.wait(seconds)

    def _search(self, attempt_deadline: float) -> BobberDetection:
        deadline = min(attempt_deadline, time.monotonic() + self.config.bobber_search_timeout)
        result = BobberDetection(False)
        while time.monotonic() < deadline and self._safe_to_send():
            region = resolve_search_region(self.config, self.window)
            self.log.debug("Capturing search region: %s (mode=%s)", region, self.config.search_mode)
            frame = self.capture.capture(region)
            result = self.detector.detect(frame, region)
            if self.config.debug:
                self.config.debug_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(self.config.debug_dir / "latest_capture.png"), frame)
                cv2.imwrite(str(self.config.debug_dir / "latest_detection.png"), self.detector.visualize(frame, result))
            if result.found:
                return result
            self._sleep(self.config.bobber_search_interval)
        return result

    def maybe_use_bait(self) -> bool:
        due = time.monotonic() - self.last_bait >= self.config.bait_interval_minutes * 60
        if not self.config.use_bait or not due or not self._safe_to_send():
            return False
        self.log.info("Applying bait")
        if self.input.use_bait(self.config.bait_key):
            self.last_bait = time.monotonic()
            self._sleep(self.config.bait_application_delay)
            return True
        return False

    def _wait_for_bite(self, detection: BobberDetection, timeout: float):
        if self.config.bite_detection_mode == "audio":
            self.log.info("Waiting for bite using WASAPI audio")
            return self.audio.wait_for_bite(timeout, self.stop_event)
        if self.config.bite_detection_mode not in {"auto", "visual"}:
            raise ValueError(f"Unsupported bite detection mode: {self.config.bite_detection_mode}")
        client = self.window.foreground_client_region()
        if client is None:
            raise SearchRegionUnavailable("Could not determine the WoW client area before visual bite detection")
        template = self._attempt_template
        if template is None:
            template = self.detector.template_by_name(detection.template_name)
        debug_path = self.config.debug_dir / "latest_visual_bite.png" if self.config.debug else None
        self.log.info("Waiting for visual bobber movement")
        if self.config.bite_detection_mode == "auto":
            result = self.auto_bite.wait_for_bite(
                detection, template, client, timeout, self.stop_event,
                lambda: self._safe_to_send() and self.window.foreground_client_region() == client,
                self.config.debug_dir / "latest_auto_bite.png" if self.config.debug else None,
            )
            if result.reason != "colour_calibration_failed":
                return result
            self.log.warning("Automatic colour calibration failed; using template fallback")
        return self.visual_bite.wait_for_bite(
            detection, template, client, timeout, self.stop_event,
            lambda: self._safe_to_send() and self.window.foreground_client_region() == client,
            debug_path,
        )

    def run_attempt(self) -> None:
        # Validate the current window area before sending the first input action.
        resolve_search_region(self.config, self.window)
        search_region = resolve_search_region(self.config, self.window)
        before_cast = self._capture_background(search_region) if self.config.bite_detection_mode == "auto" else None
        if self.config.debug and before_cast is not None:
            self.config.debug_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(self.config.debug_dir / "latest_pre_cast.png"), before_cast[-1])
        self._attempt_template = None
        cast_started = time.monotonic()
        deadline = cast_started + self.config.fishing_attempt_timeout
        self.state = BotState.CASTING; self.log.info("Casting (attempt timeout %.1fs)", self.config.fishing_attempt_timeout)
        if not self.input.cast(self.config.cast_key): return
        if self.dry_run and before_cast is not None:
            self.log.info("DRY RUN: cast manually now so automatic post-cast detection can observe the bobber")
        if not self._sleep(min(self.config.cast_delay, self._remaining(deadline))): return
        if self._remaining(deadline) <= 0: return
        self.state = BotState.SEARCHING_BOBBER
        result = BobberDetection(False)
        if before_cast is not None:
            automatic = self.automatic_finder.find(
                before_cast, search_region,
                min(deadline, time.monotonic() + self.config.bobber_search_timeout),
                self.stop_event, self._safe_to_send,
                self.config.debug_dir if self.config.debug else None,
            )
            result, self._attempt_template = automatic.detection, automatic.template
            if not result.found:
                self.log.warning("Automatic bobber search failed: %s; trying templates", automatic.reason)
        if not result.found and self.detector.templates:
            result = self._search(deadline)
            if result.found:
                self._attempt_template = self.detector.template_by_name(result.template_name)
        if not result.found:
            self.log.warning("Bobber not found (best confidence %.3f)", result.confidence); return
        self.log.info("Bobber found: x=%d y=%d confidence=%.3f template=%s", result.x, result.y, result.confidence, result.template_name)
        self.state = BotState.MOVING_TO_BOBBER
        if not self.input.move_to(result.x, result.y): return
        self.state = BotState.LISTENING
        remaining = self._remaining(deadline) - self.config.loot_safety_margin
        if remaining <= 0:
            self.log.warning("Attempt expired before bite detection"); return
        bite = self._wait_for_bite(result, remaining)
        if not bite.detected:
            self.log.warning("Bite timeout: %s", bite); return
        self.log.info("Bite detected: %s", bite)
        if self._remaining(deadline) < self.config.loot_safety_margin:
            self.log.warning("Skipping late loot: only %.2fs remain", self._remaining(deadline)); return
        self.state = BotState.LOOTING
        if not self.input.loot(): return
        if self.dry_run:
            self.log.info("DRY RUN: loot verification skipped because no click was sent")
        elif self.config.bite_detection_mode in {"auto", "visual"}:
            self.state = BotState.VERIFYING_LOOT
            client = self.window.foreground_client_region()
            if client is not None:
                tracked = BobberDetection(True, getattr(bite, "x", result.x), getattr(bite, "y", result.y),
                                           template_name=result.template_name, size=result.size)
                template = self._attempt_template
                if template is None:
                    template = self.detector.template_by_name(result.template_name)
                confirmed = self.visual_bite.wait_for_disappearance(
                    tracked, template, client,
                    min(self.config.loot_confirmation_timeout, self._remaining(deadline)),
                    self.stop_event,
                    lambda: self._safe_to_send() and self.window.foreground_client_region() == client,
                )
                self.log.info("Loot confirmed" if confirmed else "Loot not confirmed")
        self._sleep(min(self.config.post_loot_delay, self._remaining(deadline)))
        self.maybe_use_bait()

    def _capture_background(self, region):
        frames = []
        for index in range(max(1, self.config.auto_background_frames)):
            frames.append(self.capture.capture(region))
            if index + 1 < self.config.auto_background_frames:
                self._sleep(self.config.auto_background_interval)
        return frames

    @staticmethod
    def _remaining(deadline: float) -> float:
        return max(0.0, deadline - time.monotonic())

    def run(self, once: bool = False) -> None:
        if not self.detector.templates and self.config.bite_detection_mode != "auto":
            raise RuntimeError("No readable PNG templates in templates/. Use auto mode or see templates/README.md")
        self.log.info("Bot started")
        try:
            while not self.stop_event.is_set():
                if self.paused.is_set(): self._sleep(0.1); continue
                self.state = BotState.CHECK_WINDOW
                if not self.window.is_wow_active():
                    self.log.info("Waiting for WoW foreground window")
                    self._sleep(self.config.inactive_window_delay); continue
                try:
                    self.run_attempt()
                except SearchRegionUnavailable as error:
                    self.log.warning("Search skipped safely: %s", error)
                except Exception:
                    self.state = BotState.ERROR; self.log.exception("Attempt failed")
                    self.audio.close()
                if once: break
                self.state = BotState.WAITING; self._sleep(self.config.retry_delay)
        finally:
            self.stop(); self.capture.close(); self.state = BotState.STOPPED; self.log.info("Bot stopped")
