from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from config import BotConfig, Region
from .bobber_detector import BobberDetection
from .capture import ScreenCapture


@dataclass(frozen=True)
class VisualBiteResult:
    detected: bool
    peak_motion: float
    baseline_motion: float
    threshold: float
    vertical_shift: int
    template_confidence: float
    reason: str
    x: int = 0
    y: int = 0


def make_bite_region(detection: BobberDetection, client: Region, padding: int) -> Region:
    width, height = detection.size
    left = max(client.left, detection.x - width // 2 - padding)
    top = max(client.top, detection.y - height // 2 - padding)
    right = min(client.left + client.width, detection.x - width // 2 + width + padding)
    bottom = min(client.top + client.height, detection.y - height // 2 + height + padding)
    if right <= left or bottom <= top:
        raise ValueError("Bobber ROI is outside the WoW client area")
    return Region(left, top, right - left, bottom - top)


def grayscale(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.GaussianBlur(gray, (3, 3), 0)


def motion_score(previous: np.ndarray, current: np.ndarray, pixel_difference: int) -> float:
    difference = cv2.absdiff(grayscale(previous), grayscale(current))
    return float(np.count_nonzero(difference >= pixel_difference) / difference.size)


def template_position(frame: np.ndarray, template: np.ndarray) -> tuple[int, int, float]:
    source, needle = grayscale(frame), grayscale(template)
    if needle.shape[0] > source.shape[0] or needle.shape[1] > source.shape[1]:
        return 0, 0, 0.0
    result = cv2.matchTemplate(source, needle, cv2.TM_CCOEFF_NORMED)
    _, confidence, _, location = cv2.minMaxLoc(result)
    return int(location[0]), int(location[1]), float(confidence)


class VisualBiteDetector:
    def __init__(self, config: BotConfig, capture: ScreenCapture) -> None:
        self.config, self.capture = config, capture

    def wait_for_bite(
        self, detection: BobberDetection, template: np.ndarray, client_region: Region,
        timeout: float, stop_event: threading.Event, safe_to_continue: Callable[[], bool],
        debug_path: Path | None = None,
    ) -> VisualBiteResult:
        region = make_bite_region(detection, client_region, self.config.visual_bite_roi_padding)
        delay = 1.0 / max(1.0, self.config.visual_bite_fps)
        started = time.monotonic()
        previous = self.capture.capture(region)
        history: deque[tuple[float, int]] = deque(maxlen=max(3, self.config.visual_baseline_frames))
        baseline_samples: list[float] = []
        peak = threshold = 0.0
        confirmations = lost_frames = 0
        last_x, last_y = detection.x, detection.y
        initial_y = detection.y
        last_confidence = detection.confidence
        transient_at: float | None = None

        while time.monotonic() - started < timeout and not stop_event.is_set() and safe_to_continue():
            if stop_event.wait(delay):
                break
            now = time.monotonic()
            current = self.capture.capture(region)
            motion = motion_score(previous, current, self.config.visual_pixel_difference)
            local_x, local_y, confidence = template_position(current, template)
            peak, last_confidence = max(peak, motion), confidence

            if confidence >= self.config.visual_template_confidence:
                lost_frames = 0
                last_x = region.left + local_x + template.shape[1] // 2
                last_y = region.top + local_y + template.shape[0] // 2
                history.append((now, last_y))
                margin = max(2, self.config.visual_bite_roi_padding // 2)
                near_edge = (local_x < margin or local_y < margin or
                             local_x + template.shape[1] > region.width - margin or
                             local_y + template.shape[0] > region.height - margin)
                if near_edge:
                    tracked = BobberDetection(True, last_x, last_y, confidence,
                                              detection.template_name, size=detection.size)
                    new_region = make_bite_region(
                        tracked, client_region, self.config.visual_bite_roi_padding
                    )
                    if new_region != region:
                        region = new_region
                        previous = self.capture.capture(region)
                        continue
            else:
                lost_frames += 1

            if len(baseline_samples) < self.config.visual_baseline_frames:
                baseline_samples.append(motion)
                previous = current
                continue

            baseline = float(np.median(baseline_samples))
            threshold = max(self.config.visual_motion_threshold,
                            baseline * self.config.visual_motion_multiplier)
            velocity = self._vertical_velocity(history)
            downward = motion >= threshold and velocity >= self.config.visual_downward_velocity
            if downward:
                transient_at = now
                confirmations += 1
            elif confidence >= self.config.visual_template_confidence:
                confirmations = 0

            submerged = (transient_at is not None and now - transient_at <= self.config.visual_bite_window
                         and confidence < self.config.visual_template_confidence)
            if confirmations >= self.config.visual_confirmation_frames or submerged:
                reason = "downward_transient" if confirmations else "submerged_after_motion"
                return VisualBiteResult(True, peak, baseline, threshold, last_y - initial_y,
                                        confidence, reason, last_x, last_y)

            if lost_frames >= self.config.visual_tracker_lost_frames:
                reacquired = self._reacquire(template, client_region, last_x, last_y)
                if reacquired is None:
                    return VisualBiteResult(False, peak, baseline, threshold, last_y - initial_y,
                                            confidence, "tracker_lost", last_x, last_y)
                last_x, last_y, confidence = reacquired
                detection = BobberDetection(True, last_x, last_y, confidence,
                                            detection.template_name, size=detection.size)
                region = make_bite_region(detection, client_region, self.config.visual_bite_roi_padding)
                current = self.capture.capture(region)
                history.clear(); lost_frames = confirmations = 0

            if debug_path:
                self._write_debug(current, motion, threshold, velocity, confidence, debug_path)
            previous = current

        baseline = float(np.median(baseline_samples)) if baseline_samples else 0.0
        reason = "attempt_expired" if time.monotonic() - started >= timeout else "interrupted"
        return VisualBiteResult(False, peak, baseline, threshold, last_y - initial_y,
                                last_confidence, reason, last_x, last_y)

    @staticmethod
    def _vertical_velocity(history: deque[tuple[float, int]]) -> float:
        if len(history) < 2:
            return 0.0
        start_time, start_y = history[0]
        end_time, end_y = history[-1]
        elapsed = end_time - start_time
        return (end_y - start_y) / elapsed if elapsed > 0 else 0.0

    def _reacquire(self, template: np.ndarray, client: Region, old_x: int,
                   old_y: int) -> tuple[int, int, float] | None:
        frame = self.capture.capture(client)
        x, y, confidence = template_position(frame, template)
        center_x = client.left + x + template.shape[1] // 2
        center_y = client.top + y + template.shape[0] // 2
        distance = ((center_x - old_x) ** 2 + (center_y - old_y) ** 2) ** 0.5
        if confidence < self.config.visual_template_confidence or distance > self.config.visual_reacquire_radius:
            return None
        return center_x, center_y, confidence

    def wait_for_disappearance(
        self, detection: BobberDetection, template: np.ndarray, client: Region,
        timeout: float, stop_event: threading.Event, safe_to_continue: Callable[[], bool],
    ) -> bool:
        region = make_bite_region(detection, client, self.config.visual_bite_roi_padding)
        deadline = time.monotonic() + timeout
        missing = 0
        while time.monotonic() < deadline and not stop_event.is_set() and safe_to_continue():
            _, _, confidence = template_position(self.capture.capture(region), template)
            missing = missing + 1 if confidence < self.config.loot_disappearance_confidence else 0
            if missing >= self.config.loot_confirmation_frames:
                return True
            if stop_event.wait(1.0 / max(1.0, self.config.visual_bite_fps)):
                break
        return False

    @staticmethod
    def _write_debug(frame: np.ndarray, motion: float, threshold: float, velocity: float,
                     confidence: float, path: Path) -> None:
        output = frame.copy()
        text = f"motion={motion:.3f}/{threshold:.3f} vy={velocity:.1f} conf={confidence:.2f}"
        cv2.putText(output, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), output)
