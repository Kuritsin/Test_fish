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
from .visual_bite_detector import VisualBiteResult, make_bite_region


@dataclass(frozen=True)
class ColourProfile:
    hue: int
    hue_tolerance: int
    min_saturation: int
    min_value: int
    anchor_x: int
    anchor_y: int


def calibrate_colour(template: np.ndarray, config: BotConfig) -> ColourProfile | None:
    bgr = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR) if template.ndim == 2 else template
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    valid = ((hsv[:, :, 1] >= config.auto_min_saturation) &
             (hsv[:, :, 2] >= config.auto_min_value))
    if np.count_nonzero(valid) < config.auto_min_component_area:
        return None
    hues = hsv[:, :, 0][valid]
    histogram = np.bincount(hues, minlength=180)
    preferred = np.r_[np.arange(0, 16), np.arange(90, 136), np.arange(165, 180)]
    preferred_hue = int(preferred[np.argmax(histogram[preferred])])
    dominant = preferred_hue if histogram[preferred_hue] >= config.auto_min_component_area else int(np.argmax(histogram))
    hue = hsv[:, :, 0].astype(np.int16)
    distance = np.minimum(np.abs(hue - dominant), 180 - np.abs(hue - dominant))
    selected = valid & (distance <= config.auto_hue_tolerance)
    ys, xs = np.nonzero(selected)
    if xs.size < config.auto_min_component_area:
        return None
    return ColourProfile(dominant, config.auto_hue_tolerance,
                         config.auto_min_saturation, config.auto_min_value,
                         int(round(float(np.median(xs)))), int(round(float(np.median(ys)))))


def colour_mask(frame: np.ndarray, profile: ColourProfile) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    distance = np.minimum(np.abs(hue - profile.hue), 180 - np.abs(hue - profile.hue))
    mask = ((distance <= profile.hue_tolerance) &
            (hsv[:, :, 1] >= profile.min_saturation) &
            (hsv[:, :, 2] >= profile.min_value))
    return (mask.astype(np.uint8) * 255)


def find_component(mask: np.ndarray, expected: tuple[int, int], config: BotConfig) -> tuple[int, int, int] | None:
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    maximum_area = mask.size * config.auto_max_component_area_ratio
    candidates: list[tuple[float, int, int, int]] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if not config.auto_min_component_area <= area <= maximum_area:
            continue
        x, y = (int(round(value)) for value in centroids[index])
        distance = ((x - expected[0]) ** 2 + (y - expected[1]) ** 2) ** 0.5
        candidates.append((distance, x, y, area))
    if not candidates:
        return None
    _, x, y, area = min(candidates)
    return x, y, area


class AutoBiteDetector:
    """Per-cast colour tracking with an automatically calibrated median-Y strike."""

    def __init__(self, config: BotConfig, capture: ScreenCapture) -> None:
        self.config, self.capture = config, capture

    def wait_for_bite(
        self, detection: BobberDetection, template: np.ndarray, client: Region,
        timeout: float, stop_event: threading.Event, safe: Callable[[], bool],
        debug_path: Path | None = None,
    ) -> VisualBiteResult:
        profile = calibrate_colour(template, self.config)
        if profile is None:
            return VisualBiteResult(False, 0, 0, 0, 0, detection.confidence,
                                    "colour_calibration_failed", detection.x, detection.y)
        region = make_bite_region(detection, client, self.config.auto_tracking_padding)
        template_left = detection.x - detection.size[0] // 2
        template_top = detection.y - detection.size[1] // 2
        expected = (template_left + profile.anchor_x - region.left,
                    template_top + profile.anchor_y - region.top)
        positions: deque[int] = deque(maxlen=self.config.auto_baseline_window)
        lost = 0
        deadline = time.monotonic() + timeout
        delay = 1.0 / max(1.0, self.config.visual_bite_fps)
        last_x, last_y = detection.x, detection.y
        peak_drop = 0.0

        while time.monotonic() < deadline and not stop_event.is_set() and safe():
            frame = self.capture.capture(region)
            mask = colour_mask(frame, profile)
            component = find_component(mask, expected, self.config)
            if component is None:
                lost += 1
                if lost >= self.config.auto_lost_frames:
                    return VisualBiteResult(False, peak_drop, 0, 0, 0, 0,
                                            "colour_tracker_lost", last_x, last_y)
            else:
                x, y, area = component
                shift = ((x - expected[0]) ** 2 + (y - expected[1]) ** 2) ** 0.5
                if shift <= self.config.auto_max_frame_shift:
                    lost = 0
                    expected = (x, y)
                    last_x, last_y = region.left + x, region.top + y
                    if len(positions) >= self.config.auto_baseline_frames:
                        baseline = float(np.median(positions))
                        jitter = float(np.median(np.abs(np.asarray(positions) - baseline)))
                        threshold = max(float(self.config.auto_min_bite_drop),
                                        jitter * self.config.auto_jitter_multiplier)
                        drop = last_y - baseline
                        peak_drop = max(peak_drop, drop)
                        if drop >= threshold:
                            if debug_path:
                                self._debug(frame, mask, x, y, last_y, baseline, threshold, area, debug_path)
                            return VisualBiteResult(True, peak_drop, jitter, threshold,
                                                    int(drop), 1.0, "position_drop", last_x, last_y)
                    positions.append(last_y)
                    if debug_path and len(positions) >= self.config.auto_baseline_frames:
                        baseline = float(np.median(positions))
                        jitter = float(np.median(np.abs(np.asarray(positions) - baseline)))
                        threshold = max(float(self.config.auto_min_bite_drop),
                                        jitter * self.config.auto_jitter_multiplier)
                        self._debug(frame, mask, x, y, last_y, baseline, threshold, area, debug_path)
                    margin = max(4, self.config.auto_tracking_padding // 2)
                    if (x < margin or y < margin or x > region.width - margin or
                            y > region.height - margin):
                        tracked = BobberDetection(True, last_x, last_y, 1.0,
                                                  detection.template_name, size=detection.size)
                        new_region = make_bite_region(tracked, client, self.config.auto_tracking_padding)
                        if new_region != region:
                            region = new_region
                            expected = (last_x - region.left, last_y - region.top)
            if stop_event.wait(delay):
                break
        reason = "attempt_expired" if time.monotonic() >= deadline else "interrupted"
        return VisualBiteResult(False, peak_drop, 0, 0, last_y - detection.y,
                                1.0, reason, last_x, last_y)

    @staticmethod
    def _debug(frame: np.ndarray, mask: np.ndarray, x: int, local_y: int, tracked_y: int,
               baseline: float, threshold: float, area: int, path: Path) -> None:
        output = frame.copy()
        cv2.circle(output, (x, local_y), 5, (0, 255, 255), 1)
        cv2.putText(output, f"y={tracked_y} base={baseline:.1f} drop={threshold:.1f} area={area}",
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), output)
        cv2.imwrite(str(path.with_name("latest_colour_mask.png")), mask)
