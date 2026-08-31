
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable

import cv2
import numpy as np

from config import BotConfig, Region
from .bobber_detector import BobberDetection
from .capture import ScreenCapture
from .input_controller import InputController


@dataclass(frozen=True)
class TooltipMatch:
    found: bool
    confidence: float = 0.0
    box: tuple[int, int, int, int] = (0, 0, 0, 0)
    reason: str = "not_found"


@dataclass(frozen=True)
class HoverValidationResult:
    confirmed: bool
    detection: BobberDetection | None = None
    tooltip: TooltipMatch = TooltipMatch(False)
    points_checked: int = 0
    reason: str = "not_confirmed"


def detect_fishing_tooltip(before: np.ndarray, after: np.ndarray) -> TooltipMatch:
    """Find a newly appeared dark WoW panel containing a yellow title."""
    if before.shape != after.shape or after.ndim != 3:
        return TooltipMatch(False, reason="invalid_frames")
    hsv = cv2.cvtColor(after, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    yellow = ((hue >= 16) & (hue <= 42) & (saturation >= 90) & (value >= 135)).astype(np.uint8)
    changed = cv2.cvtColor(cv2.absdiff(before, after), cv2.COLOR_BGR2GRAY) >= 18
    yellow &= cv2.dilate(changed.astype(np.uint8), np.ones((5, 5), np.uint8))
    joined = cv2.morphologyEx(yellow * 255, cv2.MORPH_CLOSE, np.ones((3, 13), np.uint8))
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = TooltipMatch(False)
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width < 45 or height < 6:
            continue
        left, top = max(0, x - 14), max(0, y - 12)
        right, bottom = min(after.shape[1], x + width + 14), min(after.shape[0], y + height + 12)
        roi_hsv = cv2.cvtColor(after[top:bottom, left:right], cv2.COLOR_BGR2HSV)
        dark_ratio = float(np.mean(roi_hsv[:, :, 2] <= 75))
        yellow_ratio = float(np.mean(yellow[top:bottom, left:right] > 0))
        change_ratio = float(np.mean(changed[top:bottom, left:right]))
        confidence = min(1.0, 0.50 * dark_ratio + 5.0 * yellow_ratio + 0.35 * change_ratio)
        if dark_ratio < 0.42 or yellow_ratio < 0.012 or change_ratio < 0.08:
            continue
        match = TooltipMatch(confidence >= 0.52, confidence, (left, top, right - left, bottom - top),
                             "yellow_title_panel")
        if match.confidence > best.confidence:
            best = match
    return best


class HoverValidator:
    """Validate image proposals by hovering and observing WoW's GameTooltip."""

    def __init__(self, config: BotConfig, capture: ScreenCapture, input_controller: InputController) -> None:
        self.config, self.capture, self.input = config, capture, input_controller

    def validate(self, candidates: list[BobberDetection], client: Region, deadline: float,
                 safe: Callable[[], bool], debug_dir: Path | None = None) -> HoverValidationResult:
        checked = 0
        for candidate in candidates[:self.config.hover_max_candidates]:
            for dx, dy in self._offsets(candidate):
                if time.monotonic() >= deadline or not safe():
                    return HoverValidationResult(False, points_checked=checked, reason="interrupted")
                before = self.capture.capture(client)
                point = (candidate.x + dx, candidate.y + dy)
                if not self.input.move_to(*point):
                    return HoverValidationResult(False, points_checked=checked, reason="input_blocked")
                time.sleep(min(self.config.hover_tooltip_delay, max(0.0, deadline - time.monotonic())))
                if not safe():
                    return HoverValidationResult(False, points_checked=checked, reason="interrupted")
                after = self.capture.capture(client)
                checked += 1
                tooltip = detect_fishing_tooltip(before, after)
                self._save_debug(before, after, tooltip, debug_dir)
                if tooltip.found:
                    confirmed = BobberDetection(True, point[0], point[1], candidate.confidence,
                                                 candidate.template_name, candidate.top_left, candidate.size)
                    return HoverValidationResult(True, confirmed, tooltip, checked, "tooltip_confirmed")
        return HoverValidationResult(False, points_checked=checked, reason="tooltip_not_found")

    def _offsets(self, candidate: BobberDetection) -> list[tuple[int, int]]:
        largest = max(candidate.size) if candidate.size else 1
        step = max(3, min(self.config.hover_probe_step, largest // 3))
        return [(0, 0), (0, -step), (-step, 0), (step, 0), (0, step)][:self.config.hover_points_per_candidate]

    @staticmethod
    def _save_debug(before: np.ndarray, after: np.ndarray, match: TooltipMatch,
                    directory: Path | None) -> None:
        if directory is None:
            return
        directory.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(directory / "latest_tooltip_before.png"), before)
        output = after.copy()
        x, y, width, height = match.box
        if width and height:
            cv2.rectangle(output, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.imwrite(str(directory / "latest_tooltip_after.png"), output)