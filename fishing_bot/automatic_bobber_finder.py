from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Callable
from pathlib import Path
import json
import threading
import time

import cv2
import numpy as np

from config import BotConfig, Region
from .bobber_detector import BobberDetection
from .capture import ScreenCapture
from .calibration import CalibrationProfile, profile_colour_mask
from .candidate_scorer import CandidateFeatures, score_candidate


@dataclass(frozen=True)
class AutomaticBobberResult:
    detection: BobberDetection
    template: np.ndarray | None = None
    reason: str = "not_found"


@dataclass(frozen=True)
class BobberCandidate:
    x: int
    y: int
    width: int
    height: int
    area: int
    score: float
    features: CandidateFeatures | None = None


@dataclass(frozen=True)
class BackgroundModel:
    median: np.ndarray
    variability: np.ndarray


def build_background(frames: list[np.ndarray] | np.ndarray) -> BackgroundModel:
    items = [frames] if isinstance(frames, np.ndarray) else frames
    if not items:
        raise ValueError("Background requires at least one frame")
    grayscale = np.stack([
        cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        for frame in items
    ]).astype(np.float32)
    median = np.median(grayscale, axis=0)
    variability = np.median(np.abs(grayscale - median), axis=0)
    return BackgroundModel(median.astype(np.uint8), variability)


def preferred_colour_mask(frame: np.ndarray, config: BotConfig,
                          profile: CalibrationProfile | None = None) -> np.ndarray:
    if profile is not None:
        return profile_colour_mask(frame, profile, config.auto_hue_tolerance)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red_or_blue = ((hue <= 16) | (hue >= 164) | ((hue >= 90) & (hue <= 136)))
    mask = red_or_blue & (saturation >= config.auto_min_saturation) & (value >= config.auto_min_value)
    return mask.astype(np.uint8) * 255


def novelty_mask(before: list[np.ndarray] | np.ndarray | BackgroundModel, after: np.ndarray, config: BotConfig,
                 profile: CalibrationProfile | None = None) -> np.ndarray:
    background = before if isinstance(before, BackgroundModel) else build_background(before)
    after_gray = cv2.GaussianBlur(cv2.cvtColor(after, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    difference = cv2.absdiff(background.median, after_gray).astype(np.float32)
    threshold = np.maximum(config.auto_find_pixel_difference,
                           background.variability * config.auto_background_noise_multiplier + 8)
    changed = difference >= threshold
    colour = preferred_colour_mask(after, config, profile) > 0
    hsv = cv2.cvtColor(after, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    light_body = ((hue >= 4) & (hue <= 38) & (saturation >= 25) & (value >= 85))
    bright_metal = (saturation <= 85) & (value >= 150)
    appearance = colour | light_body | bright_metal
    nearby_change = cv2.dilate(changed.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    mask = (appearance & nearby_change).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.dilate(mask, np.ones((3, 3), np.uint8))


def find_candidates(mask: np.ndarray, config: BotConfig,
                    profile: CalibrationProfile | None = None,
                    frame: np.ndarray | None = None,
                    templates: list[np.ndarray] | None = None) -> list[BobberCandidate]:
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    center_x, center_y = mask.shape[1] / 2, mask.shape[0] / 2
    diagonal = max(1.0, (mask.shape[0] ** 2 + mask.shape[1] ** 2) ** 0.5)
    candidates: list[BobberCandidate] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        # A profile measures the coloured feather while this mask may also join the
        # cork body and metal fitting, so it must never narrow the safe defaults.
        minimum = min(config.auto_find_min_area, profile.min_area) if profile else config.auto_find_min_area
        maximum = max(config.auto_find_max_area, profile.max_area) if profile else config.auto_find_max_area
        if not minimum <= area <= maximum:
            continue
        left, top = int(stats[index, cv2.CC_STAT_LEFT]), int(stats[index, cv2.CC_STAT_TOP])
        width, height = int(stats[index, cv2.CC_STAT_WIDTH]), int(stats[index, cv2.CC_STAT_HEIGHT])
        x, y = (int(round(value)) for value in centroids[index])
        centrality = 1.0 - (((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5 / diagonal)
        features = score_candidate(frame, (left, top, width, height), config.auto_candidate_padding,
                                   templates, config.auto_template_scales) \
            if frame is not None else None
        appearance_score = features.total if features is not None else min(1.0, area / 30.0)
        # Centrality is intentionally weak: casts can land near either edge.
        score = appearance_score * 0.9 + max(0.0, centrality) * 0.1
        candidates.append(BobberCandidate(x, y, width, height, area, score, features))
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


class AutomaticBobberFinder:
    """Find the stable coloured object that appeared after a cast."""

    def __init__(self, config: BotConfig, capture: ScreenCapture) -> None:
        self.config, self.capture = config, capture
        self.profile = CalibrationProfile.load(config.calibration_profile_path)
        self.templates = [image for path in sorted(config.templates_dir.glob("*.png"))
                          if (image := cv2.imread(str(path), cv2.IMREAD_COLOR)) is not None]

    def find(
        self, before: list[np.ndarray] | np.ndarray, region: Region, deadline: float,
        stop_event: threading.Event, safe: Callable[[], bool], debug_dir: Path | None = None,
    ) -> AutomaticBobberResult:
        previous: BobberCandidate | None = None
        confirmations = 0
        background = build_background(before)
        ambiguous_seen = False
        while time.monotonic() < deadline and not stop_event.is_set() and safe():
            frame = self.capture.capture(region)
            mask = novelty_mask(background, frame, self.config, self.profile)
            candidates = find_candidates(mask, self.config, self.profile, frame, self.templates)
            candidates = [item for item in candidates if item.score >= self.config.auto_min_candidate_score]
            if len(candidates) > 1 and candidates[0].score - candidates[1].score < self.config.auto_min_score_gap:
                ambiguous_seen = True
                candidates = []
            candidate = self._nearest_confirmed(previous, candidates)
            if candidate is None:
                confirmations, previous = 0, None
            elif previous is not None:
                confirmations += 1
                previous = candidate
            else:
                confirmations, previous = 1, candidate

            if debug_dir is not None:
                self._debug(frame, mask, candidates, debug_dir)
            if previous is not None and confirmations >= self.config.auto_find_confirmation_frames:
                template, size = self._crop_template(frame, previous)
                detection = BobberDetection(
                    True, region.left + previous.x, region.top + previous.y, 1.0,
                    "<automatic>", size=size,
                )
                return AutomaticBobberResult(detection, template, "novel_colour_component")
            if stop_event.wait(self.config.bobber_search_interval):
                break
        reason = "ambiguous_candidates" if ambiguous_seen else "automatic_search_timeout"
        return AutomaticBobberResult(BobberDetection(False), reason=reason)

    def _nearest_confirmed(self, previous: BobberCandidate | None,
                           candidates: list[BobberCandidate]) -> BobberCandidate | None:
        if not candidates:
            return None
        if previous is None:
            return candidates[0]
        nearby = [candidate for candidate in candidates if
                  ((candidate.x - previous.x) ** 2 + (candidate.y - previous.y) ** 2) ** 0.5
                  <= self.config.auto_find_match_radius]
        return max(nearby, key=lambda candidate: candidate.score) if nearby else None

    def _crop_template(self, frame: np.ndarray, candidate: BobberCandidate) -> tuple[np.ndarray, tuple[int, int]]:
        padding = self.config.auto_find_template_padding
        left = max(0, candidate.x - candidate.width // 2 - padding)
        top = max(0, candidate.y - candidate.height // 2 - padding)
        right = min(frame.shape[1], candidate.x + (candidate.width + 1) // 2 + padding)
        bottom = min(frame.shape[0], candidate.y + (candidate.height + 1) // 2 + padding)
        template = frame[top:bottom, left:right].copy()
        return template, (template.shape[1], template.shape[0])

    @staticmethod
    def _debug(frame: np.ndarray, mask: np.ndarray, candidates: list[BobberCandidate], directory: Path) -> None:
        output = frame.copy()
        for candidate in candidates[:10]:
            cv2.circle(output, (candidate.x, candidate.y), 7, (0, 255, 255), 1)
            cv2.putText(output, f"{candidate.score:.2f}", (candidate.x + 5, candidate.y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        directory.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(directory / "latest_post_cast.png"), frame)
        cv2.imwrite(str(directory / "latest_novelty_mask.png"), mask)
        cv2.imwrite(str(directory / "latest_auto_candidates.png"), output)
        diagnostics = [{
            "x": candidate.x, "y": candidate.y, "width": candidate.width,
            "height": candidate.height, "area": candidate.area, "score": candidate.score,
            "features": asdict(candidate.features) if candidate.features is not None else None,
        } for candidate in candidates[:10]]
        (directory / "latest_auto_candidates.json").write_text(
            json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
        )
