from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CandidateFeatures:
    colour: float
    body: float
    diamond: float
    edges: float
    shape: float
    line: float
    template: float
    total: float


def score_candidate(frame: np.ndarray, box: tuple[int, int, int, int], padding: int = 12,
                    templates: list[np.ndarray] | None = None,
                    scales: tuple[float, ...] = (1.0,)) -> CandidateFeatures:
    """Score the feather/body/metal/line layout visible around a proposal."""
    left, top, width, height = box
    x0, y0 = max(0, left - padding), max(0, top - padding * 2)
    x1 = min(frame.shape[1], left + width + padding)
    y1 = min(frame.shape[0], top + height + padding)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return CandidateFeatures(0, 0, 0, 0, 0, 0, 0, 0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    feather = (((hue <= 18) | (hue >= 162) | ((hue >= 90) & (hue <= 138)))
               & (saturation >= 70) & (value >= 45))
    body = (hue >= 4) & (hue <= 38) & (saturation >= 25) & (saturation <= 230) & (value >= 85)
    metal = (saturation <= 85) & (value >= 150)
    pixel_count = max(1, roi.shape[0] * roi.shape[1])
    colour_score = min(1.0, float(np.count_nonzero(feather)) / max(3.0, pixel_count * 0.035))
    body_score = min(1.0, float(np.count_nonzero(body)) / max(5.0, pixel_count * 0.07))

    upper = metal[:max(1, roi.shape[0] * 2 // 3)]
    count, _, stats, _ = cv2.connectedComponentsWithStats(upper.astype(np.uint8), 8)
    compact = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)
               if 2 <= stats[i, cv2.CC_STAT_AREA] <= max(25, pixel_count * 0.12)]
    diamond_score = min(1.0, max(compact, default=0) / 8.0)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 130)
    edge_score = min(1.0, float(np.count_nonzero(edges)) / max(8.0, pixel_count * 0.12))
    aspect = width / max(1, height)
    shape_score = max(0.0, 1.0 - abs(np.log(max(0.15, aspect))) / 2.2)

    strip_width = max(3, width)
    center = min(roi.shape[1] - 1, max(0, left + width // 2 - x0))
    sx0, sx1 = max(0, center - strip_width), min(roi.shape[1], center + strip_width + 1)
    vertical = cv2.Sobel(gray[:, sx0:sx1], cv2.CV_32F, 1, 0, ksize=3)
    line_score = min(1.0, float(np.mean(np.abs(vertical) > 35)) * 9.0) if vertical.size else 0.0

    template_score = _template_score(roi, templates or [], scales)
    base = (0.25 * colour_score + 0.24 * body_score + 0.15 * diamond_score
            + 0.15 * edge_score + 0.13 * shape_score + 0.08 * line_score)
    total = base if not templates else 0.72 * base + 0.28 * template_score
    return CandidateFeatures(colour_score, body_score, diamond_score, edge_score,
                             shape_score, line_score, template_score, float(total))


def _template_score(roi: np.ndarray, templates: list[np.ndarray], scales: tuple[float, ...]) -> float:
    if not templates:
        return 0.0
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_edges = cv2.Canny(roi_gray, 45, 130)
    best = 0.0
    for template in templates:
        for scale in scales:
            width = max(3, int(round(template.shape[1] * scale)))
            height = max(3, int(round(template.shape[0] * scale)))
            if width > roi.shape[1] or height > roi.shape[0]:
                continue
            resized = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            colour_match = float(cv2.minMaxLoc(cv2.matchTemplate(roi_gray, gray, cv2.TM_CCOEFF_NORMED))[1])
            edges = cv2.Canny(gray, 45, 130)
            if np.count_nonzero(edges) >= 4:
                edge_match = float(cv2.minMaxLoc(cv2.matchTemplate(roi_edges, edges, cv2.TM_CCOEFF_NORMED))[1])
            else:
                edge_match = 0.0
            best = max(best, max(0.0, 0.6 * colour_match + 0.4 * edge_match))
    return min(1.0, best)
