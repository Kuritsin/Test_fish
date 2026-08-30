from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CalibrationSample:
    hue: float
    saturation: float
    value: float
    coloured_area: int


@dataclass(frozen=True)
class CalibrationProfile:
    hues: tuple[int, ...]
    min_saturation: int
    min_value: int
    min_area: int
    max_area: int
    sample_count: int
    schema_version: int = 2

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CalibrationProfile | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("schema_version", 1) not in {1, 2}:
                return None
            data["hues"] = tuple(int(value) for value in data["hues"])
            data["schema_version"] = 2
            return cls(**data)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None


def sample_bobber(image: np.ndarray) -> CalibrationSample:
    """Measure saturated pixels in a user-confirmed bobber crop."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    preferred_hue = ((pixels[:, 0] <= 16) | (pixels[:, 0] >= 164)
                     | ((pixels[:, 0] >= 90) & (pixels[:, 0] <= 136)))
    coloured = pixels[preferred_hue & (pixels[:, 1] >= 45) & (pixels[:, 2] >= 35)]
    if coloured.size == 0:
        coloured = pixels[(pixels[:, 1] >= 45) & (pixels[:, 2] >= 35)]
    if coloured.size == 0:
        coloured = pixels
    histogram = np.bincount(coloured[:, 0], minlength=180)
    dominant_hue = int(np.argmax(histogram))
    selected_hues = np.minimum(np.abs(coloured[:, 0].astype(np.int16) - dominant_hue),
                               180 - np.abs(coloured[:, 0].astype(np.int16) - dominant_hue)) <= 12
    selected = coloured[selected_hues]
    component_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    hsv_image = hsv
    hue_distance = np.minimum(np.abs(hsv_image[:, :, 0].astype(np.int16) - dominant_hue),
                              180 - np.abs(hsv_image[:, :, 0].astype(np.int16) - dominant_hue))
    component_mask[(hue_distance <= 12) & (hsv_image[:, :, 1] >= 45) & (hsv_image[:, :, 2] >= 35)] = 1
    count, _, stats, _ = cv2.connectedComponentsWithStats(component_mask, 8)
    component_area = max((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)), default=len(selected))
    return CalibrationSample(
        hue=float(dominant_hue),
        saturation=float(np.percentile(selected[:, 1], 20)),
        value=float(np.percentile(selected[:, 2], 20)),
        coloured_area=component_area,
    )


def build_profile(samples: list[CalibrationSample]) -> CalibrationProfile:
    if not samples:
        raise ValueError("At least one confirmed bobber sample is required")
    areas = np.asarray([sample.coloured_area for sample in samples])
    return CalibrationProfile(
        hues=tuple(sorted({int(round(sample.hue)) for sample in samples})),
        min_saturation=max(35, int(min(sample.saturation for sample in samples) * 0.75)),
        min_value=max(30, int(min(sample.value for sample in samples) * 0.75)),
        min_area=max(2, int(np.percentile(areas, 10) * 0.35)),
        max_area=max(20, int(np.percentile(areas, 90) * 2.5)),
        sample_count=len(samples),
    )


def profile_colour_mask(frame: np.ndarray, profile: CalibrationProfile, hue_tolerance: int) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    selected = np.zeros(hue.shape, dtype=bool)
    for expected in profile.hues:
        distance = np.minimum(np.abs(hue.astype(np.int16) - expected),
                              180 - np.abs(hue.astype(np.int16) - expected))
        selected |= distance <= hue_tolerance
    selected &= saturation >= profile.min_saturation
    selected &= value >= profile.min_value
    return selected.astype(np.uint8) * 255
