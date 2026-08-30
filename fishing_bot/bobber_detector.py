from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import Region


@dataclass(frozen=True)
class TemplateScore:
    name: str
    confidence: float
    location: tuple[int, int]
    size: tuple[int, int]


@dataclass(frozen=True)
class BobberDetection:
    found: bool
    x: int = 0
    y: int = 0
    confidence: float = 0.0
    template_name: str = ""
    top_left: tuple[int, int] = (0, 0)
    size: tuple[int, int] = (0, 0)


class BobberDetector:
    def __init__(self, directory: Path, confidence: float = 0.75, grayscale: bool = True) -> None:
        self.confidence = confidence
        self.grayscale = grayscale
        self.templates: list[tuple[str, np.ndarray]] = []
        self.colour_templates: dict[str, np.ndarray] = {}
        for path in sorted(directory.glob("*.png")):
            colour = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if colour is not None:
                image = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY) if grayscale else colour
                self.templates.append((path.name, image))
                self.colour_templates[path.name] = colour

    def template_by_name(self, name: str) -> np.ndarray:
        try:
            return self.colour_templates[name]
        except KeyError:
            raise KeyError(f"Unknown bobber template: {name}") from None

    def scores(self, frame: np.ndarray) -> list[TemplateScore]:
        source = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if self.grayscale and frame.ndim == 3 else frame
        scores: list[TemplateScore] = []
        for name, template in self.templates:
            height, width = template.shape[:2]
            if height > source.shape[0] or width > source.shape[1]:
                continue
            result = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED)
            _, maximum, _, location = cv2.minMaxLoc(result)
            scores.append(TemplateScore(name, float(maximum), location, (width, height)))
        return scores

    def detect(self, frame: np.ndarray, region: Region) -> BobberDetection:
        scores = self.scores(frame)
        if not scores:
            return BobberDetection(False)
        best = max(scores, key=lambda item: item.confidence)
        left, top = best.location
        width, height = best.size
        return BobberDetection(
            best.confidence >= self.confidence,
            region.left + left + width // 2,
            region.top + top + height // 2,
            best.confidence,
            best.name,
            best.location,
            best.size,
        )

    @staticmethod
    def visualize(frame: np.ndarray, result: BobberDetection) -> np.ndarray:
        output = frame.copy()
        left, top = result.top_left
        width, height = result.size
        if width and height:
            color = (0, 255, 0) if result.found else (0, 165, 255)
            cv2.rectangle(output, (left, top), (left + width, top + height), color, 2)
            cv2.putText(output, f"{result.template_name} {result.confidence:.3f}",
                        (left, max(18, top - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return output
