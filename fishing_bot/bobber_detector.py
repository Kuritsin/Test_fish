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
    def __init__(self, directory: Path, confidence: float = 0.75, grayscale: bool = True,
                 scales: tuple[float, ...] = (1.0,), edge_weight: float = 0.0) -> None:
        self.confidence = confidence
        self.grayscale = grayscale
        self.scales = scales
        self.edge_weight = min(0.8, max(0.0, edge_weight))
        self.templates: list[tuple[str, np.ndarray]] = []
        self.colour_templates: dict[str, np.ndarray] = {}
        self.template_masks: dict[str, np.ndarray] = {}
        for path in sorted(directory.glob("*.png")):
            unchanged = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if unchanged is not None:
                colour, mask = self._prepare_template(unchanged)
                image = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY) if grayscale else colour
                self.templates.append((path.name, image))
                self.colour_templates[path.name] = colour
                if mask is not None:
                    self.template_masks[path.name] = mask

    @staticmethod
    def _prepare_template(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if image.ndim != 3 or image.shape[2] != 4:
            return image[:, :, :3] if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), None
        alpha = image[:, :, 3]
        points = cv2.findNonZero((alpha >= 16).astype(np.uint8))
        if points is None:
            return image[:, :, :3], None
        left, top, width, height = cv2.boundingRect(points)
        return (image[top:top + height, left:left + width, :3].copy(),
                alpha[top:top + height, left:left + width])

    def template_by_name(self, name: str) -> np.ndarray:
        try:
            return self.colour_templates[name]
        except KeyError:
            raise KeyError(f"Unknown bobber template: {name}") from None

    def scores(self, frame: np.ndarray) -> list[TemplateScore]:
        source = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if self.grayscale and frame.ndim == 3 else frame
        scores: list[TemplateScore] = []
        source_gray = source if source.ndim == 2 else cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        source_edges = cv2.Canny(source_gray, 45, 130) if self.edge_weight else None
        for name, template in self.templates:
            template_gray = template if template.ndim == 2 else cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            for scale in self.scales:
                width = max(3, int(round(template_gray.shape[1] * scale)))
                height = max(3, int(round(template_gray.shape[0] * scale)))
                if height > source_gray.shape[0] or width > source_gray.shape[1]:
                    continue
                resized = cv2.resize(template_gray, (width, height), interpolation=cv2.INTER_AREA)
                mask = self.template_masks.get(name)
                resized_mask = (cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                                if mask is not None else None)
                method = cv2.TM_CCORR_NORMED if resized_mask is not None else cv2.TM_CCOEFF_NORMED
                result = cv2.matchTemplate(source_gray, resized, method, mask=resized_mask)
                result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)
                _, gray_score, _, location = cv2.minMaxLoc(result)
                confidence = float(gray_score)
                if source_edges is not None:
                    template_edges = cv2.Canny(resized, 45, 130)
                    if np.count_nonzero(template_edges) >= 4:
                        edge_result = cv2.matchTemplate(source_edges, template_edges, cv2.TM_CCOEFF_NORMED)
                        edge_score = float(edge_result[location[1], location[0]])
                        confidence = ((1.0 - self.edge_weight) * confidence
                                      + self.edge_weight * max(0.0, edge_score))
                scores.append(TemplateScore(name, confidence, location, (width, height)))
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

    def detect_candidates(self, frame: np.ndarray, region: Region, limit: int = 5,
                          minimum_confidence: float = 0.0) -> list[BobberDetection]:
        """Return distinct local maxima for hover validation."""
        source = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        source_edges = cv2.Canny(source, 45, 130) if self.edge_weight else None
        proposals: list[BobberDetection] = []
        for name, template in self.templates:
            template_gray = template if template.ndim == 2 else cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            for scale in self.scales:
                width = max(3, int(round(template_gray.shape[1] * scale)))
                height = max(3, int(round(template_gray.shape[0] * scale)))
                if height > source.shape[0] or width > source.shape[1]:
                    continue
                resized = cv2.resize(template_gray, (width, height), interpolation=cv2.INTER_AREA)
                mask = self.template_masks.get(name)
                resized_mask = (cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                                if mask is not None else None)
                method = cv2.TM_CCORR_NORMED if resized_mask is not None else cv2.TM_CCOEFF_NORMED
                response = cv2.matchTemplate(source, resized, method, mask=resized_mask)
                work = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
                edge_response = None
                edges = cv2.Canny(resized, 45, 130)
                if source_edges is not None and np.count_nonzero(edges) >= 4:
                    edge_response = cv2.matchTemplate(source_edges, edges, cv2.TM_CCOEFF_NORMED)
                for _ in range(limit):
                    _, gray_score, _, location = cv2.minMaxLoc(work)
                    confidence = float(gray_score)
                    if edge_response is not None:
                        confidence = ((1.0 - self.edge_weight) * confidence
                                      + self.edge_weight * max(0.0, float(edge_response[location[1], location[0]])))
                    if confidence < minimum_confidence:
                        break
                    left, top = location
                    proposals.append(BobberDetection(True, region.left + left + width // 2,
                                                     region.top + top + height // 2, confidence,
                                                     name, location, (width, height)))
                    work[max(0, top - height):min(work.shape[0], top + height + 1),
                         max(0, left - width):min(work.shape[1], left + width + 1)] = -1.0
        distinct: list[BobberDetection] = []
        for proposal in sorted(proposals, key=lambda item: item.confidence, reverse=True):
            radius = max(proposal.size[0], proposal.size[1], 16)
            if any((proposal.x - item.x) ** 2 + (proposal.y - item.y) ** 2 <= radius ** 2
                   for item in distinct):
                continue
            distinct.append(proposal)
            if len(distinct) >= limit:
                break
        return distinct

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