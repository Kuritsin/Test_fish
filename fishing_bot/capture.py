from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from config import Region


class ScreenCapture:
    def __init__(self) -> None:
        import mss
        self._mss: Any = mss.mss()

    def capture(self, region: Region) -> np.ndarray:
        bgra = np.asarray(self._mss.grab(region.as_mss()))
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

    def close(self) -> None:
        self._mss.close()

    def __enter__(self) -> "ScreenCapture":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
