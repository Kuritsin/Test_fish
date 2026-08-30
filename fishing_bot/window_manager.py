from __future__ import annotations

import ctypes
import os

from config import Region


class WindowManager:
    def __init__(self, titles: tuple[str, ...], processes: tuple[str, ...]) -> None:
        self.titles = tuple(value.lower() for value in titles)
        self.processes = tuple(value.lower() for value in processes)

    def foreground_title(self) -> str:
        if os.name != "nt":
            return ""
        user32 = ctypes.windll.user32
        handle = user32.GetForegroundWindow()
        length = user32.GetWindowTextLengthW(handle)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(handle, buffer, length + 1)
        return buffer.value

    def foreground_client_region(self) -> Region | None:
        """Return the foreground window client area in physical screen pixels."""
        if os.name != "nt":
            return None
        user32 = ctypes.windll.user32
        handle = user32.GetForegroundWindow()
        if not handle:
            return None

        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                        ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        rectangle = RECT()
        origin = POINT(0, 0)
        if not user32.GetClientRect(handle, ctypes.byref(rectangle)):
            return None
        if not user32.ClientToScreen(handle, ctypes.byref(origin)):
            return None
        width = rectangle.right - rectangle.left
        height = rectangle.bottom - rectangle.top
        if width <= 0 or height <= 0:
            return None
        return Region(origin.x, origin.y, width, height)

    def is_wow_active(self) -> bool:
        title = self.foreground_title().lower()
        return bool(title) and any(allowed in title for allowed in self.titles)
