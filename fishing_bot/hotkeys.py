from __future__ import annotations

from collections.abc import Callable


class HotkeyManager:
    def __init__(self, pause_key: str, stop_key: str, on_pause: Callable[[], None], on_stop: Callable[[], None]) -> None:
        from pynput import keyboard
        mapping = {f"<{pause_key.lower()}>": on_pause, f"<{stop_key.lower()}>": on_stop}
        self._listener = keyboard.GlobalHotKeys(mapping)

    def start(self) -> None:
        self._listener.start()

    def close(self) -> None:
        self._listener.stop()
        self._listener.join(timeout=1)
