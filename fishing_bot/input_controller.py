from __future__ import annotations

import logging
import time
from typing import Callable


class InputController:
    def __init__(self, safe_to_send: Callable[[], bool], dry_run: bool = False,
                 delay: float = 0.05, move_duration: float = 0.15) -> None:
        self.safe_to_send = safe_to_send
        self.dry_run = dry_run
        self.delay = delay
        self.move_duration = move_duration
        self.log = logging.getLogger(__name__)
        self._keyboard = self._mouse = None

    def _controllers(self):
        if self._keyboard is None:
            from pynput import keyboard, mouse
            self._keyboard, self._mouse = keyboard.Controller(), mouse.Controller()
        return self._keyboard, self._mouse

    def _allowed(self) -> bool:
        return self.safe_to_send()

    def _dry_run_success(self, action: str) -> bool:
        if not self.dry_run:
            return False
        self.log.info("DRY RUN: %s suppressed", action)
        return True

    def press_key(self, key: str) -> bool:
        if not self._allowed():
            return False
        if self._dry_run_success(f"press key {key!r}"):
            return True
        keyboard, _ = self._controllers()
        keyboard.press(key); time.sleep(self.delay); keyboard.release(key)
        return True

    def cast(self, key: str) -> bool:
        return self.press_key(key)

    def use_bait(self, key: str) -> bool:
        return self.press_key(key)

    def move_to(self, x: int, y: int) -> bool:
        if not self._allowed():
            return False
        if self._dry_run_success(f"move mouse to ({x}, {y})"):
            return True
        _, mouse = self._controllers()
        start_x, start_y = mouse.position
        steps = max(1, int(self.move_duration / 0.01))
        for step in range(1, steps + 1):
            if not self.safe_to_send():
                return False
            fraction = step / steps
            mouse.position = (int(start_x + (x - start_x) * fraction), int(start_y + (y - start_y) * fraction))
            time.sleep(self.move_duration / steps)
        return True

    def right_click(self) -> bool:
        if not self._allowed():
            return False
        if self._dry_run_success("right click"):
            return True
        from pynput.mouse import Button
        _, mouse = self._controllers()
        mouse.click(Button.right)
        return True

    def loot(self) -> bool:
        return self.right_click()

    def release_modifiers(self) -> None:
        if self._keyboard is None:
            return
        from pynput.keyboard import Key
        for key in (Key.shift, Key.ctrl, Key.alt):
            self._keyboard.release(key)
