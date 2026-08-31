from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import time

import cv2

from config import BotConfig, Region, save_search_region
from .capture import ScreenCapture
from .window_manager import WindowManager


def roi_to_screen_region(window: Region, roi: tuple[int, int, int, int]) -> Region:
    x, y, width, height = (int(value) for value in roi)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("The selected area must have a positive size")
    if x + width > window.width or y + height > window.height:
        raise ValueError("The selected area is outside the WoW client area")
    return Region(window.left + x, window.top + y, width, height)


def select_search_region(
    config: BotConfig,
    output_path: Path,
    countdown: int = 5,
    sleep: Callable[[float], None] = time.sleep,
) -> Region | None:
    """Let the user draw the search region over a snapshot of the WoW window."""
    window = WindowManager(config.wow_window_titles, config.wow_process_names)
    print("Переключитесь в World of Warcraft. Окно игры должно стать активным.")
    for remaining in range(countdown, 0, -1):
        print(f"Снимок через {remaining}...", flush=True)
        sleep(1)

    if not window.is_wow_active():
        raise RuntimeError("Активное окно не похоже на World of Warcraft")
    client_region = window.foreground_client_region()
    if client_region is None:
        raise RuntimeError("Не удалось определить клиентскую область окна WoW")

    with ScreenCapture() as capture:
        frame = capture.capture(client_region)
    title = "Select water area, then ENTER/SPACE; C or ESC cancels"
    roi = cv2.selectROI(title, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    if int(roi[2]) <= 0 or int(roi[3]) <= 0:
        print("Выбор отменён. Настройки не изменены.")
        return None

    selected = roi_to_screen_region(client_region, roi)
    save_search_region(selected, output_path)
    config.debug_dir.mkdir(parents=True, exist_ok=True)
    x, y, width, height = (int(value) for value in roi)
    preview = frame[y:y + height, x:x + width]
    cv2.imwrite(str(config.debug_dir / "selected_search_region.png"), preview)
    print(f"Область сохранена: {selected}")
    print(f"Настройки: {output_path.resolve()}")
    print("Следующий шаг: python tools/test_bobber_detection.py")
    return selected
