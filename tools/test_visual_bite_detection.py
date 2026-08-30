import _bootstrap  # noqa: F401
import threading
import time

from config import CONFIG
from fishing_bot.auto_bite_detector import AutoBiteDetector
from fishing_bot.bobber_detector import BobberDetector
from fishing_bot.capture import ScreenCapture
from fishing_bot.search_region import resolve_search_region
from fishing_bot.utils import enable_dpi_awareness
from fishing_bot.visual_bite_detector import VisualBiteDetector
from fishing_bot.window_manager import WindowManager


def main() -> int:
    enable_dpi_awareness()
    detector = BobberDetector(CONFIG.templates_dir, CONFIG.bobber_confidence, CONFIG.grayscale_matching)
    if not detector.templates:
        print("Нет PNG-шаблонов. Сначала прочитайте templates/README.md")
        return 1
    print("Забросьте удочку и переключитесь в WoW. Снимок будет сделан через 5 секунд.")
    time.sleep(5)
    window = WindowManager(CONFIG.wow_window_titles, CONFIG.wow_process_names)
    if not window.is_wow_active():
        print("Активное окно не похоже на World of Warcraft")
        return 1
    search_region = resolve_search_region(CONFIG, window)
    client_region = window.foreground_client_region()
    if client_region is None:
        print("Не удалось определить клиентскую область WoW")
        return 1
    with ScreenCapture() as capture:
        found = detector.detect(capture.capture(search_region), search_region)
        if not found.found:
            print(f"Поплавок не найден; лучший confidence={found.confidence:.3f}")
            return 1
        print(f"Поплавок найден: x={found.x} y={found.y}. Режим: {CONFIG.bite_detection_mode}.")
        watcher = (AutoBiteDetector(CONFIG, capture) if CONFIG.bite_detection_mode == "auto"
                   else VisualBiteDetector(CONFIG, capture))
        result = watcher.wait_for_bite(
            found, detector.template_by_name(found.template_name), client_region,
            CONFIG.fishing_attempt_timeout, threading.Event(), window.is_wow_active,
            CONFIG.debug_dir / ("latest_auto_bite.png" if CONFIG.bite_detection_mode == "auto"
                                else "latest_visual_bite.png"),
        )
    print("BITE" if result.detected else "TIMEOUT", result)
    return 0 if result.detected else 2


if __name__ == "__main__":
    raise SystemExit(main())
