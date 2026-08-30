import _bootstrap  # noqa: F401
import cv2
import time

from config import CONFIG
from fishing_bot.capture import ScreenCapture
from fishing_bot.search_region import resolve_search_region
from fishing_bot.utils import enable_dpi_awareness
from fishing_bot.window_manager import WindowManager

enable_dpi_awareness()
window = WindowManager(CONFIG.wow_window_titles, CONFIG.wow_process_names)
if CONFIG.search_mode == "window":
    print("Переключитесь в WoW: снимок всего окна будет сделан через 5 секунд.")
    time.sleep(5)
    if not window.is_wow_active():
        raise SystemExit("Активное окно не похоже на World of Warcraft")
region = resolve_search_region(CONFIG, window)
CONFIG.debug_dir.mkdir(exist_ok=True)
with ScreenCapture() as capture:
    image = capture.capture(region)
path = CONFIG.debug_dir / "latest_capture.png"
cv2.imwrite(str(path), image)
print(f"Saved {path} from {region} (mode={CONFIG.search_mode})")
