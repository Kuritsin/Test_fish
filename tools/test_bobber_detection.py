import _bootstrap  # noqa: F401
import cv2
import time

from config import CONFIG
from fishing_bot.bobber_detector import BobberDetector
from fishing_bot.capture import ScreenCapture
from fishing_bot.search_region import resolve_search_region
from fishing_bot.utils import enable_dpi_awareness
from fishing_bot.window_manager import WindowManager

enable_dpi_awareness()
detector = BobberDetector(CONFIG.templates_dir, CONFIG.bobber_confidence, CONFIG.grayscale_matching)
if not detector.templates:
    raise SystemExit("No PNG templates found. See templates/README.md")
window = WindowManager(CONFIG.wow_window_titles, CONFIG.wow_process_names)
if CONFIG.search_mode == "window":
    print("Переключитесь в WoW: проверка всего окна начнётся через 5 секунд.")
    time.sleep(5)
    if not window.is_wow_active():
        raise SystemExit("Активное окно не похоже на World of Warcraft")
region = resolve_search_region(CONFIG, window)
with ScreenCapture() as capture:
    frame = capture.capture(region)
for score in detector.scores(frame):
    print(f"{score.name}: confidence={score.confidence:.4f}")
best = detector.detect(frame, region)
print(f"BEST: {best.template_name}, confidence={best.confidence:.4f}, found={best.found}, x={best.x}, y={best.y}")
preview = detector.visualize(frame, best)
cv2.imshow("Bobber detection (press any key)", preview)
cv2.waitKey(0)
cv2.destroyAllWindows()
