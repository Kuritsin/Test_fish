import _bootstrap  # noqa: F401
import threading

from config import CONFIG
from fishing_bot.audio_detector import AudioDetector


def report(rms: float, baseline: float, threshold: float, spike: bool) -> None:
    print(f"rms={rms:8.1f} baseline={baseline:8.1f} threshold={threshold:8.1f} {'SPIKE' if spike else ''}")


print(f"Listening for {CONFIG.fishing_attempt_timeout} seconds (Ctrl+C to stop)...")
detector = AudioDetector(CONFIG)
try:
    result = detector.wait_for_bite(CONFIG.fishing_attempt_timeout, threading.Event(), report)
    print("BITE" if result.detected else "TIMEOUT", result)
except KeyboardInterrupt:
    detector.close()
