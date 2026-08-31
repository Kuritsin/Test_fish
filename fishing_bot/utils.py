from __future__ import annotations

import ctypes
import logging
import os


def enable_dpi_awareness() -> None:
    """Make MSS and pynput use the same physical-pixel coordinate space."""
    if os.name != "nt":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            logging.getLogger(__name__).warning("Could not enable Windows DPI awareness")


def configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
