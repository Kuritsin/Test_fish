from __future__ import annotations

import argparse
from dataclasses import replace

from config import CONFIG, USER_CONFIG_PATH
from fishing_bot.bot import FishingBot
from fishing_bot.hotkeys import HotkeyManager
from fishing_bot.region_selector import select_search_region
from fishing_bot.setup_wizard import run_setup
from fishing_bot.calibration_wizard import run_calibration
from fishing_bot.utils import configure_logging, enable_dpi_awareness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="World of Warcraft fishing automation")
    parser.add_argument("--debug", action="store_true", help="verbose logs and overwrite debug/latest_*.png")
    parser.add_argument("--dry-run", action="store_true", help="disable all keyboard and mouse input")
    parser.add_argument("--once", action="store_true", help="finish after one fishing attempt")
    parser.add_argument("--select-region", action="store_true", help="select the water area with the mouse and exit")
    parser.add_argument("--full-window", action="store_true", help="temporarily search the entire WoW client area")
    parser.add_argument("--bite-mode", choices=("auto", "visual", "audio"), help="temporarily override bite detection mode")
    parser.add_argument("--setup", action="store_true", help="configure common settings without editing Python files")
    parser.add_argument("--calibrate", action="store_true", help="run guided automatic bobber calibration")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = replace(
        CONFIG,
        debug=CONFIG.debug or args.debug,
        search_mode="window" if args.full_window else CONFIG.search_mode,
        bite_detection_mode=args.bite_mode or CONFIG.bite_detection_mode,
    )
    configure_logging(config.debug)
    enable_dpi_awareness()
    if args.setup:
        try:
            run_setup(config, USER_CONFIG_PATH)
        except (EOFError, ValueError) as error:
            print(f"Ошибка настройки: {error}")
            return 1
        return 0
    if args.calibrate:
        try:
            run_calibration(config)
        except (EOFError, RuntimeError, ValueError) as error:
            print(f"Ошибка калибровки: {error}")
            return 1
        return 0
    if args.select_region:
        try:
            select_search_region(config, USER_CONFIG_PATH)
        except (RuntimeError, ValueError) as error:
            print(f"Ошибка выбора области: {error}")
            return 1
        return 0
    bot = FishingBot(config, dry_run=args.dry_run)
    hotkeys = HotkeyManager(config.pause_key, config.stop_key, bot.toggle_pause, bot.stop)
    hotkeys.start()
    try:
        bot.run(once=args.once)
    except KeyboardInterrupt:
        bot.stop()
    finally:
        hotkeys.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
