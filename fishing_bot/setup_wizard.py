from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from config import BotConfig, save_user_settings
from .window_manager import WindowManager


def run_setup(config: BotConfig, path: Path, ask: Callable[[str], str] = input) -> None:
    print("=== Первоначальная настройка WoW Fishing Bot ===")
    window = WindowManager(config.wow_window_titles, config.wow_process_names)
    title = window.foreground_title()
    if title:
        print(f"Текущее активное окно: {title}")
    else:
        print("Окно WoW сейчас не активно — это не мешает сохранить клавиши.")

    cast_key = ask(f"Клавиша заброса [{config.cast_key}]: ").strip() or config.cast_key
    use_bait_text = ask("Использовать наживку? [Y/n]: ").strip().lower()
    use_bait = use_bait_text not in {"n", "no", "нет"}
    bait_key = config.bait_key
    if use_bait:
        bait_key = ask(f"Клавиша наживки [{config.bait_key}]: ").strip() or config.bait_key
    mode = ask("Режим поклёвки auto/visual/audio [auto]: ").strip().lower() or "auto"
    if mode not in {"auto", "visual", "audio"}:
        raise ValueError("Режим должен быть auto, visual или audio")
    save_user_settings({
        "cast_key": cast_key,
        "bait_key": bait_key,
        "use_bait": use_bait,
        "bite_detection_mode": mode,
        "search_mode": "window",
    }, path)
    print(f"Настройки сохранены: {path.resolve()}")
    print("Теперь запустите автокалибровку: python main.py --calibrate")
    print("После неё проверьте безопасный режим: python main.py --dry-run --debug --once")
