import _bootstrap  # noqa: F401

from config import CONFIG, USER_CONFIG_PATH
from fishing_bot.region_selector import select_search_region
from fishing_bot.utils import enable_dpi_awareness


def main() -> int:
    enable_dpi_awareness()
    try:
        select_search_region(CONFIG, USER_CONFIG_PATH)
    except (RuntimeError, ValueError) as error:
        print(f"Ошибка выбора области: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
