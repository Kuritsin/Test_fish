from __future__ import annotations

from typing import Protocol

from config import BotConfig, Region


class SearchRegionUnavailable(RuntimeError):
    pass


class WindowRegionProvider(Protocol):
    def foreground_client_region(self) -> Region | None: ...


def resolve_search_region(config: BotConfig, window: WindowRegionProvider) -> Region:
    """Resolve the current capture area without ever falling back to the desktop."""
    if config.search_mode == "custom":
        return config.search_region
    if config.search_mode != "window":
        raise ValueError(f"Unsupported search mode: {config.search_mode}")
    region = window.foreground_client_region()
    if region is None:
        raise SearchRegionUnavailable("Could not determine the WoW client area")
    return region
