"""System-tray icon with pause / configure / quit options.

Uses pystray + Pillow to create a small colored icon and menu.
"""

import threading
from typing import Callable

from PIL import Image, ImageDraw
import pystray


def _make_icon_image(color: str = "#4CAF50", size: int = 64) -> Image.Image:
    """Generate a simple colored square icon with an 'S' letter."""
    img = Image.new("RGB", (size, size), color)
    draw = ImageDraw.Draw(img)
    draw.text((size // 4, size // 8), "S", fill="white")
    return img


class TrayIcon:
    """System-tray icon that exposes pause/resume, trigger now, and quit."""

    def __init__(
        self,
        on_pause_toggle: Callable[[], None],
        on_trigger_now: Callable[[], None],
        on_settings: Callable[[], None],
        on_quit: Callable[[], None],
        is_paused: Callable[[], bool],
    ):
        self._on_pause_toggle = on_pause_toggle
        self._on_trigger_now = on_trigger_now
        self._on_settings = on_settings
        self._on_quit = on_quit
        self._is_paused = is_paused

        self._icon: pystray.Icon | None = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the tray icon in a background thread."""
        self._icon = pystray.Icon(
            "SquatLock",
            _make_icon_image(),
            "SquatLock",
            menu=self._build_menu(),
        )
        thread = threading.Thread(target=self._icon.run, daemon=True)
        thread.start()

    def stop(self) -> None:
        if self._icon is not None:
            self._icon.stop()

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _build_menu(self) -> pystray.Menu:
        return pystray.Menu(
            pystray.MenuItem(
                lambda _: "Resume" if self._is_paused() else "Pause",
                self._on_pause_toggle,
            ),
            pystray.MenuItem("Squat now!", self._on_trigger_now),
            pystray.MenuItem("Settings…", self._on_settings),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._on_quit),
        )
