"""SquatLock — entry point.

Wires together the timer, overlay, tray icon, and config.
"""

import sys
import threading
import tkinter as tk
from tkinter import simpledialog

import config
from overlay import LockOverlay
from timer_manager import TimerManager
from tray import TrayIcon


class App:
    """Top-level application controller."""

    def __init__(self):
        self._cfg = config.load()

        self._timer = TimerManager(
            interval_sec=self._cfg["interval_minutes"] * 60,
            on_trigger=self._trigger_overlay,
        )

        self._tray = TrayIcon(
            on_pause_toggle=self._toggle_pause,
            on_trigger_now=self._trigger_overlay,
            on_settings=self._open_settings,
            on_quit=self._quit,
            is_paused=lambda: self._timer.is_paused,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._tray.start()
        # Trigger overlay immediately on first launch, then start the timer.
        self._trigger_overlay()
        self._timer.start()
        # Keep the main thread alive (tray + timers are daemon threads).
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            self._quit()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _trigger_overlay(self) -> None:
        overlay = LockOverlay(
            squats_required=self._cfg["squats_required"],
            drop_threshold=self._cfg["drop_threshold"],
            rise_threshold=self._cfg["rise_threshold"],
            camera_index=self._cfg["camera_index"],
            on_unlock=self._on_unlock,
        )
        overlay.show()

    def _on_unlock(self) -> None:
        """Called when the user finishes the required squats."""
        self._timer.start()  # restart the countdown

    def _toggle_pause(self) -> None:
        if self._timer.is_paused:
            self._timer.resume()
        else:
            self._timer.pause()

    def _open_settings(self) -> None:
        """Show a simple tkinter dialog to edit interval + squat count."""
        root = tk.Tk()
        root.withdraw()

        interval = simpledialog.askinteger(
            "SquatLock Settings",
            "Interval (minutes):",
            initialvalue=self._cfg["interval_minutes"],
            minvalue=1,
            maxvalue=480,
            parent=root,
        )
        if interval is not None:
            self._cfg["interval_minutes"] = interval

        squats = simpledialog.askinteger(
            "SquatLock Settings",
            "Squats required:",
            initialvalue=self._cfg["squats_required"],
            minvalue=1,
            maxvalue=100,
            parent=root,
        )
        if squats is not None:
            self._cfg["squats_required"] = squats

        root.destroy()

        config.save(self._cfg)
        self._timer.interval = self._cfg["interval_minutes"] * 60
        self._timer.start()

    def _quit(self, *_args) -> None:
        self._timer.stop()
        self._tray.stop()
        sys.exit(0)


if __name__ == "__main__":
    App().run()
