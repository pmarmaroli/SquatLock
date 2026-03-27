"""Fullscreen lock overlay that shows webcam + squat counter.

Uses tkinter for the window (always-on-top, non-closable) and embeds
the OpenCV/MediaPipe video feed via PIL → tkinter PhotoImage.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import threading
import tkinter as tk
from typing import Callable

import cv2
from PIL import Image, ImageTk

from squat_detector import SquatDetector


def _get_all_monitors() -> list[tuple[int, int, int, int]]:
    """Return a list of (x, y, w, h) for each monitor."""
    monitors: list[tuple[int, int, int, int]] = []

    @ctypes.WINFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.wintypes.RECT),
        ctypes.c_double,
    )
    def callback(_hmon, _hdc, lprect, _lparam):
        r = lprect.contents
        monitors.append((r.left, r.top, r.right - r.left, r.bottom - r.top))
        return 1

    ctypes.windll.user32.EnumDisplayMonitors(None, None, callback, 0)
    return monitors


class LockOverlay:
    """Creates a fullscreen overlay that blocks the desktop until
    the user completes the required number of squats.
    """

    _REFRESH_MS = 30  # ~33 fps

    def __init__(
        self,
        squats_required: int,
        drop_threshold: float,
        rise_threshold: float,
        camera_index: int,
        on_unlock: Callable[[], None] | None = None,
    ):
        self._squats_required = squats_required
        self._camera_index = camera_index
        self._on_unlock = on_unlock
        self._drop_threshold = drop_threshold
        self._rise_threshold = rise_threshold

        self._detector: SquatDetector | None = None
        self._cap: cv2.VideoCapture | None = None

        # Build UI in a separate thread so the caller isn't blocked.
        self._thread = threading.Thread(target=self._run, daemon=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show(self) -> None:
        """Launch the overlay (non-blocking)."""
        self._thread.start()

    # ------------------------------------------------------------------
    # UI construction (runs in its own thread)
    # ------------------------------------------------------------------

    def _run(self) -> None:
        # Create detector in THIS thread to keep MediaPipe resources
        # owned by the same thread that will use them.
        self._detector = SquatDetector(self._drop_threshold, self._rise_threshold)

        self._root = tk.Tk()
        self._root.title("SquatLock")
        self._root.attributes("-fullscreen", True)
        self._root.attributes("-topmost", True)
        self._root.configure(bg="black")

        # Prevent Alt-F4 and other close attempts.
        self._root.protocol("WM_DELETE_WINDOW", lambda: None)
        self._root.bind("<Alt-F4>", lambda e: "break")
        # Capture Escape so overlay can't be dismissed.
        self._root.bind("<Escape>", lambda e: None)
        # Space bar to start calibration.
        self._root.bind("<space>", lambda e: self._detector.signal_ready())
        # Emergency exit: Ctrl+Shift+Q
        self._root.bind("<Control-Shift-Q>", lambda e: self._unlock())
        self._root.bind("<Control-Shift-q>", lambda e: self._unlock())

        # --- Block ALL other monitors with black windows ---
        self._blockers: list[tk.Toplevel] = []
        primary_x = self._root.winfo_x()
        primary_y = self._root.winfo_y()
        for mx, my, mw, mh in _get_all_monitors():
            # Skip the primary monitor (the main overlay covers it).
            if mx == primary_x and my == primary_y:
                continue
            blocker = tk.Toplevel(self._root)
            blocker.title("SquatLock")
            blocker.attributes("-topmost", True)
            blocker.configure(bg="black")
            blocker.geometry(f"{mw}x{mh}+{mx}+{my}")
            blocker.overrideredirect(True)
            blocker.protocol("WM_DELETE_WINDOW", lambda: None)
            # Show a reminder on secondary screens
            tk.Label(
                blocker,
                text="Do your squats\nto unlock!",
                font=("Segoe UI", 28, "bold"),
                fg="#555555",
                bg="black",
            ).place(relx=0.5, rely=0.5, anchor="center")
            self._blockers.append(blocker)

        # --- widgets ---
        self._video_label = tk.Label(self._root, bg="black")
        self._video_label.pack(expand=True)

        self._counter_label = tk.Label(
            self._root,
            text=self._counter_text(),
            font=("Segoe UI", 36, "bold"),
            fg="white",
            bg="black",
        )
        self._counter_label.pack(pady=20)

        self._status_label = tk.Label(
            self._root,
            text="Do your squats to unlock!  (emergency: Ctrl+Shift+Q)",
            font=("Segoe UI", 18),
            fg="#aaaaaa",
            bg="black",
        )
        self._status_label.pack()

        # Open the camera and start the update loop.
        self._cap = cv2.VideoCapture(self._camera_index)
        self._update_frame()

        self._root.mainloop()

        # Cleanup after mainloop exits.
        self._cleanup()

    # ------------------------------------------------------------------
    # Frame loop
    # ------------------------------------------------------------------

    def _update_frame(self) -> None:
        if self._cap is None or not self._cap.isOpened():
            return

        ret, frame = self._cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # mirror
            try:
                frame = self._detector.process_frame(frame)
            except Exception as e:
                print(f"[SquatLock] Detection error: {e}")

            # Convert BGR → RGB → PIL → tkinter image.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            # Scale to fit screen height while keeping aspect ratio.
            screen_h = self._root.winfo_screenheight() - 200  # room for labels
            scale = screen_h / img.height
            img = img.resize(
                (int(img.width * scale), int(img.height * scale)),
                Image.LANCZOS,
            )

            self._photo = ImageTk.PhotoImage(img)
            self._video_label.configure(image=self._photo)

            # Update counter.
            self._counter_label.configure(text=self._counter_text())

            # Check unlock condition.
            if self._detector.count >= self._squats_required:
                self._unlock()
                return

        self._root.after(self._REFRESH_MS, self._update_frame)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _counter_text(self) -> str:
        return f"{self._detector.count} / {self._squats_required} squats"

    def _unlock(self) -> None:
        """Dismiss the overlay on all screens and notify the caller."""
        for blocker in self._blockers:
            blocker.destroy()
        self._root.destroy()
        if self._on_unlock:
            self._on_unlock()

    def _cleanup(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._detector.close()
