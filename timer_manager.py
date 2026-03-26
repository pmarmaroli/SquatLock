"""Periodic timer that triggers the squat overlay.

Includes Do-Not-Disturb detection: skips the lock when a communication
app (Zoom, Teams, Slack, Discord, etc.) appears to be in an active
call or meeting.
"""

import ctypes
import ctypes.wintypes
import re
import threading
from typing import Callable

# Process names (lowercase) of known communication apps.
_COMM_PROCESSES = {
    "zoom.exe", "zoom",
    "teams.exe", "ms-teams.exe", "msteams.exe",
    "slack.exe",
    "discord.exe",
    "webex.exe", "ciscowj.exe", "atmgr.exe",
    "skype.exe", "lync.exe",
}

# Window-title patterns that suggest an ACTIVE call / meeting.
_MEETING_TITLE_PATTERNS = re.compile(
    r"zoom meeting|webinar"
    r"|screen share|sharing your screen|partage d.écran"
    r"|réunion en cours|appel en cours"
    r"|huddle|slack huddle",
    re.IGNORECASE,
)

# Teams nav tabs — these are NOT meetings, just the main Teams UI.
_TEAMS_NAV_TABS = re.compile(
    r"^(Chat|Teams|Calendar|Calls|Activity|Files|Apps|OneDrive|Assignments"
    r"|Approvals|Shifts|Viva|Search|Copilot)"
    r"\s*\|",
    re.IGNORECASE,
)

# Teams processes
_TEAMS_PROCESSES = {"teams.exe", "ms-teams.exe", "msteams.exe"}


def _is_teams_in_meeting(exe_name: str, title: str) -> bool:
    """Return True if this Teams window looks like an active meeting.

    Teams meeting windows have a title like:
      "Meeting Name | Microsoft Teams"
      "Meeting Name (General) | Microsoft Teams"
    But regular tabs look like:
      "Calendar | Calendar | Microsoft Teams"
      "Chat | Person Name | Microsoft Teams"
      "Calls | Personal | Microsoft Teams"
    """
    if exe_name not in _TEAMS_PROCESSES:
        return False
    if "| Microsoft Teams" not in title:
        return False
    # Exclude known navigation tabs
    if _TEAMS_NAV_TABS.search(title):
        return False
    # If we get here, it's a non-tab Teams window = likely a meeting
    return True


def _is_in_meeting() -> bool:
    """Return True if a communication app window with meeting-like title exists.

    Enumerates all visible windows, checks if the owning process is a
    known communication app AND the title suggests an active call.
    """
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi

    found = [False]

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))
    def enum_callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True  # continue

        # Get window title
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value

        # Get process name
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        hproc = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
        if not hproc:
            return True
        exe_buf = ctypes.create_unicode_buffer(260)
        psapi.GetProcessImageFileNameW(hproc, exe_buf, 260)
        kernel32.CloseHandle(hproc)

        exe_name = exe_buf.value.rsplit("\\", 1)[-1].lower()

        if exe_name in _COMM_PROCESSES and _MEETING_TITLE_PATTERNS.search(title):
            found[0] = True
            return False  # stop enumeration

        # Teams-specific: meeting window that isn't a nav tab
        if _is_teams_in_meeting(exe_name, title):
            found[0] = True
            return False

        return True

    user32.EnumWindows(enum_callback, 0)
    return found[0]


def _get_comm_windows() -> list[tuple[str, str]]:
    """Debug helper: return all visible comm-app windows as (exe, title) pairs."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi

    results: list[tuple[str, str]] = []
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))
    def enum_callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value

        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        hproc = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
        if not hproc:
            return True
        exe_buf = ctypes.create_unicode_buffer(260)
        psapi.GetProcessImageFileNameW(hproc, exe_buf, 260)
        kernel32.CloseHandle(hproc)

        exe_name = exe_buf.value.rsplit("\\", 1)[-1].lower()
        if exe_name in _COMM_PROCESSES or exe_name in _TEAMS_PROCESSES:
            match = (_MEETING_TITLE_PATTERNS.search(title)
                     or _is_teams_in_meeting(exe_name, title))
            tag = " [MATCH]" if match else ""
            results.append((exe_name, title + tag))
        return True

    user32.EnumWindows(enum_callback, 0)
    return results


class TimerManager:
    """Repeating timer that calls *on_trigger* every *interval_sec* seconds.

    If a communication app is in an active call/meeting at trigger time,
    the callback is skipped and the timer simply restarts.
    """

    def __init__(self, interval_sec: float, on_trigger: Callable[[], None]):
        self._interval = interval_sec
        self._on_trigger = on_trigger
        self._timer: threading.Timer | None = None
        self._paused = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def interval(self) -> float:
        return self._interval

    @interval.setter
    def interval(self, value: float) -> None:
        self._interval = value

    def start(self) -> None:
        """Start (or restart) the countdown."""
        self._cancel_timer()
        self._schedule()

    def stop(self) -> None:
        """Permanently stop the timer."""
        self._cancel_timer()

    def pause(self) -> None:
        self._paused = True
        self._cancel_timer()

    def resume(self) -> None:
        self._paused = False
        self._schedule()

    @property
    def is_paused(self) -> bool:
        return self._paused

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _schedule(self) -> None:
        self._timer = threading.Timer(self._interval, self._tick)
        self._timer.daemon = True
        self._timer.start()

    def _tick(self) -> None:
        if self._paused:
            return

        if _is_in_meeting():
            print("[SquatLock] Meeting/call detected — skipping this cycle.")
            self._schedule()
            return

        print("[SquatLock] Timer fired — launching overlay!")
        self._on_trigger()
        self._schedule()

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
