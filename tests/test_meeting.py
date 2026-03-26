"""Standalone test for meeting/call detection.

Polls every 2 seconds and prints whether a communication app
(Zoom, Teams, Slack, Discord, Webex, Skype) appears to be in
an active call. Press Ctrl+C to stop.

Usage:
    python tests/test_meeting.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from timer_manager import _is_in_meeting, _get_comm_windows


def run() -> None:
    print("Meeting detection test — press Ctrl+C to stop")
    print("Open or join a call in Zoom/Teams/Slack/etc. to test.\n")

    while True:
        in_meeting = _is_in_meeting()
        windows = _get_comm_windows()
        status = "IN A CALL" if in_meeting else "No call detected"
        print(f"[{time.strftime('%H:%M:%S')}] {status}")
        if windows:
            for exe, title in windows:
                print(f"    -> {exe}: \"{title}\"")
        time.sleep(2)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nDone.")
