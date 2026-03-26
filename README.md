# SquatLock

A Windows desktop app that periodically locks your screen and forces you to do squats (detected via webcam) to unlock it.

## Setup

1. Make sure **Python 3.11+** is installed and in your PATH
2. Double-click **`setup.bat`** — it creates a virtual environment, installs dependencies, and downloads the pose detection model
3. Double-click **`SquatLock.bat`** to launch the app

## How It Works

1. **Timer** fires every N minutes (default 45).
2. A fullscreen **overlay** blocks **all screens** and opens the webcam.
3. **MediaPipe Pose** tracks your shoulders via the laptop camera (face-to-belt view).
4. A short calibration phase (~1 sec) records your standing position.
5. A squat is counted when your shoulders drop by ≥10% of frame height **and** your arms are held horizontally, then you stand back up.
6. After reaching the target (default 10 squats), the overlay closes and the timer restarts.
7. Emergency unlock: press **Ctrl+Shift+Q**.

## System Tray Menu

| Action | Description |
|---|---|
| **Pause / Resume** | Toggle the timer |
| **Squat now!** | Trigger the overlay immediately |
| **Settings…** | Change interval and squat count |
| **Quit** | Exit the app |

## Default Configuration

Settings are persisted to `~/.squatlock_config.json`. Editable via the tray menu or directly:

```json
{
  "interval_minutes": 45,
  "squats_required": 10,
  "drop_threshold": 0.10,
  "rise_threshold": 0.04,
  "camera_index": 0
}
```

| Setting | Default | Description |
|---|---|---|
| `interval_minutes` | `45` | Minutes between each squat break |
| `squats_required` | `10` | Number of squats needed to unlock |
| `drop_threshold` | `0.10` | How far shoulders must drop (10% of frame height) to register a squat-down |
| `rise_threshold` | `0.04` | How close shoulders must return to baseline (4%) to register standing back up |
| `camera_index` | `0` | Webcam index (0 = default camera) |

## Do Not Disturb

The overlay is **skipped** when a communication app (Teams, Zoom, Slack, Discord, Webex, Skype) is detected in an active call or meeting. The timer simply restarts.

## Project Structure

```
setup.bat            # First-time setup (venv + dependencies + model)
SquatLock.bat        # Launch the app
main.py              # Entry point — wires everything together
overlay.py           # Fullscreen lock overlay with webcam display
squat_detector.py    # MediaPipe pose tracking + squat counting
timer_manager.py     # Repeating timer + DND detection
config.py            # User settings (load/save JSON)
tray.py              # System tray icon and menu
tests/
  test_pose.py       # Test squat detection with live webcam
  test_meeting.py    # Test meeting/call detection
```
