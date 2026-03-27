# SquatLock

A Windows desktop app that periodically locks your screen and forces you to do exercises (detected via webcam) to unlock it. Alternates between **squats** and **torso twists**.

## Setup

1. Make sure **Python 3.11+** is installed and in your PATH
2. Double-click **`setup.bat`** — it creates a virtual environment, installs dependencies, and downloads the pose detection model
3. Double-click **`SquatLock.bat`** to launch the app

## How It Works

1. **Timer** fires every N minutes (default 45).
2. A fullscreen **overlay** blocks **all screens** and opens the webcam.
3. Press **Space** when ready — calibration starts after a 1-second countdown.
4. **MediaPipe Pose** tracks your shoulders via the laptop camera (face-to-belt view).
5. The app **alternates** between two exercises:
   - **Squats** (default 5) — shoulders must drop ≥10% with arms held horizontally, then stand back up.
   - **Torso twists** (default 10) — standing with fists on chest, rotate left or right until shoulder width compresses to <50%, then return to neutral.
6. After reaching the target, the overlay closes and the timer restarts.
7. If you can't complete the exercise, the overlay **auto-unlocks after 2 minutes**.

## System Tray Menu

| Action | Description |
|---|---|
| **Pause / Resume** | Toggle the timer |
| **Exercise now!** | Trigger the next exercise immediately |
| **Settings…** | Change interval and squat count |
| **Quit** | Exit the app |

## Default Configuration

Settings are persisted to `~/.squatlock_config.json`. Editable via the tray menu or directly:

```json
{
  "interval_minutes": 45,
  "squats_required": 5,
  "twists_required": 10,
  "drop_threshold": 0.10,
  "rise_threshold": 0.04,
  "rotation_threshold": 0.50,
  "return_threshold": 0.75,
  "camera_index": 0
}
```

| Setting | Default | Description |
|---|---|---|
| `interval_minutes` | `45` | Minutes between each exercise break |
| `squats_required` | `5` | Number of squats needed to unlock |
| `twists_required` | `10` | Number of torso twists needed to unlock |
| `drop_threshold` | `0.10` | How far shoulders must drop (10%) to register a squat-down |
| `rise_threshold` | `0.04` | How close shoulders must return to baseline (4%) to register standing up |
| `rotation_threshold` | `0.50` | Shoulder width ratio below which a twist is detected (50%) |
| `return_threshold` | `0.75` | Shoulder width ratio above which you're back to neutral (75%) |
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
twist_detector.py    # MediaPipe pose tracking + torso twist counting
timer_manager.py     # Repeating timer + DND detection
config.py            # User settings (load/save JSON)
tray.py              # System tray icon and menu
tests/
  test_pose.py       # Test squat detection with live webcam
  test_twist.py      # Test twist detection with live webcam (press 's' to record)
  test_meeting.py    # Test meeting/call detection
```
