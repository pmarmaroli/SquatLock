"""User-configurable settings for SquatLock."""

import json
import os

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".squatlock_config.json")

DEFAULTS = {
    "interval_minutes": 45,
    "squats_required": 5,
    "twists_required": 10,
    "neck_tilts_required": 3,       # reps per side
    "neck_tilt_threshold": 0.12,    # normalised nose-to-shoulder-midpoint displacement to detect tilt
    "neck_tilt_return_threshold": 0.06,  # displacement below which head is back to neutral
    "neck_tilt_hold_min_sec": 5.0,       # minimum hold time per side rep (seconds)
    "drop_threshold": 0.10,  # shoulder must drop 10% of frame height
    "rise_threshold": 0.04,  # shoulder must return within 4% of baseline
    "rotation_threshold": 0.50,  # width ratio below which torso is rotated
    "return_threshold": 0.75,    # width ratio above which torso is back
    "camera_index": 0,
}


def load() -> dict:
    """Load config from disk, falling back to defaults."""
    config = dict(DEFAULTS)
    if os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config.update(json.load(f))
    return config


def save(config: dict) -> None:
    """Persist config to disk."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
