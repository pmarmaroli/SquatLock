"""Squat detection using MediaPipe Pose Landmarker (Tasks API).

Designed for a laptop/desktop camera that sees face-to-belt only (no
legs visible).  Detection is based on **shoulder vertical displacement**
combined with **arm horizontality**:

1.  A short calibration phase (~30 frames) records the standing baseline
    for the shoulder midpoint Y coordinate (normalized 0..1, top=0).
2.  State machine:
      STANDING  → SQUATTING  when shoulder_y drops by ≥ *drop_threshold*
                              AND arms are approximately horizontal
      SQUATTING → STANDING   when shoulder_y returns within *rise_threshold*
    One rep = full cycle STANDING → SQUATTING → STANDING.
"""

import math
import os
from collections import deque
from enum import Enum, auto

import cv2
import mediapipe as mp
import numpy as np

_vision = mp.tasks.vision
_BaseOptions = mp.tasks.BaseOptions
_draw = _vision.drawing_utils
_connections = _vision.PoseLandmarksConnections.POSE_LANDMARKS

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")

# How many frames to average for the standing baseline.
_CALIBRATION_FRAMES = 30
# How many frames to wait after the user presses space (~1 sec at 30 fps).
_COUNTDOWN_FRAMES = 30


class _State(Enum):
    WAITING = auto()      # waiting for user to press space
    COUNTDOWN = auto()    # 1-second countdown before calibration
    CALIBRATING = auto()  # recording standing baseline
    STANDING = auto()
    SQUATTING = auto()


class SquatDetector:
    """Tracks squats via webcam using upper-body vertical displacement."""

    # MediaPipe landmark indices (upper body)
    _LEFT_SHOULDER = 11
    _RIGHT_SHOULDER = 12
    _LEFT_ELBOW = 13
    _RIGHT_ELBOW = 14
    _LEFT_WRIST = 15
    _RIGHT_WRIST = 16

    # Arms must be within this many degrees of horizontal to count.
    _ARM_HORIZONTAL_TOLERANCE = 30  # degrees

    def __init__(self, drop_threshold: float = 0.10, rise_threshold: float = 0.04):
        """
        Args:
            drop_threshold: Normalised Y distance the shoulders must drop
                            below baseline to count as squatting (default 10 %
                            of frame height).
            rise_threshold: How close to baseline the shoulders must return
                            to count as standing again (default 4 %).
        """
        self.drop_threshold = drop_threshold
        self.rise_threshold = rise_threshold

        options = _vision.PoseLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = _vision.PoseLandmarker.create_from_options(options)

        self._state = _State.WAITING
        self._frame_ts = 0
        self._countdown_counter = 0  # frames elapsed in COUNTDOWN phase

        # Calibration data
        self._cal_samples: list[float] = []
        self._baseline_y: float | None = None  # standing shoulder Y

        # Smoothing: rolling average over last 5 frames
        self._y_buffer: deque[float] = deque(maxlen=5)

        # Arm horizontality: smoothed over a rolling window so that
        # a brief glitch doesn't invalidate the whole squat.
        self._arm_buffer: deque[bool] = deque(maxlen=10)
        self._arms_horizontal = False
        self.count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run pose detection, update squat count, return annotated frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_ts += 33  # ~30 fps
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts)

        shoulder_y = None
        drop = None
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            shoulder_y = self._shoulder_midpoint_y(lm)
            self._arm_buffer.append(self._check_arms_horizontal(lm))
            # Arms count as horizontal if ≥3 of the last 10 frames say so.
            self._arms_horizontal = sum(self._arm_buffer) >= 3

            if shoulder_y is not None:
                self._y_buffer.append(shoulder_y)
                smooth_y = sum(self._y_buffer) / len(self._y_buffer)
                drop = self._update_state(smooth_y)

            _draw.draw_landmarks(frame, lm, _connections)

        self._draw_hud(frame, shoulder_y, drop)
        return frame

    def signal_ready(self) -> None:
        """Called when the user presses space to start calibration countdown."""
        if self._state == _State.WAITING:
            self._state = _State.COUNTDOWN
            self._countdown_counter = 0

    def reset(self) -> None:
        """Reset counter, state, and calibration."""
        self.count = 0
        self._state = _State.WAITING
        self._countdown_counter = 0
        self._cal_samples.clear()
        self._baseline_y = None
        self._y_buffer.clear()
        self._arm_buffer.clear()

    def close(self) -> None:
        self._landmarker.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _shoulder_midpoint_y(self, landmarks) -> float | None:
        """Return the average Y of left + right shoulder, or None."""
        ls = landmarks[self._LEFT_SHOULDER]
        rs = landmarks[self._RIGHT_SHOULDER]
        if ls.presence < 0.5 or rs.presence < 0.5:
            return None
        return (ls.y + rs.y) / 2.0

    def _arm_angle_from_horizontal(self, shoulder, wrist) -> float:
        """Angle (degrees) of shoulder->wrist vector vs horizontal.

        0 = perfectly horizontal, 90 = pointing straight down.
        """
        dx = wrist.x - shoulder.x
        dy = wrist.y - shoulder.y
        return abs(math.degrees(math.atan2(dy, dx)))

    def _check_arms_horizontal(self, landmarks) -> bool:
        """Return True if both arms are approximately horizontal."""
        ls = landmarks[self._LEFT_SHOULDER]
        rs = landmarks[self._RIGHT_SHOULDER]
        lw = landmarks[self._LEFT_WRIST]
        rw = landmarks[self._RIGHT_WRIST]

        if any(p.presence < 0.5 for p in (ls, rs, lw, rw)):
            return False

        left_angle = self._arm_angle_from_horizontal(ls, lw)
        right_angle = self._arm_angle_from_horizontal(rs, rw)

        return (left_angle <= self._ARM_HORIZONTAL_TOLERANCE
                and right_angle <= self._ARM_HORIZONTAL_TOLERANCE)

    def _update_state(self, smooth_y: float) -> float | None:
        """Advance state machine. Returns current *drop* distance or None."""
        if self._state == _State.WAITING:
            return None

        if self._state == _State.COUNTDOWN:
            self._countdown_counter += 1
            if self._countdown_counter >= _COUNTDOWN_FRAMES:
                self._state = _State.CALIBRATING
            return None

        if self._state == _State.CALIBRATING:
            self._cal_samples.append(smooth_y)
            if len(self._cal_samples) >= _CALIBRATION_FRAMES:
                self._baseline_y = sum(self._cal_samples) / len(self._cal_samples)
                self._state = _State.STANDING
            return None

        # drop > 0 means shoulders moved DOWN (Y increases downward).
        drop = smooth_y - self._baseline_y

        if (self._state == _State.STANDING
                and drop > self.drop_threshold
                and self._arms_horizontal):
            self._state = _State.SQUATTING
        elif self._state == _State.SQUATTING and drop <= self.rise_threshold:
            self._state = _State.STANDING
            self.count += 1

        return drop

    def _draw_hud(self, frame: np.ndarray, shoulder_y: float | None,
                  drop: float | None) -> None:
        """Draw status info on the frame."""
        h, w = frame.shape[:2]
        y_pos = 30

        if self._state == _State.WAITING:
            # Large centered message
            msg = "STAND UP!"
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
            cv2.putText(
                frame, msg, ((w - tw) // 2, h // 2 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 200, 255), 4,
            )
            prompt = "Press SPACE when ready"
            (tw2, _), _ = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(
                frame, prompt, ((w - tw2) // 2, h // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2,
            )
            return

        if self._state == _State.COUNTDOWN:
            secs_left = max(1, (_COUNTDOWN_FRAMES - self._countdown_counter) // 30)
            msg = "GET READY!"
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
            cv2.putText(
                frame, msg, ((w - tw) // 2, h // 2 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 200, 255), 4,
            )
            countdown = f"Calibrating in {secs_left}s..."
            (tw2, _), _ = cv2.getTextSize(countdown, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(
                frame, countdown, ((w - tw2) // 2, h // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2,
            )
            return

        if self._state == _State.CALIBRATING:
            progress = len(self._cal_samples)
            cv2.putText(
                frame, f"Calibrating... stand still ({progress}/{_CALIBRATION_FRAMES})",
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
            )
            return

        if drop is not None:
            drop_pct = drop * 100
            color = (0, 0, 255) if self._state == _State.SQUATTING else (0, 255, 0)
            state_label = self._state.name

            cv2.putText(
                frame, f"State: {state_label}", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )
            y_pos += 30
            cv2.putText(
                frame, f"Drop: {drop_pct:+.1f}%  (threshold: {self.drop_threshold*100:.0f}%)",
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1,
            )
            y_pos += 25
            arms_color = (0, 255, 0) if self._arms_horizontal else (0, 0, 255)
            arms_label = "YES" if self._arms_horizontal else "NO"
            cv2.putText(
                frame, f"Arms horizontal: {arms_label}",
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, arms_color, 1,
            )

        elif shoulder_y is None:
            cv2.putText(
                frame, "Shoulders not visible", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )
