"""Torso-twist detection using MediaPipe Pose Landmarker (Tasks API).

Standing position, fists joined on the chest.  Detection is based on
**shoulder width compression**: when you rotate your torso sideways,
the apparent shoulder width (as seen by the camera) shrinks dramatically.

1.  A short calibration phase (~30 frames) records the baseline
    shoulder width while facing the camera.
2.  State machine (using width_ratio = current_width / baseline_width):
      NEUTRAL → ROTATED  when width_ratio drops below *rotation_threshold*
      ROTATED → NEUTRAL  when width_ratio recovers above *return_threshold*
    One rep = full cycle NEUTRAL → ROTATED → NEUTRAL.
"""

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

_CALIBRATION_FRAMES = 30
_COUNTDOWN_FRAMES = 30  # ~1 sec at 30 fps


class _State(Enum):
    WAITING = auto()
    COUNTDOWN = auto()
    CALIBRATING = auto()
    NEUTRAL = auto()
    ROTATED = auto()


class TwistDetector:
    """Tracks standing torso twists via shoulder width compression."""

    _LEFT_SHOULDER = 11
    _RIGHT_SHOULDER = 12

    def __init__(
        self,
        rotation_threshold: float = 0.50,
        return_threshold: float = 0.75,
    ):
        """
        Args:
            rotation_threshold: Width ratio (current/baseline) below which
                the torso counts as rotated (default 0.50 = 50%).
            return_threshold: Width ratio above which the torso counts as
                back to neutral (default 0.75 = 75%).
        """
        self.rotation_threshold = rotation_threshold
        self.return_threshold = return_threshold

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
        self._countdown_counter = 0

        # Calibration
        self._cal_w_samples: list[float] = []
        self._baseline_w: float | None = None

        # Smoothing
        self._w_buffer: deque[float] = deque(maxlen=5)

        self.count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run pose detection, update twist count, return annotated frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_ts += 33
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts)

        mid_x = None
        width_ratio = None
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            mid_x, width = self._shoulder_metrics(lm)

            if width is not None:
                self._w_buffer.append(width)
                smooth_w = sum(self._w_buffer) / len(self._w_buffer)
                width_ratio = self._update_state(smooth_w)

            _draw.draw_landmarks(frame, lm, _connections)

        self._draw_hud(frame, mid_x, width_ratio)
        return frame

    def signal_ready(self) -> None:
        """Called when the user presses space to start calibration."""
        if self._state == _State.WAITING:
            self._state = _State.COUNTDOWN
            self._countdown_counter = 0

    def reset(self) -> None:
        self.count = 0
        self._state = _State.WAITING
        self._countdown_counter = 0
        self._cal_w_samples.clear()
        self._baseline_w = None
        self._w_buffer.clear()

    def close(self) -> None:
        self._landmarker.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _shoulder_metrics(self, landmarks) -> tuple[float | None, float | None]:
        """Return (midpoint_x, width) of left + right shoulder, or (None, None)."""
        ls = landmarks[self._LEFT_SHOULDER]
        rs = landmarks[self._RIGHT_SHOULDER]
        if ls.presence < 0.5 or rs.presence < 0.5:
            return None, None
        mid_x = (ls.x + rs.x) / 2.0
        width = abs(rs.x - ls.x)
        return mid_x, width

    def _update_state(self, smooth_w: float) -> float | None:
        """Returns width_ratio (current/baseline) or None during setup."""
        if self._state == _State.WAITING:
            return None

        if self._state == _State.COUNTDOWN:
            self._countdown_counter += 1
            if self._countdown_counter >= _COUNTDOWN_FRAMES:
                self._state = _State.CALIBRATING
            return None

        if self._state == _State.CALIBRATING:
            self._cal_w_samples.append(smooth_w)
            if len(self._cal_w_samples) >= _CALIBRATION_FRAMES:
                self._baseline_w = sum(self._cal_w_samples) / len(self._cal_w_samples)
                self._state = _State.NEUTRAL
            return None

        width_ratio = smooth_w / self._baseline_w

        if self._state == _State.NEUTRAL and width_ratio < self.rotation_threshold:
            self._state = _State.ROTATED
        elif self._state == _State.ROTATED and width_ratio > self.return_threshold:
            self._state = _State.NEUTRAL
            self.count += 1

        return width_ratio

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _draw_hud(self, frame: np.ndarray, mid_x: float | None,
                  width_ratio: float | None) -> None:
        h, w = frame.shape[:2]
        y_pos = 30

        if self._state == _State.WAITING:
            msg = "STAND STRAIGHT, FISTS ON CHEST"
            (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.putText(
                frame, msg, ((w - tw) // 2, h // 2 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3,
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
            (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
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
            progress = len(self._cal_w_samples)
            cv2.putText(
                frame, f"Calibrating... stand still ({progress}/{_CALIBRATION_FRAMES})",
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
            )
            return

        if width_ratio is not None:
            ratio_pct = width_ratio * 100
            if self._state == _State.ROTATED:
                color = (0, 100, 255)
                state_label = "ROTATED"
            else:
                color = (0, 255, 0)
                state_label = "NEUTRAL"

            cv2.putText(
                frame, f"State: {state_label}", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )
            y_pos += 30
            cv2.putText(
                frame,
                f"Width: {ratio_pct:.0f}%  (rotate < {self.rotation_threshold*100:.0f}%  |  return > {self.return_threshold*100:.0f}%)",
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1,
            )

            # Visual width-ratio bar (0% to 100%)
            bar_x = 40
            bar_y = h - 60
            bar_w = w - 80
            # Background
            cv2.rectangle(frame, (bar_x, bar_y - 8), (bar_x + bar_w, bar_y + 8),
                          (60, 60, 60), -1)
            # Rotation threshold marker
            t_px = int(self.rotation_threshold * bar_w)
            cv2.line(frame, (bar_x + t_px, bar_y - 14), (bar_x + t_px, bar_y + 14),
                     (0, 0, 255), 2)
            # Return threshold marker
            r_px = int(self.return_threshold * bar_w)
            cv2.line(frame, (bar_x + r_px, bar_y - 14), (bar_x + r_px, bar_y + 14),
                     (0, 255, 0), 2)
            # Current ratio
            c_px = int(min(1.0, max(0.0, width_ratio)) * bar_w)
            cv2.circle(frame, (bar_x + c_px, bar_y), 8, color, -1)
            # Labels
            cv2.putText(frame, "0%", (bar_x - 5, bar_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            cv2.putText(frame, "100%", (bar_x + bar_w - 20, bar_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        elif mid_x is None:
            cv2.putText(
                frame, "No pose detected", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )
