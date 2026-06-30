"""Neck lateral-tilt detection using MediaPipe Pose Landmarker (Tasks API).

Sit or stand with your head in a neutral upright position.  Detection is
based on the **nose horizontal displacement** relative to the shoulder
midpoint, normalised by shoulder width — this uses landmark 0 (nose) which
is reliably tracked by the Lite model at all head angles.

``tilt_signal = (shoulder_mid_x - nose.x) / shoulder_width``

Positive → head/nose moved towards person's right (right tilt).
Negative → head/nose moved towards person's left (left tilt).

1.  A short calibration phase (~30 frames) records the neutral baseline
    for the nose tilt and shoulder width.
2.  State machine:
      NEUTRAL → TILTED_RIGHT   when tilt_signal > +tilt_threshold
                                AND expected side is "right"
      NEUTRAL → TILTED_LEFT    when tilt_signal < −tilt_threshold
                                AND expected side is "left"
      TILTED_* → NEUTRAL       when |tilt_signal| < return_threshold
                                (rep counted only when hold ≥ hold_min_sec)
    One successful rep = tilt the correct side, hold for at least
    hold_min_sec, then return to neutral.
3.  Sides are forced to alternate: right → left → right → left → …
    A tilt on the wrong side is shown as a warning but never counted.
"""

import os
import time
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
_COUNTDOWN_FRAMES = 30

# Frames that must pass (back in NEUTRAL) before the next tilt is accepted.
# Prevents immediate re-trigger and gives time to tighten/relax (~0.67 s @30 fps).
_NEUTRAL_DWELL_FRAMES = 20


class _State(Enum):
    WAITING = auto()
    COUNTDOWN = auto()
    CALIBRATING = auto()
    NEUTRAL = auto()
    TILTED_RIGHT = auto()
    TILTED_LEFT = auto()


class NeckTiltDetector:
    """Tracks bilateral neck lateral tilts with minimum hold enforcement.

    Attributes:
        right_count: Successful right-side reps completed.
        left_count:  Successful left-side reps completed.
        count:       Total individual side reps (right_count + left_count).
        is_complete: True once both sides reach reps_per_side.
    """

    # MediaPipe landmark indices
    _NOSE = 0
    _LEFT_SHOULDER = 11
    _RIGHT_SHOULDER = 12

    def __init__(
        self,
        tilt_threshold: float = 0.12,
        return_threshold: float = 0.06,
        hold_min_sec: float = 5.0,
        reps_per_side: int = 3,
    ):
        """
        Args:
            tilt_threshold:    Shoulder-width-normalised nose displacement
                               required to count as a tilt (default 0.12).
            return_threshold:  Deviation below which the head counts as back
                               to neutral; must be < tilt_threshold (default 0.06).
            hold_min_sec:      Minimum seconds the tilt must be held for a
                               rep to count (default 5 s).
            reps_per_side:     Target reps per side (default 3).
        """
        if return_threshold >= tilt_threshold:
            raise ValueError("return_threshold must be less than tilt_threshold")

        self.tilt_threshold = tilt_threshold
        self.return_threshold = return_threshold
        self.hold_min_sec = hold_min_sec
        self.reps_per_side = reps_per_side

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

        # Calibration data
        self._cal_tilt_samples: list[float] = []
        self._cal_w_samples: list[float] = []
        self._baseline_tilt: float | None = None
        self._baseline_shoulder_w: float | None = None

        # Smoothing buffers
        self._tilt_buffer: deque[float] = deque(maxlen=5)
        self._w_buffer: deque[float] = deque(maxlen=5)

        # Rep tracking
        self.right_count: int = 0
        self.left_count: int = 0
        self._expected_side = "right"  # alternating; first tilt must be right

        # Hold tracking
        self._hold_start: float | None = None

        # Neutral dwell counter — counts frames since returning to NEUTRAL
        self._neutral_frames: int = 0

        # Transient feedback message (shown for a fixed number of frames)
        self._feedback: str = ""
        self._feedback_frames: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Total individual side reps completed (right + left)."""
        return self.right_count + self.left_count

    @property
    def is_complete(self) -> bool:
        """True when both sides have reached the required rep count."""
        return (
            self.right_count >= self.reps_per_side
            and self.left_count >= self.reps_per_side
        )

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run pose detection, update tilt counts, return annotated frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_ts += 33  # ~30 fps
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts)

        tilt_signal: float | None = None
        hold_elapsed: float | None = None

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            raw_tilt, shoulder_w = self._head_metrics(lm)

            if raw_tilt is not None and shoulder_w is not None:
                self._tilt_buffer.append(raw_tilt)
                self._w_buffer.append(shoulder_w)
                smooth_tilt = sum(self._tilt_buffer) / len(self._tilt_buffer)
                smooth_w = sum(self._w_buffer) / len(self._w_buffer)
                tilt_signal, hold_elapsed = self._update_state(smooth_tilt, smooth_w)

            _draw.draw_landmarks(frame, lm, _connections)

        self._draw_hud(frame, tilt_signal, hold_elapsed)

        if self._feedback_frames > 0:
            self._feedback_frames -= 1

        return frame

    def signal_ready(self) -> None:
        """Call when the user presses space to start calibration countdown."""
        if self._state == _State.WAITING:
            self._state = _State.COUNTDOWN
            self._countdown_counter = 0

    def reset(self) -> None:
        """Reset counter, state, and calibration."""
        self.right_count = 0
        self.left_count = 0
        self._expected_side = "right"
        self._state = _State.WAITING
        self._countdown_counter = 0
        self._cal_tilt_samples.clear()
        self._cal_w_samples.clear()
        self._baseline_tilt = None
        self._baseline_shoulder_w = None
        self._tilt_buffer.clear()
        self._w_buffer.clear()
        self._hold_start = None
        self._neutral_frames = 0
        self._feedback = ""
        self._feedback_frames = 0

    def close(self) -> None:
        self._landmarker.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _head_metrics(
        self, landmarks
    ) -> tuple[float | None, float | None]:
        """Return (raw_tilt, shoulder_width) using pose landmarks, or (None, None).

        raw_tilt = (shoulder_mid_x - nose.x)
          Positive → nose has moved toward the right shoulder (right tilt).
          Negative → nose has moved toward the left shoulder (left tilt).
        Normalised by shoulder_width downstream.

        Uses nose (landmark 0) which is reliably tracked at all head angles
        by the Pose Landmarker Lite model, unlike ear landmarks.
        """
        nose = landmarks[self._NOSE]
        ls = landmarks[self._LEFT_SHOULDER]
        rs = landmarks[self._RIGHT_SHOULDER]

        if nose.presence < 0.5 or ls.presence < 0.5 or rs.presence < 0.5:
            return None, None

        shoulder_w = abs(rs.x - ls.x)
        if shoulder_w < 0.01:
            return None, None

        shoulder_mid_x = (ls.x + rs.x) / 2.0
        raw_tilt = nose.x - shoulder_mid_x  # negated so flipped webcam maps correctly
        return raw_tilt, shoulder_w

    def _update_state(
        self, smooth_tilt: float, smooth_w: float
    ) -> tuple[float | None, float | None]:
        """Advance the state machine.

        Returns (tilt_signal, hold_elapsed_sec):
          tilt_signal   — normalised displacement from neutral
                          (positive = right, negative = left)
          hold_elapsed  — seconds spent holding the current tilt, or None
        Both are None during setup phases.
        """
        if self._state == _State.WAITING:
            return None, None

        if self._state == _State.COUNTDOWN:
            self._countdown_counter += 1
            if self._countdown_counter >= _COUNTDOWN_FRAMES:
                self._state = _State.CALIBRATING
            return None, None

        if self._state == _State.CALIBRATING:
            self._cal_tilt_samples.append(smooth_tilt)
            self._cal_w_samples.append(smooth_w)
            if len(self._cal_tilt_samples) >= _CALIBRATION_FRAMES:
                self._baseline_tilt = (
                    sum(self._cal_tilt_samples) / len(self._cal_tilt_samples)
                )
                self._baseline_shoulder_w = (
                    sum(self._cal_w_samples) / len(self._cal_w_samples)
                )
                self._state = _State.NEUTRAL
                self._neutral_frames = 0
            return None, None

        # ---- Active states ------------------------------------------------
        tilt_signal = (
            (smooth_tilt - self._baseline_tilt) / self._baseline_shoulder_w
        )
        hold_elapsed: float | None = None

        if self._state == _State.NEUTRAL:
            # Wait for the dwell period before accepting a new tilt.
            if self._neutral_frames < _NEUTRAL_DWELL_FRAMES:
                self._neutral_frames += 1
                return tilt_signal, None

            if tilt_signal > self.tilt_threshold:
                if self._expected_side == "right":
                    self._state = _State.TILTED_RIGHT
                    self._hold_start = time.monotonic()
                    self._neutral_frames = 0
                else:
                    self._set_feedback("← Expected LEFT tilt next!")

            elif tilt_signal < -self.tilt_threshold:
                if self._expected_side == "left":
                    self._state = _State.TILTED_LEFT
                    self._hold_start = time.monotonic()
                    self._neutral_frames = 0
                else:
                    self._set_feedback("→ Expected RIGHT tilt next!")

        elif self._state in (_State.TILTED_RIGHT, _State.TILTED_LEFT):
            hold_elapsed = time.monotonic() - self._hold_start

            # Head has returned to neutral — evaluate the hold.
            if abs(tilt_signal) < self.return_threshold:
                side = "right" if self._state == _State.TILTED_RIGHT else "left"
                if hold_elapsed >= self.hold_min_sec:
                    if side == "right":
                        self.right_count += 1
                    else:
                        self.left_count += 1
                    self._expected_side = "left" if side == "right" else "right"
                    self._set_feedback(
                        f"✓ {side.capitalize()} rep counted!  Good job!"
                    )
                else:
                    short_by = self.hold_min_sec - hold_elapsed
                    self._set_feedback(
                        f"Too short ({hold_elapsed:.1f}s) — "
                        f"need {self.hold_min_sec:.0f}s  ({short_by:.0f}s more)"
                    )

                self._state = _State.NEUTRAL
                self._hold_start = None
                self._neutral_frames = 0

        return tilt_signal, hold_elapsed

    def _set_feedback(self, msg: str, frames: int = 90) -> None:
        self._feedback = msg
        self._feedback_frames = frames

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _draw_hud(
        self,
        frame: np.ndarray,
        tilt_signal: float | None,
        hold_elapsed: float | None,
    ) -> None:
        h, w = frame.shape[:2]

        # ---- Pre-active states -------------------------------------------

        if self._state == _State.WAITING:
            msg = "SIT OR STAND UPRIGHT"
            (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
            cv2.putText(
                frame, msg, ((w - tw) // 2, h // 2 - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 200, 255), 3,
            )
            sub = "Keep head neutral, then press SPACE"
            (tw2, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(
                frame, sub, ((w - tw2) // 2, h // 2 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
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
            progress = len(self._cal_tilt_samples)
            cv2.putText(
                frame,
                f"Calibrating neutral head... ({progress}/{_CALIBRATION_FRAMES})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
            )
            return

        # ---- Active states -----------------------------------------------

        # Progress counter — top-left
        prog = (
            f"Right: {self.right_count}/{self.reps_per_side}   "
            f"Left: {self.left_count}/{self.reps_per_side}"
        )
        cv2.putText(
            frame, prog, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
        )

        # Tilt signal gauge (horizontal bar, 200 px wide) below the counter
        if tilt_signal is not None:
            bar_w, bar_h, bx, by = 200, 14, 10, 46
            cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
            cx = bx + bar_w // 2
            clamp = max(-1.0, min(1.0, tilt_signal / self.tilt_threshold))
            fill_x = int(clamp * (bar_w // 2))
            if fill_x > 0:  # right tilt
                cv2.rectangle(
                    frame, (cx, by), (cx + fill_x, by + bar_h), (0, 180, 255), -1
                )
            elif fill_x < 0:  # left tilt
                cv2.rectangle(
                    frame, (cx + fill_x, by), (cx, by + bar_h), (255, 100, 0), -1
                )
            # Threshold markers
            cv2.line(frame, (cx + bar_w // 2, by), (cx + bar_w // 2, by + bar_h), (0, 0, 180), 2)
            cv2.line(frame, (cx - bar_w // 2, by), (cx - bar_w // 2, by + bar_h), (0, 0, 180), 2)

        # ---- Main direction cue (centered) --------------------------------

        if self._state == _State.NEUTRAL:
            if self._neutral_frames < _NEUTRAL_DWELL_FRAMES:
                # Brief pause after returning — surface the tighten cue
                cue = "Return to neutral — Tighten both sides twice"
                (tw, _), _ = cv2.getTextSize(cue, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                cv2.putText(
                    frame, cue, ((w - tw) // 2, h // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 180), 2,
                )
            else:
                if self._expected_side == "right":
                    arrow, color = "TILT RIGHT  -->", (0, 200, 255)
                else:
                    arrow, color = "<--  TILT LEFT", (255, 140, 0)
                (tw, _), _ = cv2.getTextSize(arrow, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)
                cv2.putText(
                    frame, arrow, ((w - tw) // 2, h // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 3,
                )
                sub = "Bring ear slowly to shoulder — feel the stretch"
                (tw2, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
                cv2.putText(
                    frame, sub, ((w - tw2) // 2, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 200, 200), 2,
                )

        elif self._state in (_State.TILTED_RIGHT, _State.TILTED_LEFT):
            side_lbl = "RIGHT" if self._state == _State.TILTED_RIGHT else "LEFT"
            elapsed = hold_elapsed if hold_elapsed is not None else 0.0
            remaining = max(0.0, self.hold_min_sec - elapsed)

            if remaining <= 0.0:
                hold_msg = f"HOLDING {side_lbl}  ✓  {elapsed:.1f}s"
                hold_color = (0, 255, 100)
                sub = "Return to center when ready"
            else:
                hold_msg = (
                    f"HOLDING {side_lbl}  {elapsed:.1f}s / {self.hold_min_sec:.0f}s"
                )
                hold_color = (0, 180, 255)
                sub = f"Keep holding... {remaining:.0f}s more"

            (tw, _), _ = cv2.getTextSize(hold_msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.putText(
                frame, hold_msg, ((w - tw) // 2, h // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, hold_color, 3,
            )
            (tw2, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.putText(
                frame, sub, ((w - tw2) // 2, h // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2,
            )

            # Hold progress bar
            bar_w, bar_h = 300, 18
            bx = (w - bar_w) // 2
            by = h // 2 + 58
            cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
            fill = int(min(elapsed / self.hold_min_sec, 1.0) * bar_w)
            fill_color = (0, 255, 100) if remaining <= 0.0 else (0, 180, 255)
            cv2.rectangle(frame, (bx, by), (bx + fill, by + bar_h), fill_color, -1)

        # ---- Transient feedback -------------------------------------------
        if self._feedback_frames > 0:
            alpha = min(1.0, self._feedback_frames / 30.0)
            is_good = self._feedback.startswith("✓")
            color = (
                (0, int(230 * alpha), int(100 * alpha))
                if is_good
                else (0, int(80 * alpha), int(230 * alpha))
            )
            (tw, _), _ = cv2.getTextSize(
                self._feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
            )
            cv2.putText(
                frame, self._feedback, ((w - tw) // 2, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2,
            )

        # ---- Completion banner --------------------------------------------
        if self.is_complete:
            done = "NECK STRETCH COMPLETE!"
            (tw, _), _ = cv2.getTextSize(done, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)
            cv2.putText(
                frame, done, ((w - tw) // 2, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3,
            )
