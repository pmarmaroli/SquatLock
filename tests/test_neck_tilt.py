"""Standalone test for neck lateral-tilt detection.

Opens the webcam, runs the NeckTiltDetector, and displays an annotated
feed with per-side counters and a live hold timer.

Keys:
  SPACE  — start calibration countdown
  'r'    — reset counter and re-calibrate
  's'    — start / stop recording to recordings/
  'q'    — quit

Usage:
    python tests/test_neck_tilt.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

from neck_tilt_detector import NeckTiltDetector

_RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "..", "recordings")


def run(camera_index: int = 0) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}")
        sys.exit(1)

    detector = NeckTiltDetector()
    writer: cv2.VideoWriter | None = None
    recording = False

    print(
        "Neck-tilt test — SPACE calibrate | 'q' quit | 'r' reset | 's' record"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame from camera")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Save raw frame BEFORE annotation overlay (same pattern as twist test).
        if recording and writer is not None:
            writer.write(frame.copy())

        frame = detector.process_frame(frame)

        # -- Per-side counter (large, bottom-right) --
        counter_text = (
            f"R: {detector.right_count}/{detector.reps_per_side}  "
            f"L: {detector.left_count}/{detector.reps_per_side}"
        )
        (tw, th), _ = cv2.getTextSize(
            counter_text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3
        )
        cv2.putText(
            frame, counter_text, (w - tw - 20, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3,
        )

        # Recording indicator
        if recording:
            cv2.circle(frame, (w - 30, 30), 12, (0, 0, 255), -1)
            cv2.putText(
                frame, "REC", (w - 75, 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )

        cv2.putText(
            frame,
            "SPACE calibrate | 'q' quit | 'r' reset | 's' record",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        cv2.imshow("SquatLock — Neck Tilt Test", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            detector.reset()
            print("Counter reset.")
        elif key == ord(" "):
            detector.signal_ready()
        elif key == ord("s"):
            if not recording:
                os.makedirs(_RECORDINGS_DIR, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(_RECORDINGS_DIR, f"neck_tilt_{ts}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
                recording = True
                print(f"Recording started → {path}")
            else:
                recording = False
                if writer is not None:
                    writer.release()
                    writer = None
                print("Recording stopped.")

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print(
        f"Test complete. "
        f"Right: {detector.right_count}  Left: {detector.left_count}"
    )


if __name__ == "__main__":
    run()
