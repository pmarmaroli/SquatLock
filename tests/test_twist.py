"""Standalone test for torso-twist detection.

Opens the webcam, runs the TwistDetector (shoulder horizontal shift),
and displays the annotated feed with a live twist counter.
Press 'q' to quit, 'r' to reset, SPACE to start calibration,
's' to start/stop recording a clip.

Usage:
    python tests/test_twist.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

from twist_detector import TwistDetector

_RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "..", "recordings")


def run(camera_index: int = 0) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}")
        sys.exit(1)

    detector = TwistDetector()
    writer: cv2.VideoWriter | None = None
    recording = False

    print("Twist test — SPACE start | 'q' quit | 'r' reset | 's' record")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame from camera")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Save raw frame BEFORE annotation overlay
        if recording and writer is not None:
            writer.write(frame.copy())

        frame = detector.process_frame(frame)

        # -- Twist counter (large, bottom-right) --
        counter_text = f"Twists: {detector.count}"
        (tw, th), _ = cv2.getTextSize(
            counter_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
        )
        cv2.putText(
            frame, counter_text, (w - tw - 20, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3,
        )

        # Recording indicator
        if recording:
            cv2.circle(frame, (w - 30, 30), 12, (0, 0, 255), -1)
            cv2.putText(
                frame, "REC", (w - 75, 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )

        cv2.putText(
            frame, "SPACE start | 'q' quit | 'r' reset | 's' record", (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        cv2.imshow("SquatLock — Twist Test", frame)
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
                path = os.path.join(_RECORDINGS_DIR, f"twist_{ts}.mp4")
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
    print(f"Test complete. Total twists: {detector.count}")


if __name__ == "__main__":
    run()
