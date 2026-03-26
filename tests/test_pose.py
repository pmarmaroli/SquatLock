"""Standalone test for SquatLock's squat detection.

Opens the webcam, runs the SquatDetector (shoulder-displacement based),
and displays the annotated feed with a live squat counter.
Press 'q' to quit, 'r' to reset the counter.

Usage:
    python tests/test_pose.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

from squat_detector import SquatDetector


def run(camera_index: int = 0) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}")
        sys.exit(1)

    detector = SquatDetector()

    print("SquatLock Pose test — press 'q' to quit, 'r' to reset counter")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame from camera")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Process frame (draws skeleton + state HUD)
        frame = detector.process_frame(frame)

        # -- Squat counter (large, bottom-right) --
        counter_text = f"Squats: {detector.count}"
        (tw, th), _ = cv2.getTextSize(
            counter_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
        )
        cv2.putText(
            frame, counter_text, (w - tw - 20, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3,
        )

        cv2.putText(
            frame, "Press 'q' quit | 'r' reset", (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        cv2.imshow("SquatLock — Pose Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            detector.reset()
            print("Counter reset.")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print(f"Test complete. Total squats: {detector.count}")


if __name__ == "__main__":
    run()
