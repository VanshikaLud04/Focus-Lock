"""
benchmarks/record_session.py
==============================
Records a labelled ground-truth session for benchmarking.
Press keys to annotate focus state in real-time:

    F = FOCUSED
    D = DISTRACTED
    Q = quit and save

Output: JSON file with frame-by-frame ground-truth labels.
"""

import cv2
import json
import time
from pathlib import Path


def record_ground_truth(output_path: str = "benchmarks/data/gt_session.json") -> None:
    """
    Open webcam and let the operator manually label each frame.
    Saves a JSON list: [{ts, label}, ...]
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    labels = []
    current_label = "FOCUSED"
    print("Recording. Keys: F=focused  D=distracted  Q=quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts = time.time()
        labels.append({"ts": ts, "label": current_label})

        color = (80, 210, 100) if current_label == "FOCUSED" else (60, 80, 240)
        cv2.putText(frame, f"[GT] {current_label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.imshow("Ground Truth Recording", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("f"):
            current_label = "FOCUSED"
        elif key == ord("d"):
            current_label = "DISTRACTED"
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    with open(output_path, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"Saved {len(labels)} frames to {output_path}")


if __name__ == "__main__":
    record_ground_truth()
