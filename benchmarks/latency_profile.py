"""
benchmarks/latency_profile.py
===============================
Measures per-frame inference latency for YOLO across multiple model sizes.

Usage:
    python benchmarks/latency_profile.py --frames 200
"""

import argparse
import time
import statistics
import numpy as np


MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]


def dummy_frame(h: int = 720, w: int = 1280) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def profile_yolo(model_path: str, n_frames: int) -> dict:
    """
    Run YOLO inference on n_frames dummy frames and collect timings.
    Uncomment real inference once model loading is set up.
    """
    # from ultralytics import YOLO
    # model = YOLO(model_path)
    times = []

    for _ in range(n_frames):
        frame = dummy_frame()
        t0    = time.perf_counter()
        # result = model.predict(frame, verbose=False)
        time.sleep(0.01)  # stub
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "model":   model_path,
        "frames":  n_frames,
        "mean_ms": round(statistics.mean(times), 2),
        "p50_ms":  round(statistics.median(times), 2),
        "p95_ms":  round(statistics.quantiles(times, n=20)[18], 2),
        "max_ms":  round(max(times), 2),
    }


def print_table(results: list[dict]) -> None:
    print(f"\n{'Model':<16} {'Mean ms':>8} {'P50 ms':>8} {'P95 ms':>8} {'Max ms':>8}")
    print("-" * 54)
    for r in results:
        print(f"{r['model']:<16} {r['mean_ms']:>8.1f} {r['p50_ms']:>8.1f} "
              f"{r['p95_ms']:>8.1f} {r['max_ms']:>8.1f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=100,
                        help="Number of frames to profile per model")
    args = parser.parse_args()

    print(f"Profiling {args.frames} frames per model...")
    results = [profile_yolo(m, args.frames) for m in MODELS]
    print_table(results)
