"""Benchmark MediaPipe BlazePose front-end on a real video, in the .venv_mp env.

Run:  .venv_mp\\Scripts\\python scripts/bench_mediapipe.py --video docs/video/1_incam.mp4 --frames 60
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from scripts.mediapipe_frontend import MediaPipeDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--complexity", type=int, default=0, choices=[0, 1, 2])
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")

    det = MediaPipeDetector(model_complexity=args.complexity)

    times = []
    n_det = 0
    n = 0
    try:
        while n < args.frames:
            ok, frame = cap.read()
            if not ok:
                break
            det.track(frame, 30, 1000)
            times.append(det.last_ms)
            if det.pose_results_last:
                n_det += 1
            n += 1
    finally:
        det.close()
        cap.release()

    times = np.array(times)
    print(f"\n=== video={args.video} complexity={args.complexity} "
          f"frames={n} detected={n_det} ===")
    print(f"per-frame: mean={times.mean():.1f} ms  min={times.min():.1f} "
          f"max={times.max():.1f}  p50={np.percentile(times, 50):.1f} "
          f"-> {1000.0 / times.mean():.1f} fps")


if __name__ == "__main__":
    main()
