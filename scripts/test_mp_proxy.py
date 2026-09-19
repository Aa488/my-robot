"""Integration test: MediaPipeDetectorProxy (wham_gmr) <-> server (.venv_mp).

Run with the wham_gmr python:  python scripts/test_mp_proxy.py
The proxy auto-spawns .venv_mp\\Scripts\\python.exe running the detector server.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from scripts.mp_proxy import MediaPipeDetectorProxy


def main():
    cap = cv2.VideoCapture("docs/video/1_incam.mp4")
    if not cap.isOpened():
        raise SystemExit("cannot open video")

    det = MediaPipeDetectorProxy()
    times = []
    n_det = 0
    try:
        for i in range(30):
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            det.track(frame, 30, 1000)
            times.append((time.perf_counter() - t0) * 1000.0)
            r = det.pose_results_last
            if r:
                n_det += 1
                kp, bb = r[0]["keypoints"], r[0]["bbox"]
                print(f"frame {i:2d}: kp={kp.shape} bbox={np.round(bb).astype(int)} "
                      f"server_ms={det.last_ms:.1f}")
            else:
                print(f"frame {i:2d}: no person (server_ms={det.last_ms:.1f})")
    finally:
        det.close()
        cap.release()

    times = np.array(times)
    print(f"\n=== {len(times)} frames, detected={n_det} ===")
    print(f"end-to-end (encode+pipe+MP+decode): mean={times.mean():.1f} ms "
          f"min={times.min():.1f} max={times.max():.1f}")


if __name__ == "__main__":
    main()
