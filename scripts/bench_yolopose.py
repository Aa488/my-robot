"""Benchmark single-stage YOLOv8-pose vs the two-stage YOLOv8n + ViTPose.

Run:  python scripts/bench_yolopose.py --video docs/video/1_incam.mp4 --frames 60
"""
import argparse
import os
import time

import cv2
import numpy as np

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--ckpt", default="checkpoints/yolov8n-pose.pt")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")

    model = YOLO(args.ckpt)
    times = []
    n_det = 0
    try:
        for i in range(args.frames):
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            results = model.predict(
                frame, device="cuda", classes=0, conf=args.conf,
                imgsz=args.imgsz, save=False, verbose=False,
            )
            dt = (time.perf_counter() - t0) * 1000.0
            times.append(dt)
            r = results[0]
            nk = 0 if r.keypoints is None else len(r.keypoints)
            if nk:
                n_det += 1
                kp = r.keypoints.data.cpu().numpy()   # (N, 17, 3) x,y,conf
                bb = r.boxes.xyxy.cpu().numpy()       # (N, 4)
                print(f"frame {i:2d}: {dt:6.1f} ms  persons={nk} "
                      f"kp={kp.shape} bbox={np.round(bb[0]).astype(int)}")
            else:
                print(f"frame {i:2d}: {dt:6.1f} ms  (no person)")
    finally:
        cap.release()

    times = np.array(times)
    print(f"\n=== {len(times)} frames, detected={n_det} ===")
    print(f"YOLOv8-pose: mean={times.mean():.1f} ms  min={times.min():.1f} "
          f"max={times.max():.1f}  p50={np.percentile(times, 50):.1f}")


if __name__ == "__main__":
    main()
