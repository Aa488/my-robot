"""MediaPipe BlazePose front-end: fast 33-landmark detector/tracker on CPU.

Replaces the GPU-heavy YOLOv8n + ViTPose detector (~72.7 ms) with BlazePose
(~10-15 ms on CPU), freeing the GPU entirely for the WHAM thread and removing
the GPU contention that currently adds ~70 ms of overhead.

Output: per-frame 33 landmarks, each with (x, y, z, visibility):
  - (x, y) normalized to [0,1] image coords (x right, y down).
  - z: relative depth, same scale as x (negative = closer to camera), in
        image-width units around the hip center.
  - visibility: [0,1] likelihood the landmark is visible/in-frame.

Run standalone:  .venv_mp\\Scripts\\python scripts\\mediapipe_frontend.py --video path.mp4
Run webcam:       .venv_mp\\Scripts\\python scripts\\mediapipe_frontend.py --webcam 0

Importable:  from scripts.mediapipe_frontend import MediaPipeFrontend
"""
import argparse
import time

import cv2
import mediapipe as mp
import numpy as np

from scripts.mp_convert import MP_TO_COCO, mediapipe_to_coco_keypoints  # noqa: F401

# BlazePose 33-landmark indices (subset you'll likely need for retargeting)
KP = {
    "nose": 0,
    "left_eye": 2, "right_eye": 5,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}


class MediaPipeFrontend:
    def __init__(self, model_complexity=0, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """model_complexity: 0=lite (fastest), 1=full, 2=heavy."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,          # video mode -> enables tracking
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_bgr(self, frame_bgr):
        """Return (landmarks_np, elapsed_ms).

        landmarks_np: float32 (33, 4) = (x, y, z, visibility), or None if no
        person detected in this frame.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        t0 = time.perf_counter()
        results = self.pose.process(frame_rgb)
        dt = (time.perf_counter() - t0) * 1000.0
        if results.pose_landmarks is None:
            return None, dt
        lm = results.pose_landmarks.landmark
        arr = np.array([[p.x, p.y, p.z, p.visibility] for p in lm],
                       dtype=np.float32)
        return arr, dt

    def close(self):
        self.pose.close()


class MediaPipeDetector:
    """Drop-in replacement for lib.models.preproc.detector.DetectionModel.

    Mirrors the interface the pipeline uses:
        detector.track(frame_bgr, fps, length)  ->  sets detector.pose_results_last
        detector.pose_results_last = [{'keypoints': (17,3), 'bbox': (4,), 'track_id': 0}]

    BlazePose is single-person and tracks internally (static_image_mode=False),
    so every call returns at most one pose. Runs on CPU -> frees the GPU for the
    WHAM thread (removing the ~70 ms of GPU contention from YOLO+ViTPose).
    """

    def __init__(self, model_complexity=0, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5, bbox_margin=0.15, max_dim=640):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.bbox_margin = bbox_margin
        self.max_dim = max_dim
        self.pose_results_last = []
        self.last_ms = 0.0

    def track(self, img_bgr, fps, length):
        # BlazePose returns normalized coords, so we may downscale the frame to
        # `max_dim` for speed and map the landmarks back to the original
        # resolution below. The WHAM pipeline only consumes the 2D keypoints +
        # bbox, so downscaling costs nothing in accuracy (extractor re-crops
        # the original frame to 256x256 anyway).
        img_h, img_w = img_bgr.shape[:2]
        scale = 1.0
        if max(img_h, img_w) > self.max_dim:
            scale = self.max_dim / float(max(img_h, img_w))
            img_bgr = cv2.resize(img_bgr, (int(img_w * scale), int(img_h * scale)),
                                 interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        t0 = time.perf_counter()
        results = self.pose.process(frame_rgb)
        self.last_ms = (time.perf_counter() - t0) * 1000.0

        if results.pose_landmarks is None:
            self.pose_results_last = []
            return self.pose_results_last

        lm = results.pose_landmarks.landmark
        arr = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], dtype=np.float32)
        # map normalized coords back to the ORIGINAL frame resolution
        kp2d, bbox = mediapipe_to_coco_keypoints(arr, img_w, img_h, self.bbox_margin)
        self.pose_results_last = [{
            "keypoints": kp2d,
            "bbox": bbox,
            "track_id": 0,
        }]
        return self.pose_results_last

    def close(self):
        self.pose.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default=None, help="video file path")
    ap.add_argument("--webcam", type=int, default=None, help="webcam index")
    ap.add_argument("--complexity", type=int, default=0, choices=[0, 1, 2])
    args = ap.parse_args()

    if args.video:
        cap = cv2.VideoCapture(args.video)
    elif args.webcam is not None:
        cap = cv2.VideoCapture(args.webcam)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise SystemExit("cannot open capture source")

    frontend = MediaPipeFrontend(model_complexity=args.complexity)
    times = []
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            lm, dt = frontend.process_bgr(frame)
            times.append(dt)
            n += 1
            if lm is not None:
                # Example: world-frame shoulder/hip positions
                ls = lm[KP["left_shoulder"]]
                rh = lm[KP["right_hip"]]
                print(f"frame {n:4d}  {dt:6.1f} ms  LShoulder=({ls[0]:.2f},{ls[1]:.2f},{ls[2]:.2f}) "
                      f"RHip=({rh[0]:.2f},{rh[1]:.2f},{rh[2]:.2f})")
            else:
                print(f"frame {n:4d}  {dt:6.1f} ms  (no person)")
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    finally:
        frontend.close()
        cap.release()

    times = np.array(times)
    print(f"\n=== {n} frames  mean={times.mean():.1f} ms  min={times.min():.1f} "
          f"max={times.max():.1f}  -> {1000.0 / times.mean():.1f} fps ===")


if __name__ == "__main__":
    main()
