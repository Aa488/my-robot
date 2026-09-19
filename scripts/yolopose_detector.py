"""Single-stage YOLOv8-pose detector: person bbox + 17 COCO keypoints in one pass.

Drop-in replacement for lib.models.preproc.detector.DetectionModel. Runs
in-process in the wham_gmr env (ultralytics + torch) — no subprocess, no
numpy/cv2 isolation issues — and its keypoints are COCO-17, the distribution
WHAM was trained on (unlike MediaPipe's 33 landmarks).

Replaces the two-stage YOLOv8n (bbox, ~23 ms) + ViTPose-base (keypoints,
~49.7 ms) = ~72.7 ms with a single ~27 ms forward. Accuracy sits between
ViTPose (heatmap, highest) and MediaPipe/BlazePose (regression, lowest).

Interface mirrors DetectionModel (only the parts the pipeline uses):
    detector.initialize_tracking()
    detector.track(frame_bgr, fps, length)  ->  detector.pose_results_last
    detector.pose_results_last = [{'keypoints': (17,3), 'bbox': (4,), 'track_id': int}]
"""
import numpy as np

from ultralytics import YOLO


class YOLOPoseDetector:
    def __init__(self, device, model_ckpt="checkpoints/yolov8n-pose.pt",
                 conf=0.5, imgsz=640, iou=0.7):
        self.device = device
        self.model = YOLO(model_ckpt)
        self.conf = conf
        self.imgsz = imgsz
        self.iou = iou
        self.pose_results_last = []
        self.next_id = 0

    def initialize_tracking(self):
        self.pose_results_last = []
        self.next_id = 0

    def track(self, img, fps, length):
        results = self.model.predict(
            img, device=self.device, classes=0, conf=self.conf,
            imgsz=self.imgsz, iou=self.iou, save=False, verbose=False,
        )

        poses = []
        r = results[0]
        if r.keypoints is not None and len(r.keypoints) > 0:
            kps = r.keypoints.data.cpu().numpy()      # (N, 17, 3) x, y, conf
            boxes = r.boxes.xyxy.cpu().numpy()        # (N, 4) xyxy
            # stable selection for the dominant person: rank by bbox area
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            order = np.argsort(-area)
            for rank, i in enumerate(order):
                poses.append({
                    "keypoints": kps[i].astype(np.float32),
                    "bbox": boxes[i].astype(np.float32),
                    "track_id": int(rank),
                })

        self.pose_results_last = poses
        return poses

    def close(self):
        # YOLO model needs no explicit teardown; keep the interface uniform.
        pass
