"""Pure-numpy MediaPipe -> COCO keypoint conversion (NO mediapipe import).

Imported by BOTH:
  - scripts/mediapipe_frontend.py  (runs in .venv_mp, numpy 2.x + cv2 5.0)
  - the detector proxy in the main WHAM pipeline (runs in wham_gmr, numpy 1.23)

so it must not touch mediapipe — that keeps the two environments' numpy
versions isolated. Dependencies: numpy only.
"""
import numpy as np

# MediaPipe 33-landmark -> COCO 17-keypoint indices.
# COCO order: nose, L eye, R eye, L ear, R ear, L shoulder, R shoulder,
#             L elbow, R elbow, L wrist, R wrist, L hip, R hip, L knee, R knee,
#             L ankle, R ankle.
MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def mediapipe_to_coco_keypoints(landmarks, img_w, img_h, bbox_margin=0.15):
    """Convert MediaPipe 33 landmarks -> COCO17 kp2d + xyxy bbox.

    landmarks: (33, 4) = (x_norm, y_norm, z, visibility), x/y in [0,1]
    img_w, img_h: pixel size of the ORIGINAL (unscaled) frame the pipeline
                  will crop from (bbox/keypoints are returned in these pixels).
    Returns (kp2d, bbox_xyxy):
      kp2d      (17, 3) float32 = (x_px, y_px, confidence)
      bbox_xyxy (4,)   float32 = [x1, y1, x2, y2] with `bbox_margin` padding
    """
    kp2d = np.zeros((17, 3), dtype=np.float32)
    for ci, mi in enumerate(MP_TO_COCO):
        kp2d[ci, 0] = landmarks[mi, 0] * img_w
        kp2d[ci, 1] = landmarks[mi, 1] * img_h
        kp2d[ci, 2] = float(landmarks[mi, 3])  # visibility ~ confidence

    vis = kp2d[:, 2] > 0.3
    if int(vis.sum()) < 2:
        # degenerate: fall back to a full-frame bbox so the pipeline still runs
        return kp2d, np.array([0.0, 0.0, float(img_w), float(img_h)],
                              dtype=np.float32)

    xs = kp2d[vis, 0]
    ys = kp2d[vis, 1]
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    w, h = x2 - x1, y2 - y1
    m = bbox_margin
    x1 = max(0.0, x1 - m * w)
    y1 = max(0.0, y1 - m * h)
    x2 = min(float(img_w), x2 + m * w)
    y2 = min(float(img_h), y2 + m * h)
    return kp2d, np.array([x1, y1, x2, y2], dtype=np.float32)
