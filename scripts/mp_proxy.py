"""MediaPipe detector proxy — drop-in for lib.models.preproc.detector.DetectionModel.

The actual BlazePose inference runs in a SEPARATE process using the .venv_mp
interpreter (numpy 2.x + mediapipe), because the main WHAM pipeline runs
numpy 1.23 (torch 1.11) and cannot import mediapipe. This proxy talks to that
process over a localhost TCP socket (see mediapipe_detector_server.py).

Interface mirrors DetectionModel (only the parts the pipeline uses):
    detector.track(frame_bgr, fps, length)   -> sets detector.pose_results_last
    detector.pose_results_last = [{'keypoints': (17,3), 'bbox': (4,), 'track_id': 0}]

Enable in handle_wham_gmr.py with WHAM_USE_MEDIAPIPE=1. The proxy spawns the
server automatically using .venv_mp\\Scripts\\python.exe (override with
WHAM_MEDIAPIPE_PYTHON); set WHAM_MEDIAPIPE_SPAWN=0 to connect to an already
running server instead.

Pure-numpy deps only (no mediapipe import) — safe to import from wham_gmr.
"""
import json
import os
import socket
import struct
import subprocess
import time

import cv2
import numpy as np

from scripts.mp_convert import mediapipe_to_coco_keypoints

DEFAULT_PORT = 8765


def _env_flag(name, default=False):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


class MediaPipeDetectorProxy:
    def __init__(self, host="127.0.0.1", port=None, max_dim=640,
                 bbox_margin=0.15, jpeg_quality=90, server_python=None,
                 spawn=None):
        self.host = host
        self.port = int(port or os.environ.get("WHAM_MEDIAPIPE_PORT", str(DEFAULT_PORT)))
        self.max_dim = int(os.environ.get("WHAM_MEDIAPIPE_MAX_DIM", str(max_dim)))
        self.bbox_margin = bbox_margin
        self.jpeg_quality = int(os.environ.get("WHAM_MEDIAPIPE_JPEG_QUALITY", str(jpeg_quality)))
        self.pose_results_last = []
        self.last_ms = 0.0
        self._proc = None
        self._sock = None

        if spawn is None:
            spawn = _env_flag("WHAM_MEDIAPIPE_SPAWN", True)
        if spawn:
            self._spawn_server(server_python)
        self._connect()

    # ---- server lifecycle -------------------------------------------------
    def _spawn_server(self, server_python):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        server_script = os.path.join(root, "scripts", "mediapipe_detector_server.py")

        if not server_python:
            server_python = os.environ.get("WHAM_MEDIAPIPE_PYTHON")
        if not server_python:
            candidate = os.path.join(root, ".venv_mp", "Scripts", "python.exe")
            server_python = candidate if os.path.exists(candidate) else None

        if not server_python or not os.path.exists(server_python):
            raise RuntimeError(
                "MediaPipe server python not found; set WHAM_MEDIAPIPE_PYTHON "
                "or create .venv_mp (see scripts/mediapipe_frontend.py header)"
            )

        env = dict(os.environ)
        env["WHAM_MEDIAPIPE_PORT"] = str(self.port)
        self._proc = subprocess.Popen(
            [server_python, server_script],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _connect(self, retries=100, delay=0.1):
        last = None
        for _ in range(retries):
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"MediaPipe detector server exited early (code={self._proc.returncode})"
                )
            try:
                self._sock = socket.create_connection((self.host, self.port), timeout=2.0)
                return
            except OSError as e:
                last = e
                time.sleep(delay)
        raise RuntimeError(f"cannot connect to MediaPipe detector server: {last}")

    # ---- DetectionModel-compatible interface -----------------------------
    def initialize_tracking(self):
        # Reset per-run state; BlazePose tracks internally so there is nothing
        # else to (re)initialise — the pipeline only reads pose_results_last.
        self.pose_results_last = []

    def track(self, img_bgr, fps, length):
        img_h, img_w = img_bgr.shape[:2]
        scale = 1.0
        if max(img_h, img_w) > self.max_dim:
            scale = self.max_dim / float(max(img_h, img_w))
            img_bgr = cv2.resize(
                img_bgr, (int(img_w * scale), int(img_h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            self.pose_results_last = []
            return self.pose_results_last
        jpeg = enc.tobytes()

        try:
            self._sock.sendall(struct.pack(">I", len(jpeg)) + jpeg)
            hdr = self._recv_exact(4)
            n = struct.unpack(">I", hdr)[0]
            payload = json.loads(self._recv_exact(n).decode("utf-8"))
        except (OSError, ConnectionError, json.JSONDecodeError) as e:
            raise RuntimeError(f"MediaPipe detector server error: {e}")

        self.last_ms = float(payload.get("ms", 0.0))
        lm = payload.get("landmarks")
        if lm is None:
            self.pose_results_last = []
            return self.pose_results_last

        # landmarks are normalized -> map back to ORIGINAL frame pixels
        arr = np.asarray(lm, dtype=np.float32)  # (33, 4)
        kp2d, bbox = mediapipe_to_coco_keypoints(arr, img_w, img_h, self.bbox_margin)
        self.pose_results_last = [{
            "keypoints": kp2d,
            "bbox": bbox,
            "track_id": 0,
        }]
        return self.pose_results_last

    def _recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("MediaPipe detector server closed the connection")
            buf += chunk
        return buf

    def close(self):
        if self._sock is not None:
            try:
                self._sock.sendall(struct.pack(">I", 0))  # shutdown signal
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
