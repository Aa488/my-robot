"""MediaPipe/BlazePose detector SERVER (runs in .venv_mp: numpy 2.x + mediapipe).

The main WHAM pipeline runs numpy 1.23 (torch 1.11) and cannot import mediapipe,
so detection runs here as a separate process. The pipeline talks to us over a
localhost TCP socket.

Protocol (big-endian, length-prefixed, synchronous request/response):
  client -> server:  <4-byte len><JPEG bytes of a BGR frame>
  server -> client:  <4-byte len><UTF-8 JSON>
      JSON = {"landmarks": [[x,y,z,vis] * 33] | null, "ms": <float>}

  A request with len == 0 is the shutdown signal.

Run manually (optional; the proxy spawns this by default):
  .venv_mp\\Scripts\\python scripts/mediapipe_detector_server.py
"""
import json
import os
import socket
import struct
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

HOST = "127.0.0.1"


def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("client closed the connection")
        buf += chunk
    return buf


def main():
    port = int(os.environ.get("WHAM_MEDIAPIPE_PORT", "8765"))
    complexity = int(os.environ.get("WHAM_MEDIAPIPE_COMPLEXITY", "0"))

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(1)
    print(f"[mp-server] listening on {HOST}:{port}", flush=True)

    conn, addr = srv.accept()
    print(f"[mp-server] client connected from {addr}", flush=True)
    try:
        while True:
            hdr = recv_exact(conn, 4)
            n = struct.unpack(">I", hdr)[0]
            if n == 0:
                print("[mp-server] shutdown signal received", flush=True)
                break
            jpeg = recv_exact(conn, n)
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                resp = {"landmarks": None, "ms": 0.0}
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                t0 = time.perf_counter()
                res = pose.process(rgb)
                ms = (time.perf_counter() - t0) * 1000.0
                if res.pose_landmarks is None:
                    resp = {"landmarks": None, "ms": ms}
                else:
                    lm = res.pose_landmarks.landmark
                    arr = [[p.x, p.y, p.z, p.visibility] for p in lm]
                    resp = {"landmarks": arr, "ms": ms}
            payload = json.dumps(resp).encode("utf-8")
            conn.sendall(struct.pack(">I", len(payload)) + payload)
    except ConnectionError:
        pass
    finally:
        conn.close()
        srv.close()
        pose.close()


if __name__ == "__main__":
    main()
