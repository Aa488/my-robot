#!/usr/bin/env bash
# WHAM -> GMR container entrypoint: process a video and emit robot joint angles.
#
# Usage:
#   docker run --gpus all --rm -v $PWD/out:/output wham-gmr:latest [video.mp4]
#   docker run --gpus all --rm -v $PWD/out:/output -e ROBOT=linglong2 wham-gmr:latest /input/video.mp4
#
# Outputs (under /output, created by run.sh):
#   /output/csv/live_motion.csv        robot joint-angle sequence
#   /output/pkl/my_motion.pkl          retargeted motion (python pickle)
#   /output/video/live_stream_robot.mp4  rendered robot animation (only when RECORD_GMRVIDEO=1)

set -euo pipefail

export PATH=/opt/conda/envs/wham_gmr/bin:${PATH}
export PYTHONPATH=/opt/src/wham
export WHAM_ENV=wham_gmr
export GMR_ENV=wham_gmr

# Input video: positional argument wins, then -e VIDEO, then the baked-in demo clip.
export VIDEO="${1:-${VIDEO:-/opt/src/wham/examples/drone_video.mp4}}"

# Target robot (override with -e ROBOT=...).
export ROBOT="${ROBOT:-linglong2}"

# Headless by default: MuJoCo EGL offscreen renderer, no GLFW window
# (GMR_HEADLESS_RECORD=1). Robot mp4 recording is OFF by default — rendering +
# video encoding is a real latency/GPU cost in the stream loop. Set
# -e RECORD_GMRVIDEO=1 to record the robot animation (recording is independent
# of the window switch; it does NOT force a window). To also pop a live viewer
# window, override with
#   -e GMR_HEADLESS_RECORD=0 -e MUJOCO_GL=glfw -e DISPLAY=<x-server>:0
# (DISPLAY must point at a reachable X server, e.g. VcXsrv on the Windows host).
# Skip the WHAM mesh-overlay video (needs pytorch3d).
export RECORD_GMRVIDEO="${RECORD_GMRVIDEO:-0}"
export RECORD_WHAMVIDEO="${RECORD_WHAMVIDEO:-0}"
export GMR_HEADLESS_RECORD="${GMR_HEADLESS_RECORD:-1}"
export USE_XVFB_GMR="${USE_XVFB_GMR:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# Performance defaults (override with -e): WHAM on GPU with fp16 OFF (fp16 caused
# arm jumps), GMR IK on GPU (matches run.ps1). WHAM_BBOX_DEVICE only affects the
# fallback YOLOv8n+ViTPose detector — the default single-stage YOLOv8-pose
# (WHAM_USE_YOLOPOSE=1 in run.sh) ignores it.
export WHAM_USE_AMP="${WHAM_USE_AMP:-0}"
export WHAM_BBOX_DEVICE="${WHAM_BBOX_DEVICE:-cpu}"
export GMR_TORCH_DEVICE="${GMR_TORCH_DEVICE:-cuda}"

export OUTPUT_ROOT=/output

cd /opt/src/wham
exec bash run.sh
