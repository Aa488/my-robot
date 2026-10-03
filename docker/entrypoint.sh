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
#   /output/video/live_stream_robot.mp4  rendered robot animation

set -euo pipefail

export PATH=/opt/conda/envs/wham_gmr/bin:${PATH}
export PYTHONPATH=/opt/src/wham
export WHAM_ENV=wham_gmr
export GMR_ENV=wham_gmr

# Input video: positional argument wins, then -e VIDEO, then the baked-in demo clip.
export VIDEO="${1:-${VIDEO:-/opt/src/wham/examples/drone_video.mp4}}"

# Target robot (override with -e ROBOT=...).
export ROBOT="${ROBOT:-linglong2}"

# Headless recording by default: MuJoCo EGL offscreen renderer, no GLFW window, so
# recording does a single offscreen render per frame (GMR_HEADLESS_RECORD=1). To pop
# a live viewer window, override with
#   -e GMR_HEADLESS_RECORD=0 -e MUJOCO_GL=glfw -e DISPLAY=<x-server>:0
# (DISPLAY must point at a reachable X server, e.g. VcXsrv on the Windows host).
# Robot-side rendering only; skip the WHAM mesh-overlay video (needs pytorch3d).
export RECORD_GMRVIDEO="${RECORD_GMRVIDEO:-1}"
export RECORD_WHAMVIDEO="${RECORD_WHAMVIDEO:-0}"
export GMR_HEADLESS_RECORD="${GMR_HEADLESS_RECORD:-1}"
export USE_XVFB_GMR="${USE_XVFB_GMR:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# Performance defaults (override with -e): fp16 WHAM on GPU, GMR IK on CPU,
# YOLO bbox detector on CPU (ViTPose stays on GPU).
export WHAM_USE_AMP="${WHAM_USE_AMP:-1}"
export WHAM_BBOX_DEVICE="${WHAM_BBOX_DEVICE:-cpu}"
export GMR_TORCH_DEVICE="${GMR_TORCH_DEVICE:-cpu}"

export OUTPUT_ROOT=/output

cd /opt/src/wham
exec bash run.sh
