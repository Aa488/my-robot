#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
	echo "[E2E] This script requires bash. Please run: bash run.sh"
	exit 2
fi

set -euo pipefail

# Ensure local packages are importable even when scripts are launched by path.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# =========================
# End-to-end WHAM -> GMR stream pipeline
# =========================
# Usage examples:
#   RECORD_GMRVIDEO=1 RECORD_WHAMVIDEO=0 VIDEO=examples/IMG_9732.mov bash run.sh
#   OUTPUT_ROOT=outputs/run1 RECORD_GMRVIDEO=0 VIDEO=0 bash run.sh
#   VIDEO=0 TIME=10 RECORD_GMRVIDEO=1 RECORD_WHAMVIDEO=1 bash run.sh

WHAM_ENV=${WHAM_ENV:-wham_gmr}
GMR_ENV=${GMR_ENV:-wham_gmr}
E2E_VALIDATE_ENV_ONLY=${E2E_VALIDATE_ENV_ONLY:-0}
WHAM_PYTHON=${WHAM_PYTHON:-$(which python)}
GMR_PYTHON=${GMR_PYTHON:-$(which python)}

# Validate that the selected WHAM Python can import smplx before starting the
# pipeline (mirrors run.ps1); optional env-only validation shortcut.
if ! "${WHAM_PYTHON}" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('smplx') else 1)" >/dev/null 2>&1; then
	echo "[E2E] Selected WHAM_PYTHON cannot import 'smplx': ${WHAM_PYTHON}" >&2
	exit 1
fi
if [[ "${E2E_VALIDATE_ENV_ONLY}" == "1" ]]; then
	echo "[E2E] Environment validation only; exiting before pipeline start."
	exit 0
fi

VIDEO=${VIDEO:-examples/output.mp4}
TIME=${TIME:-0}
ROBOT=${ROBOT:-linglong2}
ROBOTS=${ROBOTS:-${ROBOT}}
ROBOT_PATH=${ROBOT_PATH:-}
# GMR_FORCE_SINGLE_ROBOT=1 disables the multi-robot fan-out (keep only the first
# robot). No-op for the default single-robot setup, but guarantees one robot + one
# CSV + no scene offset even if ROBOTS lists several.
GMR_FORCE_SINGLE_ROBOT=${GMR_FORCE_SINGLE_ROBOT:-0}
export GMR_FORCE_SINGLE_ROBOT

# ROBOT_PATH supports both robot key (same choices as --robot) and custom XML path.
if [[ -n "${ROBOT_PATH}" ]]; then
	if [[ -f "${ROBOT_PATH}" || "${ROBOT_PATH}" == *.xml ]]; then
		CUSTOM_ROBOT_XML="${ROBOT_PATH}"
	else
		ROBOT="${ROBOT_PATH}"
		CUSTOM_ROBOT_XML=""
	fi
else
	CUSTOM_ROBOT_XML=""
fi

# Backward compatibility: old RECORD_VIDEO controls both sides when specific flags are not set.
LEGACY_RECORD_VIDEO=${RECORD_VIDEO:-1}
RECORD_WHAMVIDEO=${RECORD_WHAMVIDEO:-${LEGACY_RECORD_VIDEO}}
# Robot mp4 recording is OFF by default: rendering + video encoding is a real
# latency/GPU cost in the stream loop. Set RECORD_GMRVIDEO=1 to re-enable it.
RECORD_GMRVIDEO=${RECORD_GMRVIDEO:-0}
USE_XVFB_GMR=${USE_XVFB_GMR:-0}
# WHAM preview window: empty = follow RECORD_WHAMVIDEO; 1 = always show; 0 = never.
WHAM_PREVIEW=${WHAM_PREVIEW:-}

OUTPUT_ROOT=${OUTPUT_ROOT:-}
if [[ -n "${OUTPUT_ROOT}" ]]; then
	OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/stream_demo}
	STREAM_NPZ_DIR=${STREAM_NPZ_DIR:-${OUTPUT_DIR}/npz_stream}
	PKL_PATH=${PKL_PATH:-${OUTPUT_ROOT}/pkl/my_motion.pkl}
	CSV_ROOT=${CSV_ROOT:-${OUTPUT_ROOT}/csv}
	CSV_PATH=${CSV_PATH:-${OUTPUT_ROOT}/csv/live_motion.csv}
	GMR_VIDEO_PATH=${GMR_VIDEO_PATH:-${OUTPUT_ROOT}/video/live_stream_robot.mp4}
else
	OUTPUT_DIR=${OUTPUT_DIR:-output/stream_demo}
	STREAM_NPZ_DIR=${STREAM_NPZ_DIR:-${OUTPUT_DIR}/npz_stream}
	PKL_PATH=${PKL_PATH:-pkl_outputs/my_motion.pkl}
	CSV_ROOT=${CSV_ROOT:-pkl_outputs/csv}
	CSV_PATH=${CSV_PATH:-pkl_outputs/csv/live_motion.csv}
	GMR_VIDEO_PATH=${GMR_VIDEO_PATH:-videos/live_stream_robot.mp4}
fi

SMOOTH_ALPHA=${SMOOTH_ALPHA:-0.35}
HEIGHT_ADJUST=${HEIGHT_ADJUST:-1}
ROOT_ORIGIN_OFFSET=${ROOT_ORIGIN_OFFSET:-0}
# Jump detection for height_adjust: the lowest foot must rise this far above a
# running ground reference (a lower envelope of the foot height) before its
# vertical motion is kept.  Relative to the envelope, not absolute ground.
GMR_AIRBORNE_THRESHOLD=${GMR_AIRBORNE_THRESHOLD:-0.03}
export GMR_AIRBORNE_THRESHOLD
VIEWER_WARMUP_FRAMES=${VIEWER_WARMUP_FRAMES:-12}
POLL_INTERVAL=${POLL_INTERVAL:-0.05}
IDLE_TIMEOUT=${IDLE_TIMEOUT:-0}
DONE_GRACE_SEC=${DONE_GRACE_SEC:-2.0}

# WHAM performance controls (can be overridden via env before running run.sh).
WHAM_USE_AMP=${WHAM_USE_AMP:-0}
# Detector: single-stage YOLOv8-pose by default (~27ms/frame). The default
# YOLOv8n-bbox + ViTPose path was the real-time bottleneck (~4fps) because the
# bbox stage ran on CPU. Set WHAM_USE_YOLOPOSE=0 to fall back, or
# WHAM_USE_MEDIAPIPE=1 for the CPU BlazePose subprocess detector.
WHAM_USE_MEDIAPIPE=${WHAM_USE_MEDIAPIPE:-0}
WHAM_USE_YOLOPOSE=${WHAM_USE_YOLOPOSE:-1}
WHAM_MEDIAPIPE_COMPLEXITY=${WHAM_MEDIAPIPE_COMPLEXITY:-0}
WHAM_DETECT_INTERVAL=${WHAM_DETECT_INTERVAL:-1}
# Only used by the default YOLOv8n+ViTPose path (ignored when WHAM_USE_YOLOPOSE=1).
WHAM_BBOX_DEVICE=${WHAM_BBOX_DEVICE:-cpu}
WHAM_INFER_INTERVAL=${WHAM_INFER_INTERVAL:-1}
WHAM_STREAM_SEQ_LEN=${WHAM_STREAM_SEQ_LEN:-16}
WHAM_INPUT_SCALE=${WHAM_INPUT_SCALE:-1.0}
WHAM_DEDICATED_STREAM=${WHAM_DEDICATED_STREAM:-1}
# Batch-K window inference: K=2 batches two sliding windows per forward pass,
# ~2x WHAM throughput (19.7->37 fps) so 30fps input no longer drops frames. This
# removes the timing-dependent frame dropping that made runs non-deterministic
# (same video -> same robot motion). K=1 = legacy behaviour.
WHAM_BATCH_K=${WHAM_BATCH_K:-2}
# Webcam-only bounded backlog (frames kept before dropping stale ones): higher =
# smoother trajectory, lower = lower latency. 0 via WHAM_LATEST_FRAME disables.
WHAM_LATEST_FRAME=${WHAM_LATEST_FRAME:-1}
WHAM_LATEST_FRAME_KEEP=${WHAM_LATEST_FRAME_KEEP:-3}
# Adaptive keep (webcam): widen the drop window to FAST_KEEP while root speed
# exceeds SPEED_THRESHOLD m/s (fast motions like jumps), shrink back otherwise.
WHAM_ADAPTIVE_KEEP=${WHAM_ADAPTIVE_KEEP:-1}
WHAM_ADAPTIVE_FAST_KEEP=${WHAM_ADAPTIVE_FAST_KEEP:-8}
WHAM_ADAPTIVE_SPEED_THRESHOLD=${WHAM_ADAPTIVE_SPEED_THRESHOLD:-1.5}
export WHAM_USE_AMP WHAM_USE_MEDIAPIPE WHAM_USE_YOLOPOSE WHAM_MEDIAPIPE_COMPLEXITY WHAM_DETECT_INTERVAL WHAM_BBOX_DEVICE WHAM_INFER_INTERVAL WHAM_STREAM_SEQ_LEN WHAM_INPUT_SCALE WHAM_DEDICATED_STREAM WHAM_BATCH_K WHAM_PREVIEW WHAM_LATEST_FRAME WHAM_LATEST_FRAME_KEEP WHAM_ADAPTIVE_KEEP WHAM_ADAPTIVE_FAST_KEEP WHAM_ADAPTIVE_SPEED_THRESHOLD

GMR_TORCH_DEVICE=${GMR_TORCH_DEVICE:-cuda}

GMR_VIEWER_READY_TIMEOUT_SEC=${GMR_VIEWER_READY_TIMEOUT_SEC:-8}
GMR_VIEWER_THREAD_JOIN_TIMEOUT_SEC=${GMR_VIEWER_THREAD_JOIN_TIMEOUT_SEC:-10}

# One-time warmup to reduce first-run CUDA/PyTorch cold-start stalls.
E2E_WARMUP=${E2E_WARMUP:-1}
E2E_WARMUP_FORCE=${E2E_WARMUP_FORCE:-0}
E2E_WARMUP_ONCE=${E2E_WARMUP_ONCE:-1}
E2E_WARMUP_CACHE_ROOT=${E2E_WARMUP_CACHE_ROOT:-${PWD}/.cache/wham-gmr}

# GMR viewer camera controls.
CAMERA_FOLLOW=${CAMERA_FOLLOW:-1}
TRACK=${TRACK:-0}
CAMERA_LOOKAT_HEIGHT_OFFSET=${CAMERA_LOOKAT_HEIGHT_OFFSET:-0.50}
CAMERA_ELEVATION=${CAMERA_ELEVATION:-0}
CAMERA_DISTANCE_SCALE=${CAMERA_DISTANCE_SCALE:-1.5}
CAMERA_AZIMUTH=${CAMERA_AZIMUTH:-90}
# GMR rendering: GMR_HEADLESS_RECORD=1 (default) renders offscreen with no GLFW
# window; 0 shows a live RobotMotionViewer window (the real-time render). mp4
# recording is a SEPARATE switch (RECORD_GMRVIDEO), so a live window does not
# imply video encoding.
GMR_HEADLESS_RECORD=${GMR_HEADLESS_RECORD:-1}
GMR_VIEWER_MAX_WIDTH=${GMR_VIEWER_MAX_WIDTH:-960}
GMR_VIEWER_MAX_HEIGHT=${GMR_VIEWER_MAX_HEIGHT:-540}
export GMR_HEADLESS_RECORD GMR_VIEWER_MAX_WIDTH GMR_VIEWER_MAX_HEIGHT
TCP=${TCP:-0}

mkdir -p "${OUTPUT_DIR}" "$(dirname "${PKL_PATH}")" "${CSV_ROOT}" "$(dirname "${CSV_PATH}")" "$(dirname "${GMR_VIDEO_PATH}")"

GMR_READY_FLAG=${GMR_READY_FLAG:-${STREAM_NPZ_DIR}/gmr_ready.flag}
STREAM_TAIL_PATH=${STREAM_TAIL_PATH:-${STREAM_NPZ_DIR}/stream_tail.pkl}
READY_TIMEOUT_SEC=60
STARTUP_DELAY_SEC=0.5

# Clean stream artifacts before launching consumer to avoid mixing stale chunks from prior runs.
rm -f "${STREAM_NPZ_DIR}"/chunk_*.npz "${STREAM_NPZ_DIR}"/stream_done.flag "${GMR_READY_FLAG}" "${STREAM_TAIL_PATH}"

echo "[E2E] Render flags: RECORD_WHAMVIDEO=${RECORD_WHAMVIDEO} RECORD_GMRVIDEO=${RECORD_GMRVIDEO} GMR_HEADLESS_RECORD=${GMR_HEADLESS_RECORD} vid_res=${GMR_VIEWER_MAX_WIDTH}x${GMR_VIEWER_MAX_HEIGHT} USE_XVFB_GMR=${USE_XVFB_GMR} WHAM_PREVIEW=${WHAM_PREVIEW:-<follow-record>} DISPLAY=${DISPLAY:-<unset>}"
echo "[E2E] Camera params: follow=${CAMERA_FOLLOW} lookat_h=${CAMERA_LOOKAT_HEIGHT_OFFSET} elev=${CAMERA_ELEVATION} dist_scale=${CAMERA_DISTANCE_SCALE} azimuth=${CAMERA_AZIMUTH:-<auto>}"
echo "[E2E] WHAM perf params: amp=${WHAM_USE_AMP} mediapipe=${WHAM_USE_MEDIAPIPE} yolopose=${WHAM_USE_YOLOPOSE} mp_complexity=${WHAM_MEDIAPIPE_COMPLEXITY} detect_interval=${WHAM_DETECT_INTERVAL} bbox_device=${WHAM_BBOX_DEVICE} infer_interval=${WHAM_INFER_INTERVAL} seq_len=${WHAM_STREAM_SEQ_LEN} input_scale=${WHAM_INPUT_SCALE} dedicated_stream=${WHAM_DEDICATED_STREAM} batch_k=${WHAM_BATCH_K} latest_frame=${WHAM_LATEST_FRAME} latest_frame_keep=${WHAM_LATEST_FRAME_KEEP} adaptive_keep=${WHAM_ADAPTIVE_KEEP} adaptive_fast_keep=${WHAM_ADAPTIVE_FAST_KEEP} adaptive_speed_thresh=${WHAM_ADAPTIVE_SPEED_THRESHOLD}"
echo "[E2E] WHAM stream mode=tail (fixed)"
echo "[E2E] GMR torch_device=${GMR_TORCH_DEVICE}  airborne_thresh=${GMR_AIRBORNE_THRESHOLD}"
echo "[E2E] Source params: VIDEO=${VIDEO} TIME=${TIME} ROBOT=${ROBOT} ROBOTS=${ROBOTS} CSV_PATH=${CSV_PATH} CSV_ROOT=${CSV_ROOT}"
echo "[E2E] GMR viewer mode: async thread + low-latency fixed profile (ready_timeout=${GMR_VIEWER_READY_TIMEOUT_SEC}s, join_timeout=${GMR_VIEWER_THREAD_JOIN_TIMEOUT_SEC}s)"
echo "[E2E] Warmup: enabled=${E2E_WARMUP} once=${E2E_WARMUP_ONCE} force=${E2E_WARMUP_FORCE}"
if [[ "${RECORD_GMRVIDEO}" == "1" && "${USE_XVFB_GMR}" == "1" ]]; then
	echo "[E2E] GMR is running with xvfb; MuJoCo window will not appear on physical screen."
elif [[ "${RECORD_GMRVIDEO}" == "1" && -z "${DISPLAY:-}" ]]; then
	if [[ "${GMR_HEADLESS_RECORD}" == "1" ]]; then
		echo "[E2E] GMR recording in headless offscreen mode (no window needed)."
	else
		echo "[E2E] Warning: DISPLAY is empty; MuJoCo window may not appear."
	fi
fi

run_env_warmup() {
	local name="$1"
	local py_bin="$2"
	if [[ ! -x "${py_bin}" ]]; then
		echo "[E2E] Warmup warning: ${name} python not executable: ${py_bin}"
		return 0
	fi

	if ! "${py_bin}" - <<'PY'
import time

try:
    import numpy as np
    import torch

    t0 = time.time()
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        x = torch.randn(512, 512, device=dev)
        y = torch.randn(512, 512, device=dev)
        _ = x @ y
        w = torch.randn(1, 3, 128, 128, device=dev)
        k = torch.randn(16, 3, 3, 3, device=dev)
        _ = torch.nn.functional.conv2d(w, k, padding=1)
        torch.cuda.synchronize()
    else:
        x = np.random.randn(256, 256).astype(np.float32)
        _ = x @ x.T

    dt = time.time() - t0
    print(f"[E2E] Warmup python ok in {dt:.3f}s")
except Exception as e:
    print(f"[E2E] Warmup warning: {e}")
PY
	then
		echo "[E2E] Warmup warning: ${name} warmup command failed (ignored)."
	fi
}

run_e2e_warmup_if_needed() {
	if [[ "${E2E_WARMUP}" != "1" ]]; then
		return 0
	fi

	local warmup_cache_root="${E2E_WARMUP_CACHE_ROOT}"
	local warmup_marker="${warmup_cache_root}/warmup_${WHAM_ENV}_${GMR_ENV}.done"

	if [[ "${E2E_WARMUP_ONCE}" == "1" && "${E2E_WARMUP_FORCE}" != "1" && -f "${warmup_marker}" ]]; then
		echo "[E2E] Warmup already done before, skip (${warmup_marker})."
		return 0
	fi

	echo "[E2E] Running warmup for ${WHAM_ENV}/${GMR_ENV} ..."
	run_env_warmup "WHAM" "${WHAM_PYTHON}"
	run_env_warmup "GMR" "${GMR_PYTHON}"

	if [[ "${E2E_WARMUP_ONCE}" == "1" ]]; then
		mkdir -p "${warmup_cache_root}"
		date +%s > "${warmup_marker}" || true
		echo "[E2E] Warmup marker written: ${warmup_marker}"
	fi
}

run_e2e_warmup_if_needed

echo "[E2E] Starting integrated WHAM + GMR producer/consumer with ${WHAM_PYTHON} ..."
INTEGRATED_CMD=(
	"${WHAM_PYTHON}" handle_wham_gmr.py
	--video "${VIDEO}"
	--time "${TIME}"
	--output_dir "${OUTPUT_DIR}"
	--robot "${ROBOT}"
	--robots "${ROBOTS}"
	--coord_fix yup_to_zup
	--save_path "${PKL_PATH}"
	--csv_path "${CSV_PATH}"
	--csv_root "${CSV_ROOT}"
	--smooth_alpha "${SMOOTH_ALPHA}"
	--viewer_warmup_frames "${VIEWER_WARMUP_FRAMES}"
	--camera_lookat_height_offset "${CAMERA_LOOKAT_HEIGHT_OFFSET}"
	--camera_elevation "${CAMERA_ELEVATION}"
	--camera_distance_scale "${CAMERA_DISTANCE_SCALE}"
	--poll_interval "${POLL_INTERVAL}"
	--idle_timeout "${IDLE_TIMEOUT}"
	--done_grace_sec "${DONE_GRACE_SEC}"
	--torch_device "${GMR_TORCH_DEVICE}"
)

if [[ -n "${CUSTOM_ROBOT_XML}" ]]; then
	INTEGRATED_CMD+=(--robot_path "${CUSTOM_ROBOT_XML}")
fi

if [[ "${CAMERA_FOLLOW}" == "1" ]]; then
	INTEGRATED_CMD+=(--camera_follow)
else
	INTEGRATED_CMD+=(--no-camera_follow)
fi

if [[ "${TRACK}" == "1" ]]; then
	INTEGRATED_CMD+=(--track)
else
	INTEGRATED_CMD+=(--no-track)
fi

if [[ "${TCP}" == "1" ]]; then
	INTEGRATED_CMD+=(--tcp)
else
	INTEGRATED_CMD+=(--no-tcp)
fi

if [[ -n "${CAMERA_AZIMUTH}" ]]; then
	INTEGRATED_CMD+=(--camera_azimuth "${CAMERA_AZIMUTH}")
fi

INTEGRATED_CMD+=(--viewer_ready_timeout_sec "${GMR_VIEWER_READY_TIMEOUT_SEC}")
INTEGRATED_CMD+=(--viewer_thread_join_timeout_sec "${GMR_VIEWER_THREAD_JOIN_TIMEOUT_SEC}")

if [[ "${HEIGHT_ADJUST}" == "1" ]]; then
	INTEGRATED_CMD+=(--height_adjust)
else
	INTEGRATED_CMD+=(--no-height_adjust)
fi

if [[ "${ROOT_ORIGIN_OFFSET}" == "1" ]]; then
	INTEGRATED_CMD+=(--root_origin_offset)
else
	INTEGRATED_CMD+=(--no-root_origin_offset)
fi

if [[ "${RECORD_WHAMVIDEO}" == "1" ]]; then
	INTEGRATED_CMD+=(--record_whamvideo)
fi

if [[ "${RECORD_GMRVIDEO}" == "1" ]]; then
	INTEGRATED_CMD+=(--record_gmrvideo --video_path "${GMR_VIDEO_PATH}")
fi

if [[ "${RECORD_GMRVIDEO}" == "1" && "${USE_XVFB_GMR}" == "1" ]]; then
	xvfb-run -a "${INTEGRATED_CMD[@]}"
else
	"${INTEGRATED_CMD[@]}"
fi

echo "[E2E] Done."
echo "  CSV: ${CSV_PATH}"
echo "  PKL: ${PKL_PATH}"
if [[ "${RECORD_GMRVIDEO}" == "1" ]]; then
	echo "  GMR video: ${GMR_VIDEO_PATH}"
fi

