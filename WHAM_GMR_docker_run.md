# WHAM→GMR Docker 启动参考

镜像：`11111p5p/wham-gmr:latest`（Docker Hub）。

> ⚠️ 镜像必须包含 `scripts/pipeline_watchdog.py` 与 `scripts/latest_frame_queue.py`，
> 否则运行时报 `ModuleNotFoundError: No module named 'scripts.pipeline_watchdog'`。
> 增量重建：`docker build -f docker/Dockerfile.patch -t 11111p5p/wham-gmr:latest .`，然后 `docker push`。

## 模式一：视频文件 → 关节角（Windows / Linux 均可）

Windows Docker Desktop **只能**跑这个模式（透传不了摄像头）。

```powershell
# Windows PowerShell
docker run --rm --gpus all `
  -v "${PWD}\out:/output" `
  11111p5p/wham-gmr:latest
```

```bash
# Linux
docker run --rm --gpus all -v "$PWD/out:/output" 11111p5p/wham-gmr:latest
```

跑自己的视频：把视频挂载进容器，再以参数传容器内路径 `/input/xxx.mp4`。

输出：`out/csv/live_motion.csv`（关节角）、`out/pkl/my_motion.pkl`。

## 模式二：摄像头实时（仅 Linux 宿主）

Windows Docker Desktop 不支持 USB 摄像头透传；请用 Linux 宿主，或 Windows 本机 `run.ps1`。

```bash
docker run --gpus all --rm \
  --device /dev/video0:/dev/video0 \
  -e VIDEO=0 \
  -e WHAM_USE_YOLOPOSE=1 \
  -e WHAM_GPU_SCHEDULER=1 \
  -e WHAM_INFER_INTERVAL=2 \
  -e WHAM_LATEST_FRAME=1 \
  -e WHAM_LATEST_FRAME_KEEP=1 \
  -e WHAM_ADAPTIVE_KEEP=0 \
  -e CAMERA_ELEVATION=0 \
  -e CAMERA_AZIMUTH=90 \
  -e CAMERA_DISTANCE_SCALE=1.5 \
  -v "$PWD/out:/output" \
  11111p5p/wham-gmr:latest
```

## 优化参数对照

| 参数 | 值 | 作用 | 生效范围 |
|---|---|---|---|
| `WHAM_USE_YOLOPOSE` | 1 | 单阶段 YOLOv8-pose（~27ms/帧） | 视频/摄像头 |
| `WHAM_GPU_SCHEDULER` | 1 | WHAM 独占 GPU | 视频/摄像头 |
| `WHAM_INFER_INTERVAL` | 2 | 隔帧推理，算量减半（细节变粗） | 视频/摄像头 |
| `WHAM_BATCH_K` | 2 | 批量窗口推理 ~2× 吞吐 | 视频/摄像头 |
| `GMR_TORCH_DEVICE` | cuda | GMR IK 放 GPU | 视频/摄像头 |
| `GMR_RENDER_SCALE` | 0.1 | 砍 MuJoCo 渲染分辨率 | 视频/摄像头 |
| `CAMERA_ELEVATION` | 0 | 平视 | 取景 |
| `CAMERA_AZIMUTH` | 90 | 正面 | 取景 |
| `CAMERA_DISTANCE_SCALE` | 1.5 | 取景距离 | 取景 |
| `WHAM_LATEST_FRAME` | 1 | 丢旧帧保低延迟 | 仅摄像头 |
| `WHAM_LATEST_FRAME_KEEP` | 1 | 队列只留 1 帧 | 仅摄像头 |
| `WHAM_ADAPTIVE_KEEP` | 0 | 关快速运动自动放宽 | 仅摄像头 |
| `GMR_HEADLESS_RECORD` | 1 | 无头离屏（默认）；0=弹窗需 X | 渲染 |
| `RECORD_GMRVIDEO` | 0 | 不写机器人 mp4；1=写 | 渲染 |

## 备注

- Windows 摄像头实时：用本机 `run.ps1`（`$env:VIDEO="0"`），Docker 透传不了摄像头。
- Linux 摄像头：容器内固定开 `VideoCapture(0)`，所以 `--device /dev/videoN:/dev/video0` 必须映射成 `/dev/video0`。
