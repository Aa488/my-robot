# Docker 镜像使用文档（WHAM → GMR）

本镜像复刻开发机 `wham_gmr` 环境（Python 3.10 + torch 2.14.0+cu130），把 WHAM/ViTPose/检测器权重、人体模型、机器人资产、示例视频全部打进镜像，`docker run` 即可从一段视频输出机器人关节角序列。自包含、无需联网。

- 镜像仓库：`11111p5p/wham-gmr`（Docker Hub）
- 当前标签：`latest`（= 提交 `3e850e3`），历史提交标签 `bba092e`、`783c429` 等
- 镜像大小：约 26GB（含全部权重与 CUDA 运行时）

---

## 1. 快速开始

### 拉取

```bash
docker pull 11111p5p/wham-gmr:latest
```

### 冒烟测试（内置示例视频）

```bash
docker run --gpus all --rm -v $PWD/out:/output 11111p5p/wham-gmr:latest
```

跑通后 `/output/csv/live_motion.csv` 应有一行行 37 列的关节角。

### 喂自己的视频

```bash
docker run --gpus all --rm \
  -v $PWD:/input -v $PWD/out:/output \
  11111p5p/wham-gmr:latest /input/my_video.mp4
```

### 覆盖参数

```bash
docker run --gpus all --rm -v $PWD/out:/output \
  -e ROBOT=linglong2 -e WHAM_USE_YOLOPOSE=0 -e WHAM_USE_MEDIAPIPE=1 \
  11111p5p/wham-gmr:latest /input/my_video.mp4
```

---

## 2. 输入与输出

### 输入

优先级：**位置参数** > `-e VIDEO=...` > 内置示例 `examples/drone_video.mp4`。

### 输出（在 `/output` 下，由 run.sh 创建）

| 路径 | 内容 |
|---|---|
| `/output/csv/live_motion.csv` | 机器人关节角序列（无表头，逗号分隔） |
| `/output/pkl/my_motion.pkl` | 重定向后的运动 pickle |
| `/output/video/live_stream_robot.mp4` | 渲染出的机器人动画（仅 `RECORD_GMRVIDEO=1` 时产生） |

`linglong2` 的 CSV 每行 37 列：`[tx,ty,tz, qx,qy,qz,qw, 30 个关节角]`；手臂关节在第 23–36 列。

---

## 3. 环境变量参考

所有变量都可通过 `-e NAME=VALUE` 覆盖。默认值即「什么都不传」时生效的值。

### 输入 / 输出

| 变量 | 默认 | 说明 |
|---|---|---|
| `VIDEO` | 见入口 | 输入视频路径；`0` = 摄像头实时模式 |
| `TIME` | `0` | `0` 处理整段视频；`>0` 只处理前 N 秒 |
| `ROBOT` | `linglong2` | 目标机器人 |
| `ROBOTS` | = `ROBOT` | 多机器人列表 |
| `GMR_FORCE_SINGLE_ROBOT` | `0` | `1` 屏蔽多机器人分支，只处理 `ROBOT` 第一个（单机器人、单 CSV、场景无偏移） |
| `ROBOT_PATH` | 空 | 机器人 key 或自定义 XML 路径 |
| `OUTPUT_ROOT` | 空 | 设了则所有输出路径基于它，否则用 `output/`、`pkl_outputs/`、`videos/` |

### 录制 / 预览

| 变量 | 默认 | 说明 |
|---|---|---|
| `RECORD_GMRVIDEO` | `0` | 是否把渲染帧编码写入 mp4 文件（`1` 开启录制，与窗口开关独立） |
| `RECORD_WHAMVIDEO` | `1` | 是否录 WHAM 网格叠加视频（docker 入口覆盖为 `0`） |
| `RECORD_VIDEO` | `1` | 旧开关，向后兼容：同时控制上面两个 |
| `WHAM_PREVIEW` | 空 | 空=跟随 RECORD_WHAMVIDEO；`1` 始终显示；`0` 从不 |
| `GMR_HEADLESS_RECORD` | `1` | `1` 无窗口离屏渲染（默认）；`0` 弹出 MuJoCo 实时窗口（Windows 原生弹窗，**不**自动写 mp4） |
| `GMR_VIEWER_MAX_WIDTH/HEIGHT` | `960`/`540` | 离屏渲染分辨率上限 |
| `GMR_RENDER_SCALE` | `1.0` | MuJoCo 渲染分辨率缩放（0.1–1.0），如 `0.5` = 半分辨率，降 GPU 开销 |

### 相机

| 变量 | 默认 | 说明 |
|---|---|---|
| `CAMERA_FOLLOW` | `1` | 相机是否跟随机器人 |
| `TRACK` | `0` | 轨迹跟踪 |
| `CAMERA_LOOKAT_HEIGHT_OFFSET` | `0.50` | 相机注视点高度 |
| `CAMERA_ELEVATION` | `0` | 仰角 |
| `CAMERA_DISTANCE_SCALE` | `1.5` | 距离缩放 |
| `CAMERA_AZIMUTH` | `90` | 方位角 |

### 检测器（人检测 / 关键点）

| 变量 | 默认 | 说明 |
|---|---|---|
| `WHAM_USE_YOLOPOSE` | `1` | **默认单阶段 YOLOv8-pose**（~27ms/帧，bbox+17 关键点一次前向）。`0` 回退到 YOLOv8n+ViTPose |
| `WHAM_YOLOPOSE_IMGSZ` | `640` | YOLOv8-pose 输入尺寸；`320` 把 YOLO 前向砍近半，仅适合人在近处（远处/小人召回会降） |
| `WHAM_USE_MEDIAPIPE` | `0` | `1` 用 CPU BlazePose 子进程（释放 GPU） |
| `WHAM_BBOX_DEVICE` | `cpu` | 仅默认 YOLO+ViTPose 路径用；YOLOPose=1 时忽略 |
| `WHAM_DETECT_INTERVAL` | `1` | 每 N 帧检测一次（YOLOPose=1 时强制 1） |
| `WHAM_MEDIAPIPE_COMPLEXITY` | `0` | MediaPipe 复杂度 |

### WHAM 推理 / 性能

| 变量 | 默认 | 说明 |
|---|---|---|
| `WHAM_USE_AMP` | `0` | fp16。**保持 0**——开 1 会引出手臂跳变 |
| `WHAM_BATCH_K` | `2` | 批量窗口推理，K=2 约 2× 吞吐；`1` = 旧行为 |
| `WHAM_STREAM_SEQ_LEN` | `16` | 滑窗长度 |
| `WHAM_INFER_INTERVAL` | `1` | 推理间隔 |
| `WHAM_INPUT_SCALE` | `1.0` | 输入缩放 |
| `WHAM_DEDICATED_STREAM` | `1` | 专用 CUDA stream |
| `WHAM_LOG` | `1` | `0` 关闭所有日志输出（stderr 写入是实时环路里的额外延迟） |

### GMR / IK 重定向

| 变量 | 默认 | 说明 |
|---|---|---|
| `GMR_TORCH_DEVICE` | `cuda` | GMR IK 运行设备 |
| `GMR_MAX_ITER` | 代码默认 `10` | IK 单级最大迭代次数（linglong2=10，openloong=20） |
| `SMOOTH_ALPHA` | `0.35` | 关节平滑系数 |
| `HEIGHT_ADJUST` | `1` | 高度自适应（含跳跃检测） |
| `ROOT_ORIGIN_OFFSET` | `0` | 根原点偏移 |
| `GMR_AIRBORNE_THRESHOLD` | `0.03` | 腾空判定阈值（米） |

### 实时模式丢帧控制（仅摄像头 `VIDEO=0` 生效）

| 变量 | 默认 | 说明 |
|---|---|---|
| `WHAM_LATEST_FRAME` | `1` | 有界积压：超时丢弃旧帧保持低延迟；`0` 关闭 |
| `WHAM_LATEST_FRAME_KEEP` | `3` | 保留帧数上限 |
| `WHAM_ADAPTIVE_KEEP` | `1` | 快速运动（跳跃）时自动放宽丢帧窗口 |
| `WHAM_ADAPTIVE_FAST_KEEP` | `8` | 快速运动时的保留帧数 |
| `WHAM_ADAPTIVE_SPEED_THRESHOLD` | `1.5` | 触发放宽的根速度阈值（m/s） |

---

## 4. 实时模式（摄像头）

Docker 镜像主要面向**离线视频 → 机器人关节角**的提交流程。实时摄像头（`VIDEO=0`）在 **Windows Docker Desktop 上做不了设备透传**，建议在 Linux 宿主（有 `/dev/video0`）跑，或直接在 Windows 用 `run.ps1` 跑。

### 推荐实时参数

| 变量 | 值 | 说明 |
|---|---|---|
| `VIDEO` | `0` | 摄像头实时模式 |
| `ROBOT` | `linglong2` | 目标机器人 |
| `WHAM_USE_YOLOPOSE` | `1` | 单阶段 YOLOv8-pose 检测器（~27ms/帧，实时提速关键） |
| `WHAM_BATCH_K` | `2` | 批量窗口推理，~2× WHAM 吞吐 |
| `WHAM_LATEST_FRAME` | `1` | **丢帧**：超时丢弃旧帧保持低延迟（默认） |
| `WHAM_BBOX_DEVICE` | `cpu` | `WHAM_USE_YOLOPOSE=1` 时被忽略，可不设 |
| `RECORD_GMRVIDEO` | `0` | 默认不写 mp4；需要视频时设 `1` |
| `CAMERA_AZIMUTH` | `90` | 正面视角（`0` 为另一侧，二选一） |
| `CAMERA_ELEVATION` | `0` | 平视（不再俯视） |
| `CAMERA_DISTANCE_SCALE` | `1.5` | 取景距离，微调让人大小合适 |

```bash
docker run --gpus all --rm \
  --device /dev/video0:/dev/video0 \
  -e VIDEO=0 -e ROBOT=linglong2 \
  -e WHAM_USE_YOLOPOSE=1 -e WHAM_BATCH_K=2 -e WHAM_LATEST_FRAME=1 \
  -e CAMERA_AZIMUTH=90 -e CAMERA_ELEVATION=0 -e CAMERA_DISTANCE_SCALE=1.5 \
  -v $PWD/out:/output \
  11111p5p/wham-gmr:latest
```

> `WHAM_LATEST_FRAME=1` 开启丢帧保护（默认）：管线跑得比输入慢时丢弃旧帧，端到端延迟有界（~200ms 预算内）。要**一帧都不丢**再改回 `0`，代价是延迟随积压增长。
> `WHAM_BBOX_DEVICE=cpu` 只在回退到 YOLOv8n+ViTPose 检测器时才有意义。
> 机器人动画 mp4 默认**不写入**（`RECORD_GMRVIDEO=0`）。MuJoCo 渲染由 `GMR_HEADLESS_RECORD` 控制：默认 `1` 无头离屏、不弹窗；`0` 弹出实时窗口。**窗口与 mp4 完全独立**——弹窗不自动录制，需要视频时单独加 `-e RECORD_GMRVIDEO=1`。

---

## 5. 入口脚本（entrypoint.sh）覆盖的默认值

容器启动走 `docker/entrypoint.sh`，它会在 run.sh 之前**强制覆盖**几个默认值（`-e` 传入的值优先）：

| 变量 | entrypoint 覆盖值 | 原因 |
|---|---|---|
| `WHAM_USE_AMP` | `0` | 关 fp16（防手臂跳变） |
| `GMR_TORCH_DEVICE` | `cuda` | GMR IK 放 GPU（与 run.ps1 一致） |
| `WHAM_BBOX_DEVICE` | `cpu` | 默认检测器的人框阶段 |
| `RECORD_GMRVIDEO` | `0` | 默认不写机器人动画 mp4 |
| `RECORD_WHAMVIDEO` | `0` | 跳过 WHAM 网格视频（省 pytorch3d） |
| `GMR_HEADLESS_RECORD` / `USE_XVFB_GMR` / `MUJOCO_GL` | `1` / `1` / `egl` | 无头离屏渲染 |

---

## 6. run.sh 与 run.ps1 的默认值一致

`run.sh` 的所有默认值已与 `run.ps1` 对齐（`GMR_TORCH_DEVICE=cuda`、`GMR_MAX_ITER` 走代码默认、`CAMERA_FOLLOW=1`、`CAMERA_LOOKAT_HEIGHT_OFFSET=0.50`、`VIDEO` 默认 `examples/output.mp4`）。两者唯一保留的差异是平台相关逻辑：`run.sh` 有 `xvfb-run` 分支（Linux 无头渲染），`run.ps1` 直接运行（Windows 无 xvfb）。

---

## 7. 构建

### 完整镜像

```bash
docker build -f docker/Dockerfile -t wham-gmr:latest .
```

- 基础镜像：`docker.m.daocloud.io/library/ubuntu:22.04`（**裸 Ubuntu**；torch 2.14.0+cu130 自带 CUDA 13 运行时，无需 nvidia/cuda 基座）。GPU 起不来时可用 `--build-arg CUDA_BASE_IMAGE=docker.m.daocloud.io/nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04` 回退。
- 构建时跳过 `pytorch3d`（仅 WHAM 视频网格叠加，缺失时降级）与 `third-party/DPVO`（离线 SLAM，流式不 import）。

### 补丁层（只改源码时，秒级构建）

```bash
docker build -f docker/Dockerfile.patch -t wham-gmr:latest .
```

`docker/Dockerfile.patch` 在现有镜像上只重 COPY 变更的源码文件，不加权重/依赖层。

### 国内拉取基础镜像（Docker Hub 被墙时）

Dockerfile 已把 apt/pip/conda 源切到清华镜像，基础镜像走 daocloud。若仍需镜像站，任选：

```bash
# A. 一次性拉取并打本地 tag
docker pull docker.m.daocloud.io/library/ubuntu:22.04
docker tag docker.m.daocloud.io/library/ubuntu:22.04 ubuntu:22.04
```

或给 Docker Desktop 配 registry mirror（Settings → Docker Engine）：

```json
{ "registry-mirrors": ["https://docker.m.daocloud.io"] }
```

---

## 8. 导出 / 加载 tar（提交用）

```bash
# 导出
docker save 11111p5p/wham-gmr:latest | gzip > wham-gmr.tar.gz

# 评委侧加载
docker load < wham-gmr.tar.gz
docker run --gpus all --rm -v $PWD/out:/output 11111p5p/wham-gmr:latest /input/video.mp4
```

---

## 备注

- 镜像约 26GB。瘦身可只保留 `assets/unitree_g1` + `assets/linglong2` 两个机器人、只留一个 WHAM ckpt。
- 若评委机器驱动过旧无法跑 CUDA 13，需换 cu124 重编 torch（行为需小验）。
- `examples/SecretKey.csv`（腾讯云密钥）**永不进入镜像**，已由 `.dockerignore` 排除。
