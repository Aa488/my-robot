# Docker 镜像（WHAM → GMR 自包含提交）

本镜像复刻开发机 `wham_gmr` 环境（Python 3.10 + torch 2.14.0+cu130），并把权重、人体模型、机器人资产、示例视频全部打进镜像，`docker run` 即可从一段视频输出机器人关节角。

## 构建

```bash
docker build -f docker/Dockerfile -t wham-gmr:latest .
```

- 基础镜像：`nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04`（CUDA 13，匹配 cu130；可在 `--build-arg CUDA_BASE_IMAGE=...` 覆盖）。
- 构建时跳过 `pytorch3d`（仅 WHAM 视频的网格叠加，缺失时优雅降级）与 `third-party/DPVO`（离线 SLAM，流式管线不 import）。

### 国内拉取基础镜像（Docker Hub 被墙时）

Dockerfile 里唯一的 Docker Hub 拉取是基础镜像；apt / pip / conda 走的 `archive.ubuntu.com`、`pypi.org`、`download.pytorch.org`、`repo.anaconda.com` 在国内通常可达。若 `FROM` 拉取超时/失败，两种办法任选：

**A. 走镜像站拉取并打本地 tag（一次性，不改全局配置）**

```bash
docker pull docker.m.daocloud.io/nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04
docker tag docker.m.daocloud.io/nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04 \
  nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04
```

**B. 给 Docker Desktop 配 registry mirror（一劳永逸）**

Settings → Docker Engine，加入：

```json
{
  "registry-mirrors": ["https://docker.m.daocloud.io"]
}
```

Apply & Restart 后即可直接 `docker build`。

## 运行

```bash
# 用内置示例视频跑通（冒烟测试）
docker run --gpus all --rm -v $PWD/out:/output wham-gmr:latest

# 喂自己的视频
docker run --gpus all --rm -v $PWD:/input -v $PWD/out:/output \
  wham-gmr:latest /input/my_video.mp4

# 指定机器人 / 覆盖性能参数
docker run --gpus all --rm -v $PWD/out:/output \
  -e ROBOT=linglong2 -e WHAM_USE_MEDIAPIPE=1 wham-gmr:latest /input/my_video.mp4
```

**输出**（在 `/output` 下，由 `run.sh` 创建）：

| 路径 | 内容 |
|---|---|
| `/output/csv/live_motion.csv` | 机器人关节角序列 |
| `/output/pkl/my_motion.pkl` | 重定向后的人体/机器人运动 |
| `/output/video/live_stream_robot.mp4` | 渲染出的机器人动画 |

**入口参数**：位置参数 = 输入视频路径（默认内置 `examples/drone_video.mp4`）。可用 `-e` 覆盖：`ROBOT`（默认 `linglong2`）、`WHAM_USE_AMP`、`GMR_TORCH_DEVICE`、`WHAM_USE_MEDIAPIPE`、`WHAM_USE_YOLOPOSE`、`WHAM_DETECT_INTERVAL`、`WHAM_BBOX_DEVICE`、`GMR_HEADLESS_RECORD` 等（同 `run.sh`）。

## 导出 / 加载 tar

```bash
# 导出（提交用）
docker save wham-gmr:latest | gzip > wham-gmr.tar.gz

# 评委侧加载
docker load < wham-gmr.tar.gz
docker run --gpus all --rm -v $PWD/out:/output wham-gmr:latest /input/video.mp4
```

## 备注

- 镜像约 12–14GB。若需瘦身：只保留 `assets/unitree_g1` + `assets/linglong2` 两个机器人（省 ~1.1G）、只留一个 WHAM ckpt（省 ~0.5G）。
- 若评委机器驱动过旧无法运行 CUDA 13，可改 `--build-arg CUDA_BASE_IMAGE=nvidia/cuda:12.4.0-cudnn9-runtime-ubuntu22.04` 并将 torch 换成 cu124（行为需小验）。
