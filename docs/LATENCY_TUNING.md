# 实时模式加速开关参考（WHAM → GMR → linglong2）

目标：端到端延迟 ~200ms 内跑实时（`VIDEO=0` 摄像头）。这份文档覆盖实时链路里**跟延迟/吞吐直接相关**的开关。默认值以 `run.ps1` / `run.sh` 为准（`run.ps1` 里的 `_env NAME "default"`、`run.sh` 里的 `NAME=${NAME:-default}`）。

> 注意：**直接 `python handle_wham_gmr.py` 而不用 run.ps1/run.sh 时，代码 fallback 默认可能跟脚本默认不同**（尤其 `WHAM_BATCH_K`、`WHAM_USE_YOLOPOSE`）。建议走 run.ps1，或在 PowerShell 里显式 `$env:XXX` 设全。

---

## 0. 你当前的实时配置（逐行解释）

```powershell
$env:RECORD_GMRVIDEO      = "0"   # 不写机器人动画 mp4（省编码 IO）
$env:WHAM_USE_YOLOPOSE    = "1"   # 单阶段 YOLOv8-pose，~27ms/帧（旧 YOLO+ViTPose ~72ms 且 bbox 在 CPU）
$env:WHAM_YOLOPOSE_IMGSZ  = "320" # YOLO 输入尺寸，320 把 YOLO 前向砍近半（人在近处时才用，默认 640）
$env:WHAM_LOG             = "0"   # 关 loguru 日志（stderr 写入是环路延迟）
$env:GMR_RENDER_SCALE     = "0.5" # MuJoCo 渲染半分辨率（砍 CPU 光栅化像素量）
$env:GMR_HEADLESS_RECORD  = "0"   # 弹 MuJoCo 实时窗口（=1 是无头离屏，更快但不弹窗）
$env:VIDEO                = '0'   # 摄像头实时模式
$env:WHAM_LATEST_FRAME_KEEP = '1' # 积压队列只留最新 1 帧（默认 3）
$env:WHAM_LATEST_FRAME      = '1' # 丢帧总开关：1=开（默认）
$env:WHAM_BBOX_DEVICE     = 'cpu' # bbox 阶段设备（WHAM_USE_YOLOPOSE=1 时被忽略）
$env:WHAM_ADAPTIVE_KEEP   = '0'   # 关「快速运动自动放宽丢帧窗口」（默认开）
```

这套配置已经把**录制、日志、渲染分辨率、积压**这些「零代价」的都压到位了。下面按收益排序给还能继续抠的开关。

---

## 1. 还能加速的开关（按收益从大到小）

### ⭐ WHAM_USE_AMP=1 —— 最大单点收益，但有风险
```powershell
$env:WHAM_USE_AMP = "1"
```
- 作用：WHAM 前向转 **fp16**，单窗口前向 ~50ms → **~25ms**（近砍半）。
- 代价：历史实测**会引出手臂跳变**（所以之前提交把它默认关了：`run.sh/run.ps1/entrypoint` 都强制 `0`）。
- 建议：**单独开它跑一段跳跃/挥手动作为主**的视频，肉眼判断手臂是否抖。不抖就开，这是最划算的一项。代码侧限制：仅 CUDA 设备生效（`use_amp = env_flag(...) and is_cuda_device`）。

### ⭐ WHAM_BATCH_K=2 —— 消除丢帧根因
```powershell
$env:WHAM_BATCH_K = "2"   # run.ps1 已默认 2；直接 python 跑时 fallback 是 1
```
- 作用：把 K 个独立滑窗塞进同一次 GPU forward，几乎零成本吃满 SM 占用。实测（本机 diag）：
  - `1`（单窗口）：~50ms → **19.7fps**，低于 30fps 输入 → **正是丢帧根因**
  - `2`：**~37fps**（零丢帧），+~66ms 延迟，端到端 ~150ms（仍在 200ms 预算内）✅ 甜点
  - `4`：71.9fps；`8`：144.9fps（延迟超预算，不建议）
- 建议：实时模式固定 `2`。**若你直接 `python handle_wham_gmr.py`（不走 run.ps1），记得手动设 `$env:WHAM_BATCH_K="2"`**，否则回退到 1、又会丢帧。

### WHAM_INFER_INTERVAL=2 —— WHAM 少参加推理（复用上一帧姿态）
```powershell
$env:WHAM_INFER_INTERVAL = "2"   # 默认 1（每帧都推理）
```
- 作用：WHAM 每 N 帧才真跑一次前向，中间帧直接复用上一帧的 3D 姿态（丢到 wham_Q）。`2` = WHAM 计算量减半。**现在 batch 模式（WHAM_BATCH_K=2）下也生效**。
- 代价：中间帧姿态是「冻结」的上一帧，动作越快越显得卡顿。`2` 通常可接受，`3` 以上快速动作会明显抖。
- 定位：这是你要的「WHAM 少参加推理」开关，比「完全跳过 WHAM」现实得多（跳过 WHAM 就没有 3D 姿态，GMR/IK 无输入）。

### WHAM_YOLOPOSE_IMGSZ=320 —— 砍 YOLO 前向（人在近处时）
```powershell
$env:WHAM_YOLOPOSE_IMGSZ = "320"   # 默认 640
```
- 作用：YOLOv8-pose 的输入尺寸，`320` 把 YOLO 前向 ~27ms 砍近半（~13ms）。
- 代价：远处/小的人检测召回下降。**你确认人在近处，直接上 320**。

### WHAM_INPUT_SCALE < 1.0 —— 缩小 WHAM 输入
```powershell
$env:WHAM_INPUT_SCALE = "0.75"   # 默认 1.0
```
- 作用：WHAM 输入图按比例缩小，前向更快；精度略降。0.7~0.8 通常能接受。

### WHAM_STREAM_SEQ_LEN 调小 —— 缩短滑窗
```powershell
$env:WHAM_STREAM_SEQ_LEN = "8"   # 默认 16
```
- 作用：时序滑窗从 16 缩到 8，前向更快；但时序上下文变短，快速动作可能更抖。适合静态/慢速动作。

### GMR_MAX_ITER 调小 —— IK 迭代砍半
```powershell
$env:GMR_MAX_ITER = "5"   # linglong2 代码默认 10
```
- 作用：GMR/IK 每级最大迭代数减半，IK 更快；收敛精度略降。`5` 通常够，`3` 是极限。

### GMR_RENDER_SCALE 更低 / GMR_VIEWER_* 调小 —— 渲染更省
```powershell
$env:GMR_RENDER_SCALE = "0.3"   # 你已设 0.5
$env:GMR_VIEWER_MAX_WIDTH  = "640"
$env:GMR_VIEWER_MAX_HEIGHT = "360"
```
- 作用：MuJoCo 渲染是 **CPU 软件光栅化**（`mj.Renderer`，没用 GPU 光栅化），像素越少越省。`GMR_RENDER_SCALE` 是整体缩放，`GMR_VIEWER_*` 是离屏分辨率上限。

### GMR_HEADLESS_RECORD=1 —— 关弹窗换离屏
```powershell
$env:GMR_HEADLESS_RECORD = "1"   # 你已设 0（弹窗）
```
- 作用：弹窗本身有显示/同步开销；`1` 是无头离屏渲染，更快。你设 `0` 是为了看实时窗口——**这是「看效果」和「抢延迟」的取舍**，比速度时可临时改回 `1`。

---

## 2. 完整开关表（默认值 = run.ps1/run.sh）

### 检测器
| 开关 | 默认 | 说明 |
|---|---|---|
| `WHAM_USE_YOLOPOSE` | `1` | 单阶段 YOLOv8-pose（~27ms/帧）。`0` 回退 YOLO+ViTPose（~72ms，bbox 在 CPU） |
| `WHAM_YOLOPOSE_IMGSZ` | `640` | YOLOv8-pose 输入尺寸；`320` 把 YOLO 前向砍近半，仅适合人在近处 |
| `WHAM_USE_MEDIAPIPE` | `0` | `1` 用 CPU BlazePose 子进程（**依赖 `.venv_mp`，Docker 里没有，仅 Windows 开发机可用**） |
| `WHAM_DETECT_INTERVAL` | `1` | 每 N 帧检测一次（YOLOPose=1 时强制 1） |
| `WHAM_BBOX_DEVICE` | `cpu` | bbox 阶段设备；YOLOPose=1 时忽略 |
| `WHAM_DET_DEVICE` | = `device` | 检测器设备（代码侧，一般不用动） |

### WHAM 推理
| 开关 | 默认 | 说明 |
|---|---|---|
| `WHAM_USE_AMP` | `0` | fp16。**0 为默认**；`1` 近砍半前向，但引出手臂跳变，见上 |
| `WHAM_BATCH_K` | `2` | 批量窗口推理，K=2 ~2× 吞吐、消除丢帧；`1` = 单窗口旧行为 |
| `WHAM_INPUT_SCALE` | `1.0` | 输入缩放，<1 更快 |
| `WHAM_STREAM_SEQ_LEN` | `16` | 滑窗长度，调小更快、时序更短 |
| `WHAM_INFER_INTERVAL` | `1` | WHAM 推理间隔：`>1` 时中间帧复用上一帧 3D 姿态（batch 模式也生效），=「WHAM 少参加推理」 |
| `WHAM_DEDICATED_STREAM` | `1` | 专用 CUDA stream（CUDA 设备生效） |

### 丢帧 / 积压（仅 `VIDEO=0` 生效）
| 开关 | 默认 | 说明 |
|---|---|---|
| `WHAM_LATEST_FRAME` | `1` | 有界积压总开关：`1` 超时丢旧帧保延迟 |
| `WHAM_LATEST_FRAME_KEEP` | `3` | 积压保留帧数上限（`1` = 只留最新） |
| `WHAM_ADAPTIVE_KEEP` | `1` | 快速运动（跳跃）时自动放宽丢帧窗口 |
| `WHAM_ADAPTIVE_FAST_KEEP` | `8` | 快速运动时的保留帧数 |
| `WHAM_ADAPTIVE_SPEED_THRESHOLD` | `1.5` | 触发放宽的根速度阈值（m/s） |

### 渲染 / 录制
| 开关 | 默认 | 说明 |
|---|---|---|
| `GMR_HEADLESS_RECORD` | `1` | `1` 无头离屏（更快）；`0` 弹 MuJoCo 窗口 |
| `GMR_RENDER_SCALE` | `1.0` | MuJoCo 渲染分辨率缩放（0.1–1.0） |
| `GMR_VIEWER_MAX_WIDTH/HEIGHT` | `960`/`540` | 离屏渲染分辨率上限 |
| `GMR_PREVIEW_FPS` | `3.0` | GMR 预览渲染帧率 |
| `RECORD_GMRVIDEO` | `0` | 是否写机器人动画 mp4 |
| `RECORD_WHAMVIDEO` | `1`（docker 覆盖 0） | 是否录 WHAM 网格叠加视频（省 pytorch3d） |

### GMR / IK
| 开关 | 默认 | 说明 |
|---|---|---|
| `GMR_TORCH_DEVICE` | `cuda` | GMR IK 设备 |
| `GMR_MAX_ITER` | 代码 `10` | IK 单级最大迭代（linglong2=10） |
| `GMR_SMPLX_GPU` | `1` | SMPL-X 上 GPU（`1` 且 CUDA 可用时） |

### 日志 / 杂项
| 开关 | 默认 | 说明 |
|---|---|---|
| `WHAM_LOG` | `1` | `0` 关 loguru 日志（stderr 写入是环路延迟） |

---

## 3. 提速优先级小结

1. **先确认 `WHAM_BATCH_K=2` 生效**（走 run.ps1 默认就是 2；直接跑 Python 必须手动设）—— 这是丢帧根因，比什么都优先。
2. 试 `WHAM_USE_AMP=1`，实测手臂是否跳变 —— 不跳就白捡 ~25ms。
3. 再按需调 `WHAM_INPUT_SCALE` / `WHAM_STREAM_SEQ_LEN` / `GMR_MAX_ITER`，逐个开、逐个测，别一把全上（前两者牺牲精度、后者牺牲 IK 收敛）。
4. 渲染侧 `GMR_RENDER_SCALE` 已到 0.5，够用；追求极致再压 `GMR_VIEWER_*` 或临时 `GMR_HEADLESS_RECORD=1`。

> 所有「砍精度/砍上下文」的开关（INPUT_SCALE、STREAM_SEQ_LEN、GMR_MAX_ITER）都可能让手臂动作变抖或踩空，建议逐项 A/B 对比输出的 `live_motion.csv` / 机器人视频再定。
