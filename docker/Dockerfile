# WHAM -> GMR integrated pipeline, single-env image matching the Windows dev machine
# (`wham_gmr`: Python 3.10 + torch cu13). Self-contained: weights, body models, robot
# assets and a demo video are baked in so the image tar runs out of the box.
#
# Build:  docker build -f docker/Dockerfile -t wham-gmr:latest .
# Run:    docker run --gpus all --rm -v $PWD/out:/output wham-gmr:latest [video.mp4]

# Base image: bare ubuntu 22.04 via the daocloud China mirror. docker.io is NOT
# reachable from the daemon — it dials registry-1.docker.io directly with no HTTPS
# proxy (and the v2rayN node is down), while daocloud answers in ~0.2s. torch 2.13.0
# wheels bundle their own CUDA 13.0 runtime, and a WSL2 smoke test on bare Ubuntu +
# driver 581.29 returned CUDA_AVAILABLE=True, so the 1.5GB nvidia/cuda base is
# unneeded. Fallback if GPU ever fails:
#   docker build --build-arg CUDA_BASE_IMAGE=docker.m.daocloud.io/nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04 .
ARG CUDA_BASE_IMAGE=docker.m.daocloud.io/library/ubuntu:22.04
FROM ${CUDA_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    CONDA_DIR=/opt/conda \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Force direct (proxy-less) networking for the RUN steps. Docker Desktop injects
# the Windows system proxy as 127.0.0.1:7892, which is NOT reachable from inside
# the build container (WSL2 localhost != Windows host). China mirrors below are
# reachable without any proxy, so clear the proxy vars.
ENV HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy= \
    ALL_PROXY= all_proxy= NO_PROXY=* no_proxy=*

# apt via tuna mirror (direct, fast in China).
RUN sed -i 's@//archive.ubuntu.com/@//mirrors.tuna.tsinghua.edu.cn/@g; s@//security.ubuntu.com/@//mirrors.tuna.tsinghua.edu.cn/@g' /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    git \
    bzip2 \
    unzip \
    build-essential \
    pkg-config \
    cmake \
    ninja-build \
    ffmpeg \
    xvfb \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libegl1 \
    libglfw3 \
    libxi6 \
    libosmesa6 \
    libosmesa6-dev \
    && rm -rf /var/lib/apt/lists/*

# Miniconda from the tuna mirror (repo.anaconda.com is unreachable without proxy).
RUN wget -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm -f /tmp/miniconda.sh

# Point `python`/`pip` at the single runtime env (mirrors the dev `wham_gmr` env).
ENV PATH=${CONDA_DIR}/envs/wham_gmr/bin:${CONDA_DIR}/bin:${PATH}

# conda channels -> tuna mirror (repo.anaconda.com is unreachable without proxy).
RUN printf 'channels:\n  - defaults\nshow_channel_urls: true\ndefault_channels:\n  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main\n  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r\ncustom_channels:\n  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n' > /root/.condarc \
    && conda create -n wham_gmr python=3.10 -y && conda clean -afy

# pip -> tuna mirror (pypi.org is unreachable without proxy).
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# torch + torchvision, pinned to the cu130 local build (CUDA 13.0, matching the
# Windows host's driver 581.29). torch 2.13.0+cu130 has an NVRTC/JIT regression on
# Ampere (sm_86 / RTX 3060) that stalls the GPU (SM util ~4%); 2.14.0+cu130 fixes
# it (verified on host: SM ~52%). The extra index supplies the +cu130 local wheels;
# the default PyPI build of 2.14.0 moved to cu132, which the driver rejects.
RUN pip install torch==2.14.0+cu130 torchvision==0.29.0+cu130 \
    --extra-index-url https://download.pytorch.org/whl/cu130

# Build prerequisites before the C-extension packages: old setuptools (avoids the
# mmcv/pkg_resources failure), cython, and numpy 1.23 (< 1.26 keeps numpy.distutils
# for chumpy).
RUN pip install "setuptools==59.5.0" "wheel<0.46" cython numpy==1.23.5

# mmcv "light" (pure Python) — no-build-isolation avoids pkg_resources failure
# with the pinned old setuptools.
RUN pip install --no-build-isolation mmcv==1.3.9

# C-extension packages (gcc from build-essential; numpy/cython from the step above).
RUN pip install --no-build-isolation chumpy==0.70 xtcocotools==1.14.3

# Remaining WHAM + GMR deps at the exact dev-env versions.
COPY docker/requirements-linux.txt /tmp/requirements-linux.txt
RUN pip install -r /tmp/requirements-linux.txt

# Some dep in the requirements above upgrades setuptools to 84.x, which removed
# pkg_resources (dropped in setuptools>=81); mmcv/ViTPose import it. Pin back down
# so pkg_resources stays importable at runtime.
RUN pip install "setuptools<81"

# Project source (weights / body models / robot assets / demo video included via
# .dockerignore). Kept last so the heavy pip layers stay cached across source edits.
#
# The weights are split across multiple COPY layers instead of one `COPY .`: a
# single ~4GB layer OOMs buildkitd during the export/compression phase (the build
# dies at `exporting layers` with `error reading from server: EOF`). The deps-only
# ~6GB image — whose largest layer is the ~2.5GB torch RUN layer — exports clean,
# so each COPY below is kept ≤ ~2.6GB.
WORKDIR /opt/src/wham
COPY checkpoints/hmr2a.ckpt /opt/src/wham/checkpoints/
COPY checkpoints/wham_vit_bedlam_w_3dpw.pth.tar \
     checkpoints/vitpose_base_coco_aic_mpii.pth \
     checkpoints/yolov8n.pt \
     checkpoints/yolov8n-pose.pt /opt/src/wham/checkpoints/
COPY assets /opt/src/wham/assets
COPY dataset /opt/src/wham/dataset
COPY docs /opt/src/wham/docs
COPY examples /opt/src/wham/examples
COPY third-party /opt/src/wham/third-party
COPY lib /opt/src/wham/lib
COPY configs /opt/src/wham/configs
COPY scripts /opt/src/wham/scripts
COPY docker /opt/src/wham/docker
COPY general_motion_retargeting /opt/src/wham/general_motion_retargeting
COPY *.py *.txt *.sh *.ps1 LICENSE README.md TECHNICAL_REPORT.md /opt/src/wham/
# Files checked out on Windows have no exec bit; ENTRYPOINT exec-form needs it.
RUN chmod +x /opt/src/wham/docker/entrypoint.sh

ENV PYTHONPATH=/opt/src/wham \
    WHAM_ENV=wham_gmr \
    GMR_ENV=wham_gmr

ENTRYPOINT ["/opt/src/wham/docker/entrypoint.sh"]
