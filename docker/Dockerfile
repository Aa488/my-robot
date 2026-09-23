# WHAM -> GMR integrated pipeline, single-env image matching the Windows dev machine
# (`wham_gmr`: Python 3.10 + torch 2.14.0+cu130). Self-contained: weights, body
# models, robot assets and a demo video are baked in so the image tar runs out of the box.
#
# Build:  docker build -f docker/Dockerfile -t wham-gmr:latest .
# Run:    docker run --gpus all --rm -v $PWD/out:/output wham-gmr:latest [video.mp4]

ARG CUDA_BASE_IMAGE=nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04
FROM ${CUDA_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    CONDA_DIR=/opt/conda \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
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

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm -f /tmp/miniconda.sh

# Point `python`/`pip` at the single runtime env (mirrors the dev `wham_gmr` env).
ENV PATH=${CONDA_DIR}/envs/wham_gmr/bin:${CONDA_DIR}/bin:${PATH}

RUN conda create -n wham_gmr python=3.10 -y && conda clean -afy

# torch + torchvision (cu130 wheels; only available on the pytorch cu130 index).
RUN pip install --index-url https://download.pytorch.org/whl/cu130 \
      --extra-index-url https://pypi.org/simple \
      torch==2.14.0+cu130 torchvision==0.29.0+cu130

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

# Project source (weights / body models / robot assets / demo video included via
# .dockerignore). Kept last so the heavy pip layers stay cached across source edits.
WORKDIR /opt/src/wham
COPY . /opt/src/wham
# Files checked out on Windows have no exec bit; ENTRYPOINT exec-form needs it.
RUN chmod +x /opt/src/wham/docker/entrypoint.sh

ENV PYTHONPATH=/opt/src/wham \
    WHAM_ENV=wham_gmr \
    GMR_ENV=wham_gmr

ENTRYPOINT ["/opt/src/wham/docker/entrypoint.sh"]
