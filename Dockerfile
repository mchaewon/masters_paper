FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV PYTHONPATH=/VoxFormer

WORKDIR /VoxFormer

RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    git \
    wget \
    curl \
    ca-certificates \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libgomp1 \
    libffi-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    python -m pip install --upgrade pip setuptools wheel

# PyTorch 1.9.1 + CUDA 11.1
RUN pip install --no-cache-dir \
    torch==1.9.1+cu111 \
    torchvision==0.10.1+cu111 \
    torchaudio==0.9.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

# MMCV-full for CUDA 11.1 + torch 1.9.x
RUN pip install --no-cache-dir \
    mmcv-full==1.4.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# mmdet / mmseg / timm
RUN pip install --no-cache-dir \
    mmdet==2.14.0 \
    mmsegmentation==0.14.1 \
    timm

# Copy VoxFormer source
COPY . /VoxFormer


CMD ["/bin/bash"]