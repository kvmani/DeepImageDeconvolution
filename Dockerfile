# syntax=docker/dockerfile:1
# Placeholder Dockerfile for DeepImageDeconvolution

FROM python:3.10-slim

# System dependencies for common Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch (CPU wheels) and project dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . ./

# Build (CPU):
#   docker build -t kikuchi-deconv:cpu .
# Run (CPU):
#   docker run --rm -it -v "$PWD:/app" kikuchi-deconv:cpu bash
#
# GPU note:
#   Use an NVIDIA CUDA base image (e.g., nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04)
#   and install matching PyTorch CUDA wheels (see https://pytorch.org/get-started/locally/).
#   Example run:
#     docker run --rm -it --gpus all -v "$PWD:/app" kikuchi-deconv:gpu bash

CMD ["bash"]
