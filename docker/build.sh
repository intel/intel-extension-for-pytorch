#!/bin/bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage: bash $0 <IMAGE_TYPE>"
  echo "IMAGE_TYPE:= xpu-flex | xpu-max"
  echo "Example: bash $0 xpu-max"
  exit 1
fi

IMAGE_TYPE=$1

IMAGE_NAME=""
BUILD_ARGS=""

if [[ $IMAGE_TYPE == "xpu-flex" ]];then
    IMAGE_NAME=intel-extension-for-pytorch:xpu-flex
    BUILD_ARGS="--build-arg DEVICE=flex"
fi
if [[ ${IMAGE_TYPE} == "xpu-max" ]];then
    IMAGE_NAME=intel-extension-for-pytorch:xpu-max
    BUILD_ARGS="--build-arg CCL_VER=2021.9.0-43543 \
                --build-arg ONECCL_BIND_PT_VERSION=1.13.200 \
                --build-arg ONECCL_BIND_PT_WHL_URL=https://developer.intel.com/ipex-whl-stable-xpu \
                --build-arg DEVICE=max"
fi

if [[ ${IMAGE_NAME} != "" ]]; then
    docker build --build-arg http_proxy=$http_proxy \
                 --build-arg https_proxy=$https_proxy \
                 --build-arg no_proxy=$no_proxy \
                 --build-arg UBUNTU_VERSION=22.04 \
                 --build-arg PYTHON=python3.10 \
                 --build-arg ICD_VER=23.05.25593.18-601~22.04 \
                 --build-arg LEVEL_ZERO_GPU_VER=1.3.25593.18-601~22.04 \
                 --build-arg LEVEL_ZERO_VER=1.9.4+i589~22.04 \
                 --build-arg LEVEL_ZERO_DEV_VER=1.9.4+i589~22.04 \
                 --build-arg DPCPP_VER=2023.1.0-46305 \
                 --build-arg MKL_VER=2023.1.0-46342 \
                 --build-arg TORCH_VERSION=1.13.0a0+git6c9b55e \
                 --build-arg IPEX_VERSION=1.13.120+xpu \
                 --build-arg TORCHVISION_VERSION=0.14.1a0+5e8e2f1 \
                 --build-arg TORCH_WHL_URL=https://developer.intel.com/ipex-whl-stable-xpu \
                 --build-arg IPEX_WHL_URL=https://developer.intel.com/ipex-whl-stable-xpu \
                 --build-arg TORCHVISION_WHL_URL=https://developer.intel.com/ipex-whl-stable-xpu \
                 ${BUILD_ARGS} \
                 -t ${IMAGE_NAME} \
                 -f Dockerfile.xpu .
fi
