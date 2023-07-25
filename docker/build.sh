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
    IMAGE_NAME=intel/intel-extension-for-pytorch:xpu-flex
    BUILD_ARGS="--build-arg DEVICE=flex"
fi
if [[ ${IMAGE_TYPE} == "xpu-max" ]];then
    IMAGE_NAME=intel/intel-extension-for-pytorch:xpu-max
    BUILD_ARGS="--build-arg CCL_VER=2021.10.0-49084 \
                --build-arg ONECCL_BIND_PT_VERSION=2.0.100 \
                --build-arg ONECCL_BIND_PT_WHL_URL=https://developer.intel.com/ipex-whl-stable-xpu \
                --build-arg DEVICE=max"
fi

if [[ ${IMAGE_NAME} != "" ]]; then
    docker build --build-arg http_proxy=$http_proxy \
                 --build-arg https_proxy=$https_proxy \
                 --build-arg no_proxy=$no_proxy \
                 --build-arg UBUNTU_VERSION=22.04 \
                 --build-arg PYTHON=python3.10 \
                 --build-arg ICD_VER=23.17.26241.33-647~22.04 \
                 --build-arg LEVEL_ZERO_GPU_VER=1.3.26241.33-647~22.04 \
                 --build-arg LEVEL_ZERO_VER=1.11.0-647~22.04 \
                 --build-arg LEVEL_ZERO_DEV_VER=1.11.0-647~22.04 \
                 --build-arg DPCPP_VER=2023.2.0-49495 \
                 --build-arg MKL_VER=2023.2.0-49495 \
                 --build-arg TORCH_VERSION=2.0.1a0+cxx11.abi \
                 --build-arg IPEX_VERSION=2.0.110+xpu \
                 --build-arg TORCHVISION_VERSION=0.15.2a0+cxx11.abi \
                 --build-arg TORCH_WHL_URL=https://developer.intel.com/ipex-whl-stable-xpu \
                 --build-arg IPEX_WHL_URL=https://developer.intel.com/ipex-whl-stable-xpu \
                 --build-arg TORCHVISION_WHL_URL=https://developer.intel.com/ipex-whl-stable-xpu \
                 ${BUILD_ARGS} \
                 -t ${IMAGE_NAME} \
                 -f Dockerfile.xpu .
fi
