#!/bin/bash

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
                 --build-arg DPCPP_VER=2023.2.1-16 \
                 --build-arg MKL_VER=2023.2.0-49495 \
                 --build-arg TORCH_VERSION=2.0.1a0+cxx11.abi \
                 --build-arg IPEX_VERSION=2.0.110+xpu \
                 --build-arg TORCHVISION_VERSION=0.15.2a0+cxx11.abi \
                 --build-arg TORCH_WHL_URL=https://developer.intel.com/ipex-whl-stable-xpu \
                 --build-arg IPEX_WHL_URL=https://developer.intel.com/ipex-whl-stable-xpu \
                 --build-arg TORCHVISION_WHL_URL=https://developer.intel.com/ipex-whl-stable-xpu \
                 --build-arg CCL_VER=2021.10.0-49084 \
                 --build-arg ONECCL_BIND_PT_VERSION=2.0.100 \
                 --build-arg ONECCL_BIND_PT_WHL_URL=https://developer.intel.com/ipex-whl-stable-xpu \
                 -t ${IMAGE_NAME} \
                 -f Dockerfile .
fi
