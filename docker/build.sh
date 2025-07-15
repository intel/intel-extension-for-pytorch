#!/bin/bash

if [[ ${IMAGE_TYPE} = "xpu" ]]; then
    IMAGE_NAME=intel/intel-extension-for-pytorch:2.8.10-$IMAGE_TYPE
    docker build --build-arg http_proxy=$http_proxy \
                 --build-arg https_proxy=$https_proxy \
                 --build-arg no_proxy=" " \
                 --build-arg NO_PROXY=" " \
                 --build-arg UBUNTU_VERSION=22.04 \
                 --build-arg PYTHON=python3.10 \
                 --build-arg ICD_VER=25.05.32567.19-1099~22.04 \
                 --build-arg OCLOC_VER=25.05.32567.19-1099~22.04 \
                 --build-arg LEVEL_ZERO_VER=1.20.2.0-1098~22.04 \
                 --build-arg LEVEL_ZERO_DEV_VER=1.20.2.0-1098~22.04 \
                 --build-arg XPU_SMI_VER=1.2.39-69~22.04 \
                 --build-arg TORCH_VERSION=2.8.0 \
                 --build-arg IPEX_VERSION=2.8.10+xpu \
                 --build-arg TORCHVISION_VERSION=0.23.0+xpu \
                 --build-arg TORCHAUDIO_VERSION=2.8.0+xpu \
                 --build-arg ONECCL_BIND_PT_VERSION=2.8.0+xpu \
                 --build-arg INDEX_WHL_URL=https://download.pytorch.org/whl/test/xpu  \
                 --build-arg IPEX_WHL_URL=https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/ \
                 -t ${IMAGE_NAME} \
                 -f Dockerfile.prebuilt .
fi
