#!/bin/bash

if [[ ${IMAGE_TYPE} = "xpu" ]]; then
    IMAGE_NAME=intel/intel-extension-for-pytorch:2.5.10-$IMAGE_TYPE
    docker build --build-arg http_proxy=$http_proxy \
                 --build-arg https_proxy=$https_proxy \
                 --build-arg no_proxy=" " \
                 --build-arg NO_PROXY=" " \
                 --build-arg UBUNTU_VERSION=22.04 \
                 --build-arg PYTHON=python3.10 \
                 --build-arg ICD_VER=24.39.31294.20-1032~22.04 \
                 --build-arg OCLOC_VER=24.39.31294.21-1032~22.04 \
                 --build-arg LEVEL_ZERO_VER=1.17.44.0-1022~22.04 \
                 --build-arg LEVEL_ZERO_DEV_VER=1.17.44.0-1022~22.04 \
                 --build-arg XPU_SMI_VER=1.2.39-66~22.04 \
                 --build-arg TORCH_VERSION=2.5.1+cxx11.abi  \
                 --build-arg IPEX_VERSION=2.5.10+xpu \
                 --build-arg TORCHVISION_VERSION=0.20.1+cxx11.abi \
                 --build-arg TORCHAUDIO_VERSION=2.5.1+cxx11.abi \
                 --build-arg ONECCL_BIND_PT_VERSION=2.5.0+xpu \
                 --build-arg TORCH_WHL_URL=https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ \
                 --build-arg IPEX_WHL_URL=https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ \
                 --build-arg TORCHVISION_WHL_URL=https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ \
                 --build-arg TORCHAUDIO_WHL_URL=https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ \
                 --build-arg ONECCL_BIND_PT_WHL_URL=https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ \
                 -t ${IMAGE_NAME} \
                 -f Dockerfile.prebuilt .
fi
