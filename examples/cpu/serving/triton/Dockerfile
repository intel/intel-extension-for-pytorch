# Copyright (c) 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0

FROM nvcr.io/nvidia/tritonserver:23.10-py3

RUN useradd -m tritonuser && \
    echo 'tritonuser ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers

USER tritonuser
WORKDIR /home/tritonuser

ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG IPEX_VERSION

RUN python3 -m pip install --no-cache-dir \
    torch==${PYTORCH_VERSION:-2.1.0+cpu} \
    torchaudio==${TORCHAUDIO_VERSION:-2.1.0+cpu} \
    torchvision==${TORCHVISION_VERSION:-0.16.0+cpu} \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    intel_extension_for_pytorch==${IPEX_VERSION:-2.1.0}+cpu \
    -f https://developer.intel.com/ipex-whl-stable-cpu \
    configargparse \
    intel-openmp \
    numpy \
    transformers \
    tritonclient[all]

ENV PATH="/home/tritonuser/.local/bin":${PATH} \
    LD_PRELOAD="/home/tritonuser/.local/lib/libiomp5.so:/usr/lib/x86_64-linux-gnu/libtcmalloc.so":${LD_PRELOAD} \
    KMP_BLOCKTIME=1 \
    KMP_SETTINGS=1 \
    KMP_AFFINITY=granularity=fine,compact,1,0 \
    DNNL_PRIMITIVE_CACHE_CAPACITY=1024 \
    TOKENIZERS_PARALLELISM=true

CMD ["tritonserver", "--model-repository=/models", "--log-verbose", "1", "--log-error", "1", "--log-info", "1", "--exit-on-error=false", "--model-control-mode=poll"]
