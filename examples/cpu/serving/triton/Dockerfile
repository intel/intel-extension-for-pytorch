# Copyright (c) 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0

FROM nvcr.io/nvidia/tritonserver:23.10-py3

RUN useradd -m tritonuser && \
    echo 'tritonuser ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers

USER tritonuser
WORKDIR /home/tritonuser

COPY --chown=tritonuser:tritonuser requirements.txt requirements.txt
ENV PATH="/home/tritonuser/.local/bin":${PATH}
RUN python3 -m pip --no-cache-dir install -U --upgrade pip && \
    python3 -m pip --no-cache-dir install -U -r requirements.txt

ENV LD_PRELOAD="/home/tritonuser/.local/lib/libiomp5.so:/usr/lib/x86_64-linux-gnu/libtcmalloc.so":${LD_PRELOAD} \
    KMP_BLOCKTIME=1 \
    KMP_SETTINGS=1 \
    KMP_AFFINITY=granularity=fine,compact,1,0 \
    DNNL_PRIMITIVE_CACHE_CAPACITY=1024 \
    TOKENIZERS_PARALLELISM=true