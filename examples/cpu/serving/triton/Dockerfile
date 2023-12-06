# Copyright (c) 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0

FROM nvcr.io/nvidia/tritonserver:23.10-py3

COPY requirements.txt requirements.txt
RUN apt-get update &&                                              \
    apt-get install --no-install-recommends -y numactl             \
                                               google-perftools    \
                                               python3.9 &&        \
    ln -s /usr/bin/python3.9 /usr/bin/python &&                    \
    apt-get clean

RUN python3 -m pip --no-cache-dir install -U --upgrade pip &&      \
    python3 -m pip --no-cache-dir install -U -r requirements.txt

ENV LD_PRELOAD="/usr/local/lib/libiomp5.so:/usr/lib/x86_64-linux-gnu/libtcmalloc.so":${LD_PRELOAD}
ENV KMP_BLOCKTIME=1
ENV KMP_SETTINGS=1
ENV KMP_AFFINITY=granularity=fine,compact,1,0
ENV DNNL_PRIMITIVE_CACHE_CAPACITY=1024
ENV TOKENIZERS_PARALLELISM=true
