# Copyright (c) 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0

ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE} AS base
SHELL ["/bin/bash", "-c"]
RUN if [ -f /etc/apt/apt.conf.d/proxy.conf ]; then rm /etc/apt/apt.conf.d/proxy.conf; fi && \
    if [ ! -z ${HTTP_PROXY} ]; then echo "Acquire::http::Proxy \"${HTTP_PROXY}\";" >> /etc/apt/apt.conf.d/proxy.conf; fi && \
    if [ ! -z ${HTTPS_PROXY} ]; then echo "Acquire::https::Proxy \"${HTTPS_PROXY}\";" >> /etc/apt/apt.conf.d/proxy.conf; fi
RUN apt update && \
    apt full-upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends -y \
    git \
    curl \
    wget \
    vim \
    patch \
    gcc \
    g++ \
    make \
    libgomp1 \
    pkg-config \
    unzip \
    zip \
    software-properties-common \
    gnupg \
    gpg-agent
COPY ./scripts/tools/compilation_helper/basekit_driver_install_helper.sh .
RUN bash ./basekit_driver_install_helper.sh driver

WORKDIR /root

RUN curl -fsSL -v -o miniforge.sh -O https://github.com/conda-forge/miniforge/releases/download/24.7.1-2/Miniforge3-24.7.1-2-Linux-x86_64.sh && \
    bash miniforge.sh -b -p ./miniforge3 && \
    rm miniforge.sh

FROM base AS dev
# --build-arg COMPILE=ON to compile from source
ARG COMPILE
ARG AOT=pytorch
RUN bash /basekit_driver_install_helper.sh dev
COPY . ./intel-extension-for-pytorch/
RUN . ./miniforge3/bin/activate && \
    conda create -y -n compile_py310 python=3.10 && conda activate compile_py310 && \
    cd intel-extension-for-pytorch/examples/gpu/llm && \
    MODE=0; if [ -z ${COMPILE} ]; then (( MODE |= 0x06 )); else (( MODE |= 0x02 )); fi && \
    bash tools/env_setup.sh ${MODE} /opt/intel/oneapi ${AOT}

FROM base AS deploy
COPY --from=dev /root/intel-extension-for-pytorch/examples/gpu/llm ./llm
COPY --from=dev /root/intel-extension-for-pytorch/scripts/tools/compilation_helper/get_libstdcpp_lib.sh ./llm/tools
RUN apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    if [ -f /etc/apt/apt.conf.d/proxy.conf ]; then rm /etc/apt/apt.conf.d/proxy.conf; fi && \
    rm /basekit_driver_install_helper.sh
RUN . ./miniforge3/bin/activate && \
    conda create -y -n py310 python=3.10 && conda activate py310 && conda install -y libstdcxx-ng && \
    cd ./llm && \
    bash tools/env_setup.sh 1 && \
    python -m pip cache purge && \
    conda clean -a -y
RUN ENTRYPOINT=/usr/local/bin/entrypoint.sh && \
    echo "#!/bin/bash" > ${ENTRYPOINT} && \
    echo "CMDS=(); while [ \$# -gt 0 ]; do CMDS+=(\"\$1\"); shift; done;" >> ${ENTRYPOINT} && \
    echo ". ~/miniforge3/bin/activate" >> ${ENTRYPOINT} && \
    echo "conda activate py310" >> ${ENTRYPOINT} && \
    echo "\"\${CMDS[@]}\"" >> ${ENTRYPOINT} && \
    chmod +x ${ENTRYPOINT}
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
