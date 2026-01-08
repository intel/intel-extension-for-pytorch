# Copyright (c) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache 2.0

# NOTE: To build this you will need a docker version >= 19.03 and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference:
#           https://docs.docker.com/develop/develop-images/build_enhancements/

ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE} AS base
RUN if [ -f /etc/apt/apt.conf.d/proxy.conf ]; then rm /etc/apt/apt.conf.d/proxy.conf; fi && \
    if [ ! -z ${HTTP_PROXY} ]; then echo "Acquire::http::Proxy \"${HTTP_PROXY}\";" >> /etc/apt/apt.conf.d/proxy.conf; fi && \
    if [ ! -z ${HTTPS_PROXY} ]; then echo "Acquire::https::Proxy \"${HTTPS_PROXY}\";" >> /etc/apt/apt.conf.d/proxy.conf; fi
RUN apt update && \
    apt full-upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends -y \
    ca-certificates \
    git \
    curl \
    wget \
    vim \
    numactl \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    gcc-12 \
    g++-12 \
    make
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 100 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 100 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 100

WORKDIR /root

# --build-arg COMPILE=ON to compile from source
FROM base AS dev
ARG COMPILE
COPY . ./intel-extension-for-pytorch
RUN python -m venv venv && . ~/venv/bin/activate && \
    cd intel-extension-for-pytorch/examples/cpu/llm && \
    export CC=gcc && export CXX=g++ && \
    if [ -z ${COMPILE} ]; then bash tools/env_setup.sh 6; else bash tools/env_setup.sh 2; fi && \
    unset CC && unset CXX

FROM base AS deploy
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends -y \
    google-perftools \
    openssh-server \
    net-tools && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    if [ -f /etc/apt/apt.conf.d/proxy.conf ]; then rm /etc/apt/apt.conf.d/proxy.conf; fi
COPY --from=dev /root/intel-extension-for-pytorch/examples/cpu/llm ./llm
COPY --from=dev /root/intel-extension-for-pytorch/tools/get_libstdcpp_lib.sh ./llm/tools
RUN python -m venv venv && . ~/venv/bin/activate && \
    cd /usr/lib/x86_64-linux-gnu/ && ln -s libtcmalloc.so.4 libtcmalloc.so && cd && \
    cd ./llm && \
    bash tools/env_setup.sh 1 && \
    python -m pip cache purge
ARG PORT_SSH=22
RUN mkdir /var/run/sshd && \
    sed -i "s/#Port.*/Port ${PORT_SSH}/" /etc/ssh/sshd_config && \
    echo "service ssh start" >> /root/.bashrc && \
    ssh-keygen -b 4096 -f /root/.ssh/id_rsa -N "" && \
    mv /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
    echo "Host *\n    Port ${PORT_SSH}\n    IdentityFile /root/.ssh/id_rsa\n    StrictHostKeyChecking no" > /root/.ssh/config
EXPOSE ${PORT_SSH}
RUN ENTRYPOINT=/usr/local/bin/entrypoint.sh && \
    echo "#!/bin/bash" > ${ENTRYPOINT} && \
    echo "CMDS=(); while [ \$# -gt 0 ]; do CMDS+=(\"\$1\"); shift; done;" >> ${ENTRYPOINT} && \
    echo ". ~/venv/bin/activate" >> ${ENTRYPOINT} && \
    echo "echo \"**Note:** For better performance, please consider to launch workloads with command 'ipexrun'.\"" >> ${ENTRYPOINT} && \
    echo "\"\${CMDS[@]}\"" >> ${ENTRYPOINT} && \
    chmod +x ${ENTRYPOINT}
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
