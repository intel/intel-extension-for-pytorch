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
    sudo \
    git \
    wget \
    curl \
    vim \
    patch \
    gcc \
    g++ \
    make \
    pkg-config \
    software-properties-common \
    gnupg \
    gpg-agent
COPY ./tools/basekit_driver_install_helper.sh .
RUN bash ./basekit_driver_install_helper.sh driver

ARG GID_RENDER=109
RUN useradd -m -s /bin/bash ubuntu && \
    echo 'ubuntu ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers && \
    groupadd -g $GID_RENDER render && \
    usermod -a -G video,render ubuntu
USER ubuntu
WORKDIR /home/ubuntu

RUN curl -fsSL -v -o miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    bash miniconda.sh -b -p ./miniconda3 && \
    rm miniconda.sh && \
    echo "source ~/miniconda3/bin/activate" >> ./.bashrc

FROM base AS dev
# --build-arg COMPILE=ON to compile from source
ARG COMPILE
RUN bash /basekit_driver_install_helper.sh dev
COPY --chown=ubuntu:ubuntu . ./intel-extension-for-pytorch/
RUN . ./miniconda3/bin/activate && \
    conda create -y -n compile_py310 python=3.10 && conda activate compile_py310 && \
    cd intel-extension-for-pytorch/examples/gpu/inference/python/llm && \
    if [ -z ${COMPILE} ]; then MODE=6; else MODE=2; fi && \
    bash tools/env_setup.sh ${MODE} /opt/intel/oneapi/compiler/latest /opt/intel/oneapi/mkl/latest /opt/intel/oneapi/ccl/latest pvc,ats-m150,acm-g11

FROM base AS deploy
RUN bash /basekit_driver_install_helper.sh runtime && \
    sudo apt clean && \
    sudo rm -rf /var/lib/apt/lists/* && \
    if [ -f /etc/apt/apt.conf.d/proxy.conf ]; then sudo rm /etc/apt/apt.conf.d/proxy.conf; fi && \
    sudo rm /basekit_driver_install_helper.sh
COPY --from=dev --chown=ubuntu:ubuntu /home/ubuntu/intel-extension-for-pytorch/examples/gpu/inference/python/llm ./llm
COPY --from=dev --chown=ubuntu:ubuntu /home/ubuntu/intel-extension-for-pytorch/tools/get_libstdcpp_lib.sh .
RUN . ./miniconda3/bin/activate && \
    conda create -y -n py310 python=3.10 && conda activate py310 && \
    echo "conda activate py310" >> ./.bashrc && \
    ldpreload=$(bash get_libstdcpp_lib.sh) && echo "export LD_PRELOAD=${ldpreload}" >> ./.bashrc && rm get_libstdcpp_lib.sh && \
    cd ./llm && \
    bash tools/env_setup.sh 1 && \
    python -m pip cache purge && \
    conda clean -a -y
