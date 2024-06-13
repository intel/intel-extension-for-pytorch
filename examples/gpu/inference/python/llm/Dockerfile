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
    wget \
    curl \
    vim \
    patch \
    gcc \
    g++ \
    make \
    libgomp1 \
    pkg-config \
    software-properties-common \
    gnupg \
    gpg-agent
COPY ./scripts/tools/compilation_helper/basekit_driver_install_helper.sh .
RUN bash ./basekit_driver_install_helper.sh driver

WORKDIR /root

RUN curl -fsSL -v -o miniforge.sh -O https://github.com/conda-forge/miniforge/releases/download/24.1.2-0/Miniforge3-24.1.2-0-Linux-x86_64.sh && \
    bash miniforge.sh -b -p ./miniforge3 && \
    rm miniforge.sh && \
    echo "source ~/miniforge3/bin/activate" >> ./.bashrc

FROM base AS dev
# --build-arg COMPILE=ON to compile from source
ARG COMPILE
RUN bash /basekit_driver_install_helper.sh dev
COPY . ./intel-extension-for-pytorch/
RUN . ./miniforge3/bin/activate && \
    conda create -y -n compile_py310 python=3.10 && conda activate compile_py310 && \
    cd intel-extension-for-pytorch/examples/gpu/inference/python/llm && \
    if [ -z ${COMPILE} ]; then MODE=6; else MODE=2; fi && \
    bash tools/env_setup.sh ${MODE} /opt/intel/oneapi/compiler/latest /opt/intel/oneapi/mkl/latest /opt/intel/oneapi/ccl/latest /opt/intel/oneapi/mpi/latest pvc,ats-m150,acm-g11

FROM base AS deploy

COPY --from=dev /root/intel-extension-for-pytorch/examples/gpu/inference/python/llm ./llm
COPY --from=dev /root/intel-extension-for-pytorch/scripts/tools/compilation_helper/get_libstdcpp_lib.sh .
COPY --from=dev /root/intel-extension-for-pytorch/scripts/tools/compilation_helper/basekit_driver_install_helper.sh .
RUN bash ./basekit_driver_install_helper.sh runtime
RUN . ./miniforge3/bin/activate && \
    conda create -y -n py310 python=3.10 && conda activate py310 && \
    echo "conda activate py310" >> ./.bashrc && \
    ldpreload=$(bash get_libstdcpp_lib.sh) && echo "export LD_PRELOAD=${ldpreload}" >> ./.bashrc && rm get_libstdcpp_lib.sh && \
    cd ./llm && \
    bash tools/env_setup.sh 1 /opt/intel/oneapi/compiler/latest /opt/intel/oneapi/mkl/latest /opt/intel/oneapi/ccl/latest /opt/intel/oneapi/mpi/latest pvc,ats-m150,acm-g11 && \
    python -m pip cache purge && \
    conda clean -a -y
RUN apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    if [ -f /etc/apt/apt.conf.d/proxy.conf ]; then rm /etc/apt/apt.conf.d/proxy.conf; fi && \
    rm /basekit_driver_install_helper.sh
