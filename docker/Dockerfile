# syntax = docker/dockerfile:experimental
# based on https://github.com/pytorch/pytorch/blob/master/Dockerfile
# 
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference: 
#           https://docs.docker.com/develop/develop-images/build_enhancements/

ARG BASE_IMAGE=ubuntu:20.04
FROM ${BASE_IMAGE} AS dev-base
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    libjemalloc-dev \
    libjpeg-dev \
    libpng-dev \
    numactl \
    pybind11-dev \
    vim \
    && rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
ARG PYTHON_VERSION=3.8
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build pyyaml numpy ipython mkl mkl-include ninja cython typing pybind11 Pillow && \
    /opt/conda/bin/conda clean -ya

FROM dev-base AS build
COPY --from=conda /opt/conda /opt/conda
ARG TORCHVISION_VERSION=0.6.0
ARG IPEX_VERSION=v1.2.0
ARG PYTORCH_VERSION=v1.7.0
RUN --mount=type=cache,target=/opt/ccache \
    pip install torchvision==${TORCHVISION_VERSION}+cpu --no-deps \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install lark-parser hypothesis && \
    git clone -b ${IPEX_VERSION} --single-branch https://github.com/intel/intel-extension-for-pytorch && \
    cd intel-extension-for-pytorch && git submodule sync && \
    git submodule update --init --recursive && \
    git clone -b ${PYTORCH_VERSION} --single-branch https://github.com/pytorch/pytorch && \
    cd pytorch && git submodule sync && \
    git submodule update --init --recursive && \
    git apply ../torch_patches/xpu-1.7.patch && \
    USE_MKLDNN=1 USE_CUDA=0 USE_NNPACK=0 USE_CUDNN=0 \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" python setup.py install && \
    cd .. && python setup.py install && rm -rf *

FROM dev-base as dev
COPY --from=build /opt/conda /opt/conda
ARG OMP_NUM_THREADS=1
ENV OMP_NUM_THREADS ${OMP_NUM_THREADS}
ARG KMP_BLOCKTIME=1
ENV KMP_BLOCKTIME ${KMP_BLOCKTIME} 
ARG KMP_HW_SUBSET=1T
ENV KMP_HW_SUBSET ${KMP_HW_SUBSET}
ENV MALLOC_CONF "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
ENV LD_PRELOAD "/opt/conda/lib/libiomp5.so /usr/lib/x86_64-linux-gnu/libjemalloc.so"
ENV LD_LIBRARY_PATH "/opt/conda/lib/python3.8/site-packages/lib/":"/opt/conda/lib/python3.8/site-packages/torch_ipex-1.2.0-py3.8-linux-x86_64.egg/lib":${LD_LIBRARY_PATH}
