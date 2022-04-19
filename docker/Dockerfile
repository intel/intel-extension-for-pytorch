# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#
# THIS IS A GENERATED DOCKERFILE.
#
# This file was assembled from multiple pieces, whose use is documented
# throughout. Please refer to the TensorFlow dockerfiles documentation
# for more information.
FROM ubuntu:18.04
SHELL ["/bin/bash", "-c"]
RUN apt-get update -y && apt-get install -y unzip
WORKDIR /opt/installs/
ARG python_dockerfile=3.8
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    ca-certificates \
    python3-dev \
    python3-pip \
    numactl \
    autoconf \
    doxygen \
    graphviz \
    clang-tidy-10 \
    wget && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install cpplint -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    wget --quiet https://github.com/google/protobuf/releases/download/v2.6.1/protobuf-2.6.1.tar.gz && \
    tar -xzf protobuf-2.6.1.tar.gz && \
    cd protobuf-2.6.1 && \
    ./configure && \
    make && \
    make install

# Prepare the Conda environment

RUN wget --quiet https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh && \
    chmod +x anaconda3.sh && \
    ./anaconda3.sh -b -p ~/anaconda3 && \
    rm ./anaconda3.sh && \
    echo $python_dockerfile && \
    ~/anaconda3/bin/conda create -yn pytorch python=$python_dockerfile && \
    export PATH=~/anaconda3/bin/:${PATH} && \
    source activate pytorch && \
    pip install pip && \
    pip install lark-parser hypothesis tornado tqdm  Pillow  yacs opencv-python pycocotools cityscapesscripts \
    unidecode inflect librosa toml soundfile ipdb sox tensorboard jinja2 psutil pytest && \
    pip install sklearn onnx && \
    conda config --add channels intel && \
    conda install  -y openblas intel-openmp matplotlib   typing_extensions future six requests dataclasses && \
    conda install -y ninja pyyaml setuptools cmake cffi typing  intel-openmp && \
    conda install -y mkl mkl-include numpy -c intel --no-update-deps

ENV PATH ~/anaconda3/bin/:${PATH}

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/root/anaconda3/envs/pytorch/lib/
RUN echo NO | dpkg-reconfigure dash
RUN source ~/anaconda3/bin/activate pytorch && \
    pip install matplotlib Pillow pycocotools jinja2&& \
    pip install yacs opencv-python cityscapesscripts transformers && \
    conda install -y libopenblas && \
    cd /opt/installs && \
    wget --no-check-certificate https://github.com/gperftools/gperftools/releases/download/gperftools-2.7.90/gperftools-2.7.90.tar.gz && \
    tar -xzf gperftools-2.7.90.tar.gz && \
    cd gperftools-2.7.90 && \
    ./configure --prefix=$HOME/.local && \
    make && \
    make install && \
    rm -rf /opt/installs/

# Build Jemalloc
ARG JEMALLOC_SHA=c8209150f9d219a137412b06431c9d52839c7272

RUN source ~/anaconda3/bin/activate pytorch && \
    git clone  https://github.com/jemalloc/jemalloc.git && \
    cd jemalloc && \
    git checkout ${JEMALLOC_SHA} && \
    ./autogen.sh && \
    mkdir /root/lib/ && \
    ./configure --prefix=/root/lib/jemalloc/ && \
    make && \
    make install

ARG IPEX_VERSION=1.11.0
ARG IPEX_URL=https://software.intel.com/ipex-whl-stable

RUN source ~/anaconda3/bin/activate pytorch && \
    pip uninstall torch torchaudio torchvision -y && \
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir intel_extension_for_pytorch==${IPEX_VERSION} -f ${IPEX_URL}

ENV DNNL_MAX_CPU_ISA="AVX512_CORE_AMX"
ENV LD_PRELOAD="/root/lib/jemalloc/lib/libjemalloc.so:/root/anaconda3/lib/libiomp5.so:$LD_PRELOAD"
ENV PATH="~/anaconda3/bin:${PATH}"
ENV BASH_ENV=/opt/.bash_profile
WORKDIR /opt/workspace/
RUN echo "source activate pytorch" >> /opt/.bash_profile
