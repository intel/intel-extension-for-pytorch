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

ARG UBUNTU_VERSION

FROM ubuntu:${UBUNTU_VERSION}

ENV LANG=C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    apt-utils \
    build-essential \
    ca-certificates \
    clinfo \
    curl \
    git \
    gnupg2 \
    gpg-agent \
    rsync \
    sudo \
    unzip \
    wget && \
    apt-get clean && \
    rm -rf  /var/lib/apt/lists/*

RUN no_proxy=$no_proxy wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | \
    gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy max" | \
    tee  /etc/apt/sources.list.d/intel.gpu.jammy.list

ARG ICD_VER
ARG LEVEL_ZERO_GPU_VER
ARG LEVEL_ZERO_VER
ARG LEVEL_ZERO_DEV_VER

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    intel-opencl-icd=${ICD_VER} \
    intel-level-zero-gpu=${LEVEL_ZERO_GPU_VER} \
    level-zero=${LEVEL_ZERO_VER} \
    level-zero-dev=${LEVEL_ZERO_DEV_VER} && \
    apt-get clean && \
    rm -rf  /var/lib/apt/lists/*

RUN no_proxy=$no_proxy wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
   | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
   echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
   | tee /etc/apt/sources.list.d/oneAPI.list

ARG DPCPP_VER
ARG MKL_VER
ARG CCL_VER

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    intel-oneapi-runtime-dpcpp-cpp=${DPCPP_VER} \
    intel-oneapi-runtime-mkl=${MKL_VER} \
    intel-oneapi-runtime-ccl=${CCL_VER}

ARG PYTHON
RUN apt-get update && apt install -y software-properties-common 
RUN add-apt-repository -y ppa:deadsnakes/ppa 

RUN apt-cache policy $PYTHON && apt-get update && apt-get install -y \
    --no-install-recommends --fix-missing $PYTHON 

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    ${PYTHON} lib${PYTHON} python3-pip ${PYTHON}-distutils && \
    apt-get clean && \
    rm -rf  /var/lib/apt/lists/*

RUN pip --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN ln -sf $(which ${PYTHON}) /usr/local/bin/python && \
    ln -sf $(which ${PYTHON}) /usr/local/bin/python3 && \
    ln -sf $(which ${PYTHON}) /usr/bin/python && \
    ln -sf $(which ${PYTHON}) /usr/bin/python3

ARG TORCH_VERSION
ARG TORCHVISION_VERSION
ARG IPEX_VERSION
ARG ONECCL_BIND_PT_VERSION
ARG TORCH_WHL_URL
ARG IPEX_WHL_URL
ARG TORCHVISION_WHL_URL
ARG ONECCL_BIND_PT_WHL_URL

RUN python -m pip install numpy

RUN python -m pip install torch==${TORCH_VERSION} -f ${TORCH_WHL_URL} && \
    python -m pip install intel_extension_for_pytorch==${IPEX_VERSION} -f ${IPEX_WHL_URL} && \
    python -m pip install torchvision==${TORCHVISION_VERSION} -f ${TORCHVISION_WHL_URL} && \
    python -m pip install oneccl_bind_pt==${ONECCL_BIND_PT_VERSION} -f ${ONECCL_BIND_PT_WHL_URL} 

ENV LD_LIBRARY_PATH=/opt/intel/oneapi/lib:/opt/intel/oneapi/lib/intel64:$LD_LIBRARY_PATH

RUN apt update && \
    apt install -y sudo 
RUN adduser --disabled-password --gecos "" user
RUN echo 'user ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers

# Set the non-root user as the default user
USER user
