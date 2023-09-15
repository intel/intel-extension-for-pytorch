# NOTE: To build this you will need a docker version >= 19.03 and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference:
#           https://docs.docker.com/develop/develop-images/build_enhancements/

ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends -y \
    sudo \
    ca-certificates \
    git \
    curl \
    wget \
    vim \
    ccache \
    numactl \
    make \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m ubuntu
RUN echo 'ubuntu ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers
USER ubuntu
WORKDIR /home/ubuntu

RUN curl -fsSL -v -o miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    bash miniconda.sh -b -p ~/miniconda3 && \
    rm miniconda.sh && \
    echo "source ~/miniconda3/bin/activate" >> ~/.bashrc

RUN curl -fsSL -v -o compile_bundle.sh -O https://github.com/intel/intel-extension-for-pytorch/raw/llm_feature_branch/scripts/compile_bundle.sh && \
    . ~/miniconda3/bin/activate && \
    conda create -y -n py310 python=3.10 && \
    conda activate py310 && \
    bash compile_bundle.sh && \
    rm -rf intel-extension-for-pytorch llvm-project compile_bundle.sh && \
    conda install -y jemalloc gperftools -c conda-forge && \
    python -m pip install intel-openmp && \
    echo "conda activate py310" >> ~/.bashrc && \
    echo "export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so" >> ~/.bashrc && \
    echo "echo \"**Note:** For better performance, please consider to launch workloads with command 'ipexrun'.\"" >> ~/.bashrc
