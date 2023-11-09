# NOTE: To build this you will need a docker version >= 19.03 and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference:
#           https://docs.docker.com/develop/develop-images/build_enhancements/

ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE} AS base
RUN apt update && \
    apt full-upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt install -y \
    sudo \
    numactl \
    wget \
    vim \
    git \
    gcc-12 \
    g++-12 \
    make \
    curl && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 100 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 100
RUN useradd -m ubuntu && \
    echo 'ubuntu ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers
WORKDIR /home/ubuntu

FROM base AS dev
COPY . /home/ubuntu/llm
RUN chown -R ubuntu:ubuntu /home/ubuntu/llm && \
    rm /home/ubuntu/llm/Dockerfile

USER ubuntu

RUN curl -fsSL -v -o miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    bash miniconda.sh -b -p ~/miniconda3 && \
    rm miniconda.sh && \
    echo "source ~/miniconda3/bin/activate" >> ~/.bashrc

RUN . ~/miniconda3/bin/activate && \
    conda create -y -n compile_py310 python=3.10 && \
    conda activate compile_py310 && \
    cd llm && \
    bash tools/env_setup.sh 2 && \
    conda deactivate && \
    conda remove -y -n compile_py310 --all && \
    conda create -y -n py310 python=3.10 && \
    conda activate py310 && \
    bash tools/env_setup.sh 1 && \
    echo "conda activate py310" >> ~/.bashrc

FROM base AS deploy
USER ubuntu
COPY --from=dev /home/ubuntu/miniconda3 /home/ubuntu/miniconda3
COPY --from=dev /home/ubuntu/llm /home/ubuntu/llm
COPY --from=dev /home/ubuntu/.bashrc /home/ubuntu/.bashrc
RUN sudo chown -R ubuntu:ubuntu ~/miniconda3 ~/llm ~/.bashrc && \
    sudo mv ~/llm/oneCCL/build/_install /opt/oneCCL && \
    sudo chown -R root:root /opt/oneCCL && \
    rm -rf ~/llm/oneCCL && \
    sed -i "s|ONECCL_PATH=.*|ONECCL_PATH=/opt/oneCCL|" ~/llm/tools/env_activate.sh
