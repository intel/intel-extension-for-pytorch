#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: bash $0 <MODE>"
    echo "MODE: \"add-apt-repo\" for adding driver and oneAPI basekit apt repositories."
    echo "      \"driver\" for installing required driver packages."
    echo "      \"dev\" for installing required packages for compilation."
    echo "      \"runtime\" for installing required runtime packages."
    exit 1
fi
MODE=$1
if [ "${MODE}" != "add-apt-repo" ] &&
   [ "${MODE}" != "driver" ] &&
   [ "${MODE}" != "dev" ] &&
   [ "${MODE}" != "runtime" ]; then
    echo "MODE \"${MODE}\" not supported."
    echo "MODE: \"add-apt-repo\" for adding driver and oneAPI basekit apt repositories."
    echo "      \"driver\" for installing required driver packages."
    echo "      \"dev\" for installing required packages for compilation."
    echo "      \"runtime\" for installing required runtime packages."
    exit 2
fi

SUDO=''
if [ $UID -ne 0 ]; then
    SUDO='sudo'
fi

if [ "${MODE}" == "add-apt-repo" ]; then
    wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | ${SUDO} gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" | ${SUDO} tee /etc/apt/sources.list.d/intel-gpu-jammy.list
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | ${SUDO} tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | ${SUDO} tee /etc/apt/sources.list.d/oneAPI.list
    ${SUDO} apt update
fi
if [ "${MODE}" == "driver" ]; then
    ${SUDO} apt update
    ${SUDO} apt install -y intel-opencl-icd=23.30.26918.50-736~22.04 \
    level-zero=1.13.1-719~22.04 \
    intel-level-zero-gpu=1.3.26918.50-736~22.04 \
    xpu-smi=1.2.22-31~22.04
fi
if [ "${MODE}" == "dev" ]; then
    ${SUDO} apt update
    ${SUDO} sudo apt install -y level-zero-dev=1.13.1-719~22.04 \
    intel-level-zero-gpu-dev=1.3.26918.50-736~22.04 \
    intel-oneapi-dpcpp-cpp-2024.0 \
    intel-oneapi-mkl-devel-2024.0
fi
if [ "${MODE}" == "runtime" ]; then
    ${SUDO} apt update
    ${SUDO} sudo apt install -y intel-oneapi-runtime-dpcpp-cpp=2024.0.0-49819 \
    intel-oneapi-runtime-mkl=2024.0.0-49656
fi
