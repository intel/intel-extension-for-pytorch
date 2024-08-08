#!/bin/bash
set -e

if [ $# -eq 0 ]; then
    echo "Usage: bash $0 <MODE>"
    echo "MODE: \"driver\" for installing required driver packages."
    echo "      \"dev\" for installing required packages for compilation."
    echo "      \"runtime\" for installing required packages for runtime."
    exit 1
fi
MODE=$1
if [ "${MODE}" != "driver" ] &&
   [ "${MODE}" != "dev" ] &&
   [ "${MODE}" != "runtime" ] &&
   [ "${MODE}" != "runtime-dev" ]; then
    echo "MODE \"${MODE}\" not supported."
    echo "MODE: \"driver\" for installing required driver packages."
    echo "      \"dev\" for installing required packages for compilation."
    echo "      \"runtime\" for installing required packages for runtime."
    echo "      \"runtime-dev\" for installing required packages for runtime and ccl-dev "
    exit 2
fi

SUDO="null"
if [ $UID -ne 0 ]; then
    SUDO="sudo"
fi

source /etc/os-release
OS_ID=${ID}
OS_VERSION=""
if [ "${OS_ID}" = "ubuntu" ]; then
    OS_VERSION=${VERSION_CODENAME}
    if [[ ! " jammy " =~ " ${OS_VERSION} " ]]; then
        echo "Ubuntu version ${OS_VERSION} not supported"
        exit 3
    fi
elif [ "${OS_ID}" = "rhel" ] || [ "${OS_ID}" = "centos" ]; then
    OS_VERSION=${VERSION_ID}
    if [ "${OS_VERSION}" = "8" ]; then
        OS_VERSION="8.6"
    fi
    if [ "${OS_VERSION}" = "9" ]; then
        OS_VERSION="9.0"
    fi
    if [[ ! " 8.6 8.8 8.9 9.0 9.2 9.3 " =~ " ${OS_VERSION} " ]]; then
        echo "RHEL version ${OS_VERSION} not supported"
        exit 3
    fi
else
    echo "${OS_ID} not supported."
    exit 3
fi

function add-repo-driver() {
    SUDO=$1
    OS_ID=$2
    OS_VERSION=$3
    if [ "${SUDO}" = "null" ]; then
        SUDO=""
    fi

    if [ "${OS_ID}" = "ubuntu" ]; then
        wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | ${SUDO} gpg --dearmor --yes --output /usr/share/keyrings/intel-graphics.gpg
        echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu ${OS_VERSION} unified" | ${SUDO} tee /etc/apt/sources.list.d/intel-gpu-${OS_VERSION}.list
        ${SUDO} apt update
    fi
    if [ "${OS_ID}" = "rhel" ] || [ "${OS_ID}" = "centos" ]; then
          ${SUDO} dnf install -y 'dnf-command(config-manager)'
          ${SUDO} dnf config-manager --add-repo https://repositories.intel.com/gpu/rhel/${OS_VERSION}/unified/intel-gpu-${OS_VERSION}.repo
    fi
}

function add-repo-basekit() {
    SUDO=$1
    OS_ID=$2
    OS_VERSION=$3
    if [ "${SUDO}" = "null" ]; then
        SUDO=""
    fi

    if [ "${OS_ID}" = "ubuntu" ]; then
        wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | ${SUDO} tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
        echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | ${SUDO} tee /etc/apt/sources.list.d/oneAPI.list
        ${SUDO} apt update
    fi
    if [ "${OS_ID}" = "rhel" ] || [ "${OS_ID}" = "centos" ]; then
        tee > /tmp/oneAPI.repo << EOF
[oneAPI]
name=Intel® oneAPI repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
EOF
        ${SUDO} mv /tmp/oneAPI.repo /etc/yum.repos.d
    fi
}

function install-driver() {
    SUDO=$1
    OS_ID=$2
    OS_VERSION=$3
    add-repo-driver ${SUDO} ${OS_ID} ${OS_VERSION}
    if [ "${SUDO}" = "null" ]; then
        SUDO=""
    fi

    if [ "${OS_ID}" = "ubuntu" ]; then
        ${SUDO} apt update
        ${SUDO} apt install -y intel-opencl-icd=24.22.29735.27-914~22.04 \
        libze1=1.17.6-914~22.04 \
        libze-dev=1.17.6-914~22.04 \
        intel-level-zero-gpu=1.3.29735.27-914~22.04 \
        xpu-smi=1.2.35-56~22.04
    fi
    if [ "${OS_ID}" = "rhel" ] || [ "${OS_ID}" = "centos" ]; then
        ${SUDO} dnf install -y intel-opencl-24.22.29735.27 \
        level-zero-1.17.6 \
        level-zero-devel-1.17.6 \
        intel-level-zero-gpu-1.3.29735 \
        intel-ocloc-24.22.29735.27 \
        xpu-smi-1.2.35
    fi
}

function install-dev() {
    SUDO=$1
    OS_ID=$2
    OS_VERSION=$3
    add-repo-basekit ${SUDO} ${OS_ID} ${OS_VERSION}
    if [ "${SUDO}" = "null" ]; then
        SUDO=""
    fi

    if [ "${OS_ID}" = "ubuntu" ]; then
        ${SUDO} apt update
        ${SUDO} apt install -y intel-level-zero-gpu-dev=1.3.29735.27-914~22.04 \
        intel-oneapi-dpcpp-cpp-2024.2=2024.2.1-1079 \
        intel-oneapi-mkl-devel=2024.2.1-103 \
        intel-oneapi-ccl-devel=2021.13.1-31
    fi
    if [ "${OS_ID}" = "rhel" ] || [ "${OS_ID}" = "centos" ]; then
        ${SUDO} dnf install -y intel-level-zero-gpu-devel-1.3.29735 \
        intel-oneapi-dpcpp-cpp-2024.2-2024.2.1 \
        intel-oneapi-mkl-devel-2024.2.1 \
        intel-oneapi-ccl-devel-2021.13.1
    fi
}

function install-runtime() {
    SUDO=$1
    OS_ID=$2
    OS_VERSION=$3
    add-repo-basekit ${SUDO} ${OS_ID} ${OS_VERSION}
    if [ "${SUDO}" = "null" ]; then
        SUDO=""
    fi

    if [ "${OS_ID}" = "ubuntu" ]; then
        ${SUDO} apt update
        ${SUDO} apt install -y intel-oneapi-runtime-dpcpp-cpp=2024.2.1-1079 \
        intel-oneapi-runtime-mkl=2024.2.1-103 \
        intel-oneapi-runtime-ccl=2021.13.1-31
    fi
    if [ "${OS_ID}" = "rhel" ] || [ "${OS_ID}" = "centos" ]; then
        ${SUDO} dnf install -y intel-oneapi-runtime-dpcpp-cpp-2024.2.1-1079 \
        intel-oneapi-runtime-mkl-2024.2.1-103 \
        intel-oneapi-runtime-ccl-2021.13.1-31
    fi
}

function install-runtime-withdev() {
    SUDO=$1
    OS_ID=$2
    OS_VERSION=$3
    add-repo-basekit ${SUDO} ${OS_ID} ${OS_VERSION}
    if [ "${SUDO}" = "null" ]; then
        SUDO=""
    fi

    if [ "${OS_ID}" = "ubuntu" ]; then
        ${SUDO} apt update
        ${SUDO} apt install -y intel-oneapi-runtime-dpcpp-cpp=2024.2.1-1079 \
        intel-oneapi-runtime-mkl=2024.2.1-103 \
        intel-oneapi-ccl-devel=2021.13.1-31
    fi
    if [ "${OS_ID}" = "rhel" ] || [ "${OS_ID}" = "centos" ]; then
        ${SUDO} dnf install -y intel-oneapi-runtime-dpcpp-cpp-2024.2.1-1079 \
        intel-oneapi-runtime-mkl-2024.2.1-103 \
        intel-oneapi-ccl-devel-2021.13.1-31
    fi
}


for CMD in wget gpg; do
    command -v ${CMD} > /dev/null || (echo "Error: Command \"${CMD}\" not found." ; exit 1)
done

if [ "${MODE}" = "driver" ]; then
    install-driver ${SUDO} ${OS_ID} ${OS_VERSION}
fi
if [ "${MODE}" = "dev" ]; then
    install-dev ${SUDO} ${OS_ID} ${OS_VERSION}
fi
if [ "${MODE}" = "runtime" ]; then
    install-runtime ${SUDO} ${OS_ID} ${OS_VERSION}
fi
if [ "${MODE}" = "runtime-dev" ]; then
    install-runtime-withdev ${SUDO} ${OS_ID} ${OS_VERSION}
fi

