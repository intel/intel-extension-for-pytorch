#!/bin/bash
set -e

if [ $# -eq 0 ]; then
    echo "Usage: bash $0 <MODE>"
    echo "MODE: \"driver\" for installing required driver packages."
    echo "      \"dev\" for installing required packages for compilation."
    exit 1
fi
MODE=$1
if [ "${MODE}" != "driver" ] &&
   [ "${MODE}" != "dev" ]; then
    echo "MODE \"${MODE}\" not supported."
    echo "MODE: \"driver\" for installing required driver packages."
    echo "      \"dev\" for installing required packages for compilation."
    exit 2
fi
shift
DEVICE="unified"
if [ $# -gt 0 ]; then
    DEVICE=$1
fi
if [[ " unified datacenter client " =~ " ${DEVICE} " ]]; then
    if [ "${DEVICE}" = "datacenter" ]; then
        DEVICE="unified"
    fi
else
    echo "Device is not recognized. Reset it to \"unified\" for Data Center GPUs."
    DEVICE="unified"
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
    if [[ ! " jammy noble " =~ " ${OS_VERSION} " ]]; then
        echo "Ubuntu version ${OS_VERSION} not supported"
        exit 3
    fi
elif [[ " rhel centos " =~ " ${OS_ID} " ]] && [ "${DEVICE}" = "unified" ]; then
    OS_VERSION=${VERSION_ID}
    if [ "${OS_VERSION}" = "8" ]; then
        OS_VERSION="8.10"
    fi
    if [ "${OS_VERSION}" = "9" ]; then
        OS_VERSION="9.4"
    fi
    if [[ ! " 8.8 8.10 9.2 9.4 " =~ " ${OS_VERSION} " ]]; then
        echo "RHEL version ${OS_VERSION} not supported"
        exit 3
    fi
elif [ "${OS_ID}" = "opensuse-leap" ] && [ "${DEVICE}" = "unified" ]; then
    OS_VERSION=${VERSION_ID//./sp}
    if [[ ! " 15sp4 15sp5 " =~ " ${OS_VERSION} " ]]; then
        echo "SLES version ${VERSION_ID} not supported"
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
        echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu ${OS_VERSION} ${DEVICE}" | ${SUDO} tee /etc/apt/sources.list.d/intel-gpu-${OS_VERSION}.list
        ${SUDO} apt update
    fi
    if [[ " rhel centos " =~ " ${OS_ID} " ]]; then
        ${SUDO} dnf install -y 'dnf-command(config-manager)'
        ${SUDO} dnf config-manager --add-repo https://repositories.intel.com/gpu/rhel/${OS_VERSION}/unified/intel-gpu-${OS_VERSION}.repo
    fi
    if [ "${OS_ID}" = "opensuse-leap" ]; then
        ${SUDO} zypper addrepo -f -r https://repositories.intel.com/gpu/sles/${OS_VERSION}/unified/intel-gpu-${OS_VERSION}.repo
        ${SUDO} rpm --import https://repositories.intel.com/gpu/intel-graphics.key
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
        echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/intel-for-pytorch-gpu-dev all main" | ${SUDO} tee /etc/apt/sources.list.d/pti.list
        ${SUDO} apt update
    fi
    if [[ " rhel centos " =~ " ${OS_ID} " ]]; then
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
        tee > /tmp/intel-for-pytorch-gpu-dev.repo << EOF
[intel-for-pytorch-gpu-dev]
name=Intel for Pytorch GPU dev repository
baseurl=https://yum.repos.intel.com/intel-for-pytorch-gpu-dev
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
EOF
        ${SUDO} mv /tmp/intel-for-pytorch-gpu-dev.repo /etc/yum.repos.d
    fi
    if [ "${OS_ID}" = "opensuse-leap" ]; then
        ${SUDO} zypper addrepo https://yum.repos.intel.com/oneapi oneAPI
        ${SUDO} zypper addrepo https://yum.repos.intel.com/intel-for-pytorch-gpu-dev intel-for-pytorch-gpu-dev
        ${SUDO} rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
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
        POSTFIX=""
        if [ "${OS_VERSION}" = "jammy" ]; then
            POSTFIX="22.04"
        fi
        if [ "${OS_VERSION}" = "noble" ]; then
            POSTFIX="24.04"
        fi
        ${SUDO} apt update
        ${SUDO} apt install -y intel-opencl-icd=24.22.29735.27-914~${POSTFIX} \
        libze1=1.17.6-914~${POSTFIX} \
        libze-dev=1.17.6-914~${POSTFIX} \
        intel-level-zero-gpu=1.3.29735.27-914~${POSTFIX} \
        xpu-smi=1.2.35-56~${POSTFIX}
    fi
    if [[ " rhel centos " =~ " ${OS_ID} " ]]; then
        ${SUDO} dnf install -y intel-opencl-24.22.29735.27 \
        level-zero-1.17.6 \
        level-zero-devel-1.17.6 \
        intel-level-zero-gpu-1.3.29735 \
        intel-ocloc-24.22.29735.27 \
        xpu-smi-1.2.35
    fi
    if [ "${OS_ID}" = "opensuse-leap" ]; then
        ${SUDO} zypper install -y --oldpackage intel-opencl-24.22.29735.27 \
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
        POSTFIX=""
        if [ "${OS_VERSION}" = "jammy" ]; then
            POSTFIX="22.04"
        fi
        if [ "${OS_VERSION}" = "noble" ]; then
            POSTFIX="24.04"
        fi
        ${SUDO} apt update
        ${SUDO} apt install -y intel-level-zero-gpu-dev=1.3.29735.27-914~${POSTFIX} \
        intel-oneapi-dpcpp-cpp-2024.2=2024.2.1-1079 \
        intel-oneapi-mkl-devel=2024.2.1-103 \
        intel-oneapi-ccl-devel=2021.13.1-31 \
        intel-pti-dev
    fi
    if [[ " rhel centos " =~ " ${OS_ID} " ]]; then
        ${SUDO} dnf install -y intel-level-zero-gpu-devel-1.3.29735 \
        intel-oneapi-dpcpp-cpp-2024.2-2024.2.1 \
        intel-oneapi-mkl-devel-2024.2.1 \
        intel-oneapi-ccl-devel-2021.13.1 \
        intel-pti-dev
    fi
    if [ "${OS_ID}" = "opensuse-leap" ]; then
        ${SUDO} zypper install -y --oldpackage intel-level-zero-gpu-devel-1.3.29735 \
        intel-oneapi-dpcpp-cpp-2024.2-2024.2.1 \
        intel-oneapi-mkl-devel-2024.2.1 \
        intel-oneapi-ccl-devel-2021.13.1 \
        intel-pti-dev
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
