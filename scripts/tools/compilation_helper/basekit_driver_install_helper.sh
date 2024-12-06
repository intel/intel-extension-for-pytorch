#!/bin/bash
set -e

VER_ONEAPI=2025.0

#VER_ICD_C=24.39.31294.21
#VER_LIBZE_C=1.17.44.0
#VER_LIBIGC_C=1.0.17791.16
#VER_IGC_CM_C=1.0.176.54074
#VER_GMMLIB_C=22.5.2
#VER_OCLOC_C=${VER_ICD_C}
#VER_SMI_C=1.2.39
VER_DPCPP_C=${VER_ONEAPI}.1
VER_MKL_C=${VER_ONEAPI}.1
VER_CCL_C=2021.14.0
VER_PTI_C=0.10.0

#VER_ICD_U=${VER_ICD_C}-1032
#VER_LIBZE_U=${VER_LIBZE_C}-1022
#VER_LIBIGC_U=${VER_LIBIGC_C}-1032
#VER_IGC_CM_U=1.0.224-821
#VER_GMMLIB_U=${VER_GMMLIB_C}-1018
#VER_OCLOC_U=${VER_OCLOC_C}-1032
#VER_SMI_U=${VER_SMI_C}-66
VER_DPCPP_U=${VER_DPCPP_C}-1240
VER_MKL_U=${VER_MKL_C}-14
VER_CCL_U=${VER_CCL_C}-505
VER_PTI_U=${VER_PTI_C}-284

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
    fi
    if [ "${OS_ID}" = "opensuse-leap" ]; then
        ${SUDO} zypper addrepo https://yum.repos.intel.com/oneapi oneAPI
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
        #${SUDO} apt install -y intel-opencl-icd=${VER_ICD_U}~${POSTFIX} \
        #libze1=${VER_LIBZE_U}~${POSTFIX} \
        #libze-dev=${VER_LIBZE_U}~${POSTFIX} \
        #libigc1=${VER_LIBIGC_U}~${POSTFIX} \
        #libigdfcl1=${VER_LIBIGC_U}~${POSTFIX} \
        #intel-igc-cm=${VER_IGC_CM_U}~${POSTFIX} \
        #libigdgmm12=${VER_GMMLIB_U}~${POSTFIX} \
        #intel-ocloc=${VER_OCLOC_U}~${POSTFIX} \
        #xpu-smi=${VER_SMI_U}~${POSTFIX}
        ${SUDO} apt install -y intel-opencl-icd \
        libze1 \
        libze-dev \
        intel-ocloc \
        xpu-smi
    fi
    if [[ " rhel centos " =~ " ${OS_ID} " ]]; then
        #${SUDO} dnf install -y intel-opencl-${VER_ICD_C} \
        #level-zero-${VER_LIBZE_C} \
        #level-zero-devel-${VER_LIBZE_C} \
        #intel-igc-core-${VER_LIBIGC_C} \
        #intel-igc-opencl-${VER_LIBIGC_C} \
        #intel-igc-cm-${VER_IGC_CM_C} \
        #intel-gmmlib-${VER_GMMLIB_C} \
        #intel-ocloc-${VER_OCLOC_C} \
        #xpu-smi-${VER_SMI_C}
        ${SUDO} dnf install -y intel-opencl \
        level-zero \
        level-zero-devel \
        intel-ocloc \
        xpu-smi
    fi
    if [ "${OS_ID}" = "opensuse-leap" ]; then
        #${SUDO} zypper install -y --oldpackage intel-opencl-${VER_ICD_C} \
        #level-zero-${VER_LIBZE_C} \
        #level-zero-devel-${VER_LIBZE_C} \
        #intel-igc-core-${VER_LIBIGC_C} \
        #intel-igc-opencl-${VER_LIBIGC_C} \
        #intel-igc-cm-${VER_IGC_CM_C} \
        #intel-gmmlib-${VER_GMMLIB_C} \
        #intel-ocloc-${VER_OCLOC_C} \
        #xpu-smi-${VER_SMI_C}
        ${SUDO} zypper install -y intel-opencl \
        level-zero \
        level-zero-devel \
        intel-ocloc \
        xpu-smi
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
        ${SUDO} apt install -y intel-oneapi-dpcpp-cpp-${VER_ONEAPI}=${VER_DPCPP_U} \
        intel-oneapi-mkl-devel=${VER_MKL_U} \
        intel-oneapi-ccl-devel=${VER_CCL_U} \
        intel-pti-dev=${VER_PTI_U}
    fi
    if [[ " rhel centos " =~ " ${OS_ID} " ]]; then
        ${SUDO} dnf install -y intel-oneapi-dpcpp-cpp-${VER_ONEAPI}-${VER_DPCPP_C} \
        intel-oneapi-mkl-devel-${VER_MKL_C} \
        intel-oneapi-ccl-devel-${VER_CCL_C} \
        intel-pti-dev-${VER_PTI_C}
    fi
    if [ "${OS_ID}" = "opensuse-leap" ]; then
        ${SUDO} zypper install -y --oldpackage intel-oneapi-dpcpp-cpp-${VER_ONEAPI}-${VER_DPCPP_C} \
        intel-oneapi-mkl-devel-${VER_MKL_C} \
        intel-oneapi-ccl-devel-${VER_CCL_C} \
        intel-pti-dev-${VER_PTI_C}
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
