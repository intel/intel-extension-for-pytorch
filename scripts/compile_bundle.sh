#!/usr/bin/env bash
#
# Please review the system requirements before running this script
# https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installation.html
#
set -eo pipefail

VER_IPEX=xpu-main
ENABLE_ONEAPI_INTEGRATION=1

if [[ $# -lt 6 ]]; then
    echo "Usage: bash $0 <DPCPPROOT> <MKLROOT> <CCLROOT> <MPIROOT> <PTIROOT> <AOT>"
    echo "DPCPPROOT, MKLROOT, CCLROOT, MPIROOT and PTIROOT are mandatory, should be absolute or relative path to the root directory of DPC++ compiler, oneMKL, oneCCL, Intel(R) MPI and Profiling Tools Interfaces for GPU (PTI for GPU) respectively."
    echo "AOT should be set to the text string for environment variable USE_AOT_DEVLIST. Setting it to \"none\" to disable AOT."
    exit 1
fi
DPCPP_ROOT=$1
ONEMKL_ROOT=$2
ONECCL_ROOT=$3
MPI_ROOT=$4
PTI_ROOT=$5
AOT=$6
if [[ ${AOT} == "none" ]]; then
    AOT=""
fi

# Mode: Select which components to install. PyTorch and Intel® Extension for PyTorch* are always installed.
# High bit: 8 7 6 5 4 3 2 1 :Low bit
#           | | | | | | | └- torch-ccl
#           | | | | | | └--- TorchAudio
#           | | | | | └----- TorchVision
#           | | | | └------- Undefined
#           | | | └--------- Undefined
#           | | └----------- Undefined
#           | └------------- Undefined
#           └--------------- Undefined
MODE=0x07
if [ $# -gt 6 ]; then
    if [[ ! $7 =~ ^[0-9]+$ ]] && [[ ! $7 =~ ^0x[0-9a-fA-F]+$ ]]; then
        echo "Warning: Unexpected argument. Using default value."
    else
        MODE=$7
    fi
fi

# Check existance of DPCPP and ONEMKL environments
DPCPP_ENV=${DPCPP_ROOT}/env/vars.sh
if [ ! -f ${DPCPP_ENV} ]; then
    echo "DPC++ compiler environment ${DPCPP_ENV} doesn't seem to exist."
    exit 2
fi

ONEMKL_ENV=${ONEMKL_ROOT}/env/vars.sh
if [ ! -f ${ONEMKL_ENV} ]; then
    echo "oneMKL environment ${ONEMKL_ENV} doesn't seem to exist."
    exit 3
fi

CCL_ENV=${ONECCL_ROOT}/env/vars.sh
if [ ! -f ${ONECCL_ROOT}/env/vars.sh ]; then
    echo "oneCCL environment ${ONECCL_ROOT} doesn't seem to exist."
    exit 4
fi
ONEAPIROOT=${ONECCL_ROOT}/../..

MPI_ENV=${MPI_ROOT}/env/vars.sh
if [ ! -f ${MPI_ROOT}/env/vars.sh ]; then
    echo "Intel(R) MPI environment ${MPI_ROOT} doesn't seem to exist."
    exit 5
fi

PTI_ENV=${PTI_ROOT}/env/vars.sh
if [ ! -f ${PTI_ROOT}/env/vars.sh ]; then
    echo "Profiling Tools Interfaces for GPU (PTI for GPU) ${PTI_ROOT} doesn't seem to exist."
    exit 5
fi

# Check existance of required Linux commands
for APP in python git patch nproc bzip2; do
    command -v $APP > /dev/null || (echo "Error: Command \"${APP}\" not found." ; exit 6)
done

#function ver_compare() {
#    VER_MAJOR_CUR=$(echo $1 | cut -d "." -f 1)
#    VER_MINOR_CUR=$(echo $1 | cut -d "." -f 2)
#    VER_PATCH_CUR=$(echo $1 | cut -d "." -f 3)
#    VER_MAJOR_REQ=$(echo $2 | cut -d "." -f 1)
#    VER_MINOR_REQ=$(echo $2 | cut -d "." -f 2)
#    VER_PATCH_REQ=$(echo $2 | cut -d "." -f 3)
#    RET=0
#    if [[ ${VER_MAJOR_CUR} -ge ${VER_MAJOR_REQ} ]]; then
#        RET=1
#    else
#        if [[ ${VER_MAJOR_CUR} -eq ${VER_MAJOR_REQ} ]] &&
#           [[ ${VER_MINOR_CUR} -ge ${VER_MINOR_REQ} ]]; then
#            RET=2
#        else
#            if [[ ${VER_MAJOR_CUR} -eq ${VER_MAJOR_REQ} ]] &&
#               [[ ${VER_MINOR_CUR} -eq ${VER_MINOR_REQ} ]] &&
#               [[ ${VER_PATCH_CUR} -ge ${VER_PATCH_REQ} ]]; then
#                RET=3
#            fi
#        fi
#    fi
#    echo ${RET}
#}
#VER_COMP=$(ver_compare $(gcc -dumpfullversion) ${VER_GCC})
#if [ ${VER_COMP} -ne 0 ]; then
#    echo -e '\a'
#    echo "Error: GCC version cannot be equal to or newer than ${VER_GCC}."
#    echo "       Found GCC version $(gcc -dumpfullversion)"
#    exit 8
#fi

# set number of compile processes, if not already defined
MAX_JOBS_VAR=$(nproc)
if [ -z "${MAX_JOBS}" ]; then
    export MAX_JOBS=${MAX_JOBS_VAR}
fi

# Save current directory path
BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )
cd ${BASEFOLDER}

# Checkout individual components
if [ ! -d intel-extension-for-pytorch ]; then
    git clone https://github.com/intel/intel-extension-for-pytorch.git intel-extension-for-pytorch
fi
cd intel-extension-for-pytorch
if [ ! -z ${VER_IPEX} ]; then
    rm -rf * > /dev/null
    git checkout . > /dev/null
    git checkout main > /dev/null
    git pull > /dev/null
    git checkout ${VER_IPEX}
fi
git submodule sync
git submodule update --init --recursive

COMMIT_TORCH=$(python scripts/tools/compilation_helper/dep_ver_utils.py -f dependency_version.json -k pytorch:commit)
VERSION_TORCH=$(python scripts/tools/compilation_helper/dep_ver_utils.py -f dependency_version.json -k pytorch:version)
COMMIT_TORCHVISION=$(python scripts/tools/compilation_helper/dep_ver_utils.py -f dependency_version.json -k torchvision:commit)
COMMIT_TORCHAUDIO=$(python scripts/tools/compilation_helper/dep_ver_utils.py -f dependency_version.json -k torchaudio:commit)
COMMIT_TORCHCCL=$(python scripts/tools/compilation_helper/dep_ver_utils.py -f dependency_version.json -k torch-ccl:commit)
#VER_GCC=$(python scripts/tools/compilation_helper/dep_ver_utils.py -f dependency_version.json -k gcc:min-version)
cd ..

if [ "${COMMIT_TORCHVISION}" = "N/A" ]; then
    (( MODE &= 0xFB ))
fi
if [ "${COMMIT_TORCHAUDIO}" = "N/A" ]; then
    (( MODE &= 0xFD ))
fi
if [ "${COMMIT_TORCHCCL}" = "N/A" ]; then
    (( MODE &= 0xFE ))
fi

if [ ! -d pytorch ]; then
    git clone https://github.com/pytorch/pytorch.git
fi
if [ $((${MODE} & 0x04)) -ne 0 ]; then
    if [ ! -d vision ]; then
        git clone https://github.com/pytorch/vision.git
    fi
fi
if [ $((${MODE} & 0x02)) -ne 0 ]; then
    if [ ! -d audio ]; then
        git clone https://github.com/pytorch/audio.git
    fi
fi
if [ $((${MODE} & 0x01)) -ne 0 ]; then
    if [ ! -d torch-ccl ]; then
        git clone https://github.com/intel/torch-ccl.git
    fi
fi

# Checkout required branch/commit and update submodules
cd pytorch
rm -rf * > /dev/null
if [ -f .ci/docker/ci_commit_pins/triton-xpu.txt ]; then
    rm .ci/docker/ci_commit_pins/triton-xpu.txt
fi
git checkout . > /dev/null
if [ ! -z ${COMMIT_TORCH} ]; then
    git checkout main > /dev/null
    git pull > /dev/null
    git checkout ${COMMIT_TORCH}
fi
git submodule sync
git submodule update --init --recursive
git apply ../intel-extension-for-pytorch/torch_patches/*.patch
cd ..
if [ $((${MODE} & 0x04)) -ne 0 ]; then
    cd vision
    if [ ! -z ${COMMIT_TORCHVISION} ]; then
        rm -rf * > /dev/null
        git checkout . > /dev/null
        git checkout main > /dev/null
        git pull > /dev/null
        git checkout ${COMMIT_TORCHVISION}
    fi
    git submodule sync
    git submodule update --init --recursive
    cd ..
fi
if [ $((${MODE} & 0x02)) -ne 0 ]; then
    cd audio
    if [ ! -z ${COMMIT_TORCHAUDIO} ]; then
        rm -rf * > /dev/null
        git checkout . > /dev/null
        git checkout main > /dev/null
        git pull > /dev/null
        git checkout ${COMMIT_TORCHAUDIO}
    fi
    git submodule sync
    git submodule update --init --recursive
    cd ..
fi
if [ $((${MODE} & 0x01)) -ne 0 ]; then
    cd torch-ccl
    if [ ! -z "${COMMIT_TORCHCCL}" ]; then
        rm -rf * > /dev/null
        git checkout . > /dev/null
        git checkout master > /dev/null
        git pull > /dev/null
        git checkout ${COMMIT_TORCHCCL}
    fi
    git submodule sync
    git submodule update --init --recursive
    cd ..
fi

# Install python dependency
python -m pip uninstall -y torch torchvision torchaudio intel-extension-for-pytorch intel-extension-for-pytorch-deepspeed oneccl_bind_pt
python -m pip install cmake make ninja
if [ $((${MODE} & 0x04)) -ne 0 ]; then
    python -m pip install Pillow
    conda install -y conda-forge::libpng conda-forge::libjpeg-turbo
fi

ABI=1

# don't fail on external scripts
source ${DPCPP_ENV}
source ${ONEMKL_ENV}
source ${CCL_ENV}
source ${MPI_ENV}
source ${PTI_ENV}

#  PyTorch
cd pytorch
python -m pip install -r requirements.txt
python -m pip install --force-reinstall mkl-static mkl-include
export PYTORCH_BUILD_VERSION=${VERSION_TORCH}
export PYTORCH_BUILD_NUMBER=0
# Ensure cmake can find python packages when using conda or virtualenv
CMAKE_PREFIX_PATH_BK=${CMAKE_PREFIX_PATH}
if [ -n "${CONDA_PREFIX-}" ]; then
    export CMAKE_PREFIX_PATH+=${CONDA_PREFIX:-"$(dirname $(command -v conda))/../"}
elif [ -n "${VIRTUAL_ENV-}" ]; then
    export CMAKE_PREFIX_PATH+=${VIRTUAL_ENV:-"$(dirname $(command -v python))/../"}
fi
export USE_STATIC_MKL=1
export _GLIBCXX_USE_CXX11_ABI=${ABI}
export USE_NUMA=0
export USE_CUDA=0
export USE_MPI=0
python setup.py clean
if [ -d dist ]; then
    rm -rf dist
fi
python setup.py bdist_wheel 2>&1 | tee build.log
unset USE_MPI
unset USE_CUDA
unset USE_NUMA
unset _GLIBCXX_USE_CXX11_ABI
unset USE_STATIC_MKL
export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH_BK}
unset PYTORCH_BUILD_NUMBER
unset PYTORCH_BUILD_VERSION
python -m pip uninstall -y mkl-static mkl-include
python -m pip install dist/*.whl
cd ..
#  TorchVision
if [ $((${MODE} & 0x04)) -ne 0 ]; then
    cd vision
    python setup.py clean
    if [ -d dist ]; then
        rm -rf dist
    fi
    python setup.py bdist_wheel 2>&1 | tee build.log
    python -m pip install dist/*.whl
    cd ..
fi
#  TorchAudio
if [ $((${MODE} & 0x02)) -ne 0 ]; then
    cd audio
    python -m pip install -r requirements.txt
    python setup.py clean
    if [ -d dist ]; then
        rm -rf dist
    fi
    python setup.py bdist_wheel 2>&1 | tee build.log
    python -m pip install dist/*.whl
    cd ..
fi

#  Intel® Extension for PyTorch*
cd intel-extension-for-pytorch
python -m pip install -r requirements.txt
if [[ ! ${AOT} == "" ]]; then
    export USE_AOT_DEVLIST=${AOT}
fi
export BUILD_WITH_CPU=OFF
python setup.py clean
if [ -d dist ]; then
    rm -rf dist
fi
export ENABLE_ONEAPI_INTEGRATION=${ENABLE_ONEAPI_INTEGRATION}
python setup.py bdist_wheel 2>&1 | tee build_whl.log
unset ENABLE_ONEAPI_INTEGRATION
unset BUILD_WITH_CPU
if [[ ! ${AOT} == "" ]]; then
    unset USE_AOT_DEVLIST
fi
python -m pip install dist/*.whl
cd ..

#  Torch-CCL
if [ $((${MODE} & 0x01)) -ne 0 ]; then
    cd torch-ccl
    python setup.py clean
    export USE_SYSTEM_ONECCL=1
    export INTELONEAPIROOT=${ONEAPIROOT}
    COMPUTE_BACKEND=dpcpp python setup.py bdist_wheel 2>&1 | tee build.log
    unset INTELONEAPIROOT
    unset USE_SYSTEM_ONECCL
    python -m pip install dist/*.whl
    cd ..
fi

# Sanity Test
LIBSTDCPP=$(bash ./intel-extension-for-pytorch/scripts/tools/compilation_helper/get_libstdcpp_lib.sh)
if [[ ! "${LIBSTDCPP}" =~ "/usr/lib/" ]]; then
    export LD_PRELOAD=${LIBSTDCPP}
    echo "======================================================"
    echo "Note: Set environment variable \"export LD_PRELOAD=${LD_PRELOAD}\" to avoid the \"version \`GLIBCXX_N.N.NN' not found\" error."
    echo "======================================================"
fi
CMD="import torch; print(f'torch_cxx11_abi:     {torch._C._GLIBCXX_USE_CXX11_ABI}'); print(f'torch_version:       {torch.__version__}');"
if [ $((${MODE} & 0x04)) -ne 0 ]; then
    CMD="${CMD} import torchvision; print(f'torchvision_version: {torchvision.__version__}');"
fi
if [ $((${MODE} & 0x02)) -ne 0 ]; then
    CMD="${CMD} import torchaudio; print(f'torchaudio_version:  {torchaudio.__version__}');"
fi
CMD="${CMD} import intel_extension_for_pytorch as ipex; print(f'ipex_version:        {ipex.__version__}');"
#if [ $((${MODE} & 0x01)) -ne 0 ]; then
#    CMD="${CMD} import oneccl_bindings_for_pytorch as torch_ccl; print(f'torchccl_version:    {torch_ccl.__version__}');"
#fi
python -c "${CMD}"
