#!/usr/bin/env bash
#
# Please review the system requirements before running this script
# https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installation.html
#
set -eo pipefail

VER_IPEX=v2.1.20+xpu

if [[ $# -lt 3 ]]; then
    echo "Usage: bash $0 <DPCPPROOT> <MKLROOT> <CCLROOT> <MPIROOT> <AOT>"
    echo "DPCPPROOT, MKLROOT and CCLROOT are mandatory, should be absolute or relative path to the root directory of DPC++ compiler, oneMKL and oneCCL respectively."
    echo "AOT should be set to the text string for environment variable USE_AOT_DEVLIST. Setting it to \"none\" to disable AOT."
    exit 1
fi
DPCPP_ROOT=$1
ONEMKL_ROOT=$2
ONECCL_ROOT=$3
MPI_ROOT=$4
AOT=$5
if [[ ${AOT} == "none" ]]; then
    AOT=""
fi

# Mode: Select which components to install. PyTorch and Intel® Extension for PyTorch* are always installed.
# High bit: 8 7 6 5 4 3 2 1 :Low bit
#           | | | | | | | └- torch-ccl
#           | | | | | | └--- TorchAudio
#           | | | | | └----- TorchVision
#           | | | | └------- Rebuild LLVM
#           | | | └--------- Undefined
#           | | └----------- Undefined
#           | └------------- Undefined
#           └--------------- Undefined
MODE=0x07
if [ $# -gt 4 ]; then
    if [[ ! $6 =~ ^[0-9]+$ ]] && [[ ! $6 =~ ^0x[0-9a-fA-F]+$ ]]; then
        echo "Warning: Unexpected argument. Using default value."
    else
        MODE=$6
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
    exit 6
fi

MPI_ENV=${MPI_ROOT}/env/vars.sh
if [ ! -f ${MPI_ROOT}/env/vars.sh ]; then
    echo "oneCCL environment ${MPI_ROOT} doesn't seem to exist."
    exit 6
fi
ONEAPIROOT=${ONEMKL_ROOT}/../..

# Check existance of required Linux commands
for APP in python git patch pkg-config nproc bzip2; do
    command -v $APP > /dev/null || (echo "Error: Command \"${APP}\" not found." ; exit 4)
done

## Check existance of required libs
#if [ $((${MODE} & 0x02)) -ne 0 ]; then
#    for LIB_NAME in libpng libjpeg; do
#        pkg-config --exists $LIB_NAME > /dev/null || (echo "Error: \"${LIB_NAME}\" not found in pkg-config." ; exit 5)
#    done
#fi

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
#    exit 6
#fi

# set number of compile processes, if not already defined
MAX_JOBS_VAR=$(nproc)
if [ ! -z "${MAX_JOBS}" ]; then
    MAX_JOBS_VAR=${MAX_JOBS}
fi

# Save current directory path
BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
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

python -m pip install pyyaml
COMMIT_TORCH=$(python tools/yaml_utils.py -f dependency_version.yml -d pytorch -k commit)
COMMIT_TORCHVISION=$(python tools/yaml_utils.py -f dependency_version.yml -d torchvision -k commit)
COMMIT_TORCHAUDIO=$(python tools/yaml_utils.py -f dependency_version.yml -d torchaudio -k commit)
REPO_TORCHCCL=$(python tools/yaml_utils.py -f dependency_version.yml -d torch-ccl -k repo)
COMMIT_TORCHCCL=$(python tools/yaml_utils.py -f dependency_version.yml -d torch-ccl -k commit)
VER_LLVM=llvmorg-$(python tools/yaml_utils.py -f dependency_version.yml -d llvm -k version)
#VER_GCC=$(python tools/yaml_utils.py -f dependency_version.yml -d gcc -k min-version)
python -m pip uninstall -y pyyaml
cd ..

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
        git clone ${REPO_TORCHCCL} torch-ccl
    fi
fi
if [ ! -d llvm-project ]; then
    git clone https://github.com/llvm/llvm-project.git
fi

# Checkout required branch/commit and update submodules
cd pytorch
rm -rf * > /dev/null
git checkout . > /dev/null
if [ ! -z ${COMMIT_TORCH} ]; then
    git checkout main > /dev/null
    git pull > /dev/null
    git checkout ${COMMIT_TORCH}
fi
git submodule sync
git submodule update --init --recursive
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
cd llvm-project
if [ ! -z ${VER_LLVM} ]; then
    rm -rf * > /dev/null
    git checkout . > /dev/null
    git checkout main > /dev/null
    git pull > /dev/null
    git checkout ${VER_LLVM}
fi
git submodule sync
git submodule update --init --recursive
cd ..

# Install python dependency
python -m pip uninstall -y torch torchvision torchaudio intel-extension-for-pytorch oneccl_bind_pt
python -m pip install cmake ninja
if [ $((${MODE} & 0x04)) -ne 0 ]; then
    python -m pip install Pillow
    conda install -y libpng libjpeg-turbo -c conda-forge
fi

# Compile individual component
conda install -y make -c conda-forge
conda install -y sysroot_linux-64
conda install -y gcc==11.4 gxx==11.4 cxx-compiler -c conda-forge
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
#if [ -z ${CONDA_BUILD_SYSROOT} ]; then
#    source ${CONDA_PREFIX}/etc/conda/activate.d/activate-gcc_linux-64.sh
#    source ${CONDA_PREFIX}/etc/conda/activate.d/activate-gxx_linux-64.sh
#    source ${CONDA_PREFIX}/etc/conda/activate.d/activate-binutils_linux-64.sh
#fi
#if [[ ! -z ${LDFLAGS} ]]; then
#    read -a ldflags <<< "${LDFLAGS}"
#    for i in "${!ldflags[@]}"; do
#        if [[ "${ldflags[i]}" == "-Wl,--as-needed" ]]; then
#            unset 'ldflags[i]'
#            continue
#        fi
#        if [[ "${ldflags[i]}" =~ -Wl,-rpath,.* ]]; then
#            unset 'ldflags[i]'
#            continue
#        fi
#    done
#    function join { local IFS="$1"; shift; echo "$*"; }
#    export LDFLAGS=$(join ' ' "${ldflags[@]}")
#fi
ABI=1

#  PyTorch
cd pytorch
git apply ../intel-extension-for-pytorch/torch_patches/*.patch
python -m pip install -r requirements.txt
conda install --force-reinstall intel::mkl-static intel::mkl-include -y
export PYTORCH_BUILD_VERSION="${COMMIT_TORCH:1}.post0+cxx11.abi"
export PYTORCH_BUILD_NUMBER=0
# Ensure cmake can find python packages when using conda or virtualenv
if [ -n "${CONDA_PREFIX-}" ]; then
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(command -v conda))/../"}
elif [ -n "${VIRTUAL_ENV-}" ]; then
    export CMAKE_PREFIX_PATH=${VIRTUAL_ENV:-"$(dirname $(command -v python))/../"}
fi
export USE_STATIC_MKL=1
export _GLIBCXX_USE_CXX11_ABI=${ABI}
export USE_NUMA=0
export USE_CUDA=0
python setup.py clean
if [ -d dist ]; then
    rm -rf dist
fi
python setup.py bdist_wheel 2>&1 | tee build.log
unset USE_CUDA
unset USE_NUMA
unset _GLIBCXX_USE_CXX11_ABI
unset USE_STATIC_MKL
unset CMAKE_PREFIX_PATH
unset PYTORCH_BUILD_NUMBER
unset PYTORCH_BUILD_VERSION
conda remove mkl-static mkl-include -y
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
# don't fail on external scripts
source ${DPCPP_ENV}
source ${ONEMKL_ENV}
source ${CCL_ENV}
source ${MPI_ENV}
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

# Set compiler env
conda install -y gcc==12.3.0 gxx==12.3.0 cxx-compiler -c conda-forge
export PATH=${CONDA_PREFIX}/bin:${PATH}

#  LLVM
LLVM_ROOT="$(pwd)/llvm-release"
if [ $((${MODE} & 0x08)) -ne 0 ]; then
    if [ -d ${LLVM_ROOT} ]; then
        rm -rf ${LLVM_ROOT}
    fi
fi
cd llvm-project
if [ -d build ]; then
    rm -rf build
fi
if [ ! -d ${LLVM_ROOT} ]; then
    mkdir build
    cd build
    echo "***************************** cmake *****************************" > ../build.log
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${ABI}" -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF ../llvm 2>&1 | tee -a ../build.log
    echo "***************************** build *****************************" >> ../build.log
    cmake --build . -j ${MAX_JOBS_VAR} 2>&1 | tee -a ../build.log
    echo "**************************** install ****************************" >> ../build.log
    cmake -DCMAKE_INSTALL_PREFIX=${LLVM_ROOT} -P cmake_install.cmake 2>&1 | tee -a ../build.log
    #xargs rm -rf < install_manifest.txt
    cd ..
    rm -rf build
    ln -s ${LLVM_ROOT}/bin/llvm-config ${LLVM_ROOT}/bin/llvm-config-13
fi
cd ..
PATH_BK=${PATH}
LD_LIBRARY_PATH_BK=${LD_LIBRARY_PATH}
export PATH=${LLVM_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${LLVM_ROOT}/lib:$LD_LIBRARY_PATH

#  Intel® Extension for PyTorch*
cd intel-extension-for-pytorch
python -m pip install -r requirements.txt
if [[ ! ${AOT} == "" ]]; then
    export USE_AOT_DEVLIST=${AOT}
fi
export USE_LLVM=${LLVM_ROOT}
export LLVM_DIR=${USE_LLVM}/lib/cmake/llvm
export DNNL_GRAPH_BUILD_COMPILER_BACKEND=1
export IPEX_GPU_EXTRA_BUILD_OPTION="--gcc-install-dir=${CONDA_PREFIX}/lib/gcc/x86_64-conda-linux-gnu/12.3.0 -fuse-ld=lld -lrt -lpthread"
python setup.py clean
if [ -d dist ]; then
    rm -rf dist
fi
python setup.py bdist_wheel 2>&1 | tee build.log
unset IPEX_GPU_EXTRA_BUILD_OPTION
unset DNNL_GRAPH_BUILD_COMPILER_BACKEND
unset LLVM_DIR
unset USE_LLVM
if [[ ! ${AOT} == "" ]]; then
    unset USE_AOT_DEVLIST
fi
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH_BK}
export PATH=${PATH_BK}
python -m pip install dist/*.whl
cd ..

#  Torch-CCL
if [ $((${MODE} & 0x01)) -ne 0 ]; then
    cd torch-ccl
    python setup.py clean
    if [ -d ${CONDA_PREFIX}/lib/gcc/x86_64-conda-linux-gnu ]; then
        export DPCPP_GCC_INSTALL_DIR="${CONDA_PREFIX}/lib/gcc/x86_64-conda-linux-gnu/12.3.0"
    fi
    export INTELONEAPIROOT=${ONEAPIROOT}
    COMPUTE_BACKEND=dpcpp python setup.py bdist_wheel 2>&1 | tee build.log
    unset INTELONEAPIROOT
    if [ -d ${CONDA_PREFIX}/lib/gcc/x86_64-conda-linux-gnu ]; then
        unset DPCPP_GCC_INSTALL_DIR
    fi
    python -m pip install dist/*.whl
    cd ..
fi

export LD_PRELOAD=$(bash ./intel-extension-for-pytorch/tools/get_libstdcpp_lib.sh)

# Sanity Test
echo "======================================================"
echo "Note: Set environment variable \"export LD_PRELOAD=${LD_PRELOAD}\" to avoid the \"version \`GLIBCXX_N.N.NN' not found\" error."
echo "======================================================"
CMD="import torch; print(f'torch_cxx11_abi:     {torch._C._GLIBCXX_USE_CXX11_ABI}'); print(f'torch_version:       {torch.__version__}');"
if [ $((${MODE} & 0x04)) -ne 0 ]; then
    CMD="${CMD} import torchvision; print(f'torchvision_version: {torchvision.__version__}');"
fi
if [ $((${MODE} & 0x02)) -ne 0 ]; then
    CMD="${CMD} import torchaudio; print(f'torchaudio_version:  {torchaudio.__version__}');"
fi
CMD="${CMD} import intel_extension_for_pytorch as ipex; print(f'ipex_version:        {ipex.__version__}');"
if [ $((${MODE} & 0x01)) -ne 0 ]; then
    CMD="${CMD} import oneccl_bindings_for_pytorch as torch_ccl; print(f'torchccl_version:    {torch_ccl.__version__}');"
fi
python -c "${CMD}"