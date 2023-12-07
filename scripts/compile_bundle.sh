#!/usr/bin/env bash
#
# Please review the system requirements before running this script
# https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installation.html
#
set -eo pipefail

VER_IPEX=v2.1.10+xpu

if [[ $# -lt 3 ]]; then
    echo "Usage: bash $0 <DPCPPROOT> <MKLROOT> <AOT>"
    echo "DPCPPROOT and MKLROOT are mandatory, should be absolute or relative path to the root directory of DPC++ compiler and oneMKL respectively."
    echo "AOT should be set to the text string for environment variable USE_AOT_DEVLIST. Setting it to \"none\" to disable AOT."
    exit 1
fi
DPCPP_ROOT=$1
ONEMKL_ROOT=$2
AOT=$3
if [[ ${AOT} == "none" ]]; then
    AOT=""
fi

# Mode: Select which components to install. PyTorch and Intel® Extension for PyTorch* are always installed.
# High bit: 8 7 6 5 4 3 2 1 :Low bit
#           | | | | | | | └- TorchAudio
#           | | | | | | └--- TorchVision
#           | | | | | └----- Rebuild LLVM
#           | | | | └------- Undefined
#           | | | └--------- Undefined
#           | | └----------- Undefined
#           | └------------- Undefined
#           └--------------- Undefined
MODE=0x03
if [ $# -gt 3 ]; then
    if [[ ! $4 =~ ^[0-9]+$ ]] && [[ ! $4 =~ ^0x[0-9a-fA-F]+$ ]]; then
        echo "Warning: Unexpected argument. Using default value."
    else
        MODE=$4
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
    git clone https://github.com/intel/intel-extension-for-pytorch.git
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
VER_TORCH=$(python tools/yaml_utils.py -f dependency_version.yml -d pytorch -k commit)
VER_TORCHVISION=$(python tools/yaml_utils.py -f dependency_version.yml -d torchvision -k commit)
VER_TORCHAUDIO=$(python tools/yaml_utils.py -f dependency_version.yml -d torchaudio -k commit)
VER_LLVM=llvmorg-$(python tools/yaml_utils.py -f dependency_version.yml -d llvm -k version)
#VER_GCC=$(python tools/yaml_utils.py -f dependency_version.yml -d gcc -k min-version)
python -m pip uninstall -y pyyaml
cd ..

if [ ! -d pytorch ]; then
    git clone https://github.com/pytorch/pytorch.git
fi
if [ $((${MODE} & 0x02)) -ne 0 ]; then
    if [ ! -d vision ]; then
        git clone https://github.com/pytorch/vision.git
    fi
fi
if [ $((${MODE} & 0x01)) -ne 0 ]; then
    if [ ! -d audio ]; then
        git clone https://github.com/pytorch/audio.git
    fi
fi
if [ ! -d llvm-project ]; then
    git clone https://github.com/llvm/llvm-project.git
fi

# Checkout required branch/commit and update submodules
cd pytorch
rm -rf * > /dev/null
git checkout . > /dev/null
if [ ! -z ${VER_TORCH} ]; then
    git checkout main > /dev/null
    git pull > /dev/null
    git checkout ${VER_TORCH}
fi
git submodule sync
git submodule update --init --recursive
cd ..
if [ $((${MODE} & 0x02)) -ne 0 ]; then
    cd vision
    if [ ! -z ${VER_TORCHVISION} ]; then
        rm -rf * > /dev/null
        git checkout . > /dev/null
        git checkout main > /dev/null
        git pull > /dev/null
        git checkout ${VER_TORCHVISION}
    fi
    git submodule sync
    git submodule update --init --recursive
	cd ..
fi
if [ $((${MODE} & 0x01)) -ne 0 ]; then
    cd audio
    if [ ! -z ${VER_TORCHAUDIO} ]; then
        rm -rf * > /dev/null
        git checkout . > /dev/null
        git checkout main > /dev/null
        git pull > /dev/null
        git checkout ${VER_TORCHAUDIO}
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
python -m pip uninstall -y torch torchvision torchaudio intel-extension-for-pytorch
python -m pip install cmake ninja
if [ $((${MODE} & 0x02)) -ne 0 ]; then
    python -m pip install Pillow
    conda install -y libpng libjpeg-turbo -c conda-forge
fi

# Compile individual component
conda install -y make -c conda-forge
conda install -y sysroot_linux-64
conda install -y gcc==11.4 gxx==11.4 cxx-compiler -c conda-forge
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
ABI=1

#  PyTorch
cd pytorch
git apply ../intel-extension-for-pytorch/torch_patches/*.patch
python -m pip install -r requirements.txt
python -m pip install mkl-static mkl-include
mv version.txt version.txt.bk
echo "${VER_TORCH:1}a0" > version.txt
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
mv version.txt.bk version.txt
python -m pip uninstall -y mkl-static mkl-include
python -m pip install dist/*.whl
cd ..
#  TorchVision
if [ $((${MODE} & 0x02)) -ne 0 ]; then
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
if [ $((${MODE} & 0x04)) -ne 0 ]; then
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
python -m pip install dist/*.whl
export LD_PRELOAD=$(bash ./tools/get_libstdcpp_lib.sh)
cd ..

# Sanity Test
echo "======================================================"
echo "Note: Set environment variable \"export LD_PRELOAD=${LD_PRELOAD}\" to avoid the \"version \`GLIBCXX_N.N.NN' not found\" error."
echo "======================================================"
CMD="import torch; print(f'torch_cxx11_abi:     {torch._C._GLIBCXX_USE_CXX11_ABI}'); print(f'torch_version:       {torch.__version__}');"
if [ $((${MODE} & 0x02)) -ne 0 ]; then
    CMD="${CMD} import torchvision; print(f'torchvision_version: {torchvision.__version__}');"
fi
if [ $((${MODE} & 0x01)) -ne 0 ]; then
    CMD="${CMD} import torchaudio; print(f'torchaudio_version:  {torchaudio.__version__}');"
fi
CMD="${CMD} import intel_extension_for_pytorch as ipex; print(f'ipex_version:        {ipex.__version__}');"
python -c "${CMD}"
