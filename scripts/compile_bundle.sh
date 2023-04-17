#!/bin/bash
set -x
set -e

VER_LLVM="llvmorg-13.0.0"
VER_PYTORCH="v1.13.1"
VER_TORCHVISION="v0.14.1"
VER_TORCHAUDIO="v0.13.1"
VER_IPEX="xpu-master"

if [[ $# -lt 2 ]]; then
    echo "Usage: bash $0 <DPCPPROOT> <MKLROOT> [AOT]"
    echo "DPCPPROOT and MKLROOT are mandatory, should be absolute or relative path to the root directory of DPC++ compiler and oneMKL respectively."
    echo "AOT is optional, should be the text string for environment variable USE_AOT_DEVLIST."
    exit 1
fi

CC="${CC:-/usr/bin/gcc}"

if ${CC} -v 2>&1 | grep "version 11"; then
    echo "gcc version 11 is installed."
else
    echo "gcc version 11 is not installed."
    exit 2
fi

DPCPP_ROOT=$1
ONEMKL_ROOT=$2
AOT=""
if [[ $# -ge 3 ]]; then
    AOT=$3
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
which python > /dev/null 2>&1
if [[ $? -ne 0 ]]; then
    echo "Error: linux command \"python\" not found."
    exit 4
fi
which git > /dev/null 2>&1
if [[ $? -ne 0 ]]; then
    echo "Error: linux command \"git\" not found."
    exit 5
fi
which patch > /dev/null 2>&1
if [[ $? -ne 0 ]]; then
    echo "Error: linux command \"patch\" not found."
    exit 6
fi
which pkg-config > /dev/null 2>&1
if [[ $? -ne 0 ]]; then
    echo "Error: linux command \"pkg-config\" not found."
    exit 7
fi
env | grep CONDA_PREFIX > /dev/null 2>&1
CONDA=$?

# Save current directory path
BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${BASEFOLDER}

# Checkout individual components
if [ ! -d llvm-project ]; then
    git clone https://github.com/llvm/llvm-project.git
fi
if [ ! -d pytorch ]; then
    git clone https://github.com/pytorch/pytorch.git
fi
if [ ! -d vision ]; then
    git clone https://github.com/pytorch/vision.git
fi
if [ ! -d audio ]; then
    git clone https://github.com/pytorch/audio.git
fi
if [ ! -d intel-extension-for-pytorch ]; then
    git clone https://github.com/intel/intel-extension-for-pytorch.git
fi

# Checkout required branch/commit and update submodules
cd llvm-project
git checkout ${VER_LLVM}
git submodule sync
git submodule update --init --recursive
cd ../pytorch
git checkout ${VER_PYTORCH}
git submodule sync
git submodule update --init --recursive
cd ../vision
git checkout ${VER_TORCHVISION}
git submodule sync
git submodule update --init --recursive
cd ../audio
git checkout ${VER_TORCHAUDIO}
git submodule sync
git submodule update --init --recursive
cd ../intel-extension-for-pytorch
git checkout ${VER_IPEX}
git submodule sync
git submodule update --init --recursive

# Install the basic dependency cmake
python -m pip install cmake

# Compile individual component
#  LLVM
cd ../llvm-project

if [ -d build ]; then
    rm -rf build
fi
mkdir build
cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1" -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF ../llvm/
cmake --build . -j $(nproc)
LLVM_ROOT=`pwd`/../release
if [ -d ${LLVM_ROOT} ]; then
	rm -rf ${LLVM_ROOT}
fi
cmake -DCMAKE_INSTALL_PREFIX=${LLVM_ROOT}/../release/ -P cmake_install.cmake
#xargs rm -rf < install_manifest.txt
ln -s ${LLVM_ROOT}/bin/llvm-config ${LLVM_ROOT}/bin/llvm-config-13
export PATH=${LLVM_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${LLVM_ROOT}/lib:$LD_LIBRARY_PATH
cd ..

#  PyTorch
cd ../pytorch

git stash
git clean -f
git apply ../intel-extension-for-pytorch/torch_patches/*.patch
python -m pip install astunparse numpy ninja pyyaml mkl-static mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
export USE_LLVM=${LLVM_ROOT}
export LLVM_DIR=${USE_LLVM}/lib/cmake/llvm
if [[ ${CONDA} -eq 0 ]]; then
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
else
    export CMAKE_PREFIX_PATH=${VIRTUAL_ENV:-"$(dirname $(which python))/../"}
fi
export USE_STATIC_MKL=1
export _GLIBCXX_USE_CXX11_ABI=1
export USE_NUMA=0
export USE_CUDA=0
python setup.py clean
python setup.py bdist_wheel 2>&1 | tee build.log
unset USE_CUDA
unset USE_NUMA
unset _GLIBCXX_USE_CXX11_ABI
unset USE_STATIC_MKL
unset CMAKE_PREFIX_PATH
unset LLVM_DIR
unset USE_LLVM
python -m pip uninstall -y mkl-static mkl-include
python -m pip install --force-reinstall dist/*.whl

#  TorchVision
cd ../vision

conda install -y libpng jpeg
python setup.py clean
python setup.py bdist_wheel 2>&1 | tee build.log
python -m pip install --force-reinstall --no-deps dist/*.whl
python -m pip install Pillow

#  TorchAudio
cd ../audio

conda install -y bzip2
python -m pip install -r requirements.txt
python setup.py clean
python setup.py bdist_wheel 2>&1 | tee build.log
python -m pip install --force-reinstall --no-deps dist/*.whl

#  Intel® Extension for PyTorch*
cd ../intel-extension-for-pytorch

python -m pip install -r requirements.txt
source ${DPCPP_ENV}
source ${ONEMKL_ENV}
if [[ ! ${AOT} == "" ]]; then
    export USE_AOT_DEVLIST=${AOT}
fi
export USE_LLVM=${LLVM_ROOT}
export LLVM_DIR=${USE_LLVM}/lib/cmake/llvm
export DNNL_GRAPH_BUILD_COMPILER_BACKEND=1
python setup.py clean
python setup.py bdist_wheel 2>&1 | tee build.log
unset DNNL_GRAPH_BUILD_COMPILER_BACKEND
unset LLVM_DIR
unset USE_LLVM
if [[ ! ${AOT} == "" ]]; then
    unset USE_AOT_DEVLIST
fi
python -m pip install --force-reinstall dist/*.whl

# Sanity Test
cd ..
python -c "import torch; import torchvision; import torchaudio; import intel_extension_for_pytorch as ipex; print(f'torch_cxx11_abi:     {torch.compiled_with_cxx11_abi()}'); print(f'torch_version:       {torch.__version__}'); print(f'torchvision_version: {torchvision.__version__}'); print(f'torchaudio_version:  {torchaudio.__version__}'); print(f'ipex_version:        {ipex.__version__}');"
