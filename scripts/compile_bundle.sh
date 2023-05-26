#!/bin/bash
set -x
set -e

VER_IPEX="v2.1.0.dev+cpu.llm"

# Check existance of required Linux commands
for CMD in python git nproc conda; do
    command -v ${CMD} || (echo "Error: Command \"${CMD}\" not found." ; exit 4)
done

MAX_JOBS_VAR=$(nproc)
if [ ! -z "${MAX_JOBS}" ]; then
    MAX_JOBS_VAR=${MAX_JOBS}
fi

conda install -y gcc==12.3 gxx==12.3 cxx-compiler -c conda-forge

# Save current directory path
BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${BASEFOLDER}
# Checkout individual components
if [ ! -d intel-extension-for-pytorch ]; then
    git clone https://github.com/intel/intel-extension-for-pytorch.git
fi

# Checkout required branch/commit and update submodules
if [ ! -d cmake ]; then
    wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/cmake-16.0.6.src.tar.xz
    tar -xvf cmake-16.0.6.src.tar.xz
    mv cmake-16.0.6.src cmake
fi
if [ ! -d llvm ]; then
    wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/llvm-16.0.6.src.tar.xz
    tar -xvf llvm-16.0.6.src.tar.xz
    mv llvm-16.0.6.src llvm
fi
cd intel-extension-for-pytorch
if [ ! -z ${VER_IPEX} ]; then
    git checkout ${VER_IPEX}
fi
git submodule sync
git submodule update --init --recursive
cd ..

# Install dependencies
python -m pip install cmake
python -m pip install torch==2.1.0.dev20230711+cpu torchvision==0.16.0.dev20230711+cpu torchaudio==2.1.0.dev20230711+cpu --index-url https://download.pytorch.org/whl/nightly/cpu
ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

# Compile individual component
export CC=${CONDA_PREFIX}/bin/gcc
export CXX=${CONDA_PREFIX}/bin/g++
export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so

#  LLVM
LLVM_ROOT="$(pwd)/release"
if [ -d ${LLVM_ROOT} ]; then
    rm -rf ${LLVM_ROOT}
fi
mkdir ${LLVM_ROOT}
if [ -d build ]; then
    rm -rf build
fi
mkdir build
cd build
cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=${LLVM_ROOT} -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${ABI}" -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF ../llvm/
make install -j $MAX_JOBS
ln -s ${LLVM_ROOT}/bin/llvm-config ${LLVM_ROOT}/bin/llvm-config-13
export PATH=${LLVM_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${LLVM_ROOT}/lib:$LD_LIBRARY_PATH
#  Intel® Extension for PyTorch*
cd ../intel-extension-for-pytorch
python -m pip install -r requirements.txt
export USE_LLVM=${LLVM_ROOT}
export LLVM_DIR=${USE_LLVM}/lib/cmake/llvm
export DNNL_GRAPH_BUILD_COMPILER_BACKEND=1
export IPEX_VERSION=2.1.0.dev0+cpu.llm
export IPEX_VERSIONED_BUILD=0
CXXFLAGS_BK=${CXXFLAGS}
export CXXFLAGS="${CXXFLAGS} -D__STDC_FORMAT_MACROS"
python setup.py clean
python setup.py bdist_wheel 2>&1 | tee build.log
export CXXFLAGS=${CXXFLAGS_BK}
unset IPEX_VERSIONED_BUILD
unset IPEX_VERSION
unset DNNL_GRAPH_BUILD_COMPILER_BACKEND
unset LLVM_DIR
unset USE_LLVM
python -m pip install --force-reinstall dist/*.whl

# Sanity Test
cd ..
python -c "import torch; import torchvision; import torchaudio; import intel_extension_for_pytorch as ipex; print(f'torch_cxx11_abi:     {torch._C._GLIBCXX_USE_CXX11_ABI}'); print(f'torch_version:       {torch.__version__}'); print(f'torchvision_version: {torchvision.__version__}'); print(f'torchaudio_version:  {torchaudio.__version__}'); print(f'ipex_version:        {ipex.__version__}');"
