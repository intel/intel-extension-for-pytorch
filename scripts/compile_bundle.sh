#!/bin/bash
set -x
set -e

VER_LLVM="llvmorg-13.0.0"
VER_PYTORCH=""
VER_TORCHVISION=""
VER_TORCHAUDIO=""
VER_IPEX="v2.0.100+cpu"

# Check existance of required Linux commands
for CMD in gcc g++ python git nproc; do
    command -v ${CMD} || (echo "Error: Command \"${CMD}\" not found." ; exit 4)
done
echo "You are using GCC: $(gcc --version | grep gcc)"

MAX_JOBS_VAR=$(nproc)
if [ ! -z "${MAX_JOBS}" ]; then
    MAX_JOBS_VAR=${MAX_JOBS}
fi

# Save current directory path
BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${BASEFOLDER}
# Checkout individual components
if [ ! -d llvm-project ]; then
    git clone https://github.com/llvm/llvm-project.git
fi
if [ ! -d intel-extension-for-pytorch ]; then
    git clone https://github.com/intel/intel-extension-for-pytorch.git
fi

# Checkout required branch/commit and update submodules
cd llvm-project
if [ ! -z ${VER_LLVM} ]; then
    git checkout ${VER_LLVM}
fi
git submodule sync
git submodule update --init --recursive
cd ../intel-extension-for-pytorch
if [ ! -z ${VER_IPEX} ]; then
    git checkout ${VER_IPEX}
fi
git submodule sync
git submodule update --init --recursive

# Install dependencies
python -m pip install cmake
python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

# Compile individual component
#  LLVM
cd ../llvm-project
if [ -d build ]; then
    rm -rf build
fi
mkdir build
cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${ABI}" -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF ../llvm/
cmake --build . -j ${MAX_JOBS_VAR}
LLVM_ROOT="$(pwd)/../release"
if [ -d ${LLVM_ROOT} ]; then
    rm -rf ${LLVM_ROOT}
fi
cmake -DCMAKE_INSTALL_PREFIX=${LLVM_ROOT}/../release/ -P cmake_install.cmake
#xargs rm -rf < install_manifest.txt
ln -s ${LLVM_ROOT}/bin/llvm-config ${LLVM_ROOT}/bin/llvm-config-13
export PATH=${LLVM_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${LLVM_ROOT}/lib:$LD_LIBRARY_PATH
cd ..
#  Intel® Extension for PyTorch*
cd ../intel-extension-for-pytorch
python -m pip install -r requirements.txt
export USE_LLVM=${LLVM_ROOT}
export LLVM_DIR=${USE_LLVM}/lib/cmake/llvm
export DNNL_GRAPH_BUILD_COMPILER_BACKEND=1
python setup.py clean
python setup.py bdist_wheel 2>&1 | tee build.log
unset DNNL_GRAPH_BUILD_COMPILER_BACKEND
unset LLVM_DIR
unset USE_LLVM
python -m pip install --force-reinstall dist/*.whl

# Sanity Test
cd ..
python -c "import torch; import torchvision; import torchaudio; import intel_extension_for_pytorch as ipex; print(f'torch_cxx11_abi:     {torch._C._GLIBCXX_USE_CXX11_ABI}'); print(f'torch_version:       {torch.__version__}'); print(f'torchvision_version: {torchvision.__version__}'); print(f'torchaudio_version:  {torchaudio.__version__}'); print(f'ipex_version:        {ipex.__version__}');"
