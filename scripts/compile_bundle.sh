#!/bin/bash
set -eo pipefail

VER_TORCH=2.2.0.dev20231006+cpu
VER_TORCHVISION=0.17.0.dev20231006+cpu
VER_TORCHAUDIO=2.2.0.dev20231006+cpu
VER_LLVM=llvmorg-16.0.6
VER_IPEX=llm_feature_branch
VER_GCC=12.3.0

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
if [ $# -gt 0 ]; then
    if [[ ! $1 =~ ^[0-9]+$ ]] && [[ ! $1 =~ ^0x[0-9a-fA-F]+$ ]]; then
        echo "Warning: Unexpected argument. Using default value."
    else
        MODE=$1
    fi
fi

# Check existance of required Linux commands
for CMD in conda git nproc gcc g++ make; do
    command -v ${CMD} > /dev/null || (echo "Error: Command \"${CMD}\" not found." ; exit 1)
done

function ver_compare() {
    VER_MAJOR_CUR=$(echo $1 | cut -d "." -f 1)
    VER_MINOR_CUR=$(echo $1 | cut -d "." -f 2)
    VER_PATCH_CUR=$(echo $1 | cut -d "." -f 3)
    VER_MAJOR_REQ=$(echo $2 | cut -d "." -f 1)
    VER_MINOR_REQ=$(echo $2 | cut -d "." -f 2)
    VER_PATCH_REQ=$(echo $2 | cut -d "." -f 3)
    RET=0
    if [[ ${VER_MAJOR_CUR} -lt ${VER_MAJOR_REQ} ]]; then
        RET=1
    else
        if [[ ${VER_MAJOR_CUR} -eq ${VER_MAJOR_REQ} ]] &&
           [[ ${VER_MINOR_CUR} -lt ${VER_MINOR_REQ} ]]; then
            RET=2
        else
            if [[ ${VER_MAJOR_CUR} -eq ${VER_MAJOR_REQ} ]] &&
               [[ ${VER_MINOR_CUR} -eq ${VER_MINOR_REQ} ]] &&
               [[ ${VER_PATCH_CUR} -lt ${VER_PATCH_REQ} ]]; then
                RET=3
            fi
        fi
    fi
    echo ${RET}
}
VER_COMP=$(ver_compare $(gcc -dumpfullversion) ${VER_GCC})
GCC_CONDA=0
if [ ${VER_COMP} -ne 0 ]; then
    echo -e '\a'
    echo "Warning: GCC version equal to or newer than ${VER_GCC} is required."
    echo "         Found GCC version $(gcc -dumpfullversion)"
    echo "         Installing gcc and g++ 12.3 with conda"
    echo ""
    GCC_CONDA=1
    conda install -y gcc==12.3 gxx==12.3 cxx-compiler -c conda-forge
    conda update -y sysroot_linux-64
fi

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
if [ ! -z "${VER_LLVM}" ]; then
    git stash > /dev/null
    git clean -fd > /dev/null
    git checkout main > /dev/null
    git pull > /dev/null
    git checkout ${VER_LLVM}
fi
git submodule sync
git submodule update --init --recursive
cd ..
cd intel-extension-for-pytorch
if [ ! -z "${VER_IPEX}" ]; then
    git stash > /dev/null
    git clean -fd > /dev/null
    git checkout main > /dev/null
    git pull > /dev/null
    git checkout ${VER_IPEX}
fi
git submodule sync
git submodule update --init --recursive
cd ..

# Install dependencies
python -m pip install cmake
python -m pip uninstall -y torch torchvision torchaudio intel-extension-for-pytorch
python -m pip install torch==${VER_TORCH} --index-url https://download.pytorch.org/whl/nightly/cpu
if [ $((${MODE} & 0x02)) -ne 0 ]; then
    python -m pip install torchvision==${VER_TORCHVISION} --index-url https://download.pytorch.org/whl/nightly/cpu
fi
if [ $((${MODE} & 0x01)) -ne 0 ]; then
    python -m pip install torchaudio==${VER_TORCHAUDIO} --index-url https://download.pytorch.org/whl/nightly/cpu
fi
ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

# Compile individual component
if [[ ${GCC_CONDA} -eq 1 ]]; then
    export CC=${CONDA_PREFIX}/bin/gcc
    export CXX=${CONDA_PREFIX}/bin/g++
    export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
fi

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
export PATH=${LLVM_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${LLVM_ROOT}/lib:$LD_LIBRARY_PATH
cd ..
#  Intel® Extension for PyTorch*
cd intel-extension-for-pytorch
python -m pip install -r requirements.txt
export LLVM_DIR=${LLVM_ROOT}/lib/cmake/llvm
export DNNL_GRAPH_BUILD_COMPILER_BACKEND=1
CXXFLAGS_BK=${CXXFLAGS}
export CXXFLAGS="${CXXFLAGS} -D__STDC_FORMAT_MACROS"
python setup.py clean
python setup.py bdist_wheel 2>&1 | tee build.log
export CXXFLAGS=${CXXFLAGS_BK}
unset DNNL_GRAPH_BUILD_COMPILER_BACKEND
unset LLVM_DIR
python -m pip uninstall -y mkl-static mkl-include
python -m pip install dist/*.whl
cd ..

# Sanity Test
if [[ ${GCC_CONDA} -eq 1 ]]; then
    export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so
    echo "Note: Should you experience \"version \`GLIBCXX_N.N.NN' not found\" error, run command \"export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so\" and try again."
fi
CMD="import torch; print(f'torch_cxx11_abi:     {torch._C._GLIBCXX_USE_CXX11_ABI}'); print(f'torch_version:       {torch.__version__}');"
if [ $((${MODE} & 0x02)) -ne 0 ]; then
    CMD="${CMD} import torchvision; print(f'torchvision_version: {torchvision.__version__}');"
fi
if [ $((${MODE} & 0x01)) -ne 0 ]; then
    CMD="${CMD} import torchaudio; print(f'torchaudio_version:  {torchaudio.__version__}');"
fi
CMD="${CMD} import intel_extension_for_pytorch as ipex; print(f'ipex_version:        {ipex.__version__}');"
python -c "${CMD}"
