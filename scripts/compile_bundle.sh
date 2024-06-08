#!/bin/bash
set -eo pipefail

VER_IPEX=main

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
if [ $# -gt 0 ]; then
    if [[ ! $1 =~ ^[0-9]+$ ]] && [[ ! $1 =~ ^0x[0-9a-fA-F]+$ ]]; then
        echo "Warning: Unexpected argument. Using default value."
    else
        MODE=$1
    fi
fi

# Check existance of required Linux commands
for CMD in git nproc; do
    command -v ${CMD} > /dev/null || (echo "Error: Command \"${CMD}\" not found." ; exit 1)
done

# Save current directory path
BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${BASEFOLDER}

# Checkout individual components
if [ ! -d intel-extension-for-pytorch ]; then
    git clone https://github.com/intel/intel-extension-for-pytorch.git intel-extension-for-pytorch
fi
cd intel-extension-for-pytorch
if [ ! -z "${VER_IPEX}" ]; then
    rm -rf * > /dev/null
    git checkout . > /dev/null
    git checkout main > /dev/null
    git pull > /dev/null
    git checkout ${VER_IPEX}
fi
git submodule sync
git submodule update --init --recursive

python -m pip install pyyaml
VER_TORCH=$(python tools/yaml_utils.py -f dependency_version.yml -d pytorch -k version)
VER_TORCHVISION=$(python tools/yaml_utils.py -f dependency_version.yml -d torchvision -k version)
VER_TORCHAUDIO=$(python tools/yaml_utils.py -f dependency_version.yml -d torchaudio -k version)
REPO_TORCHCCL=$(python tools/yaml_utils.py -f dependency_version.yml -d torch-ccl -k repo)
COMMIT_TORCHCCL=$(python tools/yaml_utils.py -f dependency_version.yml -d torch-ccl -k commit)
REPO_LLVM=$(python tools/yaml_utils.py -f dependency_version.yml -d llvm -k repo)
VER_LLVM=$(python tools/yaml_utils.py -f dependency_version.yml -d llvm -k commit)
VER_GCC=$(python tools/yaml_utils.py -f dependency_version.yml -d gcc -k min-version)
python -m pip uninstall -y pyyaml
cd ..

if [ ! -d llvm-project ]; then
    git clone ${REPO_LLVM} llvm-project
fi
cd llvm-project
if [ ! -z "${VER_LLVM}" ]; then
    rm -rf * > /dev/null
    git checkout . > /dev/null
    git checkout main > /dev/null
    git pull > /dev/null
    git checkout ${VER_LLVM}
fi
git submodule sync
git submodule update --init --recursive
cd ..

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
set +e
command -v conda > /dev/null
EXIST_CONDA=$?

GCC_CONDA=0
command -v gcc > /dev/null
EXIST_CC=$?
command -v g++ > /dev/null
EXIST_CXX=$?
set -e
if [ ${EXIST_CC} -gt 0 ] || [ ${EXIST_CXX} -gt 0 ]; then
    echo -e '\a'
    echo "Warning: GCC not found."
    echo "         Installing gcc and g++ 12.3 from conda..."
    echo ""
    GCC_CONDA=1
else
    VER_COMP=$(ver_compare $(gcc -dumpfullversion) ${VER_GCC})
    if [ ${VER_COMP} -ne 0 ]; then
        echo -e '\a'
        echo "Warning: GCC version equal to or newer than ${VER_GCC} is required."
        echo "         Found GCC version $(gcc -dumpfullversion)."
        echo "         Installing gcc and g++ 12.3 from conda..."
        echo ""
        GCC_CONDA=1
    else
        DIR_GCC=$(which gcc)
        if [ ! -z ${CONDA_PREFIX} ] && [[ ${DIR_GCC} =~ ${CONDA_PREFIX} ]]; then
            GCC_CONDA=2
        fi
    fi
fi

MAX_JOBS_VAR=$(nproc)
if [ ! -z "${MAX_JOBS}" ]; then
    MAX_JOBS_VAR=${MAX_JOBS}
fi

# Install dependencies
python -m pip install cmake==3.28.4 make

# Compare the torch torchvision and torchaudio version
function ver_compare_eq() {
    RET=0
    if  [[ "$1" == "$2" ]]; then
            RET=1
    fi
    echo ${RET}
}
# Check if PyTorch is installed, if installed, compare the current version with compiling version
if python -c "import torch; print(torch.__version__)" &> /dev/null; then
    torch_version=$(python -c "import torch; print(torch.__version__)")
    VER_COMP_TORCH=$(ver_compare_eq ${torch_version} ${VER_TORCH})
    if python -c "import torchvision; print(torchvision.__version__)" &> /dev/null; then
        torchvision_version=$(python -c "import torchvision; print(torchvision.__version__)")
        VER_COMP_VISION=$(ver_compare_eq ${torchvision_version} ${VER_TORCHVISION})
    fi
    if python -c "import torchaudio; print(torchaudio.__version__)" &> /dev/null; then
        torchaudio_version=$(python -c "import torchaudio; print(torchaudio.__version__)")
        VER_COMP_AUDIO=$(ver_compare_eq ${torchaudio_version} ${VER_TORCHAUDIO})
    fi
    if [ ${VER_COMP_TORCH} -ne 1 ] || [ ${VER_COMP_VISION} -ne 1 ] || [ ${VER_COMP_AUDIO} -ne 1 ]; then
        if [ ! -z ${VER_COMP_TORCH} ] && [ ${VER_COMP_TORCH} -ne 1 ]; then
            printf "WARNING: Found installed torch version ${torch_version}, the required version for compiling is ${VER_TORCH}\\n"
        fi
        if [ ! -z ${VER_COMP_VISION} ] && [ ${VER_COMP_VISION} -ne 1 ]; then
            printf "         Found installed torchvision version ${torchvision_version}, the required version for compiling is ${VER_TORCHVISION}\\n"
        fi
        if [ ! -z ${VER_COMP_AUDIO} ] && [ ${VER_COMP_AUDIO} -ne 1 ]; then
            printf "         Found installed torchaudio version ${torchaudio_version}, the required version for compiling is ${VER_COMP_AUDIO}\\n"
        fi
        printf "Continue to run the compile script will replace the current torch/torchvision/torchaudio package\\n"
        printf "Are sure you want to continue the compilation? yes for continue, no for quit. [yes|no]\\n"
        printf "[yes] >>> "
        read -r ans
        ans=$(echo "${ans}" | tr '[:lower:]' '[:upper:]')
        if [ ! -z ${ans} ] && [ "${ans}" != "YES" ] && [ "${ans}" != "Y" ]; then
            printf "Aborting compilation\\n"
            exit 2
        fi
    fi
fi

python -m pip uninstall -y torch torchvision torchaudio intel-extension-for-pytorch oneccl_bind_pt
set +e
echo ${VER_TORCH} | grep "dev" > /dev/null
TORCH_DEV=$?
set -e
URL_NIGHTLY=""
if [ ${TORCH_DEV} -eq 0 ]; then
    URL_NIGHTLY="nightly/"
fi
python -m pip install torch==${VER_TORCH} --index-url https://download.pytorch.org/whl/${URL_NIGHTLY}cpu
if [ $((${MODE} & 0x04)) -ne 0 ]; then
    python -m pip install torchvision==${VER_TORCHVISION} --index-url https://download.pytorch.org/whl/${URL_NIGHTLY}cpu
fi
if [ $((${MODE} & 0x02)) -ne 0 ]; then
    python -m pip install torchaudio==${VER_TORCHAUDIO} --index-url https://download.pytorch.org/whl/${URL_NIGHTLY}cpu
fi
if [ $((${MODE} & 0x01)) -ne 0 ]; then
    if [ ! -d torch-ccl ]; then
        git clone ${REPO_TORCHCCL} torch-ccl
    fi
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
ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

# Compile individual component
if [ ${GCC_CONDA} -eq 1 ]; then
    if [ ${EXIST_CONDA} -gt 0 ]; then
        echo "Command \"conda\" not found. Exit."
        exit 2
    fi
    conda install -y sysroot_linux-64
    conda install -y gcc==12.3 gxx==12.3 cxx-compiler -c conda-forge
fi
if [ ${GCC_CONDA} -ge 1 ]; then
    if [ -z ${CONDA_BUILD_SYSROOT} ]; then
        source ${CONDA_PREFIX}/etc/conda/activate.d/activate-gcc_linux-64.sh
        source ${CONDA_PREFIX}/etc/conda/activate.d/activate-gxx_linux-64.sh
        source ${CONDA_PREFIX}/etc/conda/activate.d/activate-binutils_linux-64.sh
    fi
fi
if [[ ! -z ${LDFLAGS} ]]; then
    read -a ldflags <<< "${LDFLAGS}"
    for i in "${!ldflags[@]}"; do
        if [[ "${ldflags[i]}" == "-Wl,--as-needed" ]]; then
            unset 'ldflags[i]'
            break
        fi
    done
    function join { local IFS="$1"; shift; echo "$*"; }
    export LDFLAGS=$(join ' ' "${ldflags[@]}")
fi

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
export LLVM_DIR=${LLVM_ROOT}/lib/cmake/llvm
export DNNL_GRAPH_BUILD_COMPILER_BACKEND=1
CXXFLAGS_BK=${CXXFLAGS}
export CXXFLAGS="${CXXFLAGS} -D__STDC_FORMAT_MACROS"
python setup.py clean
python setup.py bdist_wheel 2>&1 | tee build.log
export CXXFLAGS=${CXXFLAGS_BK}
unset DNNL_GRAPH_BUILD_COMPILER_BACKEND
unset LLVM_DIR
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH_BK}
export PATH=${PATH_BK}
python -m pip uninstall -y mkl-static mkl-include
python -m pip install --force-reinstall dist/*.whl
cd ..
#  Torch-CCL
if [ $((${MODE} & 0x01)) -ne 0 ]; then
    cd torch-ccl
    python setup.py clean
    python setup.py bdist_wheel 2>&1 | tee build.log
    python -m pip install --force-reinstall dist/*.whl
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
