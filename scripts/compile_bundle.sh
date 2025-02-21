#!/usr/bin/env bash
#
# Please review the system requirements before running this script
# https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installation.html
#
set -eo pipefail

VER_IPEX=xpu-main
ENABLE_ONEAPI_INTEGRATION=1

if [[ $# -lt 2 ]]; then
    echo "Usage: bash $0 <ONEAPI_ROOT_DIR> <AOT>"
    echo "ONEAPI_ROOT_DIR should be set to the text string of the root path where oneAPI was installed in. For instance, \"/opt/intel/oneapi\"."
    echo "AOT should be set to the text string for environment variable USE_AOT_DEVLIST. Setting it to \"none\" to disable AOT. Setting it to \"pytorch\" to follow the AOT configuration in the PyTorch prebuilt binary."
    exit 1
fi

# Mode: Select which components to install. PyTorch and Intel® Extension for PyTorch* are always installed.
# High bit: 8 7 6 5 4 3 2 1 :Low bit
#           | | | | | | | └- torch-ccl
#           | | | | | | └--- TorchAudio
#           | | | | | └----- TorchVision
#           | | | | └------- Torch
#           | | | └--------- Undefined
#           | | └----------- Undefined
#           | └------------- Undefined
#           └--------------- Undefined
MODE=0x0F

DPCPP_ROOT=
ONEMKL_ROOT=
ONECCL_ROOT=
MPI_ROOT=
PTI_ROOT=
AOT=

# Pass arguments
ARGS=()
while [ $# -gt 0 ]
do
    ARGS+=($1)
    shift
done

# Check required packages version
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

VERSION_TORCH=$(python scripts/tools/compilation_helper/dep_ver_utils.py -f dependency_version.json -k pytorch:version)
VERSION_TORCHVISION=$(python scripts/tools/compilation_helper/dep_ver_utils.py -f dependency_version.json -k torchvision:version)
VERSION_TORCHAUDIO=$(python scripts/tools/compilation_helper/dep_ver_utils.py -f dependency_version.json -k torchaudio:version)
COMMIT_TORCH=$(python scripts/tools/compilation_helper/dep_ver_utils.py -f dependency_version.json -k pytorch:commit)
COMMIT_TORCHVISION=$(python scripts/tools/compilation_helper/dep_ver_utils.py -f dependency_version.json -k torchvision:commit)
COMMIT_TORCHAUDIO=$(python scripts/tools/compilation_helper/dep_ver_utils.py -f dependency_version.json -k torchaudio:commit)
COMMIT_TORCHCCL=$(python scripts/tools/compilation_helper/dep_ver_utils.py -f dependency_version.json -k torch-ccl:commit)
#COMMIT_LLVM=$(python scripts/tools/compilation_helper/dep_ver_utils.py -f dependency_version.json -k llvm:commit)
#VER_GCC=$(python scripts/tools/compilation_helper/dep_ver_utils.py -f dependency_version.json -k gcc:min-version)

if [[ ${#ARGS[@]} -lt 6 ]]; then
    ONEAPI_ROOT=${ARGS[0]}
    ARGS=(${ARGS[@]:1})
    AOT=${ARGS[0]}
    ARGS=(${ARGS[@]:1})
    VER_DPCPP=$(python scripts/tools/compilation_helper/dep_ver_utils.py -f dependency_version.json -k basekit:dpcpp-cpp-rt:version)
    VER_MKL=$(python scripts/tools/compilation_helper/dep_ver_utils.py -f dependency_version.json -k basekit:mkl-dpcpp:version)
    VER_CCL=$(python scripts/tools/compilation_helper/dep_ver_utils.py -f dependency_version.json -k basekit:oneccl-devel:version)
    VER_MPI=$(python scripts/tools/compilation_helper/dep_ver_utils.py -f dependency_version.json -k basekit:impi-devel:version)
    VER_PTI=$(python scripts/tools/compilation_helper/dep_ver_utils.py -f dependency_version.json -k basekit:intel-pti:version)
    DPCPP_ROOT="${ONEAPI_ROOT}/compiler/$(echo ${VER_DPCPP} | cut -d '.' -f 1).$(echo ${VER_DPCPP} | cut -d '.' -f 2)"
    ONEMKL_ROOT="${ONEAPI_ROOT}/mkl/$(echo ${VER_MKL} | cut -d '.' -f 1).$(echo ${VER_MKL} | cut -d '.' -f 2)"
    ONECCL_ROOT="${ONEAPI_ROOT}/ccl/$(echo ${VER_CCL} | cut -d '.' -f 1).$(echo ${VER_CCL} | cut -d '.' -f 2)"
    MPI_ROOT="${ONEAPI_ROOT}/mpi/$(echo ${VER_MPI} | cut -d '.' -f 1).$(echo ${VER_MPI} | cut -d '.' -f 2)"
    PTI_ROOT="${ONEAPI_ROOT}/pti/$(echo ${VER_PTI} | cut -d '.' -f 1).$(echo ${VER_PTI} | cut -d '.' -f 2)"
    unset ONEAPI_ROOT
else
    DPCPP_ROOT=${ARGS[0]}
    ARGS=(${ARGS[@]:1})
    ONEMKL_ROOT=${ARGS[0]}
    ARGS=(${ARGS[@]:1})
    ONECCL_ROOT=${ARGS[0]}
    ARGS=(${ARGS[@]:1})
    MPI_ROOT=${ARGS[0]}
    ARGS=(${ARGS[@]:1})
    PTI_ROOT=${ARGS[0]}
    ARGS=(${ARGS[@]:1})
    AOT=${ARGS[0]}
    ARGS=(${ARGS[@]:1})
fi
if [[ ${#ARGS[@]} -gt 0 ]]; then
    if [[ ${ARGS[0]} =~ ^[0-9]+$ ]] || [[ ${ARGS[0]} =~ ^0x[0-9a-fA-F]+$ ]]; then
        MODE=${ARGS[0]}
    else
        echo "Warning: Unexpected argument. Using default value."
    fi
fi
cd ..

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
for APP in gcc g++ python git patch nproc bzip2 wget tar zip unzip; do
    command -v $APP > /dev/null || (echo "Error: Command \"${APP}\" not found." ; exit 6)
done

# set number of compile processes, if not already defined
MAX_JOBS_VAR=$(nproc)
if [ -z "${MAX_JOBS}" ]; then
    export MAX_JOBS=${MAX_JOBS_VAR}
fi

# Save current directory path
BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )
cd ${BASEFOLDER}

# Checkout individual components
if [ "${AOT}" != "pytorch" ]; then
    if [ "${COMMIT_TORCHVISION}" = "N/A" ]; then
        (( MODE &= 0xFB ))
    fi
    if [ "${COMMIT_TORCHAUDIO}" = "N/A" ]; then
        (( MODE &= 0xFD ))
    fi
    if [ "${COMMIT_TORCHCCL}" = "N/A" ]; then
        (( MODE &= 0xFE ))
    fi
    if [ $((${MODE} & 0x06)) -ne 0 ]; then
        (( MODE |= 0x08 ))
    fi

    if [ $((${MODE} & 0x08)) -ne 0 ]; then
        if [ ! -d pytorch ]; then
            git clone https://github.com/pytorch/pytorch.git
        fi
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
    
    # Checkout required branch/commit and update submodules
    if [ $((${MODE} & 0x08)) -ne 0 ]; then
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
        #git apply ../intel-extension-for-pytorch/torch_patches/*.patch
        cd ..
    fi
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
else
    if [ "${VERSION_TORCHVISION}" = "N/A" ]; then
        (( MODE &= 0xFB ))
    fi
    if [ "${VERSION_TORCHAUDIO}" = "N/A" ]; then
        (( MODE &= 0xFD ))
    fi
    if [ $((${MODE} & 0x06)) -ne 0 ]; then
        (( MODE |= 0x08 ))
    fi
fi
if [ $((${MODE} & 0x01)) -ne 0 ]; then
    if [ ! -d torch-ccl ]; then
        git clone https://github.com/intel/torch-ccl.git
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
    
#if [ ! -d llvm-project ]; then
#    git clone https://github.com/llvm/llvm-project.git
#fi
#
#cd llvm-project
#if [ ! -z "${COMMIT_LLVM}" ]; then
#    rm -rf * > /dev/null
#    git checkout . > /dev/null
#    git checkout main > /dev/null
#    git pull > /dev/null
#    git checkout ${COMMIT_LLVM}
#fi
#git submodule sync
#git submodule update --init --recursive
#cd ..

# Install python dependency
if [ $((${MODE} & 0x08)) -ne 0 ]; then
    python -m pip uninstall -y torch torchvision torchaudio
fi
python -m pip uninstall -y intel-extension-for-pytorch intel-extension-for-pytorch-deepspeed oneccl_bind_pt
python -m pip install cmake make ninja
wget https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0-x86_64.tar.gz -O patchelf.tar.gz
if [ -d patchelf ]; then
    rm -rf patchelf
fi
mkdir patchelf
tar xvf patchelf.tar.gz -C patchelf > /dev/null
rm patchelf.tar.gz
export PATH=`pwd`/patchelf/bin:${PATH}

patchelf_so_files() {
    DIR=$1
    cd dist
    WHL=$(ls -1)
    mkdir tmp
    unzip ${WHL} -d tmp
    rm ${WHL}
    cd tmp
    find ${DIR} -maxdepth 1 -name "*.so" -exec patchelf --set-rpath '$ORIGIN:$ORIGIN/lib:$ORIGIN/../../../' --force-rpath {} \;
    find ${DIR}/lib -name "*.so" -exec patchelf --set-rpath '$ORIGIN:$ORIGIN/../../../../' --force-rpath {} \;
    zip -r ../${WHL} .
    cd ..
    rm -rf tmp
    cd ..
}

## Check gcc version
#function ver_compare() {
#    VER_MAJOR_CUR=$(echo $1 | cut -d "." -f 1)
#    VER_MINOR_CUR=$(echo $1 | cut -d "." -f 2)
#    VER_PATCH_CUR=$(echo $1 | cut -d "." -f 3)
#    VER_MAJOR_REQ=$(echo $2 | cut -d "." -f 1)
#    VER_MINOR_REQ=$(echo $2 | cut -d "." -f 2)
#    VER_PATCH_REQ=$(echo $2 | cut -d "." -f 3)
#    RET=0
#    if [[ ${VER_MAJOR_CUR} -lt ${VER_MAJOR_REQ} ]]; then
#        RET=1
#    else
#        if [[ ${VER_MAJOR_CUR} -eq ${VER_MAJOR_REQ} ]] &&
#           [[ ${VER_MINOR_CUR} -lt ${VER_MINOR_REQ} ]]; then
#            RET=2
#        else
#            if [[ ${VER_MAJOR_CUR} -eq ${VER_MAJOR_REQ} ]] &&
#               [[ ${VER_MINOR_CUR} -eq ${VER_MINOR_REQ} ]] &&
#               [[ ${VER_PATCH_CUR} -lt ${VER_PATCH_REQ} ]]; then
#                RET=3
#            fi
#        fi
#    fi
#    echo ${RET}
#}
#set +e
#command -v conda > /dev/null
#EXIST_CONDA=$?
#
#GCC_CONDA=0
#command -v gcc > /dev/null
#EXIST_CC=$?
#command -v g++ > /dev/null
#EXIST_CXX=$?
#set -e
#if [ ${EXIST_CC} -gt 0 ] || [ ${EXIST_CXX} -gt 0 ]; then
#    echo -e '\a'
#    echo "Warning: GCC not found."
#    echo "         Installing gcc and g++ 12.3 from conda..."
#    echo ""
#    GCC_CONDA=1
#else
#    VER_COMP=$(ver_compare $(gcc -dumpfullversion) ${VER_GCC})
#    if [ ${VER_COMP} -ne 0 ]; then
#        echo -e '\a'
#        echo "Warning: GCC version equal to or newer than ${VER_GCC} is required."
#        echo "         Found GCC version $(gcc -dumpfullversion)."
#        echo "         Installing gcc and g++ 12.3 from conda..."
#        echo ""
#        GCC_CONDA=1
#    else
#        DIR_GCC=$(which gcc)
#        if [ ! -z ${CONDA_PREFIX} ] && [[ ${DIR_GCC} =~ ${CONDA_PREFIX} ]]; then
#            GCC_CONDA=2
#        fi
#    fi
#fi
#
#if [ ${GCC_CONDA} -eq 1 ]; then
#    if [ ${EXIST_CONDA} -gt 0 ]; then
#        echo "Command \"conda\" not found. Exit."
#        exit 2
#    fi
#    conda install -y sysroot_linux-64 -c conda-forge
#    conda install -y gcc==12.3 gxx==12.3 cxx-compiler zstd -c conda-forge
#fi
#if [ ${GCC_CONDA} -ge 1 ]; then
#    if [ -z ${CONDA_BUILD_SYSROOT} ]; then
#        source ${CONDA_PREFIX}/etc/conda/activate.d/activate-gcc_linux-64.sh
#        source ${CONDA_PREFIX}/etc/conda/activate.d/activate-gxx_linux-64.sh
#        source ${CONDA_PREFIX}/etc/conda/activate.d/activate-binutils_linux-64.sh
#    fi
#    set +e
#    echo ${LD_LIBRARY_PATH} | grep "${CONDA_PREFIX}/lib:" > /dev/null
#    if [ $? -gt 0 ]; then
#        export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
#    fi
#    set -e
#fi
#if [[ ! -z ${LDFLAGS} ]]; then
#    read -a ldflags <<< "${LDFLAGS}"
#    for i in "${!ldflags[@]}"; do
#        if [[ "${ldflags[i]}" == "-Wl,--as-needed" ]]; then
#            unset 'ldflags[i]'
#            break
#        fi
#    done
#    function join { local IFS="$1"; shift; echo "$*"; }
#    export LDFLAGS=$(join ' ' "${ldflags[@]}")
#fi

# don't fail on external scripts
source ${DPCPP_ENV}
source ${ONEMKL_ENV}
source ${CCL_ENV}
source ${MPI_ENV}
source ${PTI_ENV}

if [ "${AOT}" != "pytorch" ]; then
    ABI=1
    if [[ ${AOT} == "none" ]]; then
        AOT=""
    fi
    
    #  PyTorch
    if [ $((${MODE} & 0x08)) -ne 0 ]; then
        cd pytorch
        python -m pip install -r requirements.txt
        python -m pip install --force-reinstall mkl-static mkl-include
        #export PYTORCH_BUILD_VERSION=${VERSION_TORCH}
        #export PYTORCH_BUILD_NUMBER=0
        # Ensure cmake can find python packages when using conda or virtualenv
        CMAKE_PREFIX_PATH_BK=${CMAKE_PREFIX_PATH}
        if [ -n "${CONDA_PREFIX-}" ]; then
            export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(command -v conda))/../'}:${CMAKE_PREFIX_PATH}"
        elif [ -n "${VIRTUAL_ENV-}" ]; then
            export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(command -v python))/../'}:${CMAKE_PREFIX_PATH}"
        fi
        export USE_STATIC_MKL=1
        export _GLIBCXX_USE_CXX11_ABI=${ABI}
        export USE_NUMA=0
        export USE_CUDA=0
        export USE_MPI=0
        export TORCH_XPU_ARCH_LIST=${AOT}
        export PYTORCH_EXTRA_INSTALL_REQUIREMENTS="intel-cmplr-lib-rt==2025.0.2|intel-cmplr-lib-ur==2025.0.2|intel-cmplr-lic-rt==2025.0.2|intel-sycl-rt==2025.0.2|tcmlib==1.2.0|umf==0.9.1|intel-pti==0.10.0"
        python setup.py clean
        if [ -d dist ]; then
            rm -rf dist
        fi
        #CFLAGS_BK=${CFLAGS}
        #CXXFLAGS_BK=${CXXFLAGS}
        #export CFLAGS="${CFLAGS} -Wno-nonnull"
        #export CXXFLAGS="${CXXFLAGS} -Wno-nonnull"
        python setup.py bdist_wheel 2>&1 | tee build.log
        #if [ ! -z ${CFLAGS_BK} ]; then
        #    export CFLAGS=${CFLAGS_BK}
        #else
        #    unset CFLAGS
        #fi
        #if [ ! -z ${CXXFLAGS_BK} ]; then
        #    export CXXFLAGS=${CXXFLAGS_BK}
        #else
        #    unset CXXFLAGS
        #fi
        unset PYTORCH_EXTRA_INSTALL_REQUIREMENTS
        unset TORCH_XPU_ARCH_LIST
        unset USE_MPI
        unset USE_CUDA
        unset USE_NUMA
        unset _GLIBCXX_USE_CXX11_ABI
        unset USE_STATIC_MKL
        if [ ! -z ${CMAKE_PREFIX_PATH_BK} ]; then
            export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH_BK}
        else
            unset CMAKE_PREFIX_PATH
        fi
        unset PYTORCH_BUILD_NUMBER
        unset PYTORCH_BUILD_VERSION
        python -m pip uninstall -y mkl-static mkl-include
        if [ ${ENABLE_ONEAPI_INTEGRATION} -eq 1 ]; then
            patchelf_so_files torch
        fi
        python -m pip install dist/*.whl
        cd ..
    fi
    #  TorchVision
    if [ $((${MODE} & 0x04)) -ne 0 ]; then
        python -m pip install Pillow
        conda install -y conda-forge::libpng conda-forge::libjpeg-turbo
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
else
    CMD_INS=""
    if [ $((${MODE} & 0x08)) -ne 0 ]; then
        CMD_INS="python -m pip install torch==${VERSION_TORCH}"
    fi
    if [ $((${MODE} & 0x04)) -ne 0 ]; then
        CMD_INS="${CMD_INS} torchvision==${VERSION_TORCHVISION}"
    fi
    if [ $((${MODE} & 0x02)) -ne 0 ]; then
        CMD_INS="${CMD_INS} torchaudio==${VERSION_TORCHAUDIO}"
    fi
    if [[ ! -z ${CMD_INS} ]]; then
        CMD_INS="${CMD_INS} --index-url https://download.pytorch.org/whl/xpu"
        eval ${CMD_INS}
    fi
    set +e
    python -m pip list | grep torch > /dev/null
    if [ $? -ne 0 ]; then
        echo "Error: PyTorch is not found in the environment."
        exit 1
    fi
    set -e
    ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
    AOT=$(python -c "import torch; arch_list = torch._C._xpu_getArchFlags().split(); print(\",\".join(arch_list));")
fi
##  LLVM
#LLVM_ROOT="$(pwd)/llvm-release"
#if [ $((${MODE} & 0x08)) -ne 0 ]; then
#    if [ -d ${LLVM_ROOT} ]; then
#        rm -rf ${LLVM_ROOT}
#    fi
#fi
#cd llvm-project
#if [ -d build ]; then
#    rm -rf build
#fi
#if [ ! -d ${LLVM_ROOT} ]; then
#    mkdir build
#    cd build
#    echo "***************************** cmake *****************************" > ../build.log
#    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${ABI}" -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF ../llvm 2>&1 | tee -a ../build.log
#    echo "***************************** build *****************************" >> ../build.log
#    cmake --build . -j ${MAX_JOBS} 2>&1 | tee -a ../build.log
#    echo "**************************** install ****************************" >> ../build.log
#    cmake -DCMAKE_INSTALL_PREFIX=${LLVM_ROOT} -P cmake_install.cmake 2>&1 | tee -a ../build.log
#    #xargs rm -rf < install_manifest.txt
#    cd ..
#    rm -rf build
#    ln -s ${LLVM_ROOT}/bin/llvm-config ${LLVM_ROOT}/bin/llvm-config-13
#fi
#cd ..
#PATH_BK=${PATH}
#LD_LIBRARY_PATH_BK=${LD_LIBRARY_PATH}
#export PATH=${LLVM_ROOT}/bin:${PATH}
#export LD_LIBRARY_PATH=${LLVM_ROOT}/lib:${LD_LIBRARY_PATH}
#  Intel® Extension for PyTorch*
cd intel-extension-for-pytorch
python -m pip install -r requirements.txt
if [[ ! ${AOT} == "" ]]; then
    export USE_AOT_DEVLIST=${AOT}
fi
export BUILD_WITH_CPU=OFF
#export LLVM_DIR=${LLVM_ROOT}/lib/cmake/llvm
#export DNNL_GRAPH_BUILD_COMPILER_BACKEND=1
#CXXFLAGS_BK=${CXXFLAGS}
#export CXXFLAGS="${CXXFLAGS} -D__STDC_FORMAT_MACROS"
python setup.py clean
if [ -d dist ]; then
    rm -rf dist
fi
ENABLE_ONEAPI_INTEGRATION=${ENABLE_ONEAPI_INTEGRATION} python setup.py bdist_wheel 2>&1 | tee build_whl.log
#if [ ! -z ${CXXFLAGS_BK} ]; then
#    export CXXFLAGS=${CXXFLAGS_BK}
#else
#    unset CXXFLAGS
#fi
#unset DNNL_GRAPH_BUILD_COMPILER_BACKEND
#unset LLVM_DIR
unset BUILD_WITH_CPU
#if [ ! -z ${LD_LIBRARY_PATH_BK} ]; then
#    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH_BK}
#else
#    unset LD_LIBRARY_PATH
#fi
#export PATH=${PATH_BK}
if [[ ! ${AOT} == "" ]]; then
    unset USE_AOT_DEVLIST
fi
if [ ${ENABLE_ONEAPI_INTEGRATION} -eq 1 ]; then
    patchelf_so_files intel_extension_for_pytorch
fi
python -m pip install dist/*.whl
cd ..

#  Torch-CCL
if [ $((${MODE} & 0x01)) -ne 0 ]; then
    cd torch-ccl
    python setup.py clean
    if [ -d dist ]; then
        rm -rf dist
    fi
    export USE_SYSTEM_ONECCL=1
    export INTELONEAPIROOT=${ONEAPIROOT}
    COMPUTE_BACKEND=dpcpp python setup.py bdist_wheel 2>&1 | tee build.log
    unset INTELONEAPIROOT
    unset USE_SYSTEM_ONECCL
    if [ ${ENABLE_ONEAPI_INTEGRATION} -eq 1 ]; then
        patchelf_so_files oneccl_bindings_for_pytorch
    fi
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
if [ $((${MODE} & 0x01)) -ne 0 ]; then
    CMD="${CMD} import oneccl_bindings_for_pytorch as torch_ccl; print(f'torchccl_version:    {torch_ccl.__version__}');"
fi
python -c "${CMD}"
