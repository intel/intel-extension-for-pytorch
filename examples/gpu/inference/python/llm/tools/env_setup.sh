#!/usr/bin/env bash
set -e

# Mode: Select to compile projects into wheel files or install wheel files compiled.
# High bit: 8 7 6 5 4 3 2 1 :Low bit
#           | | | | | | | └- Install wheel files
#           | | | | | | └--- Compile wheel files
#           | | | | | └----- Install from prebuilt wheel files
#           | | | | └------- Undefined
#           | | | └--------- Undefined
#           | | └----------- Undefined
#           | └------------- Undefined
#           └--------------- Undefined
MODE=0x03
DPCPP_ROOT=
ONEMKL_ROOT=
ONECCL_ROOT=
MPI_ROOT=
AOT=
if [[ $# -eq 0 ]]; then
    echo "Usage: bash $0 <MODE> [DPCPPROOT] [MKLROOT] [CCLROOT] [MPIROOT] [AOT]"
    echo "Set MODE to 7 to install from wheel files. Set it to 3 to compile from source. When compiling from source, you need to set arguments below."
    echo "DPCPPROOT, MKLROOT and CCLROOT should be absolute or relative path to the root directory of DPC++ compiler, oneMKL and oneCCL in oneAPI Base Toolkit respectively."
    echo "AOT should be set to the text string for environment variable USE_AOT_DEVLIST. Setting it to \"none\" to disable AOT."
    exit 1
fi

if [[ ! $1 =~ ^[0-9]+$ ]] && [[ ! $1 =~ ^0x[0-9a-fA-F]+$ ]]; then
    echo "Warning: Unexpected argument. Using default value."
else
    MODE=$1
fi
shift
if [[ $# -gt 0 ]]; then
    DPCPP_ROOT=$1
    shift
fi
if [[ $# -gt 0 ]]; then
    ONEMKL_ROOT=$1
    shift
fi
if [[ $# -gt 0 ]]; then
    ONECCL_ROOT=$1
    shift
fi
if [[ $# -gt 0 ]]; then
    MPI_ROOT=$1
    shift
fi
if [[ $# -gt 0 ]]; then
    AOT=$1
    shift
fi

# Save current directory path
BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
WHEELFOLDER=${BASEFOLDER}/../wheels
AUX_INSTALL_SCRIPT=${WHEELFOLDER}/aux_install.sh
cd ${BASEFOLDER}/..

if [ $((${MODE} & 0x02)) -eq 0 ] &&
   ([ ! -f ${WHEELFOLDER}/transformers*.whl ]); then
    (( MODE |= 0x02 ))
    echo "Expected wheel files are not found. Swith to source code compilation."
fi

if [ $((${MODE} & 0x06)) -eq 2 ] &&
   ([ -z ${DPCPP_ROOT} ] ||
   [ -z ${ONEMKL_ROOT} ] ||
   [ -z ${ONECCL_ROOT} ] ||
   [ -z ${MPI_ROOT} ] ||
   [ -z ${AOT} ]); then
    echo "Source code compilation is needed. Please set arguments DPCPP_ROOT, ONEMKL_ROOT, ONECCL_ROOT, MPI_ROOT and AOT."
    echo "DPCPPROOT, MKLROOT, CCLROOT and MPIROOT should be absolute or relative path to the root directory of DPC++ compiler, oneMKL, oneCCL and MPI in oneAPI Base Toolkit respectively."
    echo "AOT should be set to the text string for environment variable USE_AOT_DEVLIST. Setting it to \"none\" to disable AOT."
    exit 2
fi

# Check existance of required Linux commands
if [ $((${MODE} & 0x02)) -ne 0 ]; then
    # Enter IPEX root dir
    cd ../../../../..

    if [ ! -f dependency_version.yml ]; then
        echo "Please check if `pwd` is a valid Intel® Extension for PyTorch* source code directory."
        exit 3
    fi
    python -m pip install pyyaml
    VER_DS=$(python scripts/tools/compilation_helper/yaml_utils.py -f dependency_version.yml -d deepspeed -k version)
    VER_IDEX=$(python scripts/tools/compilation_helper/yaml_utils.py -f dependency_version.yml -d intel-extension-for-deepspeed -k version)
    VER_TORCHCCL=$(python scripts/tools/compilation_helper/yaml_utils.py -f dependency_version.yml -d torch-ccl -k version)
    VER_GCC=$(python scripts/tools/compilation_helper/yaml_utils.py -f dependency_version.yml -d gcc -k min-version)
    VER_TORCH=$(python scripts/tools/compilation_helper/yaml_utils.py -f dependency_version.yml -d pytorch -k version)
    COMMIT_TRANSFORMERS=$(python scripts/tools/compilation_helper/yaml_utils.py -f dependency_version.yml -d transformers -k commit)
    VER_PROTOBUF=$(python scripts/tools/compilation_helper/yaml_utils.py -f dependency_version.yml -d protobuf -k version)
    VER_LM_EVAL=$(python scripts/tools/compilation_helper/yaml_utils.py -f dependency_version.yml -d lm_eval -k version)
    VER_IPEX_MAJOR=$(grep "VERSION_MAJOR" version.txt | cut -d " " -f 2)
    VER_IPEX_MINOR=$(grep "VERSION_MINOR" version.txt | cut -d " " -f 2)
    VER_IPEX_PATCH=$(grep "VERSION_PATCH" version.txt | cut -d " " -f 2)
    VER_IPEX="${VER_IPEX_MAJOR}.${VER_IPEX_MINOR}.${VER_IPEX_PATCH}+xpu"
    python -m pip uninstall -y pyyaml
    # Enter IPEX parent dir
    cd ..

    # Check existance of required Linux commands
    for CMD in git; do
        command -v ${CMD} > /dev/null || (echo "Error: Command \"${CMD}\" is required."; exit 4;)
    done

    # Clear previous compilation output
    if [ -d ${WHEELFOLDER} ]; then
        rm -rf ${WHEELFOLDER}
    fi
    mkdir ${WHEELFOLDER}

    # Install deps
    python -m pip install cmake ninja

    echo "#!/bin/bash" > ${AUX_INSTALL_SCRIPT}
    if [ $((${MODE} & 0x04)) -ne 0 ]; then
        echo "python -m pip install torch==${VER_TORCH} intel-extension-for-pytorch-deepspeed==${VER_IDEX} intel-extension-for-pytorch==${VER_IPEX} oneccl-bind-pt==${VER_TORCHCCL} --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/" >> ${AUX_INSTALL_SCRIPT}
        python -m pip install torch==${VER_TORCH} intel-extension-for-pytorch-deepspeed==${VER_IDEX} intel-extension-for-pytorch==${VER_IPEX} oneccl-bind-pt==${VER_TORCHCCL} --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
    else
        if [ ! -f ${ONECCL_ROOT}/env/vars.sh ]; then
            echo "oneCCL environment ${ONECCL_ROOT} doesn't seem to exist."
            exit 5
        fi

        if [ ! -f ${MPI_ROOT}/env/vars.sh ]; then
            echo "MPI environment ${MPI_ROOT} doesn't seem to exist."
            exit 6
        fi

        # Install PyTorch and Intel® Extension for PyTorch*
        cp intel-extension-for-pytorch/scripts/compile_bundle.sh .
        sed -i "s/VER_IPEX=.*/VER_IPEX=/" compile_bundle.sh
        bash compile_bundle.sh ${DPCPP_ROOT} ${ONEMKL_ROOT} ${ONECCL_ROOT}  ${MPI_ROOT} ${AOT} 1
        cp pytorch/dist/*.whl ${WHEELFOLDER}
        cp intel-extension-for-pytorch/dist/*.whl ${WHEELFOLDER}
        cp intel-extension-for-pytorch/ecological_libs/deepspeed/dist/*.whl ${WHEELFOLDER}
        cp torch-ccl/dist/*.whl ${WHEELFOLDER}
        rm -rf compile_bundle.sh llvm-project llvm-release pytorch torch-ccl
        export LD_PRELOAD=$(bash intel-extension-for-pytorch/scripts/tools/compilation_helper/get_libstdcpp_lib.sh)
    fi

    #echo "python -m pip install impi-devel" >> ${AUX_INSTALL_SCRIPT}
    echo "python -m pip install cpuid accelerate datasets sentencepiece diffusers mpi4py protobuf==${VER_PROTOBUF} lm_eval==${VER_LM_EVAL} huggingface_hub py-cpuinfo " >> ${AUX_INSTALL_SCRIPT}
    echo "python -m pip install deepspeed==${VER_DS}" >> ${AUX_INSTALL_SCRIPT}

    # Install Transformers
    if [ -d transformers ]; then
        rm -rf transformers
    fi
    git clone https://github.com/huggingface/transformers.git
    cd transformers
    git checkout ${COMMIT_TRANSFORMERS}
    git apply ../intel-extension-for-pytorch/examples/gpu/inference/python/llm/tools/profile_patch
    python setup.py bdist_wheel
    cp dist/*.whl ${WHEELFOLDER}
    cd ..
    rm -rf transformers

fi
if [ $((${MODE} & 0x01)) -ne 0 ]; then
    source ${ONECCL_ROOT}/env/vars.sh
    source ${MPI_ROOT}/env/vars.sh
    python -m pip install ${WHEELFOLDER}/*.whl
    bash ${AUX_INSTALL_SCRIPT}
    rm -rf ${WHEELFOLDER}
fi

