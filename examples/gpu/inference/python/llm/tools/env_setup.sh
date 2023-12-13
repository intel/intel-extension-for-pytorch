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
AOT=
if [[ $# -eq 0 ]]; then
    echo "Usage: bash $0 <MODE> [DPCPPROOT] [MKLROOT] [CCLROOT] [AOT]"
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
    AOT=$1
    shift
fi

# Save current directory path
BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
WHEELFOLDER=${BASEFOLDER}/../wheels
AUX_INSTALL_SCRIPT=${WHEELFOLDER}/aux_install.sh
cd ${BASEFOLDER}/..

if [ ! -f ${WHEELFOLDER}/transformers*.whl ] ||
   [ ! -f ${WHEELFOLDER}/deepspeed*.whl ] ||
   [ ! -f ${WHEELFOLDER}/intel_extension_for_deepspeed*.whl ]; then
    (( MODE |= 0x02 ))
    echo "Expected wheel files are not found. Swith to source code compilation."
fi

if [ $((${MODE} & 0x06)) -eq 2 ] &&
   ([ -z ${DPCPP_ROOT} ] ||
   [ -z ${ONEMKL_ROOT} ] ||
   [ -z ${ONECCL_ROOT} ] ||
   [ -z ${AOT} ]); then
    echo "Source code compilation is needed. Please set arguments DPCPP_ROOT, ONEMKL_ROOT, ONECCL_ROOT and AOT."
    echo "DPCPPROOT, MKLROOT and CCLROOT should be absolute or relative path to the root directory of DPC++ compiler, oneMKL and oneCCL in oneAPI Base Toolkit respectively."
    echo "AOT should be set to the text string for environment variable USE_AOT_DEVLIST. Setting it to \"none\" to disable AOT."
    exit 2
fi

# Check existance of required Linux commands
for CMD in conda; do
    command -v ${CMD} > /dev/null || (echo "Error: Command \"${CMD}\" is required."; exit 3;)
done
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${CONDA_PREFIX}/lib:${LIBRARY_PATH}

if [ $((${MODE} & 0x02)) -ne 0 ]; then
    # Enter IPEX root dir
    cd ../../../../..

    if [ ! -f dependency_version.yml ]; then
        echo "Please check if `pwd` is a valid Intel® Extension for PyTorch* source code directory."
        exit 4
    fi
    python -m pip install pyyaml
    DS_REPO=$(python tools/yaml_utils.py -f dependency_version.yml -d deepspeed -k repo)
    DS_COMMIT=$(python tools/yaml_utils.py -f dependency_version.yml -d deepspeed -k commit)
    IDEX_REPO=$(python tools/yaml_utils.py -f dependency_version.yml -d intel-extension-for-deepspeed -k repo)
    IDEX_COMMIT=$(python tools/yaml_utils.py -f dependency_version.yml -d intel-extension-for-deepspeed -k commit)
    TORCHCCL_REPO=$(python tools/yaml_utils.py -f dependency_version.yml -d torch-ccl -k repo)
    TORCHCCL_COMMIT=$(python tools/yaml_utils.py -f dependency_version.yml -d torch-ccl -k commit)
    VER_TORCHCCL=$(python tools/yaml_utils.py -f dependency_version.yml -d torch-ccl -k version)
    VER_GCC=$(python tools/yaml_utils.py -f dependency_version.yml -d gcc -k min-version)
    VER_TORCH=$(python tools/yaml_utils.py -f dependency_version.yml -d pytorch -k version)
    TRANSFORMERS_COMMIT=$(python tools/yaml_utils.py -f dependency_version.yml -d transformers -k commit)
    VER_PROTOBUF=$(python tools/yaml_utils.py -f dependency_version.yml -d protobuf -k version)
    VER_IPEX_MAJOR=$(grep "VERSION_MAJOR" version.txt | cut -d " " -f 2)
    VER_IPEX_MINOR=$(grep "VERSION_MINOR" version.txt | cut -d " " -f 2)
    VER_IPEX_PATCH=$(grep "VERSION_PATCH" version.txt | cut -d " " -f 2)
    VER_IPEX="${VER_IPEX_MAJOR}.${VER_IPEX_MINOR}.${VER_IPEX_PATCH}+xpu"
    python -m pip uninstall -y pyyaml
    # Enter IPEX parent dir
    cd ..

    # Check existance of required Linux commands
    for CMD in git; do
        command -v ${CMD} > /dev/null || (echo "Error: Command \"${CMD}\" is required."; exit 5;)
    done

    # Clear previous compilation output
    if [ -d ${WHEELFOLDER} ]; then
        rm -rf ${WHEELFOLDER}
    fi
    mkdir ${WHEELFOLDER}

    # Install deps
    conda install -y cmake ninja

    echo "#!/bin/bash" > ${AUX_INSTALL_SCRIPT}
    if [ $((${MODE} & 0x04)) -ne 0 ]; then
        echo "python -m pip install torch==${VER_TORCH} intel-extension-for-pytorch==${VER_IPEX} oneccl-bind-pt==${VER_TORCHCCL} --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/" >> ${AUX_INSTALL_SCRIPT}
        python -m pip install torch==${VER_TORCH} intel-extension-for-pytorch==${VER_IPEX} oneccl-bind-pt==${VER_TORCHCCL} --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
    else
        if [ ! -f ${ONECCL_ROOT}/env/vars.sh ]; then
            echo "oneCCL environment ${ONECCL_ROOT} doesn't seem to exist."
            exit 6
        fi
        ONEAPIROOT=${ONEMKL_ROOT}/../..

        # Install PyTorch and Intel® Extension for PyTorch*
        cp intel-extension-for-pytorch/scripts/compile_bundle.sh .
        sed -i "s/VER_IPEX=.*/VER_IPEX=/" compile_bundle.sh
        bash compile_bundle.sh ${DPCPP_ROOT} ${ONEMKL_ROOT} ${AOT} 0
        cp pytorch/dist/*.whl ${WHEELFOLDER}
        cp intel-extension-for-pytorch/dist/*.whl ${WHEELFOLDER}
        rm -rf compile_bundle.sh llvm-project llvm-release pytorch
        export LD_PRELOAD=$(bash intel-extension-for-pytorch/tools/get_libstdcpp_lib.sh)

        # The following is only for DeepSpeed case
        #Install oneccl-bind-pt(also named torch-ccl)
        set +e
        function env_backup() {
            key=$1
            env | grep ${key} > /dev/null
            if [ $? -gt 0 ]; then
                echo "unset"
            else
                value=$(env | grep "^${key}=")
                echo ${value#"${key}="}
            fi
        }
        function env_recover() {
            key=$1
            value=$2
            if [ "$value" == "unset" ]; then
                unset ${key}
            else
                export ${key}=${value}
            fi
        }

        PKG_CONFIG_PATH_BK=$(env_backup PKG_CONFIG_PATH)
        ACL_BOARD_VENDOR_PATH_BK=$(env_backup ACL_BOARD_VENDOR_PATH)
        FPGA_VARS_DIR_BK=$(env_backup FPGA_VARS_DIR)
        DIAGUTIL_PATH_BK=$(env_backup DIAGUTIL_PATH)
        MANPATH_BK=$(env_backup MANPATH)
        CMAKE_PREFIX_PATH_BK=$(env_backup CMAKE_PREFIX_PATH)
        CMPLR_ROOT_BK=$(env_backup CMPLR_ROOT)
        FPGA_VARS_ARGS_BK=$(env_backup FPGA_VARS_ARGS)
        LIBRARY_PATH_BK=$(env_backup LIBRARY_PATH)
        OCL_ICD_FILENAMES_BK=$(env_backup OCL_ICD_FILENAMES)
        INTELFPGAOCLSDKROOT_BK=$(env_backup INTELFPGAOCLSDKROOT)
        LD_LIBRARY_PATH_BK=$(env_backup LD_LIBRARY_PATH)
        MKLROOT_BK=$(env_backup MKLROOT)
        NLSPATH_BK=$(env_backup NLSPATH)
        PATH_BK=$(env_backup PATH)
        CPATH_BK=$(env_backup CPATH)
        set -e
        source ${DPCPP_ROOT}/env/vars.sh
        source ${ONEMKL_ROOT}/env/vars.sh

        if [ -d torch-ccl ]; then
            rm -rf torch-ccl
        fi
        git clone ${TORCHCCL_REPO}
        cd torch-ccl
        git checkout ${TORCHCCL_COMMIT}
        git submodule sync
        git submodule update --init --recursive
        if [ -d ${CONDA_PREFIX}/lib/gcc/x86_64-conda-linux-gnu ]; then
            export DPCPP_GCC_INSTALL_DIR="${CONDA_PREFIX}/lib/gcc/x86_64-conda-linux-gnu/12.3.0"
        fi
        export INTELONEAPIROOT=${ONEAPIROOT}
        COMPUTE_BACKEND=dpcpp python setup.py bdist_wheel
        unset INTELONEAPIROOT
        if [ -d ${CONDA_PREFIX}/lib/gcc/x86_64-conda-linux-gnu ]; then
            unset DPCPP_GCC_INSTALL_DIR
        fi
        cp dist/*.whl ${WHEELFOLDER}
        python -m pip install dist/*.whl
        cd ..
        rm -rf torch-ccl

        set +e
        env_recover PKG_CONFIG_PATH ${PKG_CONFIG_PATH_BK}
        env_recover ACL_BOARD_VENDOR_PATH ${ACL_BOARD_VENDOR_PATH_BK}
        env_recover FPGA_VARS_DIR ${FPGA_VARS_DIR_BK}
        env_recover DIAGUTIL_PATH ${DIAGUTIL_PATH_BK}
        env_recover MANPATH ${MANPATH_BK}
        env_recover CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH_BK}
        env_recover CMPLR_ROOT ${CMPLR_ROOT_BK}
        env_recover FPGA_VARS_ARGS ${FPGA_VARS_ARGS_BK}
        env_recover LIBRARY_PATH ${LIBRARY_PATH_BK}
        env_recover OCL_ICD_FILENAMES ${OCL_ICD_FILENAMES_BK}
        env_recover INTELFPGAOCLSDKROOT ${INTELFPGAOCLSDKROOT_BK}
        env_recover LD_LIBRARY_PATH ${LD_LIBRARY_PATH_BK}
        env_recover MKLROOT ${MKLROOT_BK}
        env_recover NLSPATH ${NLSPATH_BK}
        env_recover PATH ${PATH_BK}
        env_recover CPATH ${CPATH_BK}
        set -e
    fi

    echo "python -m pip install impi-devel" >> ${AUX_INSTALL_SCRIPT}
    echo "python -m pip install cpuid accelerate datasets sentencepiece protobuf==${VER_PROTOBUF} huggingface_hub mpi4py mkl" >> ${AUX_INSTALL_SCRIPT}
    echo "python -m pip install lm_eval" >> ${AUX_INSTALL_SCRIPT}
    

    # Install Transformers
    if [ -d transformers ]; then
        rm -rf transformers
    fi
    git clone https://github.com/huggingface/transformers.git
    cd transformers
    git checkout ${TRANSFORMERS_COMMIT}
    git apply ../intel-extension-for-pytorch/examples/gpu/inference/python/llm/tools/profile_patch
    python setup.py bdist_wheel
    cp dist/*.whl ${WHEELFOLDER}
    cd ..
    rm -rf transformers

    # Install DeepSpeed
    python -m pip install impi-devel
    python -m pip install huggingface_hub mpi4py
    if [ -d DeepSpeed ]; then
        rm -rf DeepSpeed
    fi
    git clone ${DS_REPO}
    cd DeepSpeed
    git checkout ${DS_COMMIT}
    python -m pip install -r requirements/requirements.txt
    cd ..

    if [ -d intel-extension-for-deepspeed ]; then
        rm -rf intel-extension-for-deepspeed
    fi
    git clone ${IDEX_REPO}
    cd intel-extension-for-deepspeed
    git checkout ${IDEX_COMMIT}
    python setup.py bdist_wheel
    python -m pip install dist/*.whl
    cp dist/*.whl ${WHEELFOLDER}
    cd ..
    rm -rf intel-extension-for-deepspeed

    cd DeepSpeed
    python -m pip install mkl
    python setup.py bdist_wheel
    cp dist/*.whl ${WHEELFOLDER}
    cd ..
    rm -rf DeepSpeed
fi
if [ $((${MODE} & 0x01)) -ne 0 ]; then
    python -m pip install ${WHEELFOLDER}/*.whl
    bash ${AUX_INSTALL_SCRIPT}
    rm -rf ${WHEELFOLDER}
fi
