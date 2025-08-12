#!/usr/bin/env bash
set -e

# Save current directory path
BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )
WHEELFOLDER=${BASEFOLDER}/../wheels
AUX_INSTALL_SCRIPT=${WHEELFOLDER}/aux_install.sh
cd ${BASEFOLDER}/..

# Mode: Select to compile projects into wheel files or install wheel files compiled.
# High bit: 8 7 6 5 4 3 2 1 :Low bit
#           | | | | | | | └- Install wheel files
#           | | | | | | └--- Compile wheel files
#           | | | | | └----- Install from prebuilt wheel files for ipex, ds
#           | | | | └------- Compile DeepSpeed from source
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
if [ ! -f ${WHEELFOLDER}/lm_eval*.whl ]; then
    (( MODE |= 0x02 ))
fi

# Check existance of required Linux commands
for CMD in gcc g++; do
    command -v ${CMD} > /dev/null || (echo "Error: Command \"${CMD}\" is required."; exit 1;)
done

if [ $((${MODE} & 0x02)) -ne 0 ]; then
    # Enter IPEX root dir
    cd ../../..

    if [ ! -f dependency_version.json ]; then
        echo "Please check if `pwd` is a valid Intel® Extension for PyTorch* source code directory."
        exit 2
    fi
    COMMIT_LM_EVA=$(python tools/dep_ver_utils.py -f dependency_version.json -k lm-evaluation-harness:commit)
    COMMIT_DS_SYCL=$(python tools/dep_ver_utils.py -f dependency_version.json -k deepspeed:commit)
    VER_DS_SYCL=$(python tools/dep_ver_utils.py -f dependency_version.json -k deepspeed:version)
    VER_GCC=$(python tools/dep_ver_utils.py -f dependency_version.json -k gcc:min-version)
    VER_TORCH=$(python tools/dep_ver_utils.py -f dependency_version.json -k pytorch:version)
    VER_TRANSFORMERS=$(python tools/dep_ver_utils.py -f dependency_version.json -k transformers:version)
    VER_PROTOBUF=$(python tools/dep_ver_utils.py -f dependency_version.json -k protobuf:version)
    VER_INC=$(python tools/dep_ver_utils.py -f dependency_version.json -k neural-compressor:version)
    VER_IPEX_MAJOR=$(grep "VERSION_MAJOR" version.txt | cut -d " " -f 2)
    VER_IPEX_MINOR=$(grep "VERSION_MINOR" version.txt | cut -d " " -f 2)
    VER_IPEX_PATCH=$(grep "VERSION_PATCH" version.txt | cut -d " " -f 2)
    VER_IPEX="${VER_IPEX_MAJOR}.${VER_IPEX_MINOR}.${VER_IPEX_PATCH}+cpu"
    # Enter IPEX parent dir
    cd ..

    # Check existance of required Linux commands
    for CMD in make git; do
        command -v ${CMD} > /dev/null || (echo "Error: Command \"${CMD}\" is required."; exit 3;)
    done

    # Clear previous compilation output
    if [ -d ${WHEELFOLDER} ]; then
        rm -rf ${WHEELFOLDER}
    fi
    mkdir ${WHEELFOLDER}

    # Install deps
    python -m pip install cmake==3.28.4 ninja wheel

    echo "#!/bin/bash" > ${AUX_INSTALL_SCRIPT}
    if [ $((${MODE} & 0x04)) -ne 0 ]; then
        set +e
        echo "${VER_TORCH}" | grep "dev" > /dev/null
        TORCH_DEV=$?
        set -e
        if [ ${TORCH_DEV} -eq 0 ]; then
            echo ""
            echo "Error: Detected dependent PyTorch is a nightly built version. Installation from prebuilt wheel files is not supported. Run again to compile from source."
            exit 4
        else
            echo "python -m pip install torch==${VER_TORCH} --index-url https://download.pytorch.org/whl/cpu" >> ${AUX_INSTALL_SCRIPT}
            echo "python -m pip install intel-extension-for-pytorch==${VER_IPEX} --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/" >> ${AUX_INSTALL_SCRIPT}
            python -m pip install torch==${VER_TORCH} --index-url https://download.pytorch.org/whl/cpu
            python -m pip install intel-extension-for-pytorch==${VER_IPEX} --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
        fi
    else
        set +e
        echo ${VER_TORCH} | grep "dev" > /dev/null
        TORCH_DEV=$?
        set -e
        URL_NIGHTLY=""
        if [ ${TORCH_DEV} -eq 0 ]; then
            URL_NIGHTLY="nightly/"
        fi
        echo "python -m pip install torch==${VER_TORCH} --index-url https://download.pytorch.org/whl/${URL_NIGHTLY}cpu" >> ${AUX_INSTALL_SCRIPT}
        # Install PyTorch and Intel® Extension for PyTorch*
        python ${BASEFOLDER}/../../../../scripts/compile_bundle.py
        cp ${BASEFOLDER}/../../../../dist/*.whl ${WHEELFOLDER}
    fi

    echo "python -m pip install -r ./requirements.txt" >> ${AUX_INSTALL_SCRIPT}

    # Used for accuracy test only
    if [ -d lm-evaluation-harness ]; then
        rm -rf lm-evaluation-harness
    fi
    git clone https://github.com/EleutherAI/lm-evaluation-harness.git
    cd lm-evaluation-harness
    git checkout ${COMMIT_LM_EVA}
    python setup.py bdist_wheel
    cp dist/*.whl ${WHEELFOLDER}
    cd ..
    rm -rf lm-evaluation-harness

    # Install DeepSpeed
    if [ $((${MODE} & 0x08)) -gt 0 ]; then
        if [ -d DeepSpeed ]; then
            rm -rf DeepSpeed
        fi
        git clone https://github.com/microsoft/DeepSpeed.git
        cd DeepSpeed
        git checkout ${COMMIT_DS_SYCL}
        python -m pip install -r requirements/requirements.txt
        python setup.py bdist_wheel
        cp dist/*.whl ${WHEELFOLDER}
        cd ..
        rm -rf DeepSpeed
    else
        echo "python -m pip install deepspeed==${VER_DS_SYCL}" >> ${AUX_INSTALL_SCRIPT}
    fi

    cd ${BASEFOLDER}/..
fi
if [ $((${MODE} & 0x01)) -ne 0 ]; then
    set +e
    command -v conda > /dev/null
    EXIST_CONDA=$?
    set -e
    if [ ${EXIST_CONDA} -gt 0 ]; then
        echo "[WARNING] Command \"conda\" is not available. Please install tcmalloc manually."
    else
        conda install -y gperftools -c conda-forge
    fi
    bash ${AUX_INSTALL_SCRIPT}
    python -m pip install ${WHEELFOLDER}/*.whl
    rm -rf ${WHEELFOLDER}
fi
