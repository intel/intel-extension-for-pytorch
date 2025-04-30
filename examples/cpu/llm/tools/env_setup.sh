#!/usr/bin/env bash
set -e

# Save current directory path
BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )
WHEELFOLDER=${BASEFOLDER}/../wheels
AUX_INSTALL_SCRIPT=${WHEELFOLDER}/aux_install.sh
CCLFOLDER=${BASEFOLDER}/../oneCCL_release
cd ${BASEFOLDER}/..

# Mode: Select to compile projects into wheel files or install wheel files compiled.
# High bit: 8 7 6 5 4 3 2 1 :Low bit
#           | | | | | | | └- Install wheel files
#           | | | | | | └--- Compile wheel files
#           | | | | | └----- Install from prebuilt wheel files for ipex, torch-ccl, ds
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
if [ ! -f ${WHEELFOLDER}/lm_eval*.whl ] ||
   [ ! -d ${CCLFOLDER} ]; then
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
    VER_TORCHCCL=$(python tools/dep_ver_utils.py -f dependency_version.json -k torch-ccl:version)
    COMMIT_ONECCL=$(python tools/dep_ver_utils.py -f dependency_version.json -k oneCCL:commit)
    VER_GCC=$(python tools/dep_ver_utils.py -f dependency_version.json -k gcc:min-version)
    VER_TORCH=$(python tools/dep_ver_utils.py -f dependency_version.json -k pytorch:version)
    VER_PROTOBUF=$(python tools/dep_ver_utils.py -f dependency_version.json -k protobuf:version)
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
    if [ -d ${CCLFOLDER} ]; then
        rm -rf ${CCLFOLDER}
    fi

    # Install deps
    python -m pip install cmake==3.28.4 ninja

    echo "#!/bin/bash" > ${AUX_INSTALL_SCRIPT}
    echo "python -m pip install -r ./requirements.txt" >> ${AUX_INSTALL_SCRIPT}
    echo "python -m pip install --pre torch==2.8.0.dev20250323+cpu torchvision==0.22.0.dev20250323+cpu torchaudio==2.6.0.dev20250323+cpu --index-url https://download.pytorch.org/whl/nightly/cpu" --force-reinstall >> ${AUX_INSTALL_SCRIPT}
    echo "python -m pip install https://intel-extension-for-pytorch.s3.us-east-1.amazonaws.com/ipex_dev/cpu/intel_extension_for_pytorch-2.8.0%2Bgit37bfff1-cp310-cp310-linux_x86_64.whl" >> ${AUX_INSTALL_SCRIPT}
    echo "python -m pip install oneccl-bind-pt==${VER_TORCHCCL} --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/" >> ${AUX_INSTALL_SCRIPT}
    

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
    echo "python -m pip install deepspeed==${VER_DS_SYCL}" >> ${AUX_INSTALL_SCRIPT}

    # Install OneCCL
    if [ -d oneCCL ]; then
        rm -rf oneCCL
    fi
    git clone https://github.com/oneapi-src/oneCCL.git
    cd oneCCL
    git checkout ${COMMIT_ONECCL}
    mkdir build
    cd build
    cmake -DBUILD_EXAMPLES=FALSE -DBUILD_FT=FALSE ..
    make -j install
    cd ../..
    cp -r oneCCL/build/_install ${CCLFOLDER}
    rm -rf oneCCL

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
