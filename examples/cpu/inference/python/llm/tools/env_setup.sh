#!/usr/bin/env bash
set -e

# Save current directory path
BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
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
    cd ../../../../..

    if [ ! -f dependency_version.yml ]; then
        echo "Please check if `pwd` is a valid Intel® Extension for PyTorch* source code directory."
        exit 2
    fi
    python -m pip install pyyaml
    REPO_LM_EVA=$(python tools/yaml_utils.py -f dependency_version.yml -d lm-evaluation-harness -k repo)
    COMMIT_LM_EVA=$(python tools/yaml_utils.py -f dependency_version.yml -d lm-evaluation-harness -k commit)
    REPO_DS_SYCL=$(python tools/yaml_utils.py -f dependency_version.yml -d deepspeed -k repo)
    COMMIT_DS_SYCL=$(python tools/yaml_utils.py -f dependency_version.yml -d deepspeed -k commit)
    VER_DS_SYCL=$(python tools/yaml_utils.py -f dependency_version.yml -d deepspeed -k version)
    VER_TORCHCCL=$(python tools/yaml_utils.py -f dependency_version.yml -d torch-ccl -k version)
    REPO_ONECCL=$(python tools/yaml_utils.py -f dependency_version.yml -d oneCCL -k repo)
    COMMIT_ONECCL=$(python tools/yaml_utils.py -f dependency_version.yml -d oneCCL -k commit)
    VER_GCC=$(python tools/yaml_utils.py -f dependency_version.yml -d gcc -k min-version)
    VER_TORCH=$(python tools/yaml_utils.py -f dependency_version.yml -d pytorch -k version)
    VER_TRANSFORMERS=$(python tools/yaml_utils.py -f dependency_version.yml -d transformers -k version)
    VER_PROTOBUF=$(python tools/yaml_utils.py -f dependency_version.yml -d protobuf -k version)
    VER_INC=$(python tools/yaml_utils.py -f dependency_version.yml -d neural-compressor -k version)
    VER_IPEX_MAJOR=$(grep "VERSION_MAJOR" version.txt | cut -d " " -f 2)
    VER_IPEX_MINOR=$(grep "VERSION_MINOR" version.txt | cut -d " " -f 2)
    VER_IPEX_PATCH=$(grep "VERSION_PATCH" version.txt | cut -d " " -f 2)
    VER_IPEX="${VER_IPEX_MAJOR}.${VER_IPEX_MINOR}.${VER_IPEX_PATCH}+cpu"
    python -m pip uninstall -y pyyaml
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
    python -m pip install cmake==3.28.4 ninja unzip

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
            echo "python -m pip install intel-extension-for-pytorch==${VER_IPEX} oneccl-bind-pt==${VER_TORCHCCL} --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/" >> ${AUX_INSTALL_SCRIPT}
            python -m pip install torch==${VER_TORCH} --index-url https://download.pytorch.org/whl/cpu
            python -m pip install intel-extension-for-pytorch==${VER_IPEX} oneccl-bind-pt==${VER_TORCHCCL} --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
        fi
    else
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
        if [ ${VER_COMP} -ne 0 ]; then
            echo -e '\a'
            echo "Warning: GCC version equal to or newer than ${VER_GCC} is required."
            echo "         Found GCC version $(gcc -dumpfullversion)"
            echo "         Installing gcc and g++ 12.3 with conda"
            echo ""
            set +e
            command -v conda > /dev/null
            EXIST_CONDA=$?
            set -e
            if [ ${EXIST_CONDA} -gt 0 ]; then
                echo "[Error] Command \"conda\" is not available."
                exit 5
            else
                conda install -y sysroot_linux-64
                conda install -y gcc==12.3 gxx==12.3 cxx-compiler -c conda-forge
                if [ -z ${CONDA_BUILD_SYSROOT} ]; then
                    source ${CONDA_PREFIX}/etc/conda/activate.d/activate-gcc_linux-64.sh
                    source ${CONDA_PREFIX}/etc/conda/activate.d/activate-gxx_linux-64.sh
                    source ${CONDA_PREFIX}/etc/conda/activate.d/activate-binutils_linux-64.sh
                fi
            fi
        fi

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
        cp intel-extension-for-pytorch/scripts/compile_bundle.sh .
        sed -i "s/VER_IPEX=.*/VER_IPEX=/" compile_bundle.sh
        bash compile_bundle.sh 1
        cp intel-extension-for-pytorch/dist/*.whl ${WHEELFOLDER}
        cp torch-ccl/dist/*.whl ${WHEELFOLDER}
        rm -rf compile_bundle.sh llvm-project llvm-release torch-ccl
    fi

    echo "python -m pip install cpuid accelerate datasets sentencepiece mkl protobuf==${VER_PROTOBUF} transformers==${VER_TRANSFORMERS} neural-compressor==${VER_INC} transformers_stream_generator tiktoken" >> ${AUX_INSTALL_SCRIPT}

    # Used for accuracy test only
    if [ -d lm-evaluation-harness ]; then
        rm -rf lm-evaluation-harness
    fi
    git clone ${REPO_LM_EVA} lm-evaluation-harness
    cd lm-evaluation-harness
    git checkout ${COMMIT_LM_EVA}
    python setup.py bdist_wheel
    cp dist/*.whl ${WHEELFOLDER}
    cd ..
    rm -rf lm-evaluation-harness

    # Install DeepSpeed
    if [ $((${MODE} & 0x08)) -ne 0 ]; then
        if [ -d DeepSpeed ]; then
            rm -rf DeepSpeed
        fi
        git clone ${REPO_DS_SYCL} DeepSpeed
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

    # Install OneCCL
    if [ -d oneCCL ]; then
        rm -rf oneCCL
    fi
    git clone ${REPO_ONECCL} oneCCL
    cd oneCCL
    git checkout ${COMMIT_ONECCL}
    mkdir build
    cd build
    cmake -DBUILD_EXAMPLES=FALSE -DBUILD_FT=FALSE ..
    make -j install
    cd ../..
    cp -r oneCCL/build/_install ${CCLFOLDER}
    rm -rf oneCCL
    cd intel-extension-for-pytorch/examples/cpu/inference/python/llm
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
    if [ -f prompt.json ]; then
        rm -f prompt.json
    fi
    wget https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/prompt.json
    cd single_instance
    if [ -f prompt.json ]; then
        rm -f prompt.json
    fi
    ln -s ../prompt.json
    cd ../distributed
    if [ -f prompt.json ]; then
        rm -f prompt.json
    fi
    ln -s ../prompt.json
fi
python -m pip install numpy==1.26.4 --force-reinstall