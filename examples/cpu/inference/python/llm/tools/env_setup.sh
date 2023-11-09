#!/usr/bin/env bash
set -e

LM_EVA_COMMIT=cc9778fbe4fa1a709be2abed9deb6180fd40e7e2
# gma/run-opt-branch
DS_SYCL_COMMIT=57ff508ea592ff752fd323b383c32177d5bce7b5
TORCHCCL_COMMIT=v2.1.0+cpu
ONECCL_COMMIT=bfc879266e870b732bd165e399897419c44ad13d
VER_GCC=12.3.0
AUX_INSTALL_SCRIPT=torch_install.sh

# Mode: Select to compile projects into wheel files or install wheel files compiled.
# High bit: 8 7 6 5 4 3 2 1 :Low bit
#           | | | | | | | └- Install wheel files
#           | | | | | | └--- Compile wheel files
#           | | | | | └----- Undefined
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
if [ ! -f ${AUX_INSTALL_SCRIPT} ] ||
   [ ! -d intel-extension-for-pytorch ] ||
   [ ! -d lm-evaluation-harness ] ||
   [ ! -d torch-ccl ] ||
   [ ! -d DeepSpeedSYCLSupport ]; then
    (( MODE |= 0x02 ))
fi

if [ $((${MODE} & 0x02)) -ne 0 ]; then
    # Check existance of required Linux commands
    for CMD in conda gcc g++ make git; do
        command -v ${CMD} > /dev/null || (echo "Error: Command \"${CMD}\" is required."; exit 1;)
    done
    echo "#!/usr/bin/env bash" > ${AUX_INSTALL_SCRIPT}

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
        conda install -y gcc==12.3 gxx==12.3 cxx-compiler -c conda-forge
        conda update -y sysroot_linux-64
        export CC=${CONDA_PREFIX}/bin/gcc
        export CXX=${CONDA_PREFIX}/bin/g++
        export PATH=${CONDA_PREFIX}/bin:${PATH}
        export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
    fi

    # Install deps
    conda install -y cmake ninja mkl mkl-include

    # Install PyTorch and Intel® Extension for PyTorch*
    if [ -f compile_bundle.sh ]; then
        rm compile_bundle.sh
    fi
    wget https://github.com/intel/intel-extension-for-pytorch/raw/llm_feature_branch/scripts/compile_bundle.sh
    set +e
    grep "VER_TORCH=" compile_bundle.sh >> ${AUX_INSTALL_SCRIPT}
    grep "pip install torch==" compile_bundle.sh >> ${AUX_INSTALL_SCRIPT}
    grep "pip install torch " compile_bundle.sh >> ${AUX_INSTALL_SCRIPT}
    set -e
    bash compile_bundle.sh 0
    rm -rf compile_bundle.sh llvm-project llvm-release

    # Used for accuracy test only
    if [ -d lm-evaluation-harness ]; then
        rm -rf lm-evaluation-harness
    fi
    git clone https://github.com/EleutherAI/lm-evaluation-harness
    cd lm-evaluation-harness
    git checkout ${LM_EVA_COMMIT}
    python setup.py bdist_wheel
    cd ..

    # The following is only for DeepSpeed case
    #Install oneccl-bind-pt(also named torch-ccl)
    if [ -d torch-ccl ]; then
        rm -rf torch-ccl
    fi
    git clone https://github.com/intel/torch-ccl.git
    cd torch-ccl
    git checkout ${TORCHCCL_COMMIT}
    git submodule sync
    git submodule update --init --recursive
    python setup.py bdist_wheel
    python -m pip install dist/*.whl
    cd ..

    # Install DeepSpeed
    if [ -d DeepSpeedSYCLSupport ]; then
        rm -rf DeepSpeedSYCLSupport
    fi
    git clone https://github.com/delock/DeepSpeedSYCLSupport
    cd DeepSpeedSYCLSupport
    git checkout ${DS_SYCL_COMMIT}
    python -m pip install -r requirements/requirements.txt
    python setup.py bdist_wheel
    cd ..

    # Install OneCCL
    if [ -d oneCCL ]; then
        rm -rf oneCCL
    fi
    git clone https://github.com/oneapi-src/oneCCL.git
    cd oneCCL
    git checkout ${ONECCL_COMMIT}
    mkdir build
    cd build
    cmake ..
    make -j install
    cd ../..
fi
if [ $((${MODE} & 0x01)) -ne 0 ]; then
    conda install -y mkl
    conda install -y gperftools -c conda-forge
    bash ${AUX_INSTALL_SCRIPT}
    python -m pip install cpuid accelerate datasets sentencepiece protobuf==3.20.3 transformers==4.31.0 neural-compressor==2.3.1
    python -m pip install intel-extension-for-pytorch/dist/*.whl
    python -m pip install lm-evaluation-harness/dist/*.whl
    python -m pip install torch-ccl/dist/*.whl
    python -m pip install DeepSpeedSYCLSupport/dist/*.whl

    rm ${AUX_INSTALL_SCRIPT}
    rm -rf intel-extension-for-pytorch
    rm -rf lm-evaluation-harness
    rm -rf torch-ccl
    rm -rf DeepSpeedSYCLSupport
fi
