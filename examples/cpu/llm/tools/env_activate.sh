#!/bin/bash

# Setup environment variables for performance on Xeon
export KMP_BLOCKTIME=1
export KMP_TPAUSE=0
export KMP_FORKJOIN_BARRIER_PATTERN=dist,dist
export KMP_PLAIN_BARRIER_PATTERN=dist,dist
export KMP_REDUCTION_BARRIER_PATTERN=dist,dist

BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )
export LD_PRELOAD=$(bash ${BASEFOLDER}/get_libstdcpp_lib.sh):${LD_PRELOAD}

function set_ld_preload() {
    LIB_DIR=$1
    LIB_NAME=$2
    LIB=
    while read line; do
        LIB=${line}
        break
    done < <(find ${LIB_DIR} -name ${LIB_NAME})
    if [ ! -z ${LIB} ]; then
        export LD_PRELOAD=${LD_PRELOAD}:${LIB}
        echo "Appending ${LIB} to environment variable LD_PRELOAD."
    else
        echo "Library ${LIB_NAME} is not found. Please append it manually to environment variable LD_PRELOAD."
    fi
}

env | grep CONDA_PREFIX > /dev/null
if [ $? -eq 0 ]; then
    set_ld_preload ${CONDA_PREFIX} libiomp5.so
    set_ld_preload ${CONDA_PREFIX} libtcmalloc.so
else
    set_ld_preload /usr libiomp5.so
    set_ld_preload /usr libtcmalloc.so
fi

cd ${BASEFOLDER}/../inference

ONECCL_PATH=${BASEFOLDER}/../oneCCL_release
if [ ! -d ${ONECCL_PATH} ]; then
	echo "Warning: oneCCL is not available."
else
	source ${ONECCL_PATH}/env/setvars.sh
fi

python -m pip install -r requirements.txt
if [ -f prompt.json ]; then
	rm -f prompt.json
fi
if [ -f prompt-qwen3-moe.json ]; then
	rm -f prompt-qwen3-moe.json
fi
wget https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/prompt.json
wget https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/prompt-qwen3-moe.json

