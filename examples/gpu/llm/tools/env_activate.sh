#!/bin/bash

MSG_USAGE="Usage: source $0 [inference|fine-tuning]"
if [ $# -eq 0 ]; then
    echo ${MSG_USAGE}
    return 1
fi
MODE=$1
if [ ${MODE} != "inference" ] && [ ${MODE} != "fine-tuning" ]; then
    echo ${MSG_USAGE}
    return 2
fi

BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export LD_PRELOAD=$(bash ${BASEFOLDER}/get_libstdcpp_lib.sh)
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
export CCL_ROOT=${CONDA_PREFIX}
export TORCH_LLM_ALLREDUCE=1

cd ${BASEFOLDER}/../${MODE}
if [ ${MODE} == "fine-tuning" ]; then
    python -m pip install -r requirements.txt
fi
