#!/bin/bash

MSG_USAGE="Usage: source $0 [inference|fine-tuning|bitsandbytes|training]"
if [ $# -eq 0 ]; then
    echo ${MSG_USAGE}
    return 1
fi
MODE=$1
if [ ${MODE} != "inference" ] && [ ${MODE} != "fine-tuning" ] && [ ${MODE} != "bitsandbytes" ] && [ ${MODE} != "training" ]; then
    echo ${MSG_USAGE}
    return 2
fi

BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )
export LD_PRELOAD=$(bash ${BASEFOLDER}/get_libstdcpp_lib.sh)
export TORCH_LLM_ALLREDUCE=1

python ${BASEFOLDER}/env_activate.py ${MODE}
cd ${BASEFOLDER}/../${MODE}
