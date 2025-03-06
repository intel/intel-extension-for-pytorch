#!/bin/bash

MSG_USAGE="Usage: source $0 [inference|fine-tuning|bitsandbytes]"
if [ $# -eq 0 ]; then
    echo ${MSG_USAGE}
    return 1
fi
MODE=$1
if [ ${MODE} != "inference" ] && [ ${MODE} != "fine-tuning" ] && [ ${MODE} != "bitsandbytes" ]; then
    echo ${MSG_USAGE}
    return 2
fi

BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )
export LD_PRELOAD=$(bash ${BASEFOLDER}/get_libstdcpp_lib.sh)
export TORCH_LLM_ALLREDUCE=1

cd ${BASEFOLDER}/../${MODE}
python -m pip install -r ./requirements.txt
if [ ${MODE} == "bitsandbytes" ]; then
    git clone -b multi-backend-refactor https://github.com/bitsandbytes-foundation/bitsandbytes.git
    cd bitsandbytes
    pip install .
    cd ..
fi
python -m pip install dpcpp-cpp-rt==2025.0.4
if [ -f ./patches/transformers.patch ]; then
    PKGDIR=
    while read -r line
    do
        if [[ ! $line =~ ^\[ ]]; then
            PKGDIR=${line}
        fi
    done < <(python -c 'import transformers; print(transformers.__path__[0]);')
    if [ ! -d ${PKGDIR} ]; then
        echo "Transformers not found. Skipping performance metrics patching..."
    else
        echo "Patching Transformers for performance metrics enabling..."
        grep "token_latency" -R ${PKGDIR} > /dev/null
        if [ $? -gt 0 ]; then
            patch -d ${PKGDIR} -p3 -t < ./patches/transformers.patch
            find ${PKGDIR} -name "*.rej" | while read -r FILE; do FILE=${FILE::-4}; rm ${FILE}.rej; mv ${FILE}.orig ${FILE}; done
            find ${PKGDIR} -name "*.orig" -exec rm {} \;
        fi
    fi
fi
