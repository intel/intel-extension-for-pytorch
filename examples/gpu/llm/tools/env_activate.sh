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

BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )
export LD_PRELOAD=$(bash ${BASEFOLDER}/get_libstdcpp_lib.sh)
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
export CCL_ROOT=${CONDA_PREFIX}
export TORCH_LLM_ALLREDUCE=1

cd ${BASEFOLDER}/../${MODE}
python -m pip install -r ./requirements.txt
PKGDIR=$(python -c 'import transformers; print(transformers.__path__[0]);')
grep "token_latency" -R ${PKGDIR} > /dev/null
if [ $? -gt 0 ]; then
    patch -d ${PKGDIR} -p3 -t < ./patches/transformers.patch
    find ${PKGDIR} -name "*.rej" | while read -r FILE; do FILE=${FILE::-4}; rm ${FILE}.rej; mv ${FILE}.orig ${FILE}; done
    find ${PKGDIR} -name "*.orig" -exec rm {} \;
fi
