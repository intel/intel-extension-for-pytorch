#!/bin/bash

# Setup environment variables for performance on Xeon
export KMP_BLOCKTIME=1
export KMP_TPAUSE=0
export KMP_FORKJOIN_BARRIER_PATTERN=dist,dist
export KMP_PLAIN_BARRIER_PATTERN=dist,dist
export KMP_REDUCTION_BARRIER_PATTERN=dist,dist

BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export LD_PRELOAD=$(bash ${BASEFOLDER}/get_libstdcpp_lib.sh):${LD_PRELOAD}

env | grep CONDA_PREFIX > /dev/null
if [ $? -eq 0 ]; then
    export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so # Intel OpenMP
    # Tcmalloc is a recommended malloc implementation that emphasizes fragmentation avoidance and scalable concurrency support.
    export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
else
    echo "Conda environment is not available. You need to set environment variable LD_PRELOAD to dynamic libraries of Intel OpenMP and TcMalloc manually."
fi

ONECCL_PATH=${BASEFOLDER}/../oneCCL_release
if [ ! -d ${ONECCL_PATH} ]; then
    echo "Warning: oneCCL is not available."
else
    source ${ONECCL_PATH}/env/setvars.sh
fi
