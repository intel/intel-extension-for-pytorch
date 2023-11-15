#!/bin/bash

# Setup environment variables for performance on Xeon
export KMP_BLOCKTIME=INF
export KMP_TPAUSE=0
export KMP_SETTINGS=1
export KMP_FORJOIN_BARRIER_PATTERN=dist,dist
export KMP_PLAIN_BARRIER_PATTERN=dist,dist
export KMP_REDUCTION_BARRIER_PATTERN=dist,dist

env | grep CONDA_PREFIX > /dev/null
if [ $? -eq 0 ]; then
    export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so.6
    export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so # Intel OpenMP
    # Tcmalloc is a recommended malloc implementation that emphasizes fragmentation avoidance and scalable concurrency support.
    export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
else
    echo "Conda environment is not available. You need to set environment variable LD_PRELOAD to dynamic libraries of Intel OpenMP and TcMalloc manually."
fi

ONECCL_PATH=./oneCCL/build/_install
if [ ! -d ${ONECCL_PATH} ]; then
    echo "oneCCL is not available."
else
    source ${ONECCL_PATH}/env/setvars.sh
fi
