#!/usr/bin/env bash
#
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

ARGS=""
ARGS_PYTORCH=""

if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    echo "TEST_MODE set to THROUGHPUT"
    LOG_PREFIX="resnet50_throughput_log"
elif [[ "$TEST_MODE" == "REALTIME" ]]; then
    echo "TEST_MODE set to REALTIME"
    LOG_PREFIX="resnet50_realtime_log"
elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
    echo "TEST_MODE set to ACCURACY"
    LOG_PREFIX="resnet50_accuracy_log"
else
    echo "TEST_MODE not set to THROUGHPUT, REALTIME or ACCURACY"
    echo "Please set CORES_PER_INSTANCE and INSTANCES to specify cores per instance and number of instances"

    if [ -z "${CORES_PER_INSTANCE}" ]; then
        echo "The required environment variable CORES_PER_INSTANCE has not been set"
        exit 1
    fi
    
    if [ -z "${INSTANCES}" ]; then
        echo "The required environment variable INSTANCES has not been set"
        exit 1
    fi
    
    LOG_PREFIX="resnet50_custom_log"
fi

MODEL_DIR=${MODEL_DIR-$PWD}
if [ ! -e "${MODEL_DIR}/common/main.py"  ]; then
    echo "Could not find the script of main.py. Please set environment variable '\${MODEL_DIR}'."
    echo "From which the main.py exist at the: \${MODEL_DIR}/main.py"
    exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}
rm -rf ${OUTPUT_DIR}/summary.log
rm -rf ${OUTPUT_DIR}/results.yaml

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to bf16."
  exit 1
fi

rm -rf "${OUTPUT_DIR}/${LOG_PREFIX}_*"

if [[ $PRECISION == "bf16" ]]; then
    ARGS="$ARGS --bf16 --jit"
    echo "running bf16 path"
else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions are: bf16"
    exit 1
fi

export TORCH_INDUCTOR=1
export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so:${CONDA_PREFIX}/lib/libtcmalloc.so
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
NUMAS=`lscpu | grep 'NUMA node(s)' | awk '{print $3}'`
CORES_PER_NUMA=`expr $CORES \* $SOCKETS / $NUMAS`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

ARGS_PYTORCH="$ARGS_PYTORCH --disable-numactl --enable-jemalloc --log_path="${OUTPUT_DIR}" --log_file_prefix="./${LOG_PREFIX}_${PRECISION}""

if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    CORES_PER_INSTANCE=$CORES
    INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
    ARGS="$ARGS -e -a resnet50 ../ --dummy"
    #ARGS_PYTORCH="$ARGS_PYTORCH --throughput_mode"
    ARGS_PYTORCH="$ARGS_PYTORCH --ncores-per-instance ${CORES_PER_INSTANCE} --ninstances ${INSTANCES}"
    BATCH_SIZE=${BATCH_SIZE:-112}
elif [[ "$TEST_MODE" == "REALTIME" ]]; then
    CORES_PER_INSTANCE=4
    NUMBER_INSTANCE=`expr $CORES_PER_NUMA / $CORES_PER_INSTANCE`
    ARGS="$ARGS -e -a resnet50 ../ --dummy --weight-sharing --number-instance $NUMBER_INSTANCE"
    BATCH_SIZE=${BATCH_SIZE:-1}
elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
    python ${MODEL_DIR}/common/hub_help.py \
        --url https://download.pytorch.org/models/resnet50-0676ba61.pth
    ARGS="$ARGS --pretrained -e -a resnet50 ${DATASET_DIR}"
    BATCH_SIZE=${BATCH_SIZE:-128}
else
    echo "Running in custom mode with CORES_PER_INSTANCE=${CORES_PER_INSTANCE} and INSTANCES=${INSTANCES}"
    ARGS="$ARGS -e -a resnet50 ../ --dummy"
    ARGS_PYTORCH="$ARGS_PYTORCH --ncores-per-instance ${CORES_PER_INSTANCE} --ninstances ${INSTANCES}"
    BATCH_SIZE=${BATCH_SIZE:-112}
fi

echo "Running RN50 inference with torch.compile inductor backend."
export TORCHINDUCTOR_FREEZING=1
python -m torch.backends.xeon.run_cpu \
    ${ARGS_PYTORCH} \
    ${MODEL_DIR}/common/main.py \
    $ARGS \
    --inductor \
    --seed 2020 \
    -j 0 \
    -b $BATCH_SIZE
