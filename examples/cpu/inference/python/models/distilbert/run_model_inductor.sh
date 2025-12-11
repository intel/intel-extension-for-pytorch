#!/bin/bash
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

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"


precision="bf16"
if [[ "$PRECISION" == "bf16" ]]
then
    precision="bf16"
    ARGS="$ARGS --bf16"
    echo "### running bf16 mode"
else
    echo "The specified precision '$PRECISION' is unsupported."
    echo "Supported precisions is bf16"
    exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set, please create the output path and set it to OUTPUT_DIR"
  exit 1
fi
mkdir -p ${OUTPUT_DIR}

if [ -z "${SEQUENCE_LENGTH}" ]; then
  echo "The required environment variable SEQUENCE_LENGTH has not been set, please set the seq_length before running"
  exit 1
fi

SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
CORES=`lscpu | grep Core | awk '{print $4}'`
FINETUNED_MODEL=${FINETUNED_MODEL:-"distilbert-base-uncased-finetuned-sst-2-english"}
TOTAL_CORES=`expr $CORES \* $SOCKETS`
EVAL_SCRIPT=${EVAL_SCRIPT:-"scripts/run_glue.py"}
WORK_SPACE=${WORK_SPACE:-${OUTPUT_DIR}}

if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    ARGS="$ARGS --benchmark --perf_begin_iter 10 --perf_run_iters 100"
    BATCH_SIZE=${BATCH_SIZE:-`expr 4 \* $CORES`}
    echo "Running throughput"
    rm -rf ${OUTPUT_DIR}/distilbert_throughput*
elif [[ "$TEST_MODE" == "REALTIME" ]]; then
    ARGS="$ARGS --benchmark --perf_begin_iter 500 --perf_run_iters 2000"
    export OMP_NUM_THREADS=${CORES_PER_INSTANCE:-4}
    SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
    NUMAS=`lscpu | grep 'NUMA node(s)' | awk '{print $3}'`
    CORES_PER_NUMA=`expr $CORES \* $SOCKETS / $NUMAS`
    BATCH_SIZE=${BATCH_SIZE:-1}
    ARGS="$ARGS --use_share_weight --total_cores ${CORES_PER_NUMA} --cores_per_instance ${OMP_NUM_THREADS}"
    echo "Running realtime inference"
    rm -rf ${OUTPUT_DIR}/distilbert_latency*
elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
    BATCH_SIZE=${BATCH_SIZE:-1}
    echo "Running accuracy"
    rm -rf ${OUTPUT_DIR}/distilbert_accuracy*
elif [[ "$TEST_MODE" == "" ]]; then
    ARGS="$ARGS --benchmark --perf_begin_iter 10 --perf_run_iters 100"
    BATCH_SIZE=${BATCH_SIZE:-1}
    echo "Running in mode with custom CORES_PER_INSTANCE and INSTANCES"
fi
if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
        echo "Running inference with torch.compile inductor backend."
        export TORCHINDUCTOR_FREEZING=1
        ARGS="$ARGS --inductor"
        python -m torch.backends.xeon.run_cpu --disable-numactl --enable-jemalloc --throughput-mode --skip-cross-node-cores \
            ${EVAL_SCRIPT} $ARGS \
            --model_name_or_path   ${FINETUNED_MODEL} \
            --task_name sst2 \
            --do_eval \
            --max_seq_length ${SEQUENCE_LENGTH} \
            --output_dir ${OUTPUT_DIR} \
            --per_device_eval_batch_size $BATCH_SIZE \
            --dataloader_drop_last \
            > "${OUTPUT_DIR}/distilbert_throughput_${path}_${precision}.log" 2>&1
elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
        echo "Running inference with torch.compile inductor backend."
        export TORCHINDUCTOR_FREEZING=1
        ARGS="$ARGS --inductor"
        python -m torch.backends.xeon.run_cpu --disable-numactl \
            ${EVAL_SCRIPT} $ARGS \
            --model_name_or_path   ${FINETUNED_MODEL} \
            --task_name sst2 \
            --do_eval \
            --max_seq_length ${SEQUENCE_LENGTH} \
            --output_dir ${OUTPUT_DIR} \
            --per_device_eval_batch_size $BATCH_SIZE \
            > "${OUTPUT_DIR}/distilbert_accuracy_${path}_${precision}.log" 2>&1
elif [[ "$TEST_MODE" == "REALTIME" ]]; then
        echo "Running inference with torch.compile inductor backend."
        export TORCHINDUCTOR_FREEZING=1
        ARGS="$ARGS --inductor"
        python -m torch.backends.xeon.run_cpu --disable-numactl --ninstances $NUMAS --enable-jemalloc \
             ${EVAL_SCRIPT} $ARGS \
             --model_name_or_path   ${FINETUNED_MODEL} \
             --task_name sst2 \
             --do_eval \
             --max_seq_length ${SEQUENCE_LENGTH} \
             --output_dir ${OUTPUT_DIR} \
             --per_device_eval_batch_size $BATCH_SIZE \
             > "${OUTPUT_DIR}/distilbert_latency_${path}_${precision}.log" 2>&1
elif [[ "$TEST_MODE" == "" ]]; then
        echo "Running inference with torch.compile inductor backend."
        export TORCHINDUCTOR_FREEZING=1
        ARGS="$ARGS --inductor"
        : "${CORES_PER_INSTANCE:?Please export CORES_PER_INSTANCE=<set a value for cores per instance>}"
        : "${INSTANCES:?Please export INSTANCES=<set a value for instances>}"
        python -m torch.backends.xeon.run_cpu --disable-numactl --enable-jemalloc --ncores-per-instance ${CORES_PER_INSTANCE} --ninstances ${INSTANCES}  \
            ${EVAL_SCRIPT} $ARGS \
            --model_name_or_path   ${FINETUNED_MODEL} \
            --task_name sst2 \
            --do_eval \
            --max_seq_length ${SEQUENCE_LENGTH} \
            --output_dir ${OUTPUT_DIR} \
            --per_device_eval_batch_size $BATCH_SIZE \
            --dataloader_drop_last \
            > "${OUTPUT_DIR}/distilbert_${path}_${precision}.log" 2>&1
fi