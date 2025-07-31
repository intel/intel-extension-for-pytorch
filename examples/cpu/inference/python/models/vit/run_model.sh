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

if [[ "${TEST_MODE}" == "THROUGHPUT" ]]; then
    echo "TEST_MODE set to THROUGHPUT"
    num_warmup=${num_warmup:-"10"}
    num_iter=${num_iter:-"100"}
    ARGS="$ARGS --benchmark --perf_begin_iter ${num_warmup} --perf_run_iters ${num_iter}"
    LOG_PREFIX="throughput_log"
elif [[ "${TEST_MODE}" == "ACCURACY" ]]; then
    echo "TEST_MODE set to ACCURACY"
    LOG_PREFIX="accuracy_log"
elif [[ "${TEST_MODE}" == "REALTIME" ]]; then
    echo "TEST_MODE set to REALTIME"
    num_warmup=${num_warmup:-"10"}
    num_iter=${num_iter:-"100"}
    ARGS="$ARGS --benchmark --perf_begin_iter ${num_warmup} --perf_run_iters ${num_iter} "
    LOG_PREFIX="realtime_log"
    if [[ -z "${CORE_PER_INSTANCE}" ]]; then
        echo "The required environment variable CORE_PER_INSTANCE has not been set, please set the cores_per_instance before running, e.g. export CORE_PER_INSTANCE=4"
        exit 1
    fi
    export OMP_NUM_THREADS=${CORE_PER_INSTANCE}
else
    echo "Please set TEST_MODE to THROUGHPUT, ACCURACY, OR REALTIME"
    exit 1
fi

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set, please create the output path and set it to OUTPUT_DIR"
  exit 1
fi

precision="fp32"
if [[ "${PRECISION}" == "bf16" ]]
then
    precision="bf16"
    ARGS="$ARGS --bf16"
    echo "### running bf16 mode"
elif [[ "${PRECISION}" == "fp16" ]]
then
    precision=fp16
    ARGS="$ARGS --fp16_cpu"
    echo "### running fp16 mode"
elif [[ "${PRECISION}" == "fp32" ]]
then
    echo "### running fp32 mode"
elif [[ "${PRECISION}" == "bf32" ]]
then
    precision="bf32"
    ARGS="$ARGS --bf32 --auto_kernel_selection"
    echo "### running bf32 mode"
elif [[ "${PRECISION}" == "int8-fp32" ]]
then
    precision="int8-fp32"
    ARGS="$ARGS --int8"
    echo "### running int8-fp32 mode"
elif [[ "${PRECISION}" == "int8-bf16" ]]
then
    precision="int8-bf16"
    ARGS="$ARGS --bf16 --int8"
    echo "### running int8-bf16 mode"
elif [[ "${PRECISION}" == "fp8-fp32" ]]
then
    precision="fp8-fp32"
    ARGS="$ARGS --fp8 --fp8_config fp8_configure.json"
    echo "### running fp8-fp32 mode"
elif [[ "${PRECISION}" == "fp8-bf16" ]]
then
    precision="fp8-bf16"
    ARGS="$ARGS --bf16 --fp8 --fp8_config fp8_configure.json"
    echo "### running fp8-bf16 mode"
else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions are: fp32, bf32, bf16, fp16, int8-fp32, int8-bf16, fp8-fp32, fp8-bf16"
    exit 1
fi

mkdir -p ${OUTPUT_DIR}
CORES=`lscpu | grep 'Core(s)' | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
FINETUNED_MODEL=${FINETUNED_MODEL:-"google/vit-base-patch16-224"}
DATASET_DIR=${DATASET_DIR:-"None"}
DATASET_ARGS=""
if [[ "1" == ${DUMMY_INPUT} && "${TEST_MODE}" != "ACCURACY" ]];then
    DATASET_ARGS="--dataset_name dummy"
elif [[ "None" == ${DATASET_DIR} ]];then
    DATASET_ARGS="--dataset_name imagenet-1k"
else
    DATASET_ARGS="--train_dir ${DATASET_DIR}/train --validation_dir ${DATASET_DIR}/val"
fi
EVAL_SCRIPT=${EVAL_SCRIPT:-"./transformers/examples/pytorch/image-classification/run_image_classification.py"}
WORK_SPACE=${WORK_SPACE:-${OUTPUT_DIR}}

rm -rf ${OUTPUT_DIR}/${LOG_PREFIX}*
if [[ "${TEST_MODE}" == "REALTIME" ]]; then
    BATCH_SIZE=${BATCH_SIZE:-1}
    CORES_PER_INSTANCE=${OMP_NUM_THREADS}
    TOTAL_CORES=`expr $CORES \* $SOCKETS`
    INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
    INSTANCES_PER_SOCKET=`expr $INSTANCES / $SOCKETS`
    echo "Running inference realtime with torch.compile inductor backend."
    export TORCHINDUCTOR_FREEZING=1
    python -m torch.backends.xeon.run_cpu --disable-numactl --latency-mode --enable_tcmalloc --log_path=${OUTPUT_DIR} \
        ${EVAL_SCRIPT} $ARGS \
        --inductor \
        --model_name_or_path   ${FINETUNED_MODEL} \
        --do_eval \
        --output_dir ${OUTPUT_DIR} \
        --per_device_eval_batch_size $BATCH_SIZE \
        ${DATASET_ARGS} \
        --remove_unused_columns False 2>&1 | tee ${OUTPUT_DIR}/latency_log_${path}_${precision}_${mode}.log
elif [[ "${TEST_MODE}" == "THROUGHPUT" ]]; then
    BATCH_SIZE=${BATCH_SIZE:-`expr 4 \* $CORES`}
    echo "Running inference throughput with torch.compile inductor backend."
    export TORCHINDUCTOR_FREEZING=1
    python -m torch.backends.xeon.run_cpu --disable-numactl --throughput-mode --enable_tcmalloc --log_path=${OUTPUT_DIR} \
        ${EVAL_SCRIPT} $ARGS \
        --inductor \
        --model_name_or_path   ${FINETUNED_MODEL} \
        --do_eval \
        --output_dir ${OUTPUT_DIR} \
        --per_device_eval_batch_size $BATCH_SIZE \
        ${DATASET_ARGS} \
        --remove_unused_columns False 2>&1 | tee ${OUTPUT_DIR}/throughput_log_${path}_${precision}_${mode}.log
elif [[ "${TEST_MODE}" == "ACCURACY" ]]; then
    echo "Running inference accuracy with torch.compile inductor backend."
    export TORCHINDUCTOR_FREEZING=1
    BATCH_SIZE=${BATCH_SIZE:-1}
    python -m torch.backends.xeon.run_cpu --disable-numactl --log_path=${OUTPUT_DIR} \
        ${EVAL_SCRIPT} $ARGS \
        --inductor \
        --model_name_or_path   ${FINETUNED_MODEL} \
        --do_eval \
        --output_dir ${OUTPUT_DIR} \
        --per_device_eval_batch_size $BATCH_SIZE \
        ${DATASET_ARGS} \
        --remove_unused_columns False 2>&1 | tee ${OUTPUT_DIR}/accuracy_log_${path}_${precision}_${mode}.log
fi
latency="N/A"
throughput="N/A"
accuracy="N/A"

if [[ "${TEST_MODE}" == "THROUGHPUT" ]]; then
    throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/throughput_log* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
    BEGIN {
            sum = 0;
        i = 0;
        }
        {
            sum = sum + $1;
    i++;
        }
    END   {
    sum = sum / i;
    printf("%.3f", sum);
    }')
    echo "--------------------------------Performance Summary per NUMA Node--------------------------------"
    echo ""vit-base";"throughput";${precision};${BATCH_SIZE};${throughput}" | tee -a ${WORK_SPACE}/summary.log
elif [[ "${TEST_MODE}" == "ACCURACY" ]]; then
    accuracy=$(cat ${OUTPUT_DIR}/accuracy_log* | grep "eval_accuracy" |sed -e 's/.*= //;s/[^0-9.]//g')
    f1=$(cat ${OUTPUT_DIR}/accuracy_log* | grep "eval_f1" |sed -e 's/.*= //;s/[^0-9.]//g')
    echo ""vit-base";"accuracy";${precision};${BATCH_SIZE};${accuracy}" | tee -a ${WORK_SPACE}/summary.log
elif [[ "${TEST_MODE}" == "REALTIME" ]]; then
    throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/latency_log* |sed -e 's/.*Throughput://;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_PER_SOCKET '
    BEGIN {
        sum = 0;
    i = 0;
        }
        {
            sum = sum + $1;
    i++;
        }
    END   {
    sum = sum / i * INSTANCES_PER_SOCKET;
        printf("%.2f", sum);
    }')
    p99_latency=$(grep 'P99 Latency' ${OUTPUT_DIR}/latency_log* |sed -e 's/.*P99 Latency//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_PER_SOCKET '
    BEGIN {
        sum = 0;
        i = 0;
        }
        {
            sum = sum + $1;
            i++;
        }
    END   {
        sum = sum / i;
        printf("%.3f ms", sum);
    }')
    echo $INSTANCES_PER_SOCKET
    echo "--------------------------------Performance Summary per Socket--------------------------------"
    echo ""vit-base";"latency";${precision};${BATCH_SIZE};${throughput}" | tee -a ${WORK_SPACE}/summary.log
    echo ""vit-base";"p99_latency";${precision};${BATCH_SIZE};${p99_latency}" | tee -a ${WORK_SPACE}/summary.log
fi

yaml_content=$(cat << EOF
results:
- key : throughput
  value: $throughput
  unit: fps
- key: latency
  value: $latency
  unit: ms
- key: accuracy
  value: $accuracy
  unit: AP
EOF
)

echo "$yaml_content" >  $OUTPUT_DIR/results.yaml
echo "YAML file created."
