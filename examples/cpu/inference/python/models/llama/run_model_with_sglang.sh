#!/bin/bash

ARGS=""

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

FINETUNED_MODEL=${FINETUNED_MODEL:-"meta-llama/Llama-3.1-8B-Instruct"}
NUMAS=`lscpu | grep 'NUMA node(s)' | awk '{print $3}'`

if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    echo "Running Multi-instance Throughput Inference"
    export LOG_PREFIX="throughput_log"
    BATCH_SIZE=${BATCH_SIZE:-1}
    export KMP_BLOCKTIME=1
    rm -rf ${OUTPUT_DIR}/throughput_log*
    export usecase=throughput
    CORES=`lscpu | grep Core | awk '{print $4}'`
    SOCKETS=`lscpu | grep Socket | awk '{print $2}'`

elif [[ "$TEST_MODE" == "REALTIME" ]]; then
    echo "Running Multi-instance Realtime Inference"
    export LOG_PREFIX="latency_log"
    BATCH_SIZE=${BATCH_SIZE:-1}
    export KMP_BLOCKTIME=-1
    rm -rf ${OUTPUT_DIR}/latency_log*
    export usecase=latency
    CORES=`lscpu | grep Core | awk '{print $4}'`
    SOCKETS=`lscpu | grep Socket | awk '{print $2}'`

else
    echo "Please set TEST_MODE to THROUGHPUT or REALTIME"
    exit
fi

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set, please create the output path and set it to OUTPUT_DIR"
  exit 1
fi

mkdir -p ${OUTPUT_DIR}

if [[ "${PRECISION}" == "bf16" ]]
then
    precision="bf16"
    ARGS="$ARGS --dtype bfloat16 "
    echo "### running bf16 mode"
elif [[ "${PRECISION}" == "fp16" ]]
then
    precision=fp16
    ARGS="$ARGS --dtype float16 "
    echo "### running fp16 mode"
elif [[ "${PRECISION}" == "int8" ]]
then
    precision="int8"
    ARGS="$ARGS --quantization w8a8_int8 "
    echo "### running int8 mode"
elif [[ "${PRECISION}" == "fp8" ]]
then
    precision="fp8"
    echo "### running fp8 mode"
else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions are: bf16, fp16, int8, fp8"
    exit 1
fi


if [ -z "${OUTPUT_TOKEN}" ]; then
    echo "The required environment variable OUTPUT_TOKEN has not been set, please set before running, e.g. export OUTPUT_TOKEN=32"
    exit 1
fi

if [ -z "${INPUT_TOKEN}" ]; then
    echo "The required environment variable INPUT_TOKEN has not been set, please set before running (choice in 32 64 128 512 1024 2016 ), e.g. export INPUT_TOKEN=1024"
    exit 1
fi

get_core_count() {
    local range_str=$1
    local total=0
    IFS=',' read -ra ranges <<< "$range_str"
    for r in "${ranges[@]}"; do
        if [[ "$r" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            start=${BASH_REMATCH[1]}
            end=${BASH_REMATCH[2]}
            total=$((total + end - start + 1))
        elif [[ "$r" =~ ^[0-9]+$ ]]; then
            total=$((total + 1))
        fi
    done
    echo "$total"
}

for ((i=1; i<NUMAS; i++)); do
    echo "Launching task on NUMA node $i..."

    CPU_LIST=$(lscpu | grep "NUMA node$i CPU(s):" | awk -F: '{print $2}' | sed 's/^ //g' | cut -d',' -f1 | tr -d ' ')
    CORES_PER_INSTANCE=$(get_core_count "$CPU_LIST")
    LOG_FILE="${OUTPUT_DIR}/${usecase}_${precision}_bs${BATCH_SIZE}_node${i}.log"
    echo "OMP_NUM_THREADS=${CORES_PER_INSTANCE} numactl -C ${CPU_LIST} -m ${i} python3 -m sglang.bench_one_batch ${ARGS} --batch-size ${BATCH_SIZE} --model ${FINETUNED_MODEL} --trust-remote-code --device cpu --input ${INPUT_TOKEN} --output ${OUTPUT_TOKEN} --disable-overlap-schedule --prompt-filename prompt.json 2>&1 | tee ${LOG_FILE} &"
    OMP_NUM_THREADS=${CORES_PER_INSTANCE} numactl -C ${CPU_LIST} -m ${i} python3 -m sglang.bench_one_batch ${ARGS} --batch-size ${BATCH_SIZE} --model ${FINETUNED_MODEL} --trust-remote-code --device cpu --input ${INPUT_TOKEN} --output ${OUTPUT_TOKEN} --disable-overlap-schedule --prompt-filename prompt.json 2>&1 | tee ${LOG_FILE} &
done
CPU_LIST=$(lscpu | grep "NUMA node0 CPU(s):" | awk -F: '{print $2}' | sed 's/^ //g' | cut -d',' -f1 | tr -d ' ')
CORES_PER_INSTANCE=$(get_core_count "$CPU_LIST")
LOG_FILE="${OUTPUT_DIR}/${usecase}_${precision}_bs${BATCH_SIZE}_node0.log"
echo "OMP_NUM_THREADS=${CORES_PER_INSTANCE} numactl -C ${CPU_LIST} -m 0 python3 -m sglang.bench_one_batch ${ARGS} --batch-size ${BATCH_SIZE} --model ${FINETUNED_MODEL} --trust-remote-code --device cpu --input ${INPUT_TOKEN} --output ${OUTPUT_TOKEN} --disable-overlap-schedule --prompt-filename prompt.json 2>&1 | tee ${LOG_FILE}"
OMP_NUM_THREADS=${CORES_PER_INSTANCE} numactl -C ${CPU_LIST} -m 0 python3 -m sglang.bench_one_batch ${ARGS} --batch-size ${BATCH_SIZE} --model ${FINETUNED_MODEL} --trust-remote-code --device cpu --input ${INPUT_TOKEN} --output ${OUTPUT_TOKEN} --disable-overlap-schedule --prompt-filename prompt.json 2>&1 | tee ${LOG_FILE}

wait