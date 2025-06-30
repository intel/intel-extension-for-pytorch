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

FINETUNED_MODEL=${FINETUNED_MODEL:-"meta-llama/Llama-2-7b-hf"}
NUMAS=`lscpu | grep 'NUMA node(s)' | awk '{print $3}'`

if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    echo "Running Multi-instance Throughput Inference"
    if [[ "${PRECISION}" == "bf16" ]]; then
        export BF16_OPTIMIZED_THROUGHPUT=1
    fi
    export LOG_PREFIX="throughput_log"
    BATCH_SIZE=${BATCH_SIZE:-1}
    export KMP_BLOCKTIME=1
    rm -rf ${OUTPUT_DIR}/throughput_log*
    export usecase=throughput
    NUM_WARMUP=${NUM_WARMUP:-10}
    NUM_ITER=${NUM_ITER:-20}
    ARGS="$ARGS  --benchmark --num-warmup ${NUM_WARMUP} --num-iter $NUM_ITER --token-latency"
    CORES=`lscpu | grep Core | awk '{print $4}'`
    SOCKETS=`lscpu | grep Socket | awk '{print $2}'`

elif [[ "$TEST_MODE" == "REALTIME" ]]; then
    echo "Running Multi-instance Realtime Inference"
    export LOG_PREFIX="latency_log"
    BATCH_SIZE=${BATCH_SIZE:-1}
    export KMP_BLOCKTIME=-1
    rm -rf ${OUTPUT_DIR}/latency_log*
    export usecase=latency
    NUM_WARMUP=${NUM_WARMUP:-10}
    NUM_ITER=${NUM_ITER:-20}
    ARGS="$ARGS  --benchmark --num-warmup ${NUM_WARMUP} --num-iter $NUM_ITER --token-latency"
    CORES=`lscpu | grep Core | awk '{print $4}'`
    SOCKETS=`lscpu | grep Socket | awk '{print $2}'`

elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
    echo "TEST_MODE set to ACCURACY"
    export LOG_PREFIX="accuracy_log"
    ARGS="$ARGS --accuracy_only  --lambada"
    rm -rf ${OUTPUT_DIR}/*accuracy*
else
    echo "Please set TEST_MODE to THROUGHPUT, REALTIME or ACCURACY"
    exit
fi

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set, please create the output path and set it to OUTPUT_DIR"
  exit 1
fi

mkdir -p ${OUTPUT_DIR}
EVAL_SCRIPT=${EVAL_SCRIPT:-"${PWD}/run_llm_inductor_greedy.py"}
WORK_SPACE=${WORK_SPACE:-${OUTPUT_DIR}}
TORCH_INDUCTOR=${TORCH_INDUCTOR:-"0"}

if [[ "${PRECISION}" == "fp32" ]]
then
    precision="fp32"
    ARGS="$ARGS --dtype fp32 "
    echo "### running fp32 mode"
elif [[ "${PRECISION}" == "bf16" ]]
then
    precision="bf16"
    ARGS="$ARGS --dtype bf16 "
    echo "### running bf16 mode"
elif [[ "${PRECISION}" == "fp16" ]]
then
    precision=fp16
    ARGS="$ARGS --dtype fp16"
    echo "### running fp16 mode"
elif [[ "${PRECISION}" == "bf32" ]]
then
    precision="bf32"
    ARGS="$ARGS --dtype bf32"
    echo "### running bf32 mode"
elif [[ "${PRECISION}" == "int8-fp32" ]]
then
    precision="int8-fp32"
    ARGS="$ARGS --dtype int8 --int8-qconfig   ${OUTPUT_DIR}/${MODEL_HF}-qconfig.json"
    echo "### running int8-fp32 mode"
elif [[ "${PRECISION}" == "int8-bf16" ]] || [[ "${PRECISION}" == "int8" ]]
then
    precision="int8-bf16"
    ARGS="$ARGS --dtype int8-bf16 --int8_bf16_mixed --int8-qconfig ${OUTPUT_DIR}/${MODEL_HF}-qconfig.json"
    echo "### running int8-bf16 mode"
else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions are: fp32, bf32, bf16, fp16, int8-fp32, int8-bf16"
    exit 1
fi


if [[ "$TEST_MODE" != "ACCURACY" ]]; then
    if [ -z "${OUTPUT_TOKEN}" ]; then
        echo "The required environment variable OUTPUT_TOKEN has not been set, please set before running, e.g. export OUTPUT_TOKEN=32"
        exit 1
    fi

    if [ -z "${INPUT_TOKEN}" ]; then
        echo "The required environment variable INPUT_TOKEN has not been set, please set before running (choice in 32 64 128 512 1024 2016 ), e.g. export INPUT_TOKEN=1024"
        exit 1
    fi

    echo "### running with torch.compile inductor backend"
    if [[ "${PRECISION}" == *"int8"* ]];then
        if [ "${INT8_QUANT_TYPE}" == "sq" ];then
            ARGS="$ARGS --smooth_quant "
        else
            ARGS="$ARGS --torchao  --weight-only-quant --weight-dtype INT8 "
        fi
    fi
    if [ "${INDUCTOR_PROFILE}" == "1" ];then
        ARGS+=" --profile "
    fi
    export TORCHINDUCTOR_FREEZING=1
    python -m torch.backends.xeon.run_cpu --disable-numactl --throughput-mode --skip-cross-node-cores --enable_tcmalloc --log_path=${OUTPUT_DIR} \
        ${EVAL_SCRIPT} $ARGS \
        --inductor \
        -m ${FINETUNED_MODEL} \
        --max-new-tokens ${OUTPUT_TOKEN} \
        --input-tokens  ${INPUT_TOKEN} \
        --batch-size $BATCH_SIZE > ${OUTPUT_DIR}/${usecase}_log_${precision}.log

    latency=($(grep -i 'inference-latency:' ${OUTPUT_DIR}/${usecase}_log_${PRECISION}* |sed -e 's/.*atency: //;s/[^0-9.]//g;s/\.$//' |awk '
        BEGIN {
            num = 0;
            sum = 0;
        }{
            num ++;
            sum += $1;
        }END {
            if(num > 0) {
                printf("%.6f", sum / num);
            }else {
                printf("0  0");
            }
        }
    '))

    first_latency=($(grep -i 'first-token-latency:' ${OUTPUT_DIR}/${usecase}_log_${PRECISION}*  |sed -e 's/.*atency://;s/[^0-9.]//g;s/\.$//' |awk '
    BEGIN {
        num = 0;
        sum = 0;
    }{
        num ++;
        sum += $1;
    }END {
        if(num > 0) {
            printf("%.6f", sum / num);
        }else {
            printf("0");
        }
    }
    '))
    rest_token_latency=($(grep -i '^rest-token-latency:' ${OUTPUT_DIR}/${usecase}_log_${PRECISION}*  |sed -e 's/.*atency://;s/[^0-9.]//g;s/\.$//' |awk '
    BEGIN {
        num = 0;
        sum = 0;
    }{
        num ++;
        sum += $1;
    }END {
        if(num > 0) {
            printf("%.6f", sum / num);
        }else {
            printf("0");
        }
    }
    '))
    P90_rest_token_latency=($(grep -i 'P90-rest-token-latency:' ${OUTPUT_DIR}/${usecase}_log_${PRECISION}*  |sed -e 's/.*atency://;s/[^0-9.]//g;s/\.$//' |awk '
    BEGIN {
        num = 0;
        sum = 0;
    }{
        num ++;
        sum += $1;
    }END {
        if(num > 0) {
            printf("%.6f", sum / num);
        }else {
            printf("0");
        }
    }
    '))


    token_per_sec=($(awk -v output_token=$OUTPUT_TOKEN -v total=$latency -v batch=$BATCH_SIZE -v first_token=${first_latency}} '
    BEGIN {
        thp = batch*(output_token-1)/(total-first_token);
        printf("%.3f", thp);
    }
    '))

    first_token_thp=($(awk -v output_token=$OUTPUT_TOKEN -v total=$latency -v batch=$BATCH_SIZE -v first_token=${first_latency}} '
    BEGIN {
        thp = batch*(1)/(first_token);
        printf("%.3f", thp);
    }
    '))

    echo "--------------------------------Performance Summary per NUMA Node--------------------------------"
    echo "${FINETUNED_MODEL};Input/Output Token;${INPUT_TOKEN}/${OUTPUT_TOKEN};${LOG_PREFIX};"total-latency";${PRECISION};${BATCH_SIZE}; ${latency} " |tee -a ${OUTPUT_DIR}/summary.log
    echo "${FINETUNED_MODEL};Input/Output Token;${INPUT_TOKEN}/${OUTPUT_TOKEN};${LOG_PREFIX};"first-token-latency";${PRECISION};${BATCH_SIZE}; ${first_latency} " |tee -a ${OUTPUT_DIR}/summary.log
    echo "${FINETUNED_MODEL};Input/Output Token;${INPUT_TOKEN}/${OUTPUT_TOKEN};${LOG_PREFIX};"rest-token-latency";${PRECISION};${BATCH_SIZE}; ${rest_token_latency} " |tee -a ${OUTPUT_DIR}/summary.log
    echo "${FINETUNED_MODEL};Input/Output Token;${INPUT_TOKEN}/${OUTPUT_TOKEN};${LOG_PREFIX};"P90-rest-token-latency";${PRECISION};${BATCH_SIZE}; ${P90_rest_token_latency} " |tee -a ${OUTPUT_DIR}/summary.log
    echo "${FINETUNED_MODEL};Input/Output Token;${INPUT_TOKEN}/${OUTPUT_TOKEN};${LOG_PREFIX};"token_per_sec";${PRECISION};${BATCH_SIZE}; ${token_per_sec} " |tee -a ${OUTPUT_DIR}/summary.log
    echo "${FINETUNED_MODEL};Input/Output Token;${INPUT_TOKEN}/${OUTPUT_TOKEN};${LOG_PREFIX};"first_token_thp";${PRECISION};${BATCH_SIZE}; ${first_token_thp} " |tee -a ${OUTPUT_DIR}/summary.log

    first_token_latency=$( grep "first-token-latency;" ${OUTPUT_DIR}/summary.log | awk '{print $NF}' )
    rest_token_latency=$( grep ";rest-token-latency;" ${OUTPUT_DIR}/summary.log | awk '{print $NF}' )

    ## Single instance throughput calculation
    first_token_throughput=$( echo "(1/$first_token_latency)*${BATCH_SIZE}" | bc -l )
    rest_token_throughput=$( echo "(1/$rest_token_latency)*${BATCH_SIZE}" | bc -l )
    accuracy="N/A"

else
    first_token_latency="N/A"
    rest_token_latency="N/A"
    first_token_throughput="N/A"
    rest_token_throughput="N/A"
    BATCH_SIZE=${BATCH_SIZE:-1}
    echo "Running Accuracy Inference"

    echo "### running with torch.compile inductor backend"
    if [[ "${PRECISION}" == *"int8"* ]];then
        if [ "${INT8_QUANT_TYPE}" == "sq" ];then
            ARGS="$ARGS --smooth_quant "
        else
            ARGS="$ARGS --torchao  --weight-only-quant --weight-dtype INT8 "
        fi
    fi
    export TORCHINDUCTOR_FREEZING=1
    python -m torch.backends.xeon.run_cpu --disable-numactl --log_path=${OUTPUT_DIR} \
        ${EVAL_SCRIPT} $ARGS \
        --inductor \
        --model-name-or-path ${FINETUNED_MODEL} > ${OUTPUT_DIR}/LLaMa_${PRECISION}_accuracy.log

    accuracy=$(cat ${OUTPUT_DIR}/LLaMa_${PRECISION}_accuracy* | grep "Accuracy:" |sed -e 's/.*= //;s/[^0-9.]//g')

    echo "${FINETUNED_MODEL};"accuracy";${PRECISION};${BATCH_SIZE};${accuracy}" | tee -a ${OUTPUT_DIR}/summary.log
fi

yaml_content=$(cat << EOF
results:
- key: first token throughput
  value: $first_token_throughput
- key: rest token throughput
  value: $rest_token_throughput
- key: first token latency
  value: $first_token_latency
- key: rest token latency
  value: $rest_token_latency
- key: accuracy
  value: $accuracy
EOF
)

echo "$yaml_content" >  $OUTPUT_DIR/results.yaml
echo "YAML file created."
