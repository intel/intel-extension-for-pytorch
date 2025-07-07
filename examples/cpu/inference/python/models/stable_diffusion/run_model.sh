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

LOG_PREFIX=""
ARGS_LAUNCH=""

if [[ "${TEST_MODE}" == "THROUGHPUT" ]]; then
    echo "TEST_MODE set to THROUGHPUT"
elif [[ "${TEST_MODE}" == "ACCURACY" ]]; then
    echo "TEST_MODE set to ACCURACY"
    if [[ "${DISTRIBUTED}" == "True" || "${DISTRIBUTED}" == "TRUE" ]]; then
        echo "Running distributed inference accuracy"
        CORES=`lscpu | grep 'Core(s)' | awk '{print $4}'`
        CORES_PER_INSTANCE=$CORES
        SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
        TOTAL_CORES=`expr $CORES \* $SOCKETS`
        NNODES=${NNODES:-1}
        HOSTFILE=${HOSTFILE:-./hostfile}
        NUM_RANKS=$(( NNODES * SOCKETS ))
    fi
elif [[ "${TEST_MODE}" == "REALTIME" ]]; then
    echo "TEST_MODE set to REALTIME"
else
    echo "Please set TEST_MODE to THROUGHPUT or REALTIME or ACCURACY"
    exit
fi

MODEL_DIR=${MODEL_DIR-$PWD}

if [ ! -e "${MODEL_DIR}/inference.py" ]; then
  echo "Could not find the script of inference.py. Please set environment variable '\${MODEL_DIR}'."
  echo "From which the inference.py exist at the: \${MODEL_DIR}/inference.py"
  exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR \${DATASET_DIR} does not exist"
  exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [[ "${PRECISION}" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

ARGS=""
if [[ "${PRECISION}" == "bf16" ]]; then
    ARGS="$ARGS --precision=bf16"
    echo "### running bf16 datatype"
elif [[ "${PRECISION}" == "fp16" ]]; then
    ARGS="$ARGS --precision=fp16"
    echo "### running fp16 datatype"
elif [[ "${PRECISION}" == "int8-bf16" ]]; then
    ARGS="$ARGS --precision=int8-bf16"
    if [[ "${MODE}" == "compile-inductor" ]]; then
        if [ ! -f "${INT8_MODEL}" ]; then
            echo "The required file INT8_MODEL does not exist"
            exit 1
        fi
        ARGS="$ARGS --quantized_model_path=${INT8_MODEL}"
    else
        echo "For int8-bf16 datatype, the specified mode '${MODE}' is unsupported."
        echo "Supported mode is: compile-inductor"
        exit 1
    fi
    echo "### running int8-bf16 datatype"
elif [[ "${PRECISION}" == "int8-fp32" ]]; then
    ARGS="$ARGS --precision=int8-fp32"
    if [[ "${MODE}" == "compile-inductor" ]]; then
        if [ ! -f "${INT8_MODEL}" ]; then
            echo "The required file INT8_MODEL does not exist"
            exit 1
        fi
        ARGS="$ARGS --quantized_model_path=${INT8_MODEL}"
    else
        echo "For int8-fp32 datatype, the specified mode '${MODE}' is unsupported."
        echo "Supported mode is: compile-inductor"
        exit 1
    fi
    echo "### running int8-fp32 datatype"
elif [[ "${PRECISION}" == "fp8-bf16" ]]; then
    ARGS="$ARGS --precision=fp8-bf16"
    if [[ "${MODE}" == "compile-inductor" ]]; then
        echo "### running fp8-bf16 datatype"
    else
        echo "For fp8-bf16 datatype, the specified mode '${MODE}' is unsupported."
        echo "Supported mode is: compile-inductor"
        exit 1
    fi
elif [[ "${PRECISION}" == "fp8-fp32" ]]; then
    ARGS="$ARGS --precision=fp8-fp32"
    if [[ "${MODE}" == "compile-inductor" ]]; then
        echo "### running fp8-fp32 datatype"
    else
        echo "For fp8-fp32 datatype, the specified mode '${MODE}' is unsupported."
        echo "Supported mode is: compile-inductor"
        exit 1
    fi
elif [[ "${PRECISION}" == "fp32" ]]; then
    echo "### running fp32 datatype"
else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions are: fp32, fp16, bf16, int8-bf16, int8-fp32, fp8-bf16, fp8-fp32"
    exit 1
fi

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_BLOCKTIME=200
export KMP_AFFINITY=granularity=fine,compact,1,0

if [[ "${TEST_MODE}" == "THROUGHPUT" ]]; then
    LOG_PREFIX="stable_diffusion_${PRECISION}_inference_throughput"
    ARGS_LAUNCH="$ARGS_LAUNCH --throughput_mode"
    num_warmup=${num_warmup:-"1"}
    num_iter=${num_iter:-"10"}
    ARGS="$ARGS --benchmark -w ${num_warmup} -i ${num_iter}"
elif [[ "${TEST_MODE}" == "ACCURACY" ]]; then
    if [[ "${DISTRIBUTED}" == "True" || "${DISTRIBUTED}" == "TRUE" ]]; then
        LOG_PREFIX="stable_diffusion_${PRECISION}_dist_inference_accuracy"
        oneccl_bindings_for_pytorch_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))")
        source $oneccl_bindings_for_pytorch_path/env/setvars.sh
        ARGS_LAUNCH="$ARGS_LAUNCH --nnodes ${NNODES} --hostfile ${HOSTFILE} --logical-cores-for-ccl --ccl-worker-count 8"
        ARGS="$ARGS --accuracy --dist-backend ccl"
    else
        LOG_PREFIX="stable_diffusion_${PRECISION}_inference_accuracy"
        ARGS_LAUNCH="$ARGS_LAUNCH --ninstances 1"
        ARGS="$ARGS --accuracy"
    fi
else
    CORES=`lscpu | grep Core | awk '{print $4}'`
    SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
    NUMAS=`lscpu | grep 'NUMA node(s)' | awk '{print $3}'`
    CORES_PER_NUMA=`expr $CORES \* $SOCKETS / $NUMAS`
    CORES_PER_INSTANCE=4
    export OMP_NUM_THREADS=$CORES_PER_INSTANCE
    NUMBER_INSTANCE=`expr $CORES_PER_NUMA / $CORES_PER_INSTANCE`
    LOG_PREFIX="stable_diffusion_${PRECISION}_inference_realtime"
    ARGS_LAUNCH="$ARGS_LAUNCH --ninstances $NUMAS"
    num_warmup=${num_warmup:-"1"}
    num_iter=${num_iter:-"1"}
    ARGS="$ARGS --benchmark -w ${num_warmup} -i ${num_iter} --weight-sharing --number-instance $NUMBER_INSTANCE"
fi

if [ "${MODE}" == "eager" ]; then
    echo "### running eager mode"
elif [[ "${MODE}" == "compile-inductor" ]]; then
    export TORCHINDUCTOR_FREEZING=1
    export TORCHINDUCTOR_CPP_ENABLE_TILING_HEURISTIC=0
    export TORCHINDUCTOR_ENABLE_LINEAR_BINARY_FOLDING=1
    ARGS="$ARGS --compile_inductor"
    echo "### running torch.compile with inductor backend"
else
    echo "The specified mode '${MODE}' is unsupported."
    echo "Supported mode are: eager, compile-inductor"
    exit 1
fi

rm -rf ${OUTPUT_DIR}/${LOG_PREFIX}*

python -m torch.backends.xeon.run_cpu --disable-numactl --log-path ${OUTPUT_DIR} \
    --enable_tcmalloc \
    ${ARGS_LAUNCH} \
    --log_path=${OUTPUT_DIR} \
    ${MODEL_DIR}/inference.py \
    --dataset_path=${DATASET_DIR} \
    $ARGS 2>&1 | tee ${OUTPUT_DIR}/stable_diffusion_${PRECISION}_inference_throughput.log

wait

if [[ "${TEST_MODE}" == "REALTIME" ]]; then
    TOTAL_CORES=`expr $CORES \* $SOCKETS`
    INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
    INSTANCES_PER_SOCKET=`expr $INSTANCES / $SOCKETS`

    latency=$(grep 'Throughput:' ${OUTPUT_DIR}/${LOG_PREFIX}* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_PER_SOCKET '
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
            printf("%.4f", sum);
    }')
    echo "--------------------------------Performance Summary per Instance --------------------------------"
    echo ""stable_diffusion";"latency";${PRECISION};${latency}" | tee -a ${OUTPUT_DIR}/summary.log
elif [[ "${TEST_MODE}" == "THROUGHPUT" ]]; then
    throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/${LOG_PREFIX}* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
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
            printf("%.4f", sum);
    }')
    echo "--------------------------------Performance Summary per Instance --------------------------------"
    echo ""stable_diffusion";"throughput";${PRECISION};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
elif [[ "${TEST_MODE}" == "ACCURACY" ]]; then
    accuracy=$(grep 'FID:' ${OUTPUT_DIR}/${LOG_PREFIX}* |sed -e 's/.*FID//;s/[^0-9.]//g')
    echo ""stable_diffusion";"FID";${PRECISION};${accuracy}" | tee -a ${OUTPUT_DIR}/summary.log
fi

if [[ -z $throughput ]]; then
    throughput="N/A"
fi
if [[ -z $accuracy ]]; then
    accuracy="N/A"
fi
if [[ -z $latency ]]; then
    latency="N/A"
fi

yaml_content=$(cat << EOF
results:
- key : throughput
  value: $throughput
  unit: samples/sec
- key: latency
  value: $latency
  unit: s
- key: accuracy
  value: $accuracy
  unit: FID
EOF
)

echo "$yaml_content" >  $OUTPUT_DIR/results.yaml
echo "YAML file created."
