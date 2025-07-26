#!/bin/bash
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
set -x
ARGS=""
EXTRA_ARGS=""

MODEL_DIR=${MODEL_DIR-$PWD}

if [[ "${TEST_MODE}" == "THROUGHPUT" ]]; then
    echo "TEST_MODE set to THROUGHPUT"
    BATCH_SIZE=${BATCH_SIZE:-256}
    LOG_PREFIX=dlrm_inference_performance_log
    if [ -z "${DATASET_DIR}" ]; then
        echo "DATASET_DIR are not set, will use dummy generated dataset"
        export EXTRA_ARGS="$EXTRA_ARGS --multi_hot_distribution_type uniform "
        export EXTRA_ARGS="$EXTRA_ARGS --multi_hot_sizes 3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 "
    else
        export EXTRA_ARGS="$EXTRA_ARGS --synthetic_multi_hot_criteo_path $DATASET_DIR "
    fi
elif [[ "${TEST_MODE}" == "ACCURACY" ]]; then
    echo "TEST_MODE set to ACCURACY"
    BATCH_SIZE=${BATCH_SIZE:-65536}
    LOG_PREFIX=dlrm_inference_accuarcy_log
    if [ -z "${DATASET_DIR}" ]; then
        echo "The required environment variable DATASET_DIR has not been set"
        exit 1
    fi
    if [ -z "${WEIGHT_DIR}" ]; then
        echo "The required environment variable WEIGHT_DIR has not been set"
        exit 1
    fi
    export EXTRA_ARGS="$EXTRA_ARGS --synthetic_multi_hot_criteo_path $DATASET_DIR --test_auroc --snapshot-dir $WEIGHT_DIR "
else
    echo "Please set TEST_MODE to THROUGHPUT or ACCURACY"
    exit 1
fi

if [ ! -e "${MODEL_DIR}/dlrm_main.py"  ]; then
    echo "Could not find the script of dlrm_s_pytorch.py. Please set environment variable '\${MODEL_DIR}'."
    echo "From which the dlrm_s_pytorch.py exist at the: \${MODEL_DIR}/dlrm_main.py"
    exit 1
fi

export MODEL_SCRIPT=${MODEL_DIR}/dlrm_main.py
export INT8_CONFIG=${MODEL_DIR}/int8_configure.json

echo "PRECISION: ${PRECISION}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

TORCH_INDUCTOR=${TORCH_INDUCTOR:-"0"}
AOT_INDUCTOR=${AOT_INDUCTOR:-"0"}
# if the number of cores are not equal on different numa node
# or for TORCHINDUCTOR=1 we will lanuch 2 process per numa
ENABLE_2ND_PROCESS=${ENABLE_2ND_PROCESS:-"0"}
MANUALLY_LAUNCH=${MANUALLY_LAUNCH:-"0"}
if [[ "1" == ${TORCH_INDUCTOR} ]];then
    ENABLE_2ND_PROCESS=1
    MANUALLY_LAUNCH=1
fi

EVAL_BATCH=${EVAL_BATCH:-300}
echo "EVAL_BATCH: $EVAL_BATCH"

if [[ $PRECISION == "bf16" ]]; then
    ARGS="$ARGS --dtype bf16"
    echo "running bf16 path"
elif [[ $PRECISION == "fp32" ]]; then
    echo "running fp32 path"
    ARGS="$ARGS --dtype fp32"
elif [[ $PRECISION == "bf32" ]]; then
    echo "running bf32 path"
    ARGS="$ARGS --dtype bf32"
elif [[ $PRECISION == "fp16" ]]; then
    echo "running fp16 path"
    ARGS="$ARGS --dtype fp16"
elif [[ $PRECISION == "tf32" ]]; then
    echo "running tf32 path"
    ARGS="$ARGS --dtype tf32"
elif [[ $PRECISION == "int8" ]]; then
    if [[ "0" == ${TORCH_INDUCTOR} ]];then
      echo "prepare int8 weight"
      bash ${MODEL_DIR}/prepare_int8.sh
    fi
    echo "running int8 path"
    ARGS="$ARGS --dtype int8 --int8-configure-dir ${INT8_CONFIG}"
else
    echo "The specified PRECISION '${PRECISION}' is unsupported."
    echo "Supported PRECISIONs are: fp32, fp16, bf16, bf32, int8"
    exit 1
fi

LOG=${OUTPUT_DIR}_${LOG_PREFIX}_${PRECISION}.log

if [ -z "${BATCH_SIZE}" ]; then
  export BATCH_SIZE=512
fi

export launcher_cmd="-m torch.backends.xeon.run_cpu --disable-numactl --throughput-mode --enable-jemalloc"

if [[ $PLOTMEM == "true" ]]; then
pip install memory_profiler matplotlib
export mrun_cmd="mprof run --python -o ${MEMLOG}"
unset launcher_cmd
fi

export COMMON_ARGS=""

if [[ "${TEST_MODE}" == "THROUGHPUT" ]]; then
    if [[ $ENABLE_TORCH_PROFILE == "true" ]]; then
        ARGS="$ARGS --profile"
    fi
    CORES_PER_SOCKET=`lscpu | grep "Core(s) per socket" | awk '{print $4}'`
    SOCKETS=`lscpu | grep "Socket(s)" | awk '{print $2}'`
    NUMA_NODES=`lscpu | grep "NUMA node(s)" | awk '{print $3}'`
    NUMA_NODES_PER_SOCKETS=`expr $NUMA_NODES / $SOCKETS`
    CORES_PER_NUMA_NODE=`expr $CORES_PER_SOCKET / $NUMA_NODES_PER_SOCKETS`
    COMMON_ARGS="${COMMON_ARGS} --benchmark --limit_val_batches $EVAL_BATCH"
    if [[ "0" == ${MANUALLY_LAUNCH} ]]; then
        COMMON_ARGS="${COMMON_ARGS} --share-weight-instance=$CORES_PER_NUMA_NODE"
    fi
    export OMP_NUM_THREADS=1
else
    # Not set limit_val_batches and use full dataset for accuracy test
    COMMON_ARGS="${COMMON_ARGS}"
fi

FP8=${FP8:-"0"}
if [[ "1" == ${FP8} ]];then
    COMMON_ARGS="${COMMON_ARGS} --fp8"
fi

COMMON_ARGS="${COMMON_ARGS} \
    --embedding_dim 128 \
    --dense_arch_layer_sizes 512,256,128 \
    --over_arch_layer_sizes 1024,1024,512,256,1 \
    --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 \
    --epochs 1 \
    --pin_memory \
    --mmap_mode \
    --batch_size $BATCH_SIZE \
    --interaction_type=dcn \
    --dcn_num_layers=3 \
    --dcn_low_rank_dim=512 \
    --log-freq 10 \
    --inference-only \
    $EXTRA_ARGS $ARGS"

if [[ "0" == ${TORCH_INDUCTOR} ]];then
    $mrun_cmd python $launcher_cmd $MODEL_SCRIPT $COMMON_ARGS --ipex-optimize --jit --ipex-merged-emb-cat 2>&1 | tee $LOG
else
    export TORCHINDUCTOR_FREEZING=1
    if [[ "0" == ${AOT_INDUCTOR} ]];then
        if [[ "${TEST_MODE}" != "THROUGHPUT" ]]; then
            $mrun_cmd python $MODEL_SCRIPT $COMMON_ARGS --inductor 2>&1 | tee $LOG
        else
            if [[ "0" != ${MANUALLY_LAUNCH} ]];then
                python launch.py $SOCKETS $NUMA_NODES $CORES_PER_SOCKET $ENABLE_2ND_PROCESS 0 "python $MODEL_SCRIPT $COMMON_ARGS --inductor"  2>&1 | tee $LOG
            else
                $mrun_cmd python $launcher_cmd $MODEL_SCRIPT $COMMON_ARGS --inductor 2>&1 | tee $LOG
            fi
        fi
    else
        aoti_model_dir=$(python $MODEL_SCRIPT $COMMON_ARGS --inductor --aot-inductor | grep "aoti-model")
        echo $aoti_model_dir
        python launch.py $SOCKETS $NUMA_NODES $CORES_PER_SOCKET $ENABLE_2ND_PROCESS 1 $aoti_model_dir  2>&1 | tee $LOG
    fi
fi

wait
throughput="N/A"
accuracy="N/A"
latency="N/A"
if [[ $PLOTMEM == "true" ]]; then
mprof plot ${MEMLOG} -o ${MEMPIC}
fi

if [[ "${TEST_MODE}" == "THROUGHPUT" ]]; then
    throughput=$(grep 'Throughput:' "${LOG}" |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk -v enable_2nd_process="$ENABLE_2ND_PROCESS" '
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
            if (enable_2nd_process == "1") {
                sum = sum * 2;
            }
            printf("%.3f", sum);
    }')
    echo "--------------------------------Performance Summary per NUMA Node--------------------------------"
    echo ""dlrm-v2";"throughput";${PRECISION};${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
else
    accuracy=$(grep 'Final AUROC:' $LOG |sed -e 's/.*Final AUROC//;s/[^0-9.]//g')
    echo ""dlrm-v2";"auc";${PRECISION};${BATCH_SIZE};${accuracy}" | tee -a ${OUTPUT_DIR}/summary.log
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
