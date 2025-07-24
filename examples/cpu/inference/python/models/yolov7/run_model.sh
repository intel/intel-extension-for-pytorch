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

MODEL_DIR=${MODEL_DIR-$PWD}

if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    echo "TEST_MODE set to THROUGHPUT"
    mode=throughput
elif [[ "$TEST_MODE" == "REALTIME" ]]; then
    echo "TEST_MODE set to REALTIME"
    mode=latency
elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
    echo "TEST_MODE set to ACCURACY"
    mode=accuracy
else
    echo "Please set TEST_MODE to THROUGHPUT, REALTIME or ACCURACY"
    exit
fi

if [ ! -d "${DATASET_DIR}/coco" ]; then
  echo "The DATASET_DIR \${DATASET_DIR}/coco does not exist"
  exit 1
fi

if [ ! -e "${CHECKPOINT_DIR}/yolov7.pt" ]; then
  echo "The CHECKPOINT_DIR \${CHECKPOINT_DIR}/yolov7.pt does not exist"
  exit 1
fi

cd $DATASET_DIR
DATASET_DIR=$(pwd)
cd -

cd $CHECKPOINT_DIR
CHECKPOINT_DIR=$(pwd)
cd -

MODEL_DIR=${MODEL_DIR-$PWD}

if [ ! -e "${MODEL_DIR}/yolov7.patch"  ]; then
    echo "Could not find the script of yolov7.patch. Please set environment variable '\${MODEL_DIR}'."
    echo "From which the yolov7.patch exist at the: \${MODEL_DIR}/yolov7.patch"
    exit 1
else
    TMP_PATH=$(pwd)
    cd "${MODEL_DIR}/"
    if [ ! -d "yolov7" ]; then
        git clone https://github.com/WongKinYiu/yolov7.git yolov7
        cd yolov7
        cp ../inference.py .
        git checkout a207844
        git apply ../yolov7.patch
        pip install -r requirements.txt
    else
        cd yolov7
    fi
    cd $TMP_PATH
fi

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

cd ${OUTPUT_DIR}
OUTPUT_DIR=$(pwd)
cd -


if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to int8, fp32, bf32, bf16, or fp16."
  exit 1
fi

cd "${MODEL_DIR}/yolov7"
if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    rm -rf ${OUTPUT_DIR}/yolov7_throughput_log*
    ARGS="--checkpoint-dir $CHECKPOINT_DIR --weights yolov7.pt"
    ARGS="$ARGS --img 640 -e --performance --data data/coco.yaml --dataset-dir $DATASET_DIR --conf-thres 0.001 --iou 0.65 --device cpu --drop-last"
    MODE_ARGS="--throughput-mode"
    # default value, you can fine-tune it to get perfect performance.
    BATCH_SIZE=${BATCH_SIZE:-40}
    CORES=`lscpu | grep 'Core(s)' | awk '{print $4}'`
    SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
    TOTAL_CORES=`expr $CORES \* $SOCKETS`
    export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
    export KMP_BLOCKTIME=1
    export KMP_AFFINITY=granularity=fine,compact,1,0

elif [[ "$TEST_MODE" == "REALTIME" ]]; then
    rm -rf ${OUTPUT_DIR}/yolov7_latency_log*
    BATCH_SIZE=${BATCH_SIZE:-1}
    ARGS="--checkpoint-dir $CHECKPOINT_DIR --weights yolov7.pt"
    ARGS="$ARGS --img 640 -e --performance --data data/coco.yaml --dataset-dir $DATASET_DIR --conf-thres 0.001 --iou 0.65 --device cpu --drop-last"
    CORES=`lscpu | grep 'Core(s)' | awk '{print $4}'`
    SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
    NUMAS=`lscpu | grep 'NUMA node(s)' | awk '{print $3}'`
    CORES_PER_INSTANCE=4
    export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
    export KMP_BLOCKTIME=1
    export KMP_AFFINITY=granularity=fine,compact,1,0
    export OMP_NUM_THREADS=$CORES_PER_INSTANCE
    MODE_ARGS="$MODE_ARGS --latency-mode"

elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
    rm -rf ${OUTPUT_DIR}/yolov7_accuracy_log*
    BATCH_SIZE=${BATCH_SIZE:-40}
    ARGS="--checkpoint-dir $CHECKPOINT_DIR --weights yolov7.pt"
    ARGS="$ARGS --img 640 -e --data data/coco.yaml --dataset-dir $DATASET_DIR --conf-thres 0.001 --iou 0.65 --device cpu"
    MODE_ARGS="$MODE_ARGS"
fi


if [[ $PRECISION == "int8" ]]; then
    echo "running int8 path"
    if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
        NUMA_NODES=`lscpu | grep 'NUMA node(s)' | awk '{print $3}'`
        CORES_PER_NODE=`expr $TOTAL_CORES / $NUMA_NODES`
        BATCH_SIZE=${BATCH_SIZE:-`expr $CORES_PER_NODE \* 8`}
    fi
    ARGS="$ARGS --int8"
elif [[ $PRECISION == "bf16" ]]; then
    echo "running bf16 path"
    if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
        BATCH_SIZE=${BATCH_SIZE:-80}
    fi
    ARGS="$ARGS --bf16"
elif [[ $PRECISION == "bf32" ]]; then
    echo "running bf32 path"
    if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
        BATCH_SIZE=${BATCH_SIZE:-80}
    fi
    ARGS="$ARGS --bf32"
elif [[ $PRECISION == "fp16" ]]; then
    echo "running fp16 path"
    if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
        BATCH_SIZE=${BATCH_SIZE:-80}
    fi
    ARGS="$ARGS --fp16"
elif [[ $PRECISION == "fp32" ]]; then
    echo "running fp32 path"
    if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
        BATCH_SIZE=${BATCH_SIZE:-40}
    fi
else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions are: fp32, fp16, bf16, int8, bf32"
    exit 1
fi


echo "Running yolov7 inference with torch.compile inductor backend."

if [ -e "${DATASET_DIR}/coco/val2017.cache"  ]; then
    rm -f ${DATASET_DIR}/coco/val2017.cache
    echo "Removed existing cache file ${DATASET_DIR}/coco/val2017.cache"
fi
python ${MODEL_DIR}/yolov7/inference.py $ARGS --inductor --batch-size $BATCH_SIZE --prepare-dataloader

export TORCHINDUCTOR_FREEZING=1
python -m torch.backends.xeon.run_cpu --disable-numactl \
    --enable-tcmalloc \
    $MODE_ARGS \
    --log_path=${OUTPUT_DIR} \
    ${MODEL_DIR}/yolov7/inference.py \
    $ARGS \
    --inductor \
    --batch-size $BATCH_SIZE 2>&1 | tee ${OUTPUT_DIR}/yolov7_${mode}_log_${PRECISION}.log


wait
cd -

throughput="N/A"
accuracy="N/A"
latency="N/A"

if [[ "$TEST_MODE" == "REALTIME" ]]; then
    TOTAL_CORES=`expr $CORES \* $SOCKETS`
    INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
    INSTANCES_PER_SOCKET=`expr $INSTANCES / $SOCKETS`

    throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/yolov7_latency_log* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_PER_SOCKET '
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

    latency=$(grep 'Inference latency ' ${OUTPUT_DIR}/yolov7_latency_log* |sed -e 's/.*Inference latency //;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_PER_SOCKET '
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
    echo "--------------------------------Performance Summary per Socket--------------------------------"
    echo "yolov7;"latency";${PRECISION};${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
    echo "yolov7;"p99_latency";${PRECISION};${BATCH_SIZE};${latency}" | tee -a ${OUTPUT_DIR}/summary.log

elif [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    throughput=$(grep 'Throughput:'  ${OUTPUT_DIR}/yolov7_${mode}_log_${PRECISION}* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
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
    echo "yolov7;"throughput";${PRECISION};${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log

elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
    accuracy=$(grep -F 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ' \
        ${OUTPUT_DIR}/yolov7_accuracy_log_${PRECISION}*.log | \
        awk -F '=' '{print $NF}')
    echo "yolov7;"accuracy";${PRECISION};${BATCH_SIZE};${accuracy}" | tee -a ${OUTPUT_DIR}/summary.log
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
