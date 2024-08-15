
#!/bin/bash

#
# Copyright (c) 2021 Intel Corporation
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

MAXSTEP=${MAXSTEP:--1}

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set, please create the output path and set it to OUTPUT_DIR"
  exit 1
fi

if [[ "$1" == "bf16" ]]
then
    precision="bf16"
    ARGS="$ARGS --bf16 "
    echo "### running bf16 mode"
elif [[ "$1" == "fp32" ]]
then
    echo "### running fp32 mode"
else
    echo "The specified precision '$1' is unsupported."
    echo "Supported precisions are: fp32, bf16"
    exit 1
fi


MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"meta-llama/Llama-2-7b-hf"}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-32}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-""}

#DDP settings
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export MASTER_ADDR=`head -1 hostfile`
CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`
NNODES=${NNODES:-1}
HOSTFILE=${HOSTFILE:-./hostfile}
DATASET=${DATASET:-"./alpaca_data.json"}
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0

<< EOF
# specific oneCCL settings depending on any CPU cluster
export CCL_WORKER_COUNT=8
export CCL_LOG_LEVEL=info
export CCL_BF16=avx512bf
export CCL_ATL_TRANSPORT=ofi
export CCL_MNIC_COUNT=2
export CCL_MNIC=local
export CCL_MNIC_NAME=irdma1,irdma5
export CCL_ALLREDUCE=ring
export CCL_WORKER_COUNT=8

for (( i = $SOCKETS; i < 2*$SOCKETS; i++ )); do  # pin CCL workers to HT
  START_CORE=$(( i * CORES ))
  for (( j = 0; j < $CCL_WORKER_COUNT; j++)); do
   CCL_WORKER_AFFINITY="${CCL_WORKER_AFFINITY} $((START_CORE + j))"
  done
done

export CCL_WORKER_AFFINITY=`echo ${CCL_WORKER_AFFINITY} | tr " " ","`
EOF

# specific Fabric settings depending on your network hardware status
# export FI_PROVIDER=psm3
# export PSM3_IDENTIFY=1
# export PSM3_ALLOW_ROUTERS=1
# export PSM3_RDMA=1
# export PSM3_PRINT_STATS=0
# export PSM3_RV_MR_CACHE_SIZE=8192
# export PSM3_KASSIST_MODE=none
# export FI_PSM3_CONN_TIMEOUT=100
# export PSM3_HAL=sockets


oneccl_bindings_for_pytorch_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh

python -m intel_extension_for_pytorch.cpu.launch \
    --memory-allocator tcmalloc \
    --nnodes ${NNODES} \
    --hostfile ${HOSTFILE} \
    --logical-cores-for-ccl --ccl_worker_count 2 \
    ./finetune.py  $ARGS \
    --base_model ${MODEL_NAME_OR_PATH} \
    --attn_implementation ${ATTN_IMPLEMENTATION} \
    --data_path ${DATASET} \
    --output_dir ${OUTPUT_DIR} \
    --micro_batch_size ${LOCAL_BATCH_SIZE} \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length \
    --max_steps ${MAXSTEP} 


