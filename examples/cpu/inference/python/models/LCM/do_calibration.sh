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

INT8_MODEL=${INT8_MODEL:-"quantized_model.pt2"}

mkdir -p ${OUTPUT_DIR}

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_BLOCKTIME=200
export KMP_AFFINITY=granularity=fine,compact,1,0

export TORCHINDUCTOR_FREEZING=1
export TORCHINDUCTOR_CPP_ENABLE_TILING_HEURISTIC=0
export TORCHINDUCTOR_ENABLE_LINEAR_BINARY_FOLDING=1

python -m torch.backends.xeon.run_cpu --disable-numactl \
            --log_path ${OUTPUT_DIR} \
            ${MODEL_DIR}/inference.py \
            --model_name_or_path="SimianLuo/LCM_Dreamshaper_v7" \
            --dataset_path=${DATASET_DIR} \
            --quantized_model_path=${INT8_MODEL} \
            --compile_inductor \
            --precision=int8-bf16 \
            --calibration
