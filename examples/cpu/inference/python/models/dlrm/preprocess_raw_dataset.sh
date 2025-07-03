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

MODEL_DIR=${MODEL_DIR-$PWD}
PREPRECESS_SCRIPT=${MODEL_DIR}/scripts/process_Criteo_1TB_Click_Logs_dataset.sh
GET_MULTI_HOT_SCRIPTS=${MODEL_DIR}/scripts/materialize_synthetic_multihot_dataset.py

if [ ! -e "$PREPRECESS_SCRIPT"  ]; then
    echo "Could not find the script of process_Criteo_1TB_Click_Logs_dataset.sh. Please set environment variable '\${MODEL_DIR}'."
    echo "From which the process_Criteo_1TB_Click_Logs_dataset.sh exist at the: \${MODEL_DIR}/models/recommendation/pytorch/torchrec_dlrm/scripts/process_Criteo_1TB_Click_Logs_dataset.sh"
    exit 1
fi

if [ ! -e "$GET_MULTI_HOT_SCRIPTS"  ]; then
    echo "Could not find the script of materialize_synthetic_multihot_dataset.py. Please set environment variable '\${MODEL_DIR}'."
    echo "From which the materialize_synthetic_multihot_dataset.py exist at the: \${MODEL_DIR}/models/recommendation/pytorch/torchrec_dlrm/scripts/materialize_synthetic_multihot_dataset.sh"
    exit 1
fi

if [ -z "${RAW_DIR}" ]; then
  echo "The required environment variable RAW_DIR has not been set"
  exit 1
fi

if [ -z "${TEMP_DIR}" ]; then
  echo "The required environment variable TEMP_DIR has not been set"
  exit 1
fi

if [ -z "${PREPROCESSED_DIR}" ]; then
  echo "The required environment variable PREPROCESSED_DIR has not been set"
  exit 1
fi

if [ -z "${MULTI_HOT_DIR}" ]; then
  echo "The required environment variable MULTI_HOT_DIR has not been set"
  exit 1
fi

bash $PREPRECESS_SCRIPT ${RAW_DIR} ${TEMP_DIR} ${PREPROCESSED_DIR}

python $GET_MULTI_HOT_SCRIPTS \
    --in_memory_binary_criteo_path $PREPROCESSED_DIR \
    --output_path $MULTI_HOT_DIR \
    --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 \
    --multi_hot_sizes 3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 \
    --multi_hot_distribution_type uniform
