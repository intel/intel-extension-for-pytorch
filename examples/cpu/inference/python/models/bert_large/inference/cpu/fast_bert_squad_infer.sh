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

MODEL_DIR=${MODEL_DIR-../../../../../..}
SCRIPT=${SCRIPT:-${MODEL_DIR}/models/language_modeling/pytorch/fast_bert/squad/run_squad.py}
EVAL_DATA_FILE=${EVAL_DATA_FILE:-"${PWD}/squad1.1/dev-v1.1.json"}
FINETUNED_MODEL=${FINETUNED_MODEL:-bert_squad_model}
NUMA_ARGS=""
if command -v numactl >& /dev/null ; then
if [ "x$MPI_LOCALRANKID" != "x" ] ; then
  REAL_NUM_NUMA_NODES=`lscpu | grep "NUMA node(s):" | awk '{print $NF}'`
  PPNUMA=$(( MPI_LOCALNRANKS / REAL_NUM_NUMA_NODES ))
  if [ $PPNUMA -eq 0 ] ; then 
    if [ "x$SINGLE_SOCKET_ONLY" == "x1" ] ; then 
      NUMA_ARGS="numactl -m 0 "
    fi
  else
    NUMARANK=$(( MPI_LOCALRANKID / PPNUMA ))
    NUMA_ARGS="$NUMA_ARGS $GDB_ARGS "
  fi
  NUM_RANKS=$PMI_SIZE
else
  NUMA_ARGS="numactl -m 0 "
  NUM_RANKS=1
fi
fi

$NUMA_RAGS $GDB_ARGS python -u ${SCRIPT} \
  --model_type bert \
  --model_name_or_path ${FINETUNED_MODEL} \
  --do_eval \
  --do_lower_case \
  --predict_file $EVAL_DATA_FILE \
  --per_gpu_eval_batch_size 24 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/ \
  $@

