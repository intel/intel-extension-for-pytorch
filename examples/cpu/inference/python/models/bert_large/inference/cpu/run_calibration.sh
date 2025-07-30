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


#export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
ARGS=""
precision=fp32
if [[ "$1" == "bf16" ]]
then
    ARGS="$ARGS --bf16"
    precision=bf16
    echo "### running bf16 mode"
fi

if [[ "$1" == "int8" ]]
then
    ARGS="$ARGS --int8"
    precision=int8
    echo "### running int8 mode"
fi

if [[ "$1" == "fp8" ]]
then
    ARGS="$ARGS --fp8"
    precision=fp8
    echo "### running fp8 mode"
fi
rm -f calibration_log*
INT8_CONFIG=${INT8_CONFIG:-"configure.json"}
FP8_CONFIG=${FP8_CONFIG:-"fp8_state_dict.pt"}
BATCH_SIZE=${BATCH_SIZE:-8}
EVAL_DATA_FILE=${EVAL_DATA_FILE:-"${PWD}/squad1.1/dev-v1.1.json"}
FINETUNED_MODEL=${FINETUNED_MODEL:-bert_squad_model}
OUTPUT_DIR=${OUTPUT_DIR:-"${PWD}"}
EVAL_SCRIPT=${EVAL_SCRIPT:-"./transformers/examples/legacy/question-answering/run_squad.py"}
work_space=${work_space:-"${OUTPUT_DIR}"}

if [[ "$precision" == "int8" ]]
then
    python -m intel_extension_for_pytorch.cpu.launch --log_dir=${OUTPUT_DIR} --log_file_prefix="calibration_log" $EVAL_SCRIPT $ARGS --model_type bert --model_name_or_path ${FINETUNED_MODEL}  --do_eval --do_lower_case --predict_file $EVAL_DATA_FILE  --per_gpu_eval_batch_size $BATCH_SIZE --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --tokenizer_name bert-large-uncased-whole-word-masking-finetuned-squad --int8_config ${INT8_CONFIG} --do_calibration --calibration_iters 80 2>&1 | tee $LOG_0
elif [[ "$precision" == "fp8" ]]
then
    python -m intel_extension_for_pytorch.cpu.launch --log_dir=${OUTPUT_DIR} --log_file_prefix="accuracy_log" $EVAL_SCRIPT $ARGS --model_type bert --model_name_or_path ${FINETUNED_MODEL}  --do_eval --do_lower_case --predict_file $EVAL_DATA_FILE  --per_gpu_eval_batch_size $BATCH_SIZE --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --tokenizer_name bert-large-uncased-whole-word-masking-finetuned-squad --ipex --fp8_config ${FP8_CONFIG} --do_calibration --calibration_iters 80 2>&1 | tee $LOG_0
fi
