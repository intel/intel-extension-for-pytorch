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

if [ "$DDP" == 'false' ]; then
    echo "Running single-node training"
    if [[ "$TRAINING_PHASE" == '1' ]]; then
        echo "Running phase 1 training"
        ARGS="--benchmark"
        precision=fp32
        batch_size=${batch_size:-224}
    elif [ "$TRAINING_PHASE" == '2' ]; then
        echo "Running phase 2 training"
        ARGS="--benchmark"
        precision=fp32
        batch_size=${batch_size:-28}
    else
        echo "Please set TRAINING_PHASE to 1 or 2"
        exit 1
    fi
elif [[ "$DDP" == 'true' ]]; then
    echo "Running distributed training"
    oneccl_bindings_for_pytorch_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))")
    source $oneccl_bindings_for_pytorch_path/env/setvars.sh
    if [[ "$TRAINING_PHASE" == '1' ]]; then
        ARGS="--benchmark"
        precision=fp32
        batch_size=${batch_size:-224}
    elif [[ "$TRAINING_PHASE" == '2' ]]; then
        ARGS="--benchmark"
        precision=fp32
        batch_size=${batch_size:-28}
    else
        echo "Please set TRAINING_PHASE to 1 or 2"
        exit 1
    fi
else
    echo "Please set DDP to true or false"
    exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  exit 1
fi

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET has not been set"
  exit 1
fi


MODEL_DIR=${MODEL_DIR-$PWD}

if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

if [[ "$PRECISION" == "bf16" ]]; then
    ARGS="$ARGS --bf16"
    precision=bf16
    batch_size=${batch_size:-448}
    echo "### running bf16 mode"
elif [[ $PRECISION == "bf32" ]]; then
    echo "### running BF32 mode"
    ARGS="$ARGS --bf32"
    precision=bf32
elif [[ $DDP == 'false' && $PRECISION == "fp16" ]]; then
    echo "### running FP16 mode"
    ARGS="$ARGS --fp16"
    precision=fp16
elif [[ $DDP == 'true' && $PRECISION == "fp16" ]]; then
    echo "### running BF32 mode"
    ARGS="$ARGS --fp16"
    precision=bf32
elif [[ $DDP == 'false' && $PRECISION == "fp8" ]]; then
    echo "### running FP8 mode"
    ARGS="$ARGS --fp8"
    precision=fp8
elif [[ $PRECISION == "fp32" || $PRECISION == "avx-fp32" ]]; then
    echo "### running FP32 mode"

else
    echo "The specified precision '$PRECISION' is unsupported."
    echo "Supported precisions for single-node training are: fp32, bf32, avx-fp32, bf16, fp8"
    echo "Supported precisions for distributed training are: fp32, bf16, bf32"
    exit 1
fi

if [ "$DDP" == 'false' ]; then
    export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000";
    if [[ "$TRAINING_PHASE" == '1' ]]; then
        BERT_MODEL_CONFIG=${BERT_MODEL_CONFIG-~/dataset/checkpoint/config.json}
        rm -rf ${OUTPUT_DIR}/throughput_log_phase1_*
        rm -rf ${OUTPUT_DIR}/model_save_${PRECISION}
    elif [[ "$TRAINING_PHASE" == '2' ]]; then
        PRETRAINED_MODEL=${PRETRAINED_MODEL:-~/dataset/checkpoint/}
        rm -rf ${OUTPUT_DIR}/throughput_log_phase2_*
    fi
elif [ "$DDP" == 'true' ]; then
    if [[ "$TRAINING_PHASE" == '1' ]]; then
        BERT_MODEL_CONFIG=${BERT_MODEL_CONFIG-~/dataset/checkpoint/config.json}
        SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
        NNODES=${NNODES:-1}
        HOSTFILE=${HOSTFILE:-./hostfile}
        rm -rf ${OUTPUT_DIR}/throughput_log_phase1_*
    elif [[ "$TRAINING_PHASE" == '2' ]]; then
        PRETRAINED_MODEL=${PRETRAINED_MODEL:-~/dataset/checkpoint/}
        SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
        NNODES=${NNODES:-1}
        HOSTFILE=${HOSTFILE:-./hostfile}
        rm -rf ${OUTPUT_DIR}/throughput_log_phase2_*
    fi
fi

DATASET_DIR=${DATASET_DIR:-~/dataset/}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-${MODEL_DIR}/run_pretrain_mlperf.py}
OUTPUT_DIR=${OUTPUT_DIR:-${PWD}}
work_space=${work_space:-${OUTPUT_DIR}}

latency="N/A"
accuracy="N/A"
throughput="N/A"

if [[ "$DDP" == "false" ]]; then
    if [[ "$TRAINING_PHASE" == "1" ]]; then
        NUM_RANKS=1
        LBS=$(( batch_size / NUM_RANKS ))
        params="--train_batch_size=$LBS     --learning_rate=3.5e-4     --opt_lamb_beta_1=0.9     --opt_lamb_beta_2=0.999     --warmup_proportion=0.0     --warmup_steps=0.0     --start_warmup_step=0     --max_steps=13700    --max_predictions_per_seq=76      --do_train   --train_mlm_accuracy_window_size=0     --target_mlm_accuracy=0.720     --weight_decay_rate=0.01     --max_samples_termination=4500000     --eval_iter_start_samples=150000 --eval_iter_samples=150000     --eval_batch_size=16  --gradient_accumulation_steps=1 --num_samples_per_checkpoint 1 --min_samples_to_start_checkpoints 1 --log_freq 1 "

        TORCH_INDUCTOR=${TORCH_INDUCTOR:-"0"}
        if [[ "0" == ${TORCH_INDUCTOR} ]];then
            python -m intel_extension_for_pytorch.cpu.launch --nodes-list 0 --memory-allocator jemalloc --log_file_prefix="${OUTPUT_DIR}/throughput_log_phase1_${precision}" ${TRAIN_SCRIPT} \
                --input_dir ${DATASET_DIR}/2048_shards_uncompressed_128/ \
                --eval_dir ${DATASET_DIR}/eval_set_uncompressed/ \
                --model_type 'bert' \
                --benchmark \
                --ipex \
                --output_dir $OUTPUT_DIR/model_save_${PRECISION} \
                --dense_seq_output \
                --config_name ${BERT_MODEL_CONFIG} \
                $ARGS \
                $params 2>&1 | tee ${OUTPUT_DIR}/throughput_log_phase1_${precision}.log
        else
            python -m torch.backends.xeon.run_cpu --disable-numactl --node_id 0 --enable-jemalloc --log_path=${OUTPUT_DIR} ${TRAIN_SCRIPT} \
                --input_dir ${DATASET_DIR}/2048_shards_uncompressed_128/ \
                --eval_dir ${DATASET_DIR}/eval_set_uncompressed/ \
                --model_type 'bert' \
                --benchmark \
                --inductor \
                --output_dir $OUTPUT_DIR/model_save_${PRECISION} \
                --dense_seq_output \
                --config_name ${BERT_MODEL_CONFIG} \
                $ARGS \
                $params 2>&1 | tee ${OUTPUT_DIR}/throughput_log_phase1_${precision}.log
        fi
        throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/throughput_log_phase1_${precision}* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
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
        echo ""BERT";"training phase1 throughput";${precision}; ${batch_size};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
    elif [[ "$TRAINING_PHASE" == "2" ]]; then
        NUM_RANKS=1
        LBS=$(( batch_size / NUM_RANKS ))
        params="--train_batch_size=$LBS     --learning_rate=3.5e-4     --opt_lamb_beta_1=0.9     --opt_lamb_beta_2=0.999     --warmup_proportion=0.0     --warmup_steps=0.0     --start_warmup_step=0     --max_steps=13700     --phase2    --max_predictions_per_seq=76      --do_train     --skip_checkpoint     --train_mlm_accuracy_window_size=0     --target_mlm_accuracy=0.720     --weight_decay_rate=0.01     --max_samples_termination=4500000     --eval_iter_start_samples=150000 --eval_iter_samples=150000     --eval_batch_size=16  --gradient_accumulation_steps=1     --log_freq=0 "

        TORCH_INDUCTOR=${TORCH_INDUCTOR:-"0"}
        if [[ "0" == ${TORCH_INDUCTOR} ]];then
            python -m intel_extension_for_pytorch.cpu.launch --nodes-list 0 --memory-allocator jemalloc --log_file_prefix="${OUTPUT_DIR}/throughput_log_phase2_${precision}" ${TRAIN_SCRIPT} \
                --input_dir ${DATASET_DIR}/2048_shards_uncompressed_512/ \
                --eval_dir ${DATASET_DIR}/eval_set_uncompressed/ \
                --model_type 'bert' \
                --model_name_or_path ${PRETRAINED_MODEL} \
                --benchmark \
                --ipex \
                --dense_seq_output \
                --output_dir $OUTPUT_DIR/model_save_${PRECISION} \
                $ARGS \
                $params 2>&1 | tee ${OUTPUT_DIR}/throughput_log_phase2_${precision}.log
        else
            python -m torch.backends.xeon.run_cpu --disable-numactl --node_id 0 --enable-jemalloc --log_path=${OUTPUT_DIR} ${TRAIN_SCRIPT} \
                --input_dir ${DATASET_DIR}/2048_shards_uncompressed_512/ \
                --eval_dir ${DATASET_DIR}/eval_set_uncompressed/ \
                --model_type 'bert' \
                --model_name_or_path ${PRETRAINED_MODEL} \
                --benchmark \
                --inductor \
                --dense_seq_output \
                --output_dir $OUTPUT_DIR/model_save_${PRECISION} \
                $ARGS \
                $params 2>&1 | tee ${OUTPUT_DIR}/throughput_log_phase2_${precision}.log
        fi
        throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/throughput_log_phase2_${precision}* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
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
        echo ""BERT";"training phase2 throughput";${precision}; ${batch_size};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
    fi
elif [[ "$DDP" == "true" ]]; then
    if [[ "$TRAINING_PHASE" == "1" ]]; then
        NUM_RANKS=$(( NNODES * SOCKETS ))
        LBS=$(( batch_size / NUM_RANKS ))
        params="--train_batch_size=$LBS     --learning_rate=3.5e-4     --opt_lamb_beta_1=0.9     --opt_lamb_beta_2=0.999     --warmup_proportion=0.0     --warmup_steps=0.0     --start_warmup_step=0     --max_steps=13700   --max_predictions_per_seq=76      --do_train     --skip_checkpoint     --train_mlm_accuracy_window_size=0     --target_mlm_accuracy=0.720     --weight_decay_rate=0.01     --max_samples_termination=4500000     --eval_iter_start_samples=150000 --eval_iter_samples=150000     --eval_batch_size=16  --gradient_accumulation_steps=1     --log_freq=0 "

        # export FI_PROVIDER=psm3
        # export PSM3_HAL=sockets

        TORCH_INDUCTOR=${TORCH_INDUCTOR:-"0"}
        if [[ "0" == ${TORCH_INDUCTOR} ]];then
            python -m intel_extension_for_pytorch.cpu.launch --nnodes ${NNODES} --hostfile ${HOSTFILE}  --log_dir=${OUTPUT_DIR} --log_file_prefix="./throughput_log_phase1_${precision}" ${TRAIN_SCRIPT} \
                --input_dir ${DATASET_DIR}/2048_shards_uncompressed_128/ \
                --eval_dir ${DATASET_DIR}/eval_set_uncompressed/ \
                --model_type 'bert' \
                --ipex \
                --output_dir $OUTPUT_DIR/model_save_${PRECISION}  \
                --dense_seq_output \
                --config_name ${BERT_MODEL_CONFIG} \
                $ARGS \
                $params \
            2>&1 | tee ${OUTPUT_DIR}/throughput_log_phase1_${precision}.log
        else
            python -m intel_extension_for_pytorch.cpu.launch --nnodes ${NNODES} --hostfile ${HOSTFILE}  --log_dir=${OUTPUT_DIR} --log_file_prefix="./throughput_log_phase1_${precision}" ${TRAIN_SCRIPT} \
                --input_dir ${DATASET_DIR}/2048_shards_uncompressed_128/ \
                --eval_dir ${DATASET_DIR}/eval_set_uncompressed/ \
                --model_type 'bert' \
                --inductor \
                --output_dir $OUTPUT_DIR/model_save_${PRECISION}  \
                --dense_seq_output \
                --config_name ${BERT_MODEL_CONFIG} \
                $ARGS \
                $params \
            2>&1 | tee ${OUTPUT_DIR}/throughput_log_phase1_${precision}.log
        fi
        # For the summary of results
        wait
        throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/throughput_log_phase1_${precision}* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
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
        echo ""BERT";"training phase1 distributed throughput";${precision}; ${batch_size};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
    elif [[ "$TRAINING_PHASE" == "2" ]]; then
        NUM_RANKS=$(( NNODES * SOCKETS ))
        LBS=$(( batch_size / NUM_RANKS ))
        params="--train_batch_size=$LBS     --learning_rate=3.5e-4     --opt_lamb_beta_1=0.9     --opt_lamb_beta_2=0.999     --warmup_proportion=0.0     --warmup_steps=0.0     --start_warmup_step=0     --max_steps=13700     --phase2    --max_predictions_per_seq=76      --do_train     --skip_checkpoint     --train_mlm_accuracy_window_size=0     --target_mlm_accuracy=0.720     --weight_decay_rate=0.01     --max_samples_termination=4500000     --eval_iter_start_samples=150000 --eval_iter_samples=150000     --eval_batch_size=16  --gradient_accumulation_steps=1     --log_freq=0 "

        # export FI_PROVIDER=psm3
        # export PSM3_HAL=sockets

        TORCH_INDUCTOR=${TORCH_INDUCTOR:-"0"}
        if [[ "0" == ${TORCH_INDUCTOR} ]];then
            python -m intel_extension_for_pytorch.cpu.launch --nnodes ${NNODES} --hostfile ${HOSTFILE}  --log_dir=${OUTPUT_DIR} --log_file_prefix="./throughput_log_phase2_${precision}" ${TRAIN_SCRIPT} \
                --input_dir ${DATASET_DIR}/2048_shards_uncompressed_512/ \
                --eval_dir ${DATASET_DIR}/eval_set_uncompressed/ \
                --model_type 'bert' \
                --ipex \
                --model_name_or_path ${PRETRAINED_MODEL} \
                --output_dir $OUTPUT_DIR/model_save_${PRECISION} \
                --dense_seq_output \
                $ARGS \
                $params \
                2>&1 | tee ${OUTPUT_DIR}/throughput_log_phase2_${precision}.log
        else
            python -m intel_extension_for_pytorch.cpu.launch --nnodes ${NNODES} --hostfile ${HOSTFILE}  --log_dir=${OUTPUT_DIR} --log_file_prefix="./throughput_log_phase2_${precision}" ${TRAIN_SCRIPT} \
                --input_dir ${DATASET_DIR}/2048_shards_uncompressed_512/ \
                --eval_dir ${DATASET_DIR}/eval_set_uncompressed/ \
                --model_type 'bert' \
                --inductor \
                --model_name_or_path ${PRETRAINED_MODEL} \
                --output_dir $OUTPUT_DIR/model_save_${PRECISION} \
                --dense_seq_output \
                $ARGS \
                $params \
                2>&1 | tee ${OUTPUT_DIR}/throughput_log_phase2_${precision}.log
        fi

        # For the summary of results
        wait
        throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/throughput_log_phase2_${precision}* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
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
        echo ""BERT";"training phase2 distributed throughput";${precision}; ${batch_size};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
    fi
fi

yaml_content=$(cat << EOF
results:
- key : throughput
  value: $throughput
  unit: sentence/s
- key: latency
  value: $latency
  unit: s
- key: accuracy
  value: $accuracy
  unit: f1
EOF
)

echo "$yaml_content" >  $OUTPUT_DIR/results.yaml
echo "YAML file created."
