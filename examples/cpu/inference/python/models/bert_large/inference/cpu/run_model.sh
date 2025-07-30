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

if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    echo "TEST_MODE set to THROUGHPUT"
    ARGS=${ARGS:-""}
    num_warmup=${num_warmup:-"15"}
    num_iter=${num_iter:-"40"}
    ARGS="$ARGS --benchmark"
    precision=fp32
elif [[ "$TEST_MODE" == "REALTIME" ]]; then
    echo "TEST_MODE set to REALTIME"
    ARGS=${ARGS:-""}
    num_warmup=${num_warmup:-"20"}
    num_iter=${num_iter:-"100"}
    ARGS="$ARGS --benchmark"
    precision=fp32
elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
    echo "TEST_MODE set to ACCURACY"
    ARGS=""
    precision=fp32
else
    echo "Please set TEST_MODE to THROUGHPUT, REALTIME or ACCURACY"
    exit
fi

if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

if [[ "$PRECISION" == "bf16" ]]
then
    precision=bf16
    ARGS="$ARGS --bf16"
    echo "### running bf16 mode"
elif [[ "$PRECISION" == "fp16" ]]
then
    precision=fp16
    ARGS="$ARGS --fp16_cpu"
    echo "### running fp16 mode"

elif [[ "$PRECISION" == "bf32" ]]
then
    precision=bf32
    ARGS="$ARGS --bf32"
    echo "### running bf32 mode"
elif [[ "$PRECISION" == "int8" || "$PRECISION" == "avx-int8" ]]
then
    precision=int8
    ARGS="$ARGS --int8 --int8_bf16"
    echo "### running int8 mode"
elif [[ "$PRECISION" == "fp32" || "$PRECISION" == "avx-fp32" ]]
then
    precision=fp32
    echo "### running fp32 mode"
else
    echo "Please set PRECISION to : fp32, int8, bf32, bf16, avx-int8 or avx-fp32"
    exit
fi

if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    BATCH_SIZE=${BATCH_SIZE:-56}
    #export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000";
elif [[ "$TEST_MODE" == "REALTIME" ]]; then
    #export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000";
    export OMP_NUM_THREADS=4
    CORES=`lscpu | grep 'Core(s)' | awk '{print $4}'`
    SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
    NUMAS=`lscpu | grep 'NUMA node(s)' | awk '{print $3}'`
    CORES_PER_NUMA=`expr $CORES \* $SOCKETS / $NUMAS`
    BATCH_SIZE=${BATCH_SIZE:-1}
elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
    BATCH_SIZE=${BATCH_SIZE:-8}
fi

EVAL_DATA_FILE=${EVAL_DATA_FILE:-"${PWD}/squad1.1/dev-v1.1.json"}
FINETUNED_MODEL=${FINETUNED_MODEL:-"bert_squad_model"}
OUTPUT_DIR=${OUTPUT_DIR:-${PWD}}
EVAL_SCRIPT=${EVAL_SCRIPT:-"${PWD}/transformers/examples/legacy/question-answering/run_squad.py"}
work_space=${work_space:-${OUTPUT_DIR}}
INT8_CONFIG=${INT8_CONFIG:-"${PWD}/configure.json"}
FP8_CONFIG=${FP8_CONFIG:-"fp8_state_dict.pt"}

TORCH_INDUCTOR=${TORCH_INDUCTOR:-"0"}

if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    if [ "$WEIGHT_SHARING" ]; then
        CORES=`lscpu | grep Core | awk '{print $4}'`
        SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
        TOTAL_CORES=`expr $CORES \* $SOCKETS`
        CORES_PER_INSTANCE=$CORES
        INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
        LAST_INSTANCE=`expr $INSTANCES - 1`
        INSTANCES_PER_SOCKET=`expr $INSTANCES / $SOCKETS`

        echo "Running Bert_Large inference throughput with runtime extension enabled."
        STREAM_PER_INSTANCE=$CORES_PER_INSTANCE
        BATCH_SIZE=$STREAM_PER_INSTANCE
        for i in $(seq 0 $LAST_INSTANCE); do
            numa_node_i=`expr $i / $INSTANCES_PER_SOCKET`
            start_core_i=`expr $i \* $CORES_PER_INSTANCE`
            end_core_i=`expr $start_core_i + $CORES_PER_INSTANCE - 1`
            LOG_i="${OUTPUT_DIR}/throughput_log_${PRECISION}_${i}.log"

            ARGS="$ARGS --use_multi_stream_module"
            ARGS="$ARGS --num_streams $STREAM_PER_INSTANCE"
            ARGS="$ARGS --instance_number $numa_node_i"

            numactl -C $start_core_i-$end_core_i --membind=$numa_node_i python ${EVAL_SCRIPT} $ARGS --model_type bert --model_name_or_path ${FINETUNED_MODEL} --tokenizer_name bert-large-uncased-whole-word-masking-finetuned-squad  --do_eval --do_lower_case --predict_file $EVAL_DATA_FILE --per_gpu_eval_batch_size $BATCH_SIZE --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --perf_begin_iter ${num_warmup} --use_jit --ipex --perf_run_iters ${num_iter} --int8_config ${INT8_CONFIG} \
            2>&1 | tee ${LOG_i} &
        done
        wait
    elif [[ "0" == ${TORCH_INDUCTOR} ]];then
        if [[ "$PRECISION" == "int8" || "$PRECISION" == "avx-int8" ]]; then
            python -m intel_extension_for_pytorch.cpu.launch --throughput_mode --memory-allocator jemalloc --log_dir=${OUTPUT_DIR} --log_file_prefix="./throughput_log_${precision}" ${EVAL_SCRIPT} $ARGS --model_type bert --model_name_or_path ${FINETUNED_MODEL} --tokenizer_name bert-large-uncased-whole-word-masking-finetuned-squad  --do_eval --do_lower_case --predict_file $EVAL_DATA_FILE --per_gpu_eval_batch_size $BATCH_SIZE --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --perf_begin_iter ${num_warmup} --inductor --ipex --perf_run_iters ${num_iter} --int8_config ${INT8_CONFIG}
        else
            python -m intel_extension_for_pytorch.cpu.launch --throughput_mode --memory-allocator jemalloc --log_dir=${OUTPUT_DIR} --log_file_prefix="./throughput_log_${precision}" ${EVAL_SCRIPT} $ARGS --model_type bert --model_name_or_path ${FINETUNED_MODEL} --tokenizer_name bert-large-uncased-whole-word-masking-finetuned-squad  --do_eval --do_lower_case --predict_file $EVAL_DATA_FILE --per_gpu_eval_batch_size $BATCH_SIZE --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --perf_begin_iter ${num_warmup} --use_jit --ipex --perf_run_iters ${num_iter} --int8_config ${INT8_CONFIG}
        fi
    else
        echo "Running Bert_Large inference with torch.compile() indutor backend enabled."
        export TORCHINDUCTOR_FREEZING=1
        python -m torch.backends.xeon.run_cpu --disable-numactl --throughput_mode --enable_jemalloc --log_path=${OUTPUT_DIR} ${EVAL_SCRIPT} $ARGS --model_type bert --model_name_or_path ${FINETUNED_MODEL} --tokenizer_name bert-large-uncased-whole-word-masking-finetuned-squad  --do_eval --do_lower_case --predict_file $EVAL_DATA_FILE --per_gpu_eval_batch_size $BATCH_SIZE --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --perf_begin_iter ${num_warmup} --inductor --perf_run_iters ${num_iter} --int8_config ${INT8_CONFIG} 2>&1 | tee ${OUTPUT_DIR}/throughput_log_${precision}.log
    fi
elif [[ "$TEST_MODE" == "REALTIME" ]]; then
    if [[ "0" == ${TORCH_INDUCTOR} ]];then
        if [[ "$PRECISION" == "int8" || "$PRECISION" == "avx-int8" ]]; then
            python -m intel_extension_for_pytorch.cpu.launch --ninstances ${NUMAS} --log_dir=${OUTPUT_DIR} --log_file_prefix="./latency_log_${precision}" ${EVAL_SCRIPT} $ARGS --model_type bert --model_name_or_path ${FINETUNED_MODEL} --tokenizer_name bert-large-uncased-whole-word-masking-finetuned-squad  --do_eval --do_lower_case --predict_file $EVAL_DATA_FILE  --per_gpu_eval_batch_size $BATCH_SIZE --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --perf_begin_iter ${num_warmup} --perf_run_iters ${num_iter} --inductor --ipex --int8_config ${INT8_CONFIG} --use_share_weight --total_cores ${CORES_PER_NUMA}
        else
            python -m intel_extension_for_pytorch.cpu.launch --ninstances ${NUMAS} --log_dir=${OUTPUT_DIR} --log_file_prefix="./latency_log_${precision}" ${EVAL_SCRIPT} $ARGS --model_type bert --model_name_or_path ${FINETUNED_MODEL} --tokenizer_name bert-large-uncased-whole-word-masking-finetuned-squad  --do_eval --do_lower_case --predict_file $EVAL_DATA_FILE  --per_gpu_eval_batch_size $BATCH_SIZE --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --perf_begin_iter ${num_warmup} --perf_run_iters ${num_iter} --use_jit --ipex --int8_config ${INT8_CONFIG} --use_share_weight --total_cores ${CORES_PER_NUMA}
        fi
    else
        echo "Running Bert_Large inference with torch.compile() indutor backend enabled."
        export TORCHINDUCTOR_FREEZING=1
        python -m torch.backends.xeon.run_cpu --disable-numactl --ninstances ${NUMAS} --log_path=${OUTPUT_DIR} ${EVAL_SCRIPT} $ARGS --model_type bert --model_name_or_path ${FINETUNED_MODEL} --tokenizer_name bert-large-uncased-whole-word-masking-finetuned-squad  --do_eval --do_lower_case --predict_file $EVAL_DATA_FILE  --per_gpu_eval_batch_size $BATCH_SIZE --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --perf_begin_iter ${num_warmup} --perf_run_iters ${num_iter} --inductor --int8_config ${INT8_CONFIG} --use_share_weight --total_cores ${CORES_PER_NUMA} 2>&1 | tee ${OUTPUT_DIR}/latency_log_${precision}.log
    fi
    CORES_PER_INSTANCE=4
    TOTAL_CORES=`expr $CORES \* $SOCKETS`
    INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
    INSTANCES_PER_SOCKET=`expr $INSTANCES / $SOCKETS`
elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
    if [ ${WEIGHT_SHAREING} ]; then
        CORES=`lscpu | grep Core | awk '{print $4}'`
        SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
        TOTAL_CORES=`expr $CORES \* $SOCKETS`
        CORES_PER_INSTANCE=$CORES
        INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
        LAST_INSTANCE=`expr $INSTANCES - 1`
        INSTANCES_PER_SOCKET=`expr $INSTANCES / $SOCKETS`

        numa_node_i=0
        start_core_i=0
        end_core_i=`expr $start_core_i + $CORES_PER_INSTANCE - 1`
        LOG_0="${OUTPUT_DIR}/accuracy_log_${PRECISION}.log"

        echo "Running Bert_Large inference throughput with runtime extension enabled."
        STREAM_PER_INSTANCE=$CORES_PER_INSTANCE

        #export OMP_NUM_THREADS=`expr $BATCH_SIZE \/ $STREAM_PER_INSTANCE`
        BATCH_SIZE=$STREAM_PER_INSTANCE
        ARGS="$ARGS --use_multi_stream_module"
        ARGS="$ARGS --num_streams $STREAM_PER_INSTANCE"
        ARGS="$ARGS --instance_number $numa_node_i"

        numactl -C $start_core_i-$end_core_i --membind=$numa_node_i python $EVAL_SCRIPT $ARGS --model_type bert --model_name_or_path ${FINETUNED_MODEL}  --do_eval --do_lower_case --predict_file $EVAL_DATA_FILE  --per_gpu_eval_batch_size $BATCH_SIZE --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --tokenizer_name bert-large-uncased-whole-word-masking-finetuned-squad --use_jit --ipex --int8_config ${INT8_CONFIG} \
        2>&1 | tee $LOG_0
    elif [[ "0" == ${TORCH_INDUCTOR} ]]; then
        if [[ "$PRECISION" == "fp8" ]]; then
            python -m intel_extension_for_pytorch.cpu.launch --log_dir=${OUTPUT_DIR} --log_file_prefix="accuracy_log" $EVAL_SCRIPT $ARGS --model_type bert --model_name_or_path ${FINETUNED_MODEL}  --do_eval --do_lower_case --predict_file $EVAL_DATA_FILE  --per_gpu_eval_batch_size $BATCH_SIZE --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --tokenizer_name bert-large-uncased-whole-word-masking-finetuned-squad --ipex --fp8_config ${FP8_CONFIG} 2>&1 | tee $LOG_0
        elif [[ "$PRECISION" == "int8" || "$PRECISION" == "avx-int8" ]]; then
            python -m intel_extension_for_pytorch.cpu.launch --log_dir=${OUTPUT_DIR} --log_file_prefix="accuracy_log" $EVAL_SCRIPT $ARGS --model_type bert --model_name_or_path ${FINETUNED_MODEL}  --do_eval --do_lower_case --predict_file $EVAL_DATA_FILE  --per_gpu_eval_batch_size $BATCH_SIZE --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --tokenizer_name bert-large-uncased-whole-word-masking-finetuned-squad --inductor --ipex --int8_config ${INT8_CONFIG} 2>&1 | tee $LOG_0
        else
            python -m intel_extension_for_pytorch.cpu.launch --log_dir=${OUTPUT_DIR} --log_file_prefix="accuracy_log" $EVAL_SCRIPT $ARGS --model_type bert --model_name_or_path ${FINETUNED_MODEL}  --do_eval --do_lower_case --predict_file $EVAL_DATA_FILE  --per_gpu_eval_batch_size $BATCH_SIZE --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --tokenizer_name bert-large-uncased-whole-word-masking-finetuned-squad --use_jit --ipex --int8_config ${INT8_CONFIG} 2>&1 | tee $LOG_0
        fi
    else
        echo "Running Bert_Large inference with torch.compile() indutor backend enabled."
        export TORCHINDUCTOR_FREEZING=1
        python -m torch.backends.xeon.run_cpu --disable-numactl --log_path=${OUTPUT_DIR} $EVAL_SCRIPT $ARGS --model_type bert --model_name_or_path ${FINETUNED_MODEL}  --do_eval --do_lower_case --predict_file $EVAL_DATA_FILE  --per_gpu_eval_batch_size $BATCH_SIZE --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --tokenizer_name bert-large-uncased-whole-word-masking-finetuned-squad --inductor --int8_config ${INT8_CONFIG} 2>&1 | tee ${OUTPUT_DIR}/accuracy_log_${PRECISION}.log
    fi
fi

throughput="0"
latency="0"
accuracy="0"
if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/throughput_log_${PRECISION}* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
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
    echo ""BERT";"throughput";${precision}; ${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
elif [[ "$TEST_MODE" == "REALTIME" ]]; then
    throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/latency_log_${PRECISION}* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_PER_SOCKET '
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

    latency=$(grep 'P99 Latency' ${OUTPUT_DIR}/latency_log_${PRECISION}* |sed -e 's/.*P99 Latency//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_PER_SOCKET '
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
    echo $INSTANCES_PER_SOCKET
    echo "--------------------------------Performance Summary per Socket--------------------------------"
    echo ""BERT";"latency";${precision}; ${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
    echo ""BERT";"p99_latency";${precision}; ${BATCH_SIZE};${latency}" | tee -a ${OUTPUT_DIR}/summary.log
elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
    accuracy=$(grep 'Results:' ${OUTPUT_DIR}/accuracy_log_${PRECISION}*|awk -F ' ' '{print $12}' | awk -F ',' '{print $1}')
    echo ""BERT";"f1";${precision}; ${BATCH_SIZE};${accuracy}" | tee -a ${OUTPUT_DIR}/summary.log
fi

yaml_content=$(cat << EOF
results:
- key : throughput
  value: $throughput
  unit: examples per second
- key: latency
  value: $latency
  unit: seconds per example
- key: accuracy
  value: $accuracy
  unit: percentage
EOF
)

echo "$yaml_content" >  $OUTPUT_DIR/results.yaml
echo "YAML file created."
