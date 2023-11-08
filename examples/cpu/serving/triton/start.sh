#!/bin/bash

# Copyright (c) 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0

source "$(pwd)"/config.properties

print_help() {
    echo ""
    echo "Usage: $0 [arg]"
    echo ""
    echo "   This script starts Docker container for host, client or localhost scenario."
    echo "   If no arguments are specified script will start Docker containers for localhost."
    echo ""
    echo "   Available arguments:"
    echo "   client              - starts client container."
    echo "   host                - starts host container."
    echo "   localhost (default) - starts client and host containers on the same instance."
    echo ""
    exit 2
}

start_host() {
    # Prepare Triton Host Server backend
    [[ "${model_name}" == *"ov" ]] && common_dir="bert_common_ov" || common_dir="bert_common"
    if [[ "${model_name}" == "bert_base"* ]]; then
        model_fill="csarron\\/bert-base-uncased-squad-v1"
        queue_fill="700"
    elif [[ "${model_name}" == "bert_large"* ]]; then
        model_fill="bert-large-uncased-whole-word-masking-finetuned-squad"
        queue_fill="1500"
    else
        echo "Wrong model name"
        print_help
    fi
    backend_dir="$(pwd)/backend/${model_name}" 
    [ ! -d "${backend_dir}" ] && mkdir -p "$(pwd)/backend/" && cp -fr "$(pwd)/model_utils/${common_dir}" "${backend_dir}" && cp "$(pwd)/model_utils/config.pbtxt" "${backend_dir}"
    sed -i "s/model_placeholder/${model_fill}/g" "${backend_dir}"/config.pbtxt 
    sed -i "s/queue_placeholder/${queue_fill}/g" "${backend_dir}"/config.pbtxt 

    # Run Triton Host Server for specified model
    CORE_NUMBER=$(lscpu | grep 'Core(s) per socket:' | awk '{print $4}')
    docker run -it --read-only --rm -e OMP_NUM_THREADS=$CORE_NUMBER --privileged --shm-size=1g -p${1}:8000 -p8001:8001 -p8002:8002 --tmpfs /tmp:rw,noexec,nosuid,size=1g --tmpfs /root/.cache/:rw,noexec,nosuid,size=4g -v$(pwd)/backend:/models --name ai_inference_host ai_inference:v1 numactl -C 0-"$((CORE_NUMBER - 1))" -m 0 tritonserver --model-repository=/models --log-verbose 1 --log-error 1 --log-info 1
}

start_client() {
    docker run -id --read-only --net=host --name "${image_name}_client" "${image_name}${image_tag}"
}

declare -l argument=${1:-"localhost"}

# Validate model_name
case "$model_name" in
    "bert_base") ;;
    "bert_large") ;;
    "bert_base_ov") ;;
    "bert_large_ov") ;;
    *)
        echo ""
        echo "Provided wrong model name"
        echo ""
        exit 1
        ;;
esac

echo ""
case "$argument" in
    "client")
        echo ""
        echo "Starting client server"
        echo ""
        start_client
        ;;
    "host")
        echo ""
        echo "Starting host server"
        echo ""
        start_host 80
        ;;
    "localhost")
        echo ""
        echo "Starting both client and host server for localhost deployment."
        echo ""
        start_client
        start_host 8000
        ;;
    *)
	    print_help
        ;;
esac
echo ""