#!/bin/bash

# Copyright (c) 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0

source "$(pwd)"/config.properties

print_help() {
    echo ""
    echo "Usage: $0 [arg]"
    echo ""
    echo "   This script will send perf_analyzer requests to Triton Host Server."
    echo ""
    echo "   Available arguments:"
    echo "   [ -s | --sequence_length ] sequence_length (default = 128)     - size of input sequence tokens. Must be greater than 0."
    echo "                                                                    sequence_length can't exceed 128 for Bert Base model and 256 for Bert Large model."
    echo "   [ -r | --run_mode ]        run_mode        (default = sync)    - run mode of perf_analyzer. To choose with:"
    echo "                                                                      * sync          - send syncronous requests"
    echo "                                                                      * async         - send asyncronous requests"
    echo "   [ -c | --concurrency ]     concurrency     (default = 1)       - concurency range of requests to be send. Must be greater than 0."
    echo "   [ -n | --number_request ]  number_request  (default = 1000)    - number of requests to be send. Must be greater than 0."
    echo ""
    echo ""
    echo "   Example cmd: run_test.sh --sequence_length 128 --run_mode sync --concurrency 1 --number_request 1000"
    echo ""
    exit 2
}

if [ $# -eq 0 ]; then
    echo "Script will run with default arguments"
else 
    short_args=s:,r:,c:,n:,h
    long_args=sequence_length:,run_mode:,concurrency:,number_request:,help
    args=$(getopt --alternative --name run_test --options $short_args --longoptions $long_args -- "$@") 
    eval set -- "$args"
    
    while : 
    do
        case "$1" in
            -s | --sequence_length )
            sequence_length="$2"
            shift 2
            ;;
            -r | --run_mode )
            run_mode="$2"
            shift 2
            ;;
            -c | --concurrency )
            concurrency="$2"
            shift 2
            ;;
            -n | --number_request )
            number_request="$2"
            shift 2
            ;;
            -h | --help )
            print_help
            ;;
            --)
            shift;
            break
            ;;
            *)
            echo "Unexpected option: $1"
            print_help
            break
            ;;
        esac
    done
fi

default=("Arguments to be filled with default values: ")
declare -i default_check=0
[ -z "$sequence_length" ] && declare -i sequence_length=128 && default+=("sequence_length ") && default_check=$((default_check + 1))
[ -z "$run_mode" ] && declare -l run_mode="sync" && default+=("run_mode ") && default_check=$((default_check + 1))
[ -z "$concurrency" ] && declare -i concurrency=1 && default+=("concurrency ") && default_check=$((default_check + 1))
[ -z "$number_request" ] && declare -i number_request=1000 && default+=("number_request ") && default_check=$((default_check + 1))

# Validate integer variables
case $sequence_length in
    ''|*[!0-9]*) echo "sequence_length must be an integer" && print_help ;;
    *) ;;
esac
case $concurrency in
    ''|*[!0-9]*) echo "concurrency must be an integer" && print_help ;;
    *) ;;
esac
case $number_request in
    ''|*[!0-9]*) echo "number_request must be an integer" && print_help ;;
    *) ;;
esac

[ $sequence_length -le 0 ] && echo "sequence_length must be greater than 0" && print_help
if [[ "${model_name}" == "bert_base"* ]]; then
    [ $sequence_length -gt 128 ] && echo "sequence_length can't exceed 128 for Bert Base model" && print_help
elif [[ "${model_name}" == "bert_large"* ]]; then
     [ $sequence_length -gt 256 ] && echo "sequence_length can't exceed 256 for Bert Large model" && print_help
fi
[ $concurrency -le 0 ] && echo "concurrency must be greater than 0" && print_help
[ $number_request -le 0 ] && echo "number_request must be greater than 0" && print_help


[ $default_check -gt 0 ] && ( echo "" && echo "${default[@]}" && echo "" )

[ "${ip_address}" = localhost ] && port=8000 || port=80

cmd="perf_analyzer -u ${ip_address}:${port} -m ${model_name} --shape INPUT0:${sequence_length} --input-data zero --${run_mode} --concurrency-range ${concurrency} --measurement-mode count_windows --measurement-request-count ${number_request}"

{ 
    echo "" &&
    echo "Starting perf analyzer for ${model_name} with sequence length ${sequence_length}, run_mode ${run_mode}, number_request ${number_request}, concurrency ${concurrency} on ${ip_address}:${port}" &&
    echo "" &&
    docker exec -it "${image_name}_client" ${cmd} &&
    echo ""
} || { 
    echo "" &&
    echo "See above message. If failure when receiving data from the peer occured wait for all models to be loaded on host side or check connection with host." &&
    echo ""
}