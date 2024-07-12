#!/bin/bash

# Accuracy check task
task=piqa
## GPT-J
Accuracy_lmeval_gpt-j-6b() {
    model=EleutherAI/gpt-j-6B
    sub_model_name=gpt-j-6B
    dir=accuracy/${model}/task${task}
    mkdir -p ${dir}
    LLM_ACC_TEST=1 python -u run_generation.py -m ${model} --sub-model-name ${sub_model_name} --ipex --dtype float16 --accuracy-only --acc-tasks ${task} 2>&1 | tee log_acc
    mv log_acc ${dir}
}


## Llama-7b
Accuracy_lmeval_llama-7b() {
    model=decapoda-research/llama-7b-hf
    sub_model_name=llama-7b
    dir=accuracy/${model}/task${task}
    mkdir -p ${dir}
    LLM_ACC_TEST=1 python -u run_generation.py -m ${model} --sub-model-name ${sub_model_name} --ipex --dtype float16 --accuracy-only --acc-tasks ${task} 2>&1 | tee log_acc
    mv log_acc ${dir}
}


## Llama-13b
Accuracy_lmeval_llama-13b() {
    model=decapoda-research/llama-13b-hf
    sub_model_name=llama-13b
    dir=accuracy/${model}/task${task}
    mkdir -p ${dir}
    LLM_ACC_TEST=1 python -u run_generation.py -m ${model} --sub-model-name ${sub_model_name} --ipex --dtype float16 --accuracy-only --acc-tasks ${task} 2>&1 | tee log_acc
    mv log_acc ${dir}
}


## Llama2-7b
Accuracy_lmeval_llama2-7b() {
    model=meta-llama/Llama-2-7b-hf
    sub_model_name=llama2-7b
    dir=accuracy/${model}/task${task}
    mkdir -p ${dir}
    LLM_ACC_TEST=1 python -u run_generation.py -m ${model} --sub-model-name ${sub_model_name} --ipex --dtype float16 --accuracy-only --acc-tasks ${task} 2>&1 | tee log_acc
    mv log_acc ${dir}
}


## Llama2-13b
Accuracy_lmeval_llama2-13b() {
    model=meta-llama/Llama-2-13b-hf
    sub_model_name=llama2-13b
    dir=accuracy/${model}/task${task}
    mkdir -p ${dir}
    LLM_ACC_TEST=1 python -u run_generation.py -m ${model} --sub-model-name ${sub_model_name} --ipex --dtype float16 --accuracy-only --acc-tasks ${task} 2>&1 | tee log_acc
    mv log_acc ${dir}
}


## OPT
Accuracy_lmeval_opt-6.7b() {
    model=facebook/opt-6.7b
    sub_model_name=opt-6.7b
    dir=accuracy/${model}/task${task}
    mkdir -p ${dir}
    LLM_ACC_TEST=1 python -u run_generation.py -m ${model} --sub-model-name ${sub_model_name} --ipex --dtype float16 --accuracy-only --acc-tasks ${task} 2>&1 | tee log_acc
    mv log_acc ${dir}
}


## BLOOM
Accuracy_lmeval_bloom-7b() {
    model=bigscience/bloom-7b1
    sub_model_name=bloom-7b
    dir=accuracy/${model}/task${task}
    mkdir -p ${dir}
    LLM_ACC_TEST=1 python -u run_generation.py -m ${model} --sub-model-name ${sub_model_name} --dtype float16 --ipex --accuracy-only --acc-tasks ${task} 2>&1 | tee log_acc
    mv log_acc ${dir}
}


main() {

    Accuracy_lmeval_gpt-j-6b
    Accuracy_lmeval_llama2-7b
    Accuracy_lmeval_llama2-13b
    Accuracy_lmeval_opt-6.7b
    Accuracy_lmeval_bloom-7b
}

main


