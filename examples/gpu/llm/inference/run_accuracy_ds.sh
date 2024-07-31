#!/bin/bash

# Accuracy check task
task=piqa

# GPT-J-6b
Accuracy_lmeval_gpt-j-6b() {
    model=EleutherAI/gpt-j-6B
    sub_model_name=gpt-j-6B
    dir=accuracy/${model}/task${task}_ranknum2
    mkdir -p ${dir}
    LLM_ACC_TEST=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py -m ${model} --sub-model-name ${sub_model_name} --ipex --dtype float16 --accuracy-only --acc-tasks ${task} 2>&1 | tee log_acc_ds
    mv log_acc_ds ${dir}
}


## Llama-7b
Accuracy_lmeval_llama-7b() {
    model=decapoda-research/llama-7b-hf
    sub_model_name=llama-7b
    dir=accuracy/${model}/task${task}_ranknum2
    mkdir -p ${dir}
    LLM_ACC_TEST=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py -m ${model} --sub-model-name ${sub_model_name} --ipex --dtype float16 --accuracy-only --acc-tasks ${task} 2>&1 | tee log_acc_ds
    mv log_acc_ds ${dir}
}


## Llama-13b
Accuracy_lmeval_llama-13b() {
    model=decapoda-research/llama-13b-hf
    sub_model_name=llama-13b
    dir=accuracy/${model}/task${task}_ranknum2
    mkdir -p ${dir}
    LLM_ACC_TEST=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py -m ${model} --sub-model-name ${sub_model_name} --ipex --dtype float16 --accuracy-only --acc-tasks ${task} 2>&1 | tee log_acc_ds
    mv log_acc_ds ${dir}
}


## Llama2-7b
Accuracy_lmeval_llama2-7b() {
    model=meta-llama/Llama-2-7b-hf
    sub_model_name=llama2-7b
    dir=accuracy/${model}/task${task}_ranknum2
    mkdir -p ${dir}
    LLM_ACC_TEST=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py -m ${model} --sub-model-name ${sub_model_name} --ipex --dtype float16 --accuracy-only --acc-tasks ${task} 2>&1 | tee log_acc_ds
    mv log_acc_ds ${dir}
}


## Llama2-13b
Accuracy_lmeval_llama2-13b() {
    model=meta-llama/Llama-2-13b-hf
    sub_model_name=llama2-13b
    dir=accuracy/${model}/task${task}_ranknum2
    mkdir -p ${dir}
    LLM_ACC_TEST=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py -m ${model} --sub-model-name ${sub_model_name} --ipex --dtype float16 --accuracy-only --acc-tasks ${task} 2>&1 | tee log_acc_ds
    mv log_acc_ds ${dir}

}


## Llama2-34b
Accuracy_lmeval_llama2-34b() {
    model=codellama/CodeLlama-34b-hf
    sub_model_name=llama2-34b
    dir=accuracy/${model}/task${task}_ranknum2
    mkdir -p ${dir}
    LLM_ACC_TEST=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py -m ${model} --sub-model-name ${sub_model_name} --ipex --dtype float16 --accuracy-only --acc-tasks ${task} 2>&1 | tee log_acc_ds
    mv log_acc_ds ${dir}
}


## Llama2-70b
Accuracy_lmeval_llama2-70b() {
    model=meta-llama/Llama-2-70b-hf
    sub_model_name=llama2-70b
    dir=accuracy/${model}/task${task}_ranknum4
    mkdir -p ${dir}
    LLM_ACC_TEST=1 mpirun -np 4 --prepend-rank python -u run_generation_with_deepspeed.py -m ${model} --sub-model-name ${sub_model_name} --ipex --dtype float16 --accuracy-only --acc-tasks ${task} 2>&1 | tee log_acc_ds
    mv log_acc_ds ${dir}
}


## Falcon-40b
Accuracy_lmeval_falcon-40b() {
    model=tiiuae/falcon-40b
    sub_model_name=falcon-40b
    dir=accuracy/${model}/task${task}_ranknum2
    mkdir -p ${dir}
    LLM_ACC_TEST=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py -m ${model} --sub-model-name ${sub_model_name} --ipex --dtype float16 --accuracy-only --acc-tasks ${task} 2>&1 | tee log_acc_ds
    mv log_acc_ds ${dir}
}


## OPT-6.7b
Accuracy_lmeval_opt-6.7b() {
    model=facebook/opt-6.7b
    sub_model_name=opt-6.7b
    dir=accuracy/${model}/task${task}_ranknum2
    mkdir -p ${dir}
    LLM_ACC_TEST=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py -m ${model} --sub-model-name ${sub_model_name} --ipex --dtype float16 --accuracy-only --acc-tasks ${task} 2>&1 | tee log_acc_ds
    mv log_acc_ds ${dir}
}


## OPT-30b
Accuracy_lmeval_opt-30b() {
    model=facebook/opt-30b
    sub_model_name=opt-30b
    dir=accuracy/${model}/task${task}_ranknum2
    mkdir -p ${dir}
    LLM_ACC_TEST=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py -m ${model} --sub-model-name ${sub_model_name} --ipex --dtype float16 --accuracy-only --acc-tasks ${task} 2>&1 | tee log_acc_ds
    mv log_acc_ds ${dir}
}


## BLOOM-7b
Accuracy_lmeval_bloom-7b() {
    model=bigscience/bloom-7b1
    sub_model_name=bloom-7b
    dir=accuracy/${model}/task${task}_ranknum2
    mkdir -p ${dir}
    LLM_ACC_TEST=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py -m ${model} --sub-model-name ${sub_model_name} --ipex --dtype float16 --accuracy-only --acc-tasks ${task} 2>&1 | tee log_acc_ds
    mv log_acc_ds ${dir}
}


## BLOOM-176b
Accuracy_lmeval_bloom-176b() {
    model=bigscience/bloom
    sub_model_name=bloom-176b
    dir=accuracy/${model}/task${task}_ranknum8
    mkdir -p ${dir}
    LLM_ACC_TEST=1 mpirun -np 8 --prepend-rank python -u run_generation_with_deepspeed.py -m ${model} --sub-model-name ${sub_model_name} --ipex --dtype float16 --accuracy-only --acc-tasks ${task} 2>&1 | tee log_acc_ds
    mv log_acc_ds ${dir}
}


main() {
    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2

    Accuracy_lmeval_gpt-j-6b
    Accuracy_lmeval_llama-7b
    Accuracy_lmeval_llama-13b
    Accuracy_lmeval_llama2-7b
    Accuracy_lmeval_llama2-13b
    Accuracy_lmeval_llama2-34b
    Accuracy_lmeval_llama2-70b
    Accuracy_lmeval_falcon-40b
    Accuracy_lmeval_opt-6.7b
    Accuracy_lmeval_opt-30b
    Accuracy_lmeval_bloom-7b
    Accuracy_lmeval_bloom-176b
}

main
