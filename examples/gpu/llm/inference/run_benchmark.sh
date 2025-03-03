# if you want to get the --token-latency breakdown, please follow the bkc
# git clone https://github.com/huggingface/transformers.git
# cd transformers
# git checkout v4.31.0
# git apply gpu-models/LLM/profile_patch
# pip install setup.py

# export TOKENIZERS_PARALLELISM=false
# the PoC weekly check:
# beam=1, bs=1, input=1024, out=128
# beam=4, bs=1, input=1024, out=128
beam=4
bs=1
input=1024
out=128
iter=10


## GPT-J
Run_benchmark_gpt-j-6b() {
    model=EleutherAI/gpt-j-6B
    sub_model_name=gpt-j-6B
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beam ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## Llama-7b
Run_benchmark_llama-7b() {
    model=decapoda-research/llama-7b-hf
    sub_model_name=llama-7b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## Llama-13b
Run_benchmark_llama-13b() {
    model=decapoda-research/llama-13b-hf
    sub_model_name=llama-13b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## Llama2-7b
Run_benchmark_llama2-7b() {
    model=meta-llama/Llama-2-7b-hf
    sub_model_name=llama2-7b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## Llama2-13b
Run_benchmark_llama2-13b() {
    model=meta-llama/Llama-2-13b-hf
    sub_model_name=llama2-13b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}



## Llama3-8b
Run_benchmark_llama3-8b() {
    model=meta-llama/Meta-Llama-3-8B
    sub_model_name=llama3-8b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## Llama3.2-1b
Run_benchmark_llama3.2-1b() {
    model=Llama-3.2-1B-model-id
    sub_model_name=llama3-8b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## Llama3.2-3b
Run_benchmark_llama3.2-3b() {
    model=Llama-3.2-3B-model-id
    sub_model_name=llama3-8b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}

## OPT
Run_benchmark_opt-6.7b() {
    model=facebook/opt-6.7b
    sub_model_name=opt-6.7b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency --use-static-cache 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --use-static-cache
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## BLOOM
Run_benchmark_bloom-7b() {
    model=bigscience/bloom-7b1
    sub_model_name=bloom-7b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency --use-static-cache 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --use-static-cache
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## Baichuan2-13b
Run_benchmark_baichuan2-13b-chat() {
    model=baichuan-inc/Baichuan2-13B-Chat
    sub_model_name=baichuan2-13b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency --disable-auto-cast 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## QWen-7b
Run_benchmark_qwen-7b() {
    model=Qwen/Qwen-7B-Chat
    sub_model_name=qwen-7b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## QWen1.5-7b
Run_benchmark_qwen1.5-7b() {
    model=Qwen/Qwen1.5-7B-Chat
    sub_model_name=qwen1.5-7b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## QWen2-7b
Run_benchmark_qwen2-7b() {
    model=Qwen/Qwen2-7B-Instruct
    sub_model_name=qwen2-7b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## QWen2.5-0.5b
Run_benchmark_qwen2.5-0.5b() {
    model=Qwen/Qwen2.5-0.5B
    sub_model_name=qwen2.5-0.5b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## ChatGLM3-6b-chat
Run_benchmark_chatglm3-6b-chat() {
    model=THUDM/chatglm3-6b
    sub_model_name=chatglm3-6b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## Phi3-mini
Run_benchmark_Phi3-mini() {
    model=microsoft/Phi-3-mini-4k-instruct
    sub_model_name=phi3-mini
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-hf-code --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-hf-code --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## Phi3-small
Run_benchmark_Phi3-small() {
    model=microsoft/Phi-3-small-128k-instruct
    sub_model_name=phi3-small
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## GLM4-9b-chat
Run_benchmark_glm4-9b-chat() {
    model=THUDM/glm-4-9b-chat
    sub_model_name=glm-4-9b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## GLM4-9b-chat-hf
Run_benchmark_glm4-9b-chat-hf() {
    model=THUDM/glm-4-9b-chat-hf
    sub_model_name=glm-4-9b-hf
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --use-static-cache --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## Mistral-7B
Run_benchmark_Mistral-7B() {
    model=mistralai/Mistral-7B-Instruct-v0.2
    sub_model_name=mistral
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


main() {

    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2

    Run_benchmark_gpt-j-6b
    Run_benchmark_llama-7b
    Run_benchmark_llama-13b
    Run_benchmark_llama2-7b
    Run_benchmark_llama2-13b
    Run_benchmark_llama3-8b
    Run_benchmark_llama3.2-1b
    Run_benchmark_llama3.2-3b
    Run_benchmark_opt-6.7b
    Run_benchmark_bloom-7b
    Run_benchmark_baichuan2-13b-chat
    Run_benchmark_qwen-7b
    Run_benchmark_qwen1.5-7b
    Run_benchmark_qwen2-7b
    Run_benchmark_qwen2.5-0.5b
    Run_benchmark_chatglm3-6b-chat
    Run_benchmark_Phi3-mini
    Run_benchmark_Phi3-small
    Run_benchmark_glm4-9b-chat
    Run_benchmark_glm4-9b-chat-hf
    Run_benchmark_Mistral-7B
}

main
