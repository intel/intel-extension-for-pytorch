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
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beam ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
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
    model=/mllnvme0/llama2-7b
    sub_model_name=llama2-7b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## Llama2-13b
Run_benchmark_llama2-13b() {
    model=/mllnvme0/llama2-13b
    sub_model_name=llama2-13b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## OPT
Run_benchmark_opt-6.7b() {
    model=facebook/opt-6.7b
    sub_model_name=opt-6.7b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}


## BLOOM
Run_benchmark_bloom-7b() {
    model=bigscience/bloom-7b1
    sub_model_name=bloom-7b
    dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}

main() {

    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
    export ENABLE_SDP_FUSION=1

    export HF_HOME=/mllnvme0/huggingface/
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_EVALUATE_OFFLINE=1

    Run_benchmark_gpt-j-6b
    Run_benchmark_llama-7b
    Run_benchmark_llama-13b
    Run_benchmark_llama2-7b
    Run_benchmark_llama2-13b
    Run_benchmark_opt-6.7b
    Run_benchmark_bloom-7b
}

main
