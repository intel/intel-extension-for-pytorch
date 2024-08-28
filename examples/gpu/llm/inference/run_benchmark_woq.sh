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


## QWen-7b
Run_benchmark_qwen-7b_int4() {
    model=Qwen/Qwen-7B-Chat
    sub_model_name=qwen-7b
    dir=int4_perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation_woq.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation_woq.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}

## GPT-J-6B
Run_benchmark_gpt-j-6b_int4() {
    model=EleutherAI/gpt-j-6B
    sub_model_name=gpt-j-6B
    dir=int4_perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation_woq.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation_woq.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}

## Llama2-7b
Run_benchmark_llama2-7b_int4() {
    model=meta-llama/Llama-2-7b-hf
    sub_model_name=llama2-7b
    dir=int4_perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation_woq.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation_woq.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}

## Llama2-70b
Run_benchmark_llama2-70b_int4() {
    model=meta-llama/Llama-2-70b-hf
    sub_model_name=llama2-70b
    dir=int4_perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation_woq.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation_woq.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}

## Phi3-mini
Run_benchmark_Phi3-mini() {
    model=microsoft/Phi-3-mini-4k-instruct
    sub_model_name=phi3-mini
    dir=int4_perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
    mkdir -p ${dir}
    python -u run_generation_woq.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
    mv log_e2e ${dir}
    PROFILE=1 python -u run_generation.py --benchmark -m ${model} --sub-model-name ${sub_model_name} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
    mv profile*pt ${dir}
    mv trace.json ${dir}
}

main() {

    Run_benchmark_qwen-7b_int4
    Run_benchmark_gpt-j-6b_int4
    Run_benchmark_llama2-7b_int4
    Run_benchmark_llama2-70b_int4
    Run_benchmark_Phi3-mini
}

main
