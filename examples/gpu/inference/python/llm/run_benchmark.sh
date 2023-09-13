# if you want to get the --token-latency breakdown, please follow the bkc
# git clone https://github.com/huggingface/transformers.git
# cd transformers
# git checkout v4.29.2
# git apply gput-models/LLM/profile_patch
# pip install setup.py

export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export ENABLE_SDP_FUSION=1
export ZE_AFFINITY_MASK=0.1

export HF_HOME=~/.cache/huggingface/
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1

# the PoC weekly check:
# beam=1, bs=1, input=1024, out=128
# beam=4, bs=1, input=1024, out=128
beam=4
bs=1
input=1024
out=128
iter=10

model=EleutherAI/gpt-j-6B
dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
mkdir -p ${dir}
python -u run_generation.py --benchmark -m ${model} --num-beam ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
PROFILE=1 python -u run_generation.py --benchmark -m ${model} --num-beam ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
mv log_e2e ${dir}
mv profile*pt ${dir}
mv trace.json ${dir}

model=decapoda-research/llama-7b-hf
dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
mkdir -p ${dir}
python -u run_generation.py --benchmark -m ${model} --num-beam ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
PROFILE=1 python -u run_generation.py --benchmark -m ${model} --num-beam ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
mv log_e2e ${dir}
mv profile*pt ${dir}
mv trace.json ${dir}


model=decapoda-research/llama-13b-hf
dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
mkdir -p ${dir}
python -u run_generation.py --benchmark -m ${model} --num-beam ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
PROFILE=1 python -u run_generation.py --benchmark -m ${model} --num-beam ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
mv log_e2e ${dir}
mv profile*pt ${dir}
mv trace.json ${dir}


model=facebook/opt-6.7b
dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
mkdir -p ${dir}
python -u run_generation.py --benchmark -m ${model} --num-beam ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
PROFILE=1 python -u run_generation.py --benchmark -m ${model} --num-beam ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
mv log_e2e ${dir}
mv profile*pt ${dir}
mv trace.json ${dir}


model=bigscience/bloom-7b1
dir=perf/${model}/beam${beam}_bs${bs}_input${input}_out${out}
mkdir -p ${dir}
python -u run_generation.py --benchmark -m ${model} --num-beam ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency 2>&1 | tee log_e2e
PROFILE=1 python -u run_generation.py --benchmark -m ${model} --num-beam ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16
mv log_e2e ${dir}
mv profile*pt ${dir}
mv trace.json ${dir}
