#!/bin/sh
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export ENABLE_SDP_FUSION=1

export HF_HOME=~/huggingface/
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1

LLM_ACC_TEST=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py -m bigscience/bloom-7b1 --accuracy-only --ipex
LLM_ACC_TEST=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py -m facebook/opt-30b --accuracy-only --ipex
LLM_ACC_TEST=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py -m EleutherAI/gpt-j-6b --accuracy-only --ipex
