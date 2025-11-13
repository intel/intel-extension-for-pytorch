# LLaMA3.1 8B inference (generation)

## Description

This document has instructions for running [LLaMA3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) inference (generation) using Intel-optimized PyTorch and SGLang.

## Benchmarking with SGLang
### Preparation

```
# Create conda ENV
conda create -n sglang python=3.10
conda activate sglang

# Install SGLang
git clone https://github.com/blzheng/sglang.git

cd sglang

git checkout llm_po

pip install -e "python[all_cpu]"

conda install -y libsqlite=3.48.0

pip uninstall torch torchvision
git clone https://github.com/yanbing-j/pytorch -b yanbing/tf32_dev_branch_for_test
cd pytorch
git submodule sync
git submodule update --init --recursive
conda install cmake ninja
pip install -r requirements.txt
pip install mkl-static mkl-include
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
python setup.py install
cd ..
python -m pip install torchvision --index-url https://download.pytorch.org/whl/nightly/cpu/ --no-deps

# Build sgl-kernel
conda install -y libnuma numactl

cd sgl-kernel
cp pyproject_cpu.toml pyproject.toml
pip install uv
pip install scikit-build-core
SGLANG_CPU_FP8_BRGEMM=1 uv build --wheel -Cbuild-dir=build . --color=always --no-build-isolation
pip install "$(find dist -name 'sgl_kernel-*.whl' | sort | tail -n 1)" --force-reinstall
cd ..

conda install -y gperftools -c conda-forge
pip install intel-openmp==2024.2.0

# Set IOMP and tcmalloc Preload for better performance
export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libtcmalloc.so

# Download the prompt file
wget -O prompt.json https://intel-extension-for-pytorch.s3.us-east-1.amazonaws.com/miscellaneous/llm/prompt-1024-1.txt

```

### Performance
1. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **TEST_MODE** (THROUGHPUT, REALTIME)              | `export TEST_MODE=THROUGHPUT`                  |
| **ENABLE_TP**              | `export ENABLE_TP=1`                  |
| **OUTPUT_DIR**               |                               `export OUTPUT_DIR=<path to an output directory>`                               |
| **FINETUNED_MODEL**    | # Test BF16/FP16: `export FINETUNED_MODEL="meta-llama/Llama-3.1-8B-Instruct"` <br> # Test INT8: `export FINETUNED_MODEL="RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"` <br> Test FP8: `export FINETUNED_MODEL="Intel/llama-3.1-8b-instruct-fp8"`         |
| **PRECISION**     |                  `export PRECISION=bf16` (bf16, fp16, int8, fp8) |
| **INPUT_TOKEN**    |    `export INPUT_TOKEN=32 (choice in [32 64 128 256 512 1024 2016 2048 4096 8192]`    |
| **OUTPUT_TOKEN**    |   `export OUTPUT_TOKEN=32`      |
| **MODEL_DIR**               |                               `export MODEL_DIR=$(pwd)`                               |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=256 (using BATCH_SIZE=1 for realtime mode, using BATCH_SIZE=N for throughput mode (N could be further tuned according to the testing host, by default using 1)`                                |
| **ATTN_BACKEND** (optional) | # For the platforms that do not support Intel AMX: `export ATTN_BACKEND=torch_native` |
2. Command lines
```
bash run_model_with_sglang.sh
```

### Accuracy
1. Server:
```
export SGLANG_USE_CPU_ENGINE=1

# BF16
python -m sglang.launch_server --model ${FINETUNED_MODEL} --trust-remote-code --device cpu --disable-overlap-schedule --dtype bfloat16  --mem-fraction-static 0.8 --max-total-tokens 65536 --disable-radix-cache --tp 6 --enable-torch-compile

# FP16
python -m sglang.launch_server --model ${FINETUNED_MODEL} --trust-remote-code --device cpu --disable-overlap-schedule --dtype float16  --mem-fraction-static 0.8 --max-total-tokens 65536 --disable-radix-cache --tp 6 --enable-torch-compile

# INT8
python -m sglang.launch_server --model ${FINETUNED_MODEL} --trust-remote-code --device cpu --disable-overlap-schedule --quantization w8a8_int8  --mem-fraction-static 0.8 --max-total-tokens 65536 --disable-radix-cache --tp 6 --enable-torch-compile

# FP8
SGLANG_LLAMA_BRGEMM_FP8A8=1 python -m sglang.launch_server --model ${FINETUNED_MODEL} --trust-remote-code --device cpu --disable-overlap-schedule  --mem-fraction-static 0.8 --max-total-tokens 65536 --disable-radix-cache --tp 6 --enable-torch-compile

```

2. Client:
```
cd sglang/benchmark/mmlu
bash download_data.sh
python bench_sglang.py --parallel 1
```

## Benchmarking with TorchInductor
### Preparation
```
# Create virtual environment `venv` and activate it:
python3 -m venv venv
. ./venv/bin/activate

# Run setup.sh
source ./setup.sh

# Install PyTorch, Torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu/

# Install Intel OpenMP and TCMalloc
pip install packaging intel-openmp accelerate
conda install -y gperftools -c conda-forge

# Set IOMP and tcmalloc Preload for better performance
export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libtcmalloc.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:$LD_PRELOAD

```

### Inference
1. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **TEST_MODE** (THROUGHPUT, ACCURACY, REALTIME)              | `export TEST_MODE=THROUGHPUT`                  |
| **OUTPUT_DIR**               |                               `export OUTPUT_DIR=<path to an output directory>`                               |
| **FINETUNED_MODEL**    | `export FINETUNED_MODEL="meta-llama/Llama-3.1-8B-Instruct"`         |
| **PRECISION**     |                  `export PRECISION=bf16` (fp32, bf32, bf16, fp16, int8) |
| **DATASET** |`export DATASET=mmlu` (For accuracy mode only. Supported datasets: mmlu, gsm8k_cot_llama, lambada)|
| **INPUT_TOKEN**    |    `export INPUT_TOKEN=32 (choice in [32 64 128 256 512 1024 2016], we prefer to benchmark on 32 and 2016)`    |
| **OUTPUT_TOKEN**    |   `export OUTPUT_TOKEN=32 (32 is preferred, while you could set any other length)`      |
| **MODEL_DIR**               |                               `export MODEL_DIR=$(pwd)`                               |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=256`                                |

2. Command lines
```
bash run_model_with_inductor.sh
```