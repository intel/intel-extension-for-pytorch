# DeepSeek R1 inference (generation)

## Description

This document has instructions for running [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) inference (generation) using Intel-optimized SGLang.

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
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu/

# Build sgl-kernel
conda install -y libnuma numactl

cd sgl-kernel
python setup.py install

cd ..

conda install -y gperftools -c conda-forge
pip install intel-openmp==2024.2.0

# Set IOMP and tcmalloc Preload for better performance
export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libtcmalloc.so

# Download the prompt file
wget --no-proxy -O prompt.json http://mlpc.intel.com/downloads/LLM/prompt-qwen3-1-16new.json

```

### Performance
#### 1. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **FINETUNED_MODEL**    | `# Test FP8: export FINETUNED_MODEL="deepseek-ai/DeepSeek-R1"`        |
| **INPUT_TOKEN**    |    `export INPUT_TOKEN=32 (for offline mode with prompt.json file, it is fixed with 1K length)`    |
| **OUTPUT_TOKEN**    |   `export OUTPUT_TOKEN=32`      |                        |
| **BATCH_SIZE** (optional)    |  `export BATCH_SIZE=16 (using BATCH_SIZE=1 for realtime mode, using BATCH_SIZE=N for throughput mode (N could be further tuned according to the testing host)`                                |

#### 2. Command lines

**The following command lines are for demonstration purposes. Modify the arguments and thread binding according to your requirements and CPU type.**

**Please avoid cross NUMA node memory access when setting SGLANG_CPU_OMP_THREADS_BIND.**

`SGLANG_CPU_OMP_THREADS_BIND` specifies the CPU cores dedicated to the OpenMP threads. `--tp` sets the TP size. Below are the example of running without TP and with TP = 6. By changing `--tp` and `SGLANG_CPU_OMP_THREADS_BIND` accordingly, you could set TP size to other values and specifiy the core binding for each rank.


##### 2.1 Bench one batch
```sh
# TP = 6, 43 OpenMP threads of rank0 are bound on 0-42 CPU cores, and the OpenMP threads of rank1 are bound on 43-85 CPU cores, etc.
# prompt file is needed for offline mode test, to activate MoE experts for different batches.
SGLANG_CPU_OMP_THREADS_BIND="0-42|43-85|86-127|128-170|171-213|214-255" python3 -m sglang.bench_one_batch --batch-size {BATCH_SIZE} --input {INPUT_TOKEN} --output {OUTPUT_TOKEN} --model  deepseek-ai/DeepSeek-R1  --trust-remote-code --device cpu --tp 6 --prompt-file prompt.json
```


### Accuracy (MMLU)
Dataset downloading (only needs to be done once)
```sh
cd /path/to/sglang/

wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar xf data.tar
```

Server
```sh
# R1 FP8
SGLANG_CPU_OMP_THREADS_BIND="0-42|43-85|86-127|128-170|171-213|214-255" python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1 --trust-remote-code --device cpu --log-requests --log-requests-level 1 --disable-overlap-schedule --tp 6 --chunked-prefill-size 2048 --max-running-requests 8
```

Client
```sh
python benchmark/mmlu/bench_sglang.py
```
