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

# Build sgl-kernel
conda install -y libnuma numactl

cd sgl-kernel
cp pyproject_cpu.toml pyproject.toml
pip install uv
pip install scikit-build-core
SGLANG_CPU_FP8_BRGEMM=1 uv build --wheel -Cbuild-dir=build . --color=always --no-build-isolation
pip install dist/sgl_kernel-0.2.5-cp310-cp310-linux_x86_64.whl --force-reinstall
cd ..

conda install -y gperftools -c conda-forge
pip install intel-openmp==2024.2.0

# Set IOMP and tcmalloc Preload for better performance
export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libtcmalloc.so

# Download the prompt file
wget -O prompt.json https://intel-extension-for-pytorch.s3.us-east-1.amazonaws.com/miscellaneous/llm/prompt-1024-1.txt

```

### Performance
#### 1. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **FINETUNED_MODEL**    | `# Test FP8: export FINETUNED_MODEL="deepseek-ai/DeepSeek-R1"`        |
| **INPUT_TOKEN**    |    `export INPUT_TOKEN=1024 (for offline mode with prompt.json file, it is fixed with 1K length)`    |
| **OUTPUT_TOKEN**    |   `export OUTPUT_TOKEN=32`      |                        |
| **BATCH_SIZE** (optional)    |  `export BATCH_SIZE=16 (using BATCH_SIZE=1 for realtime mode, using BATCH_SIZE=N for throughput mode (N could be further tuned according to the testing host)`                                |

#### 2. Command lines

**The following command lines are for demonstration purposes. Modify the arguments and thread binding according to your requirements and CPU type.**

**Please avoid cross NUMA node memory access when setting SGLANG_CPU_OMP_THREADS_BIND.**

`SGLANG_CPU_OMP_THREADS_BIND` specifies the CPU cores dedicated to the OpenMP threads. `--tp` sets the TP size. Below are the example of running without TP and with TP = 6. By changing `--tp` and `SGLANG_CPU_OMP_THREADS_BIND` accordingly, you could set TP size to other values and specifiy the core binding for each rank.


##### 2.1 Bench one batch
```sh
# TP = 6, 43 OpenMP threads of rank0 are bound on 0-42 CPU cores, and the OpenMP threads of rank1 are bound on 43-85 CPU cores, etc.
SGLANG_USE_CPU_ENGINE=1 SGLANG_DEEPSEEK_FP8A8=1 SGLANG_CPU_OMP_THREADS_BIND="0-42|43-85|86-127|128-170|171-213|214-255" python3 -m sglang.bench_one_batch --batch-size ${BATCH_SIZE} --input ${INPUT_TOKEN} --output ${OUTPUT_TOKEN} --model ${FINETUNED_MODEL} --trust-remote-code --device cpu --tp 6 --prompt-file prompt.json --mem-fraction-static 0.8 --max-total-tokens 65536 --enable-torch-compile
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
SGLANG_USE_CPU_ENGINE=1 SGLANG_DEEPSEEK_FP8A8=1 SGLANG_CPU_OMP_THREADS_BIND="0-42|43-85|86-127|128-170|171-213|214-255" python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1 --trust-remote-code --device cpu --log-requests --log-requests-level 1 --disable-overlap-schedule --tp 6 --chunked-prefill-size 2048 --max-running-requests 8 --mem-fraction-static 0.8 --max-total-tokens 65536 --enable-torch-compile
```

Client
```sh
python benchmark/mmlu/bench_sglang.py
```
