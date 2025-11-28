# GPT-J 6B inference (generation)

## Description

This document has instructions for running [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6b) inference (generation) using Intel-optimized PyTorch.

## Benchmarking with TorchInductor
### Preparation
```
# Create virtual environment `venv` and activate it:
python3 -m venv venv
. ./venv/bin/activate

# Run setup.sh
source ./setup.sh

# Install PyTorch, Torchvision
pip install torch torchvision torchao --index-url https://download.pytorch.org/whl/nightly/cpu/

# Install Intel OpenMP and TCMalloc
pip install packaging intel-openmp accelerate
conda install -y gperftools -c conda-forge

# Set IOMP and tcmalloc Preload for better performance
export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libtcmalloc.so:$(realpath venv)/lib/libiomp5.so:$LD_PRELOAD

```

### Inference
1. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **TEST_MODE** (THROUGHPUT, ACCURACY, REALTIME)              | `export TEST_MODE=THROUGHPUT`                  |
| **OUTPUT_DIR**               |                               `export OUTPUT_DIR=<path to an output directory>`                               |
| **FINETUNED_MODEL**    | `export FINETUNED_MODEL="EleutherAI/gpt-j-6b"`         |
| **PRECISION**     |                  `export PRECISION=bf16` (fp32, bf32, bf16, fp16, int8) |
| **DATASET** |`export DATASET=lambada` (For accuracy mode only.)|
| **INPUT_TOKEN**    |    `export INPUT_TOKEN=32 (choice in [32 64 128 256 512 1024 2016], we prefer to benchmark on 32 and 2016)`    |
| **OUTPUT_TOKEN**    |   `export OUTPUT_TOKEN=32 (32 is preferred, while you could set any other length)`      |
| **MODEL_DIR**               |                               `export MODEL_DIR=$(pwd)`                               |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=256`                                |

2. Command lines
```
bash run_model_with_inductor.sh
```