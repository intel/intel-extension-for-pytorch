# Text Generation

We provide the inference benchmarking scripts for large language models text generation.<br />
Support large language model families, including LLaMA 2, GPT-J, OPT, and Bloom.<br />
The scripts include both single instance and distributed (DeepSpeed) use cases.<br />
The scripts cover model generation inference with low precions cases for different models with best perf and accuracy (fp16 AMP and weight only quantization).<br />

# Supported Model List

| MODEL FAMILY | Verified < MODEL ID > (Huggingface hub)| FP16 | Weight only quantization INT4 |
|---|:---:|:---:|:---:|
|LLAMA 2| "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-70b-hf" | ✅ | ❎ |
|GPT-J| "EleutherAI/gpt-j-6b" | ✅ | ✅ |
|OPT|"facebook/opt-6.7b", "facebook/opt-30b"| ✅ | ❎ |
|Bloom|"bigscience/bloom-7b1", "bigscience/bloom"| ✅ | ❎ |

*Note*: The above verified models (including other models in the same model family, like "codellama/CodeLlama-7b-hf" from LLAMA family) are well supported with all optimizations like indirect access KV cache and fused ROPE. For other LLM model families, we are working in progress to cover those optimizations, which will expand the model list above.

# Supported platform

\* PVC(1550/1100): support all the models in model list<br />
\* ATS-M, Arc: support GPT-J (EleutherAI/gpt-j-6b)

# Environment Setup

1. Get the Intel® Extension for PyTorch\* source code

```bash
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout v2.1.10+xpu
git submodule sync
git submodule update --init --recursive
```

2.a. It is highly recommended to build a Docker container from the provided `Dockerfile` for single-instance executions.

```bash
# Build an image with the provided Dockerfile by compiling Intel® Extension for PyTorch* from source
DOCKER_BUILDKIT=1 docker build -f examples/gpu/inference/python/llm/Dockerfile --build-arg GID_RENDER=$(getent group render | sed -E 's,^render:[^:]*:([^:]*):.*$,\1,') --build-arg COMPILE=ON -t ipex-llm:2.1.10 .

# Build an image with the provided Dockerfile by installing from Intel® Extension for PyTorch* prebuilt wheel files
DOCKER_BUILDKIT=1 docker build -f examples/cpu/inference/python/llm/Dockerfile --build-arg GID_RENDER=$(getent group render | sed -E 's,^render:[^:]*:([^:]*):.*$,\1,') -t ipex-llm:2.1.10 .

# Run the container with command below
docker run --rm -it --privileged --device=/dev/dri --ipc=host ipex-llm:2.1.10 bash

# When the command prompt shows inside the docker container, enter llm examples directory
cd llm
```

2.b. Alternatively, you can take advantage of a provided environment configuration script to setup an environment without using a docker container.

```bash
# Make sure you have GCC >= 11 is installed on your system.
# Create a conda environment
conda create -n llm python=3.10 -y
conda activate llm

# Setup the environment with the provided script
cd examples/gpu/inference/python/llm
# If you want to install Intel® Extension for PyTorch\* from prebuilt wheel files, use the command below:
bash ./tools/env_setup.sh 7
# If you want to install Intel® Extension for PyTorch\* from source, use the commands below:
bash ./tools/env_setup.sh 3 <DPCPP_ROOT> <ONEMKL_ROOT> <AOT>
export LD_PRELOAD=$(bash ../../../../../tools/get_libstdcpp_lib.sh)
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
```

\* `DPCPP_ROOT` is the placeholder for path where DPC++ compile was installed to. By default, it is `/opt/intel/oneapi/compiler/latest`.<br />
\* `ONEMKL_ROOT` is the placeholder for path where oneMKL was installed to. By default, it is `/opt/intel/oneapi/mkl/latest`.<br />
\* `AOT` is a text string to enable `Ahead-Of-Time` compilation for specific GPU models. Check [tutorial](../../../../../docs/tutorials/technical_details/AOT.md) for details.<br />

3. Once an environment is configured with either method above, set necessary environment variables with an environment variables activation script.

```bash
# Activate environment variables
source ./tools/env_activate.sh
```


# Run Models Generations

| Benchmark mode | FP16 | Weight only quantization INT4 |
|---|:---:|:---:|
|Single instance | ✅ | ✅ |
| Distributed (autotp) |  ✅ | ❎ |

## Example usages of one-click Python script
You can run LLM with a one-click bash script "run_benchmark.sh" for all inference cases.
```
bash run_benchmark.sh
```

### Single Instance Performance

Note: We only support LLM optimizations with datatype float16, so please don't change datatype to float32 or bfloat16.

```bash
# fp16 benchmark
python -u run_generation.py --benchmark -m ${model} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency
```

Notes:

(1) By default, generations are based on bs = 1, input token size = 1024, output toke size = 128, iteration num = 10 and "beam search", and beam size = 4. For beam size = 1 and other settings, please export env settings, such as: "beam=1", "input=32", "output=32", "iter=5".

### Distributed Performance with DeepSpeed

You can run LLM with a one-click bash script "run_benchmark_ds.sh" for all distributed inference cases.
```
bash run_benchmark_ds.sh
```

```bash
# distributed env setting
source ${ONECCL_DIR}/build/_install/env/setvars.sh
# fp16 benchmark
mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency
```

Notes:

(1) By default, generations are based on bs = 1, input token size = 1024, output toke size = 128, iteration num = 10 and "beam search", and beam size = 4. For beam size = 1 and other settings, please export env settings, such as: "beam=1", "input=32", "output=32", "iter=5".

# Advanced Usage

## Weight only quantization with low precision checkpoint (Experimental)

Using INT4 weights can further improve performance by reducing memory bandwidth. However, direct per-channel quantization of weights to INT4 probably results in poor accuracy. Some algorithms can modify weights through calibration before quantizing weights to minimize accuracy drop. GPTQ is one of such algorithms. You may generate modified weights and quantization info (scales, zero points) for a certain model with a dataset for specified tasks by such algorithms. The results are saved as a `state_dict` in a `.pt` file. We provided a script here to run GPT-J .

### Single Instance GPT-J Weight only quantization Performance

```bash
# quantization benchmark
#To run quantization performance, you need to firstly get the quantized weight with the following step (1) and then run the performance benchmark with the following step (2)
## (1) Get the quantized weight
download link: https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/xpu/gptj_int4_weight_master.pt
export weight_path = path-to-your-weight

## (2) Run quantization performance test
python -u run_generation.py --device xpu --ipex --dtype float16 --input-tokens ${input} --max-new-tokens ${out}  --token-latency --benchmark  --num-beams ${beam}  -m ${model} --woq --woq_checkpoint_path ${weight_path}
```

### Single Instance GPT-J Weight only quantization INT4 Accuracy

```bash
# we use "lambada_standard" task to check accuracy
LLM_ACC_TEST=1 python -u run_generation.py -m ${model} --ipex --dtype float16 --accuracy-only --acc-tasks ${task} --woq --woq_checkpoint_path ${weight_path}
```

## Single Instance Accuracy

```bash
Accuracy test {TASK_NAME}, choice in this [link](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md), by default we use "lambada_standard"

# one-click bash script
bash run_accuracy.sh

# float16
LLM_ACC_TEST=1 python -u run_generation.py -m ${model} --ipex --dtype float16 --accuracy-only --acc-tasks ${task}
```

## Distributed Accuracy with DeepSpeed

```bash
# Run distributed accuracy with 2 ranks of one node for float16 with ipex
source ${ONECCL_DIR}/build/_install/env/setvars.sh

# one-click bash script
bash run_accuracy_ds.sh

# float16
LLM_ACC_TEST=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py -m ${model} --ipex --dtype float16 --accuracy-only --acc-tasks ${task} 2>&1
```
