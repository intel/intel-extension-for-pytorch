# Text Generation

Here you can find the inference benchmarking scripts for large language models (LLM) text generation. These scripts:

- Support LLaMA 2, GPT-J, OPT, and Bloom model families
- Include both single instance and distributed (DeepSpeed) use cases
- Cover model generation inference with low precision cases for different models with best performance and accuracy (fp16 AMP and weight only quantization)

## Supported Models

| MODEL FAMILY | Verified < MODEL ID > (Huggingface hub)| FP16 | Weight only quantization INT4 |
|---|:---:|:---:|:---:|
|LLAMA 2| "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-70b-hf" | ✅ | ❎ |
|GPT-J| "EleutherAI/gpt-j-6b" | ✅ | ✅ |
|OPT|"facebook/opt-6.7b", "facebook/opt-30b"| ✅ | ❎ |
|Bloom|"bigscience/bloom-7b1", "bigscience/bloom"| ✅ | ❎ |

**Note**: The verified models mentioned above (including other models in the same model family, like "codellama/CodeLlama-7b-hf" from LLAMA family) are well-supported with all optimizations like indirect access KV cache and fused ROPE. For other LLM model families, we are actively working to implement these optimizations, which will be reflected in the expanded model list above.

## Supported Platforms

\* PVC(1550/1100): support all the models in the model list above<br />
\* ATS-M, Arc: support GPT-J (EleutherAI/gpt-j-6b)

## Environment Setup

1. Get the Intel® Extension for PyTorch\* source code:

```bash
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout v2.1.10+xpu
git submodule sync
git submodule update --init --recursive
```

2. Do one of the following:
   
   a. (Recommended) Build a Docker container from the provided `Dockerfile` for single-instance executions.
    
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
   b. Alternatively, use the provided environment configuration script to set up environment without using a docker container:

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
         bash ./tools/env_setup.sh 3 <DPCPP_ROOT> <ONEMKL_ROOT> <ONECCL_ROOT> <AOT>
         export LD_PRELOAD=$(bash ../../../../../tools/get_libstdcpp_lib.sh)
         export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
         source <DPCPP_ROOT>/env/vars.sh
         source <ONEMKL_ROOT>/env/vars.sh
         source <ONECCL_ROOT>/env/vars.sh
         source <MPI_ROOT>/env/vars.sh
     ```
     where <br />
     - `DPCPP_ROOT` is the path to the DPC++ compiler. By default, it is `/opt/intel/oneapi/compiler/latest`.<br />
     - `ONEMKL_ROOT` is the path to oneMKL. By default, it is `/opt/intel/oneapi/mkl/latest`.<br />
     - `ONECCL_ROOT` is the path to oneCCL. By default, it is `/opt/intel/oneapi/ccl/latest`.<br />
     - `MPI_ROOT` is the path to oneAPI MPI library. By default, it is `/opt/intel/oneapi/mpi/latest`.<br />
     - `AOT` is a text string to enable `Ahead-Of-Time` compilation for specific GPU models. Check [tutorial](../../../../../docs/tutorials/technical_details/AOT.md) for details.<br />

3. Set necessary environment variables with the environment variables activation script.

```bash
# Activate environment variables
source ./tools/env_activate.sh
```


## Run Models Generation

| Benchmark mode | FP16 | Weight only quantization INT4 |
|---|:---:|:---:|
|Single instance | ✅ | ✅ |
| Distributed (autotp) |  ✅ | ❎ |

### Run with Bash Script
For all inference cases, can run LLM with the one-click bash script `run_benchmark.sh`:
```
bash run_benchmark.sh
```

#### Single Instance Performance

Note: We only support LLM optimizations with datatype float16, so please don't change datatype to float32 or bfloat16.

```bash
# fp16 benchmark
python -u run_generation.py --benchmark -m ${model} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency
```

**Note**: By default, generations are based on `bs = 1`, input token size = 1024, output toke size = 128, iteration num = 10 and `beam search`, and beam size = 4. For beam size = 1 and other settings, please export env settings, such as: `beam=1`, `input=32`, `output=32`, `iter=5`.

#### Distributed Performance with DeepSpeed

For all distributed inference cases, run LLM with the one-click bash script `run_benchmark_ds.sh`:
```
bash run_benchmark_ds.sh
```

```bash
# distributed env setting
source ${ONECCL_DIR}/build/_install/env/setvars.sh
# fp16 benchmark
mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${out} --device xpu --ipex --dtype float16 --token-latency
```

**Note**: By default, generations are based on `bs = 1`, input token size = 1024, output toke size = 128, iteration num = 10 and `beam search`, and beam size = 4. For beam size = 1 and other settings, please export env settings, such as: `beam=1`, `input=32`, `output=32`, `iter=5`.

## Advanced Usage

### Weight only quantization with low precision checkpoint (Experimental)

Using INT4 weights can further improve performance by reducing memory bandwidth. However, direct per-channel quantization of weights to INT4 may result in poor accuracy. Some algorithms can modify weights through calibration before quantizing weights to minimize accuracy drop. GPTQ is one of such algorithms. You may generate modified weights and quantization info (scales, zero points) for a certain model with a dataset for specified tasks by such algorithms. The results are saved as a `state_dict` in a `.pt` file. We provide the script here to run GPT-J .

#### Single Instance GPT-J Weight only quantization Performance

```bash
# quantization benchmark
#To run quantization performance, you need to firstly get the quantized weight with step (1) and then run the performance benchmark with step (2)
## (1) Get the quantized weight
download link: https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/xpu/gptj_int4_weight_master.pt
export weight_path = path-to-your-weight

## (2) Run quantization performance test
python -u run_generation.py --device xpu --ipex --dtype float16 --input-tokens ${input} --max-new-tokens ${out}  --token-latency --benchmark  --num-beams ${beam}  -m ${model} --woq --woq_checkpoint_path ${weight_path}
```

#### Single Instance GPT-J Weight only quantization INT4 Accuracy

```bash
# we use "lambada_standard" task to check accuracy
LLM_ACC_TEST=1 python -u run_generation.py -m ${model} --ipex --dtype float16 --accuracy-only --acc-tasks ${task} --woq --woq_checkpoint_path ${weight_path}
```

### Single Instance Accuracy

```bash
Accuracy test {TASK_NAME}, choice in this [link](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md), by default we use "lambada_standard"

# one-click bash script
bash run_accuracy.sh

# float16
LLM_ACC_TEST=1 python -u run_generation.py -m ${model} --ipex --dtype float16 --accuracy-only --acc-tasks ${task}
```

### Distributed Accuracy with DeepSpeed

```bash
# Run distributed accuracy with 2 ranks of one node for float16 with ipex
source ${ONECCL_DIR}/build/_install/env/setvars.sh

# one-click bash script
bash run_accuracy_ds.sh

# float16
LLM_ACC_TEST=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py -m ${model} --ipex --dtype float16 --accuracy-only --acc-tasks ${task} 2>&1
```
