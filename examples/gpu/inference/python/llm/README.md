# Text Generation

Here you can find the inference benchmarking scripts for large language models (LLM) text generation. These scripts:

- Support LLaMA 2, GPT-J, OPT, and Bloom model families
- Include both single instance and distributed (DeepSpeed) use cases
- Cover model generation inference with low precision cases for different models with best performance and accuracy (fp16 AMP and weight only quantization)



## Optimized Models

| MODEL FAMILY | Verified < MODEL ID > (Huggingface hub)| FP16 | Weight only quantization INT4 |
|---|:---:|:---:|:---:|
|LLAMA 2| "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-70b-hf" | ✅ | ❎ |
|GPT-J| "EleutherAI/gpt-j-6b" | ✅ | ✅ |
|OPT|"facebook/opt-6.7b", "facebook/opt-30b"| ✅ | ❎ |
|Bloom|"bigscience/bloom-7b1", "bigscience/bloom"| ✅ | ❎ |


**Note**: The verified models mentioned above (including other models in the same model family, like "codellama/CodeLlama-7b-hf" from LLAMA family) are well-supported with all optimizations like indirect access KV cache and fused ROPE. For other LLM families, we are actively working to implement these optimizations, which will be reflected in the expanded model list above. 

## Supported Platforms

\* PVC(1550/1100): support all the models in the model list above<br /> 


## Environment Setup

Note: The instructions in this section will setup an environment with a latest source build of IPEX on `xpu-main` branch.
If you would like to use stable IPEX release versions, please refer to the instructions in [the release branch](https://github.com/intel/intel-extension-for-pytorch/blob/v2.1.10%2Bxpu/examples/gpu/inference/python/llm/README.md#environment-setup),
in which IPEX is installed via prebuilt wheels using `pip install` rather than source code building.

### [Recommended] Docker-based environment setup with compilation from source



```bash
# Get the Intel® Extension for PyTorch* source code
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout xpu-main
git submodule sync
git submodule update --init --recursive

# Build an image with the provided Dockerfile by compiling Intel® Extension for PyTorch* from source
docker build -f examples/gpu/inference/python/llm/Dockerfile --build-arg GID_RENDER=$(getent group render | sed -E 's,^render:[^:]*:([^:]*):.*$,\1,') --build-arg COMPILE=ON -t ipex-llm:main .

# Run the container with command below
docker run --privileged -it --rm --device /dev/dri:/dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
--ipc=host --net=host --cap-add=ALL -v /lib/modules:/lib/modules --workdir /workspace  \
--volume `pwd`/examples/gpu/inference/python/llm/:/workspace/llm ipex-llm:main /bin/bash


# When the command prompt shows inside the docker container, enter llm examples directory
cd llm

# Activate environment variables
source ./tools/env_activate.sh
```

### Conda-based environment setup with compilation from source

Make sure the driver and Base Toolkit are installed without using a docker container. Refer to [Installation Guide](https://intel.github.io/intel-extension-for-pytorch/#installation?platform=gpu&version=v2.1.10%2Bxpu&os=linux%2Fwsl2&package=source).



```bash

# Get the Intel® Extension for PyTorch* source code
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout xpu-main
git submodule sync
git submodule update --init --recursive

# Make sure you have GCC >= 11 is installed on your system.
# Create a conda environment
conda create -n llm python=3.10 -y
conda activate llm
conda install pkg-config
# Setup the environment with the provided script
cd examples/gpu/inference/python/llm
# If you want to install Intel® Extension for PyTorch\* from source, use the commands below:
bash ./tools/env_setup.sh 3 <DPCPP_ROOT> <ONEMKL_ROOT> <ONECCL_ROOT> <AOT>
export LD_PRELOAD=$(bash ../../../../../tools/get_libstdcpp_lib.sh)
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
source ./tools/env_activate.sh

```

where <br />
- `AOT` is a text string to enable `Ahead-Of-Time` compilation for specific GPU models. Check [tutorial](../../../../../docs/tutorials/technical_details/AOT.md) for details.<br />


 
## Run Models Generation

| Benchmark mode | FP16 | Weight only quantization INT4 |
|---|:---:|:---:|
|Single instance | ✅ | ✅ |
| Distributed (autotp) |  ✅ | ❎ |


Note: During usage, you may need to log in to your Hugging Face account to access model files. Refer to [HuggingFace Login](https://huggingface.co/docs/huggingface_hub/quick-start#login)

### Run with Bash Script

For all inference cases, can run LLM with the one-click bash script `run_benchmark.sh`:
```
bash run_benchmark.sh
```

#### Single Instance Performance

Note: We only support LLM optimizations with datatype float16, so please don't change datatype to float32 or bfloat16.

```bash
# fp16 benchmark
python -u run_generation.py --benchmark -m ${model} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${output} --device xpu --ipex --dtype float16 --token-latency
```

**Note**: By default, generations are based on `bs = 1`, input token size = 1024, output toke size = 128, iteration num = 10 and `beam search`, and beam size = 4. For beam size = 1 and other settings, please export env settings, such as: `beam=1`, `input=32`, `output=32`, `iter=5`.

#### Distributed Performance with DeepSpeed

For all distributed inference cases, run LLM with the one-click bash script `run_benchmark_ds.sh`:
```
bash run_benchmark_ds.sh
```

```bash
# fp16 benchmark
mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py --benchmark -m ${model} --num-beams ${beam} --num-iter ${iter} --batch-size ${bs} --input-tokens ${input} --max-new-tokens ${output} --device xpu --ipex --dtype float16 --token-latency
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
export weight_path=path-to-your-weight

## (2) Run quantization performance test
python -u run_generation.py --device xpu --ipex --dtype float16 --input-tokens ${input} --max-new-tokens ${output}  --token-latency --benchmark  --num-beams ${beam}  -m ${model} --woq --woq_checkpoint_path ${weight_path}
```

#### Single Instance GPT-J Weight only quantization INT4 Accuracy

```bash
# we use "lambada_standard" task to check accuracy
LLM_ACC_TEST=1 python -u run_generation.py --device xpu --ipex --dtype float16 -m ${model} --accuracy-only --acc-tasks ${task} --woq --woq_checkpoint_path ${weight_path}
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
# one-click bash script
bash run_accuracy_ds.sh

# float16
LLM_ACC_TEST=1 mpirun -np 2 --prepend-rank python -u run_generation_with_deepspeed.py -m ${model} --ipex --dtype float16 --accuracy-only --acc-tasks ${task} 2>&1
```
