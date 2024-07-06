# Text Generation

Here you can find the inference benchmarking scripts for large language models (LLM) text generation. These scripts:

- Support Llama 2, GPT-J, Qwen, OPT, Bloom model families and some other Chinese models such as ChatGLMv3-6B and Baichuan2-13B. 
- Include both single instance and distributed (DeepSpeed) use cases for FP16 optimization.
- Cover model generation inference with low precision cases for different models with best performance and accuracy (fp16 AMP and weight only quantization)


## Optimized Models

Currently, only support Transformers 4.31.0. Support for newer versions of Transformers and more models will be available in the future.

| MODEL FAMILY | Verified < MODEL ID > (Huggingface hub)| FP16 | Weight only quantization INT4 | Optimized on Intel® Data Center GPU Max Series (1550/1100) | Optimized on Intel® Arc™ A-Series Graphics (A770) |
|---|:---:|:---:|:---:|:---:|:---:|
|Llama 2| "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-70b-hf" | ✅ | ✅|✅ | ✅|
|GPT-J| "EleutherAI/gpt-j-6b" | ✅ | ✅ |✅ | ✅|
|Qwen|"Qwen/Qwen-7B"| ✅ | ✅ |✅ | ✅|
|OPT|"facebook/opt-6.7b", "facebook/opt-30b"| ✅ | ❎ |✅ | ❎ |
|Bloom|"bigscience/bloom-7b1", "bigscience/bloom"| ✅ | ❎ |✅ | ❎ |
|ChatGLM3-6B|"THUDM/chatglm3-6b"| ✅ | ❎ |✅ | ❎ |
|Baichuan2-13B|"baichuan-inc/Baichuan2-13B-Chat"| ✅ | ❎ |✅ | ❎ |


**Note**: The verified models mentioned above (including other models in the same model family, like "codellama/CodeLlama-7b-hf" from LLAMA family) are well-supported with all optimizations like indirect access KV cache and fused ROPE. For other LLM families, we are actively working to implement these optimizations, which will be reflected in the expanded model list above. 

## Supported Platforms

\* Intel® Data Center GPU Max Series (1550/1100) and Optimized on Intel® Arc™ A-Series Graphics (A770) : support all the models in the model list above.<br />


## Environment Setup

Note: The instructions in this section will setup an environment with a latest source build of IPEX on `xpu-main` branch.
If you would like to use stable IPEX release versions, please refer to the instructions in [the release branch](https://github.com/intel/intel-extension-for-pytorch/blob/v2.1.30%2Bxpu/examples/gpu/inference/python/llm/README.md#environment-setup),
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

Make sure the driver and Base Toolkit are installed without using a docker container. Refer to [Installation Guide](https://intel.github.io/intel-extension-for-pytorch/#installation?platform=gpu&version=v2.1.30%2Bxpu&os=linux%2Fwsl2&package=source).



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
# Setup the environment with the provided script
cd examples/gpu/inference/python/llm
# If you want to install Intel® Extension for PyTorch\* from source, use the commands below:
bash ./tools/env_setup.sh 3 <DPCPP_ROOT> <ONEMKL_ROOT> <ONECCL_ROOT> <MPI_ROOT> <AOT>
conda deactivate
conda activate llm
source ./tools/env_activate.sh

```

where <br />
- `AOT` is a text string to enable `Ahead-Of-Time` compilation for specific GPU models. Check [tutorial](../../../../../docs/tutorials/technical_details/AOT.md) for details.<br />


 
## Run Models

| Benchmark mode | FP16 | Weight only quantization INT4 |
|---|:---:|:---:|
|Single instance | ✅ | ✅ |
| Distributed (autotp) |  ✅ | ❎ |


Note: During the execution, you may need to log in your Hugging Face account to access model files. Refer to [HuggingFace Login](https://huggingface.co/docs/huggingface_hub/quick-start#login)

### Run with Bash Script

Run all inference cases with the one-click bash script `run_benchmark.sh`:
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


### Weight Only Quantization with low precision checkpoint (Prototype)

Using INT4 weights can further improve performance by reducing memory bandwidth. However, direct per-channel quantization of weights to INT4 may result in poor accuracy. Some algorithms can modify weights through calibration before quantizing weights to minimize accuracy drop. You may generate modified weights and quantization info (scales, zero points) for a Llama 2/GPT-J/Qwen models with a dataset for specified tasks by such algorithms. We recommend intel extension for transformer to quantize the LLM model.

Check [WOQ INT4](../../../../../docs/tutorials/llm/int4_weight_only_quantization.md) for more details.

#### Install intel-extension-for-transformers and intel-neural-compressor 

```
pip install neural-compressor
pip install intel-extension-for-transformers
```

#### Install other required packages


```
pip install tiktoken einops transformers_stream_generator
```

#### Run the weight only quantization and inference

```python
bash run_benchmark_woq.sh
```

>Note:
> * Saving quantized model should be executed before the optimize_transformers function is called.
> * The optimize_transformers function is designed to optimize transformer-based models within frontend Python modules, with a particular focus on Large Language Models (LLMs). It provides optimizations for both model-wise and content-generation-wise. The detail of `optimize_transformers`, please refer to [Transformers Optimization Frontend API](../../../../../docs/tutorials/llm/llm_optimize_transformers.md).


