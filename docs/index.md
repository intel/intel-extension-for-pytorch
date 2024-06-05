# Intel® Extension for PyTorch\* Large Language Model (LLM) Feature Get Started For Qwen2 models

Intel® Extension for PyTorch\* provides dedicated optimization for running Qwen2 models faster, including technical points like paged attention, ROPE fusion, etc. And a set of data types are supported for various scenarios, including BF16, Weight Only Quantization, etc. 
# 1. Environment Setup

There are several environment setup methodologies provided. You can choose either of them according to your usage scenario. The Docker-based ones are recommended.

## 1.1 [RECOMMENDED] Docker-based environment setup with pre-built wheels

```bash
# Get the Intel® Extension for PyTorch\* source code
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout 2.3-qwen-2
git submodule sync
git submodule update --init --recursive

# Build an image with the provided Dockerfile by installing from Intel® Extension for PyTorch\* prebuilt wheel files
DOCKER_BUILDKIT=1 docker build -f examples/cpu/inference/python/llm/Dockerfile -t ipex-llm:qwen2 .

# Run the container with command below
docker run --rm -it --privileged ipex-llm:qwen2 bash

# When the command prompt shows inside the docker container, enter llm examples directory
cd llm

# Activate environment variables
source ./tools/env_activate.sh
```

## 1.2 Conda-based environment setup with pre-built wheels

```bash
# Get the Intel® Extension for PyTorch\* source code
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout 2.3-qwen-2
git submodule sync
git submodule update --init --recursive

# Create a conda environment (pre-built wheel only available with python=3.10)
conda create -n llm python=3.10 -y
conda activate llm

# Setup the environment with the provided script
# A sample "prompt.json" file for benchmarking is also downloaded
cd examples/cpu/inference/python/llm
bash ./tools/env_setup.sh 7

# Activate environment variables
source ./tools/env_activate.sh
```
<br>

# 2. How To Run Qwen2 with ipex.llm

**ipex.llm provides a single script to facilitate running generation tasks as below:**

```
# if you are using a docker container built from commands above in Sec. 1.1, the placeholder LLM_DIR below is /home/ubuntu/llm
# if you are using a conda env created with commands above in Sec. 1.2, the placeholder LLM_DIR below is intel-extension-for-pytorch/examples/cpu/inference/python/llm
cd <LLM_DIR>
python run.py --help # for more detailed usages
```

| Key args of run.py | Notes |
|---|---|
| model id | "--model-name-or-path" or "-m" to specify the <QWEN2_MODEL_ID_OR_LOCAL_PATH>, it is model id from Huggingface or downloaded local path |
| generation | default: beam search (beam size = 4), "--greedy" for greedy search |
| input tokens | provide fixed sizes for input prompt size, use "--input-tokens" for <INPUT_LENGTH> in [1024, 2048, 4096, 8192, 16384, 32768]; if "--input-tokens" is not used, use "--prompt" to choose other strings as prompt inputs|
| output tokens | default: 32, use "--max-new-tokens" to choose any other size |
| batch size |  default: 1, use "--batch-size" to choose any other size |
| token latency |  enable "--token-latency" to print out the first or next token latency |
| generation iterations |  use "--num-iter" and "--num-warmup" to control the repeated iterations of generation, default: 100-iter/10-warmup |
| streaming mode output | greedy search only (work with "--greedy"), use "--streaming" to enable the streaming generation output |

*Note:* You may need to log in your HuggingFace account to access the model files. Please refer to [HuggingFace login](https://huggingface.co/docs/huggingface_hub/quick-start#login).

## 2.1 Usage of running Qwen2 models

The _\<QWEN2_MODEL_ID_OR_LOCAL_PATH\>_ in the below commands specifies the Qwen2 model you will run, which can be found from [HuggingFace Models](https://huggingface.co/models).

### 2.1.1 Run generation with multiple instances on multiple CPU numa nodes

#### 2.1.1.1 Prepare:

```bash
unset KMP_AFFINITY
```

In the DeepSpeed cases below, we recommend "--shard-model" to shard model weight sizes more even for better memory usage when running with DeepSpeed.

If using "--shard-model", it will save a copy of the shard model weights file in the path of "--output-dir" (default path is "./saved_results" if not provided).
If you have used "--shard-model" and generated such a shard model path (or your model weights files are already well sharded), in further repeated benchmarks, please remove "--shard-model", and replace "-m <QWEN2_MODEL_ID_OR_LOCAL_PATH>" with "-m <shard model path>" to skip the repeated shard steps.

Besides, the standalone shard model function/scripts are also provided in section 2.1.1.4, in case you would like to generate the shard model weights files in advance before running distributed inference.

#### 2.1.1.2 BF16:

- Command:
```bash
deepspeed --bind_cores_to_rank  run.py --benchmark -m <QWEN2_MODEL_ID_OR_LOCAL_PATH> --dtype bfloat16 --ipex  --greedy --input-tokens <INPUT_LENGTH> --autotp --shard-model
```

#### 2.1.1.3 Weight-only quantization (INT8):

By default, for weight-only quantization, we use quantization with [Automatic Mixed Precision](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html) inference ("--quant-with-amp") to get peak performance and fair accuracy.
For weight-only quantization with deepspeed, we quantize the model then run the benchmark. The quantized model won't be saved.

- Command:
```bash
deepspeed --bind_cores_to_rank run.py  --benchmark -m <QWEN2_MODEL_ID_OR_LOCAL_PATH> --ipex --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --greedy --input-tokens <INPUT_LENGTH>  --autotp --shard-model --output-dir "saved_results"
```

#### 2.1.1.4 How to Shard Model weight files for Distributed Inference with DeepSpeed

To save memory usage, we could shard the model weights files under the local path before we launch distributed tests with DeepSpeed.

```
cd ./utils
# general command:
python create_shard_model.py -m <QWEN2_MODEL_ID_OR_LOCAL_PATH>  --save-path ./local_qwen2_model_shard
# After sharding the model, using "-m ./local_qwen2_model_shard" in later tests
```

### 2.1.2 Run generation with single instance on a single numa node
#### 2.1.2.1 BF16:

- Command:
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py --benchmark -m <QWEN2_MODEL_ID_OR_LOCAL_PATH> --dtype bfloat16 --ipex --greedy --input-tokens <INPUT_LENGTH> 
```

#### 2.1.2.2 Weight-only quantization (INT8):

By default, for weight-only quantization, we use quantization with [Automatic Mixed Precision](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html) inference ("--quant-with-amp") to get peak performance and fair accuracy.

- Command:
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list>  python run.py  --benchmark -m <QWEN2_MODEL_ID_OR_LOCAL_PATH> --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --output-dir "saved_results"  --greedy --input-tokens <INPUT_LENGTH>
```

#### 2.1.2.3 Notes:

(1) [_numactl_](https://linux.die.net/man/8/numactl) is used to specify memory and cores of your hardware to get better performance. _\<node N\>_ specifies the [numa](https://en.wikipedia.org/wiki/Non-uniform_memory_access) node id (e.g., 0 to use the memory from the first numa node). _\<physical cores list\>_ specifies phsysical cores which you are using from the _\<node N\>_ numa node. You can use [_lscpu_](https://man7.org/linux/man-pages/man1/lscpu.1.html) command in Linux to check the numa node information.

(2) For all quantization benchmarks, both quantization and inference stages will be triggered by default. For quantization stage, it will auto-generate the quantized model named "best_model.pt" in the "--output-dir" path, and for inference stage, it will launch the inference with the quantized model "best_model.pt".  For inference-only benchmarks (avoid the repeating quantization stage), you can also reuse these quantized models for by adding "--quantized-model-path <output_dir + "best_model.pt">" .

## Miscellaneous Tips
Intel® Extension for PyTorch\* also provides dedicated optimization for many other Large Language Models (LLM), which cover a set of data types that are supported for various scenarios. For more details, please check this [Intel® Extension for PyTorch\* doc](https://github.com/intel/intel-extension-for-pytorch/blob/release/2.3/README.md).