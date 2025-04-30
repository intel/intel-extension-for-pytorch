# Intel® Extension for PyTorch\* Large Language Model (LLM) Feature Get Started For Qwen3

Intel® Extension for PyTorch\* provides dedicated optimization for running Qwen3 models faster, including technical points like paged attention, ROPE fusion, MoE optimizations, etc.

<br>

# 1. Model Quantization with Intel® AutoRound

[intel/auto-round](https://github.com/intel/auto-round) has provided the recipes for Qwen3 series models.

- Weight only quantization INT4 for [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)

We can complete the quantization process following
[the example command](https://github.com/intel/auto-round/blob/main/docs/Qwen3-14B-sym-recipe.md#generate-the-model)
with such changes on the arguments:

a. Remove `--format 'auto_round'` and `--device 0`

b. Add `--format 'auto_gptq'` and `--asym`

- Weight only quantization INT8 for [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)

The quantization process can be completed with the following command list.

```bash
git clone -b enable_llama4_int8_baseline https://github.com/intel/auto-round.git
cd auto-round
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .[cpu]
sh run_qwen3.sh <INPUT_BF16_MODEL_PATH> <OUTPUT_INT8_MODEL_PATH>
```

<br>

# 2. Environment Setup for Inference

There are several environment setup methodologies provided.
You can choose either of them according to your usage scenario.
The Docker-based one is recommended.

## 2.1 [RECOMMENDED] Docker-based environment setup

```bash
# Get the Intel® Extension for PyTorch\* source code
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout 2.8-pre-qwen3
git submodule sync
git submodule update --init --recursive

# Build an image with the provided Dockerfile
# To have a custom ssh server port for multi-nodes run, please add --build-arg PORT_SSH=<CUSTOM_PORT> ex: 2345, otherwise use the default 22 SSH port
docker build -f examples/cpu/llm/Dockerfile --build-arg PORT_SSH=2345 -t ipex-llm:qwen3 .

# Run the container with command below
docker run --rm -it --privileged -v /dev/shm:/dev/shm ipex-llm:qwen3 bash

# When the command prompt shows inside the docker container, enter llm examples directory
cd llm

# Activate environment variables
source ./tools/env_activate.sh
```

## 2.2 Conda-based environment setup

```bash
# Get the Intel® Extension for PyTorch\* source code
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout 2.8-pre-qwen3
git submodule sync
git submodule update --init --recursive

# Create a conda environment (pre-built wheel only available with python=3.10)
conda create -n llm python=3.10 -y
conda activate llm

# Setup the environment with the provided script
cd examples/cpu/llm
bash ./tools/env_setup.sh

# Activate environment variables
source ./tools/env_activate.sh
```

<br>

# 3. Run Qwen3 with ipex.llm

## 3.1 Running quantized Qwen/Qwen3-14B

```bash
export model=<QUANT_QWEN3_14B_MODEL_PATH>
# Create a soft link for the used prompt.json file
cd distributed
ln -s ../prompt.json
cd ..
```

Run the model with command

```bash
deepspeed --bind_cores_to_rank --num_accelerators 3 --bind_core_list 0-41,43-84,86-127 run.py --benchmark -m ${model} --input-tokens 1024 --max-new-tokens 1024 --ipex-weight-only-quantization --weight-dtype INT4 --quant-with-amp --ipex --autotp --greedy --token-latency
```

## 3.2 Running quantized Qwen/Qwen3-30B-A3B

```bash
export model=<QUANT_QWEN3_30B_MODEL_PATH>
# Change the soft link for the prompt file for MoE model
cd distributed
rm -f prompt.json
ln -s ../prompt-qen3-moe.json prompt.json
cd ..
```

Run the model with command

```bash
deepspeed --bind_cores_to_rank --num_accelerators 3 --bind_core_list 0-41,43-84,86-127 run.py --benchmark -m ${model} --input-tokens 1024 --max-new-tokens 1024 --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --ipex --autotp --greedy --token-latency
```

*Note:* In the example command tensor parallel 3 (TP3) is applied. Please change it per your server configuration.
But there is a limit that the TP rank number cannot surpass the number of KV_HEAD of the models.
Specifically, NUM_KV_HEAD of Qwen3-30B MoE model is 4, so TP rank number cannot be larger than 4.

More explanations about the arguments:

| Key args of run.py | Notes |
|---|---|
| model id | "--model-name-or-path" or "-m" to specify the <QWEN3_MODEL_PATH> |
| generation | default: beam search (beam size = 4), "--greedy" for greedy search |
| input tokens or prompt | provide fixed sizes for input prompt size, use "--input-tokens" for <INPUT_LENGTH> in [1024, 2048, 4096, 8192, 32768, 130944]; if "--input-tokens" is not used, use "--prompt" to choose other strings as inputs|
| output tokens | default: 32, use "--max-new-tokens" to choose any other size |
| batch size |  default: 1, use "--batch-size" to choose any other size |
| token latency |  enable "--token-latency" to print out the first and next token latency |
| generation iterations | use "--num-iter" and "--num-warmup" to control the repeated iterations of generation, default: 100-iter/10-warmup |
| streaming mode output | greedy search only (work with "--greedy"), use "--streaming" to enable the streaming generation output |

<br>

# 4. Miscellaneous Tips

Intel® Extension for PyTorch\* also provides dedicated optimization for many other Large Language Models (LLM), which cover a set of data types that are supported for various scenarios.
For more details, please check [the Intel® Extension for PyTorch\* regular release doc](https://github.com/intel/intel-extension-for-pytorch/blob/v2.7.0%2Bcpu/examples/cpu/llm/README.md).