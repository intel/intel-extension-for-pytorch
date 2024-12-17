# Intel® Extension for PyTorch\* Large Language Model (LLM) Feature Get Started For Falcon 3 models

Intel® Extension for PyTorch\* provides dedicated optimization for running Falcon 3 models faster, including technical points like paged attention, ROPE fusion, etc. And a set of data types are supported for various scenarios, including Weight Only Quantization INT4 (prototype), etc.

# 1. Environment Setup

There are several environment setup methodologies provided. You can choose either of them according to your usage scenario. The Docker-based ones are recommended.

## 1.1 [RECOMMENDED] Docker-based environment setup with pre-built wheels

```bash
# Get the Intel® Extension for PyTorch\* source code
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout 2.6-falcon-3
git submodule sync
git submodule update --init --recursive

# Build an image with the provided Dockerfile by compiling Intel® Extension for PyTorch\* from source
# To have a custom ssh server port for multi-nodes run, please add --build-arg PORT_SSH=<CUSTOM_PORT> ex: 2345, otherwise use the default 22 SSH port
docker build -f examples/cpu/llm/Dockerfile --build-arg PORT_SSH=2345 -t ipex-llm:2.6.0-preview .

# Run the container with command below
docker run --rm -it --privileged -v /dev/shm:/dev/shm ipex-llm:2.6.0-preview bash

# When the command prompt shows inside the docker container, enter llm examples directory
cd llm

# Activate environment variables for infernece
source ./tools/env_activate.sh inference
```

## 1.2 Conda-based environment setup with pre-built wheels

```bash
# Get the Intel® Extension for PyTorch\* source code
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout 2.6-falcon-3
git submodule sync
git submodule update --init --recursive

# Create a conda environment (pre-built wheel only available with python=3.10)
conda create -n llm python=3.10 -y
conda activate llm

# Setup the environment with the provided script
cd examples/cpu/llm
bash ./tools/env_setup.sh 7

# Activate environment variables for infernece
source ./tools/env_activate.sh inference
```
<br>

# 2. How To Run Falcon 3 with ipex.llm

**ipex.llm provides a single script to facilitate running generation tasks as below:**

```
# if you are using a docker container built from commands above in Sec. 1.1, the placeholder LLM_DIR below is ~/llm
# if you are using a conda env created with commands above in Sec. 1.2, the placeholder LLM_DIR below is intel-extension-for-pytorch/examples/cpu/inference/python/llm
cd <LLM_DIR>
python run.py --help # for more detailed usages
```

| Key args of run.py | Notes |
|---|---|
| model id | "--model-name-or-path" or "-m" to specify the <FALCON3_MODEL_ID_OR_LOCAL_PATH>, it is model id from Huggingface or downloaded local path |
| generation | default: beam search (beam size = 4), "--greedy" for greedy search |
| input tokens or prompt | provide fixed sizes for input prompt size, use "--input-tokens" for <INPUT_LENGTH> in [128, 1024, 2048]; if "--input-tokens" is not used, use "--prompt" to choose other strings as inputs|
| output tokens | default: 32, use "--max-new-tokens" to choose any other size for <OUTPUT_LENGTH> |
| batch size |  default: 1, use "--batch-size" to choose any other size |
| token latency |  enable "--token-latency" to print out the first or next token latency |
| generation iterations |  use "--num-iter" and "--num-warmup" to control the repeated iterations of generation, default: 100-iter/10-warmup |
| streaming mode output | greedy search only (work with "--greedy"), use "--streaming" to enable the streaming generation output |

*Note:* You may need to log in your HuggingFace account to access the model files. Please refer to [HuggingFace login](https://huggingface.co/docs/huggingface_hub/quick-start#login).

## 2.1 Usage of running Falcon 3 7B models

The _\<FALCON3_MODEL_ID_OR_LOCAL_PATH\>_ in the below commands specifies the Falcon 3 model you will run, which can be found from [HuggingFace Models](https://huggingface.co/models).

### 2.1.1 Run text generation with Falcon 3 7B model using Weight-only quantization (INT4) per CPU numa node

#### 2.1.1.1 Quantization:
You can use the auto-round tool to generate the INT4 WOQ model by yourself with the following steps:
- Environment installation:
```bash
pip install auto-round
```
- Command (quantize):
```bash
auto-round  \
--model <FALCON3_MODEL_ID_OR_LOCAL_PATH> \
--nsamples 512 \
--seqlen 2048 \
--iters 1000 \
--model_dtype "float16" \
--format 'auto_awq' \
--output_dir <INT4_MODEL_SAVE_PATH>
```

Optionally, you can also checkout the model card to download the existing quantized INT4 WOQ model of Falcon 3 7B [here](https://huggingface.co/OPEA/falcon-three-7b-int4-sym-inc) in the HF hub.
```bash
git clone https://huggingface.co/OPEA/falcon-three-7b-int4-sym-inc
cd falcon-three-7b-int4-sym-inc
git checkout e9aa317 # autoawq format
cd ..
export <INT4_MODEL_SAVE_PATH> = ./falcon-three-7b-int4-sym-inc
```
#### 2.1.1.2 Benchmark:
- Command (benchmark):
```bash
cd <LLM_DIR>
# Text generation inference
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list>  python run.py  --benchmark -m <FALCON3_MODEL_ID_OR_LOCAL_PATH> --ipex-weight-only-quantization --quant-with-amp --low-precision-checkpoint <INT4_MODEL_SAVE_PATH> --output-dir "saved_results"  --greedy --input-tokens <INPUT_LENGTH> --max-new-tokens <OUTPUT_LENGTH> 

# Note that to get the best throughput on multiple CPU numa nodes, we could further tune how many cores per instance and batch sizes (according to the latency requirement) to run multiple instances at the same time. 
```

#### 2.1.1.3 Notes:

(1) [_numactl_](https://linux.die.net/man/8/numactl) is used to specify memory and cores of your hardware to get better performance. _\<node N\>_ specifies the [numa](https://en.wikipedia.org/wiki/Non-uniform_memory_access) node id (e.g., 0 to use the memory from the first numa node). _\<physical cores list\>_ specifies phsysical cores which you are using from the _\<node N\>_ numa node. You can use [_lscpu_](https://man7.org/linux/man-pages/man1/lscpu.1.html) command in Linux to check the numa node information.

(2) For all quantization benchmarks, both quantization and inference stages will be triggered by default. For quantization stage, it will auto-generate the quantized model named "best_model.pt" in the "--output-dir" path, and for inference stage, it will launch the inference with the quantized model "best_model.pt".  For inference-only benchmarks (avoid the repeating quantization stage), you can also reuse these quantized models for by adding "--quantized-model-path <output_dir + "best_model.pt">" .


## Miscellaneous Tips
Intel® Extension for PyTorch\* also provides dedicated optimization for many other Large Language Models (LLM), which cover a set of data types that are supported for various scenarios. For more details, please check this [Intel® Extension for PyTorch\* doc](https://github.com/intel/intel-extension-for-pytorch/blob/main/README.md).
