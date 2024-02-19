# LLM Optimizations Overview

We have achieved various optimizations for Large Language Models (LLMs) in IPEX, and provided the inference benchmarking scripts for LLM performance and accuracy tests.<br/>
The scripts include both single instance and distributed (DeepSpeed) use cases.<br/>
The scripts cover model generation inference with low precions cases for different models with best perf and accuracy (bf16 AMP，static quantization and weight only quantization).<br/>

# Optimized Model List

| MODEL FAMILY | Verified <MODEL ID> (Huggingface hub)| FP32/BF16 | Weight only quantzation INT8 | Weight only quantization INT4| Static quantization INT8 |
|---|:---:|:---:|:---:|:---:|:---:|
|LLAMA| "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-70b-hf" | ✅ | ✅ | ✅ | ✅ | 
|GPT-J| "EleutherAI/gpt-j-6b" | ✅ | ✅ | ✅ | ✅ | 
|GPT-NEOX| "EleutherAI/gpt-neox-20b", "databricks/dolly-v2-12b" | ✅ | ✅ | ✅ | ✅ | 
|FALCON|"tiiuae/falcon-40b" | ✅ | ✅ |  ✅ | ✅ | 
|OPT|"facebook/opt-30b", "facebook/opt-1.3b"| ✅ | ✅ |  ✅ | ✅ | 
|Bloom|"bigscience/bloom", "bigscience/bloom-1b7"| ✅ | ✅ |  ✅ | ✅ |
|CodeGen|"Salesforce/codegen-2B-multi"| ✅ | ✅ |  ✅ | ✅ |
|Baichuan|"baichuan-inc/Baichuan2-13B-Chat", "baichuan-inc/Baichuan2-7B-Chat", "baichuan-inc/Baichuan-13B-Chat"| ✅ | ✅ |  ✅ | ✅ |
|ChatGLM|"THUDM/chatglm3-6b", "THUDM/chatglm2-6b"| ✅ | ✅ |  ✅ | ✅ |
|GPTBigCode|"bigcode/starcoder"| ✅ | ✅ |  ✅ | ✅ |
|T5|"google/flan-t5-xl"| ✅ | ✅ |  ✅ | ✅ |
|Mistral|"mistralai/Mistral-7B-v0.1"| ✅ | ✅ |  ✅ | ✅ |
|MPT|"mosaicml/mpt-7b"| ✅ | ✅ |  ✅ | ✅ |
|Mixtral|"mistralai/Mixtral-8x7B-v0.1"| ✅ | ✅ |  ✅ | ✅ |
|StableLM|"stabilityai/stablelm-2-1_6b"| ✅ | ✅ |  ✅ | ✅ |
|Qwen|"Qwen/Qwen-7B-Chat"| ✅ | ✅ |  ✅ | ✅ |

*Note*: The above verified models (including other models in the same model family, like "codellama/CodeLlama-7b-hf" from LLAMA family) are well supported with all optimizations like indirect access KV cache, fused ROPE, and prepacked TPP Linear (fp32/bf16). For other LLM model families, we are working in progress to cover those optimizations, which will expand the model list above.

# Environment Setup

There are several environment setup methodologies provided. You can choose either of them according to your usage scenario. The Docker-based ones are recommended.

## [Not Applicable] Docker-based environment setup with pre-built wheels

```bash
# Get the Intel® Extension for PyTorch\* source code
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git submodule sync
git submodule update --init --recursive

# Build an image with the provided Dockerfile by installing from Intel® Extension for PyTorch\* prebuilt wheel files
DOCKER_BUILDKIT=1 docker build -f examples/cpu/inference/python/llm/Dockerfile -t ipex-llm:main .

# Run the container with command below
docker run --rm -it --privileged ipex-llm:main bash

# When the command prompt shows inside the docker container, enter llm examples directory
cd llm

# Activate environment variables
source ./tools/env_activate.sh
```

## [Recommended] Docker-based environment setup with compilation from source

```bash
# Get the Intel® Extension for PyTorch\* source code
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git submodule sync
git submodule update --init --recursive

# Build an image with the provided Dockerfile by compiling Intel® Extension for PyTorch\* from source
DOCKER_BUILDKIT=1 docker build -f examples/cpu/inference/python/llm/Dockerfile --build-arg COMPILE=ON -t ipex-llm:main .

# Run the container with command below
docker run --rm -it --privileged ipex-llm:main bash

# When the command prompt shows inside the docker container, enter llm examples directory
cd llm

# Activate environment variables
source ./tools/env_activate.sh
```

## [Not Applicable] Conda-based environment setup with pre-built wheels

```bash
# Get the Intel® Extension for PyTorch\* source code
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git submodule sync
git submodule update --init --recursive

# GCC 12.3 is required. Installation can be taken care of by the environment configuration script.
# Create a conda environment
conda create -n llm python=3.10 -y
conda activate llm

# Setup the environment with the provided script
# A sample "prompt.json" file for benchmarking is also downloaded
cd examples/cpu/inference/python/llm
bash ./tools/env_setup.sh 7

# Activate environment variables
source ./tools/env_activate.sh
```

## Conda-based environment setup with compilation from source

```bash
# Get the Intel® Extension for PyTorch\* source code
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git submodule sync
git submodule update --init --recursive

# GCC 12.3 is required. Installation can be taken care of by the environment configuration script.
# Create a conda environment
conda create -n llm python=3.10 -y
conda activate llm

# Setup the environment with the provided script
# A sample "prompt.json" file for benchmarking is also downloaded
cd examples/cpu/inference/python/llm
bash ./tools/env_setup.sh

# Activate environment variables
source ./tools/env_activate.sh
```

# Run Models Generations

| Benchmark mode | FP32/BF16 | Weight only quantzation INT8 | Weight only quantization INT4 | Static quantization INT8 |
|---|:---:|:---:|:---:|:---:|
|Single instance | ✅ | ✅ | ✅ | ✅ |
| Distributed (autotp) |  ✅ | ✅ | ❎ | ❎ |

**We provide the one-click Python script "run.py" for all generation inference cases.**

```
python run.py --help # for more detailed usages
```
| Key args of run.py | Notes |
|---|:---:|
| generation | default: beam search (beam size = 4), "--greedy" for greedy search |
| input tokens | default: 32, provide fixed sizes for input prompt size, use "--input-tokens" for [32, 64, 128, 256, 512, 1024, 2016, 2017, 2048, 4096, 8192]; if "--input-tokens" is not used, use "--prompt" to choose other strings as inputs|
| output tokens | default: 32, use "--max-new-tokens" to choose any other size |
| batch size |  default: 1, use "--batch-size" to choose any other size |
| token latency |  enable "--token-latency" to print out the first or next token latency |
| generation iterations |  use "--num-iter" and "--num-warmup" to control the repeated iterations of generation, default: 100-iter/10-warmup |

*Note:* You may need to log in your HuggingFace account to access the model files. Please refer to [HuggingFace login](https://huggingface.co/docs/huggingface_hub/quick-start#login).

## Quick start usage example commands for LLaMA2-7b
### Benchmark generation performance
```bash
# The following "OMP_NUM_THREADS" and "numactl" settings are based on the assumption that
# the target server has 56 physical cores per numa socket, and we benchmark with 1 socket.
# Please adjust the settings per your hardware.

# Running FP32 model
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype float32

# Running FP32 model with IPEX
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype float32 --ipex 

# Running BF16 model with IPEX
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex 

# INT8 static quantization
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-hf --ipex-smooth-quant --qconfig-summary-file <path to "llama-2-7b_qconfig.json"> --output-dir "saved_results"

# INT8 weight-only quantization
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-hf --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --output-dir "saved_results" 

# INT4 weight-only quantization
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-hf --ipex-weight-only-quantization --weight-dtype INT4  --gptq --quant-with-amp --output-dir "saved_results" 

# Distributed inference in BF16
deepspeed --bind_cores_to_rank  run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex  --autotp --shard-model

# Distributed inference in weight-only quantization INT8
deepspeed --bind_cores_to_rank  run.py --benchmark -m meta-llama/Llama-2-7b-hf --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp  --autotp --shard-model --output-dir "saved_results"
```
### Accuracy test
For the quantized models used in accuracy tests below, we can reuse the model files that are named "best_model.pt" in the "--output-dir" path ([generated during inference performance tests above](#generation_sq)).

Check [Advanced Usage](#accuracy-test) for details.

```bash
# The following "OMP_NUM_THREADS" and "numactl" settings are based on the assumption that
# the target server has 56 physical cores per numa socket, and we benchmark with 1 socket.
# Please adjust the settings per your hardware.

# run_accuracy.py script is inside single_instance directory.
cd single_instance

# Running FP32 model
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py  -m meta-llama/Llama-2-7b-hf --dtype float32 --ipex --tasks lambada_openai

# Running BF16 model
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py  -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex --tasks lambada_openai

# Quantization
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py -m meta-llama/Llama-2-7b-hf --quantized-model-path "./saved_results/best_model.pt" --dtype int8  --tasks lambada_openai


# run_accuracy_with_deepspeed.py script is inside distributed directory.
cd distributed
unset KMP_AFFINITY

# Distributed inference in FP32
deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model  meta-llama/Llama-2-7b-hf --dtype float32 --ipex --tasks lambada_openai 

# Distributed inference in BF16
deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model  meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex --tasks lambada_openai 

# Distributed inference with Weight-Only Quantization
deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model  meta-llama/Llama-2-7b-hf --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --tasks lambada_openai  
```

## General usage example commands

### Model specifications
There are some model-specific requirements to be aware of, as follows:

- For ChatGLM models, the default torch_dtype is float16 in config.json. We need to replace the "float16" with "float32" in config.json.
- For MPT models from the remote hub, we need to modify the config.json to use the modeling_mpt.py in transformers. Therefore, in the following scripts, we need to pass an extra configuration file like "--config-file=model_config/mosaicml_mpt-7b_config.json".
- For GPT-NEOX (model id "EleutherAI/gpt-neox-20b") quantizations path, do not use "--quant-with-amp" flag due to accuracy loss.
- For Falcon models from remote hub, we need to modify the config.json to use the modeling_falcon.py in transformers. Therefore, in the following scripts, we need to pass an extra configuration file like "--config-file=model_config/tiiuae_falcon-40b_config.json". This is optional for FP32/BF16 but needed for quantizations.
### Single Instance inference

#### FP32:
- Command:
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py --benchmark -m <MODEL_ID> --dtype float32 --ipex
```
- An example of llama2 7b model:
```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype float32 --ipex
```

#### BF16:
- Command:
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex
```
- An example of llama2 7b model:
```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex
```

#### Static quantization (int8):
We use the SmoothQuant algorithm to get good accuracy of static quantization, which is a popular method for LLM models. Besides, by default, we enable quantization mixed fp32 inference (non-quantized OPs run with fp32 dtype). To get better performance, please add "--quant-with-amp" to enable quantization with [Automatic Mixed Precision](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html) inference (non-quantized OPs run with bf16 dtype, which may affect the accuracy).
- Command:
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py  --benchmark -m <MODEL_ID> --ipex-smooth-quant --qconfig-summary-file <path to the qconfig of the model_id> --output-dir "saved_results" 

```
- An example of llama2 7b model:
```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-hf --ipex-smooth-quant --qconfig-summary-file <path to "llama-2-7b_qconfig.json"> --output-dir "saved_results"
```

We provide the following qconfig summary files with good quality (calibration on "NeelNanda/pile-10k" dataset and evaluate accuracy on "lambada_openai" dataset):
| Model ID | Download links |
|---|:---:|
| meta-llama/Llama-2-7b-hf | link |
| meta-llama/Llama-2-13b-hf | link |
| meta-llama/Llama-2-70b-hf | link |
| EleutherAI/gpt-j-6b | link |
| tiiuae/falcon-40b | link |
| facebook/opt-30b | link |
| facebook/opt-1.3b | link |
| baichuan-inc/Baichuan2-7B-Chat | link |
| baichuan-inc/Baichuan2-13B-Chat | link |
| THUDM/chatglm2-6b | link |

If you would like to generate qconfig summary files (due to changes on model variants or calibration dataset), we provide the [autotune API](../../../../../docs/tutorials/features/sq_recipe_tuning_api.md) and its [tuning examples](llm_sq_recipes.md), which allows an automatic global smoothquant tuning, and automatic layer-by-layer tuning provided by Intel® Neural Compressor for the best accuracy.

#### Weight-only quantization:
By default, for weight-only quantization, we use quantization with [Automatic Mixed Precision](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html) inference ("--quant-with-amp") to get peak performance and fair accuracy.
- Command (Int8):
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list>  python run.py  --benchmark -m <MODEL_ID> --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --output-dir "saved_results" 
```
The command above works for most models we listed. However, to get better accuracy for the following models, some changes to the command are needed.
| Model ID | Changes to command |
| - | - |
| bigcode/starcoder | Add "`--group-size 128`" |
| Baichuan-inc/Baichuan-13B-Chat | Remove "`--quant-with-amp`" |
| Baichuan-inc/Baichuan2-13B-Chat | Add "`--group-size 64`" |
| bigscience/bloom-1b7 | Remove "`--quant-with-amp`"; add "`--group-size 128`" |
| EleutherAI/gpt-neox-20b | Remove "`--quant-with-amp`"; add "`--group-size 256`" |
| facebook/opt-30b | Remove "`--quant-with-amp`" |
| databricks/dolly-v2-12b | Remove "`--quant-with-amp`"; add "`--lowp-mode FP32`" |


- Command (Int4):
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py  --benchmark -m <MODEL_ID> --ipex-weight-only-quantization --weight-dtype INT4 --gptq --quant-with-amp --output-dir "saved_results" 
```
- An Int8 example of llama2 7b model:
```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-hf --ipex-weight-only-quantization  --weight-dtype INT8 --quant-with-amp --output-dir "saved_results" 
```
- An Int4 example of llama2 7b model:
```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-hf --ipex-weight-only-quantization --weight-dtype INT4 --gptq --quant-with-amp --output-dir "saved_results" 
```

#### Notes:

(1) [_numactl_](https://linux.die.net/man/8/numactl) is used to specify memory and cores of your hardware to get better performance. _\<node N\>_ specifies the [numa](https://en.wikipedia.org/wiki/Non-uniform_memory_access) node id (e.g., 0 to use the memory from the first numa node). _\<physical cores list\>_ specifies phsysical cores which you are using from the _\<node N\>_ numa node (e.g., 0-56 from the first numa node). You can use [_lscpu_](https://man7.org/linux/man-pages/man1/lscpu.1.html) command in Linux to check the numa node information.

(2) The _\<MODEL_ID\>_ (e.g., "meta-llama/Llama-2-13b-hf") specifies the model you will run. we provide some _Verified \<MODEL ID\>_ in the [Optimized Model List](#optimized-model-list). You can also try other models from [HuggingFace Models](https://huggingface.co/models).

(3) <a name="generation_sq">for all quantization benchmarks</a>, both quantization and inference stages will be triggered by default. For quantization stage, it will auto-generate the quantized model named "best_model.pt" in the "--output-dir" path, and for inference stage, it will launch the inference with the quantized model "best_model.pt".  For inference-only benchmarks (avoid the repeating quantization stage), you can also reuse these quantized models for by adding "--quantized-model-path <output_dir + "best_model.pt">" . 


### Distributed inference with DeepSpeed (autoTP)

#### Prepare:

```bash
unset KMP_AFFINITY
```

In the DeepSpeed cases below, we recommend "--shard-model" to shard model weight sizes more even for better memory usage when running with DeepSpeed.

If using "--shard-model", it will save a copy of the shard model weights file in the path of "--output-dir" (default path is "./saved_results" if not provided).
If you have used "--shard-model" and generated such a shard model path (or your model weights files are already well sharded), in further repeated benchmarks, please remove "--shard-model", and replace "-m <MODEL_ID>" with "-m <shard model path>" to skip the repeated shard steps.

Besides, the standalone shard model function/scripts are also provided in the [Advanced Usage](#how-to-shard-model-for-distributed-tests-with-deepspeed-autotp) section, in case you would like to generate the shard model weights files in advance before running distributed inference.

#### FP32:
- Command:
```bash
deepspeed --bind_cores_to_rank  run.py --benchmark -m <MODEL_ID> --dtype float32 --ipex --autotp --shard-model
```
- An example of llama2 7b model:
```bash
deepspeed --bind_cores_to_rank  run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype float32 --ipex --autotp --shard-model
```

#### BF16:
- Command:
```bash
deepspeed --bind_cores_to_rank  run.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex  --autotp --shard-model
```
- An example of llama2 7b model:
```bash
deepspeed --bind_cores_to_rank  run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex  --autotp --shard-model
```

#### Weight-only quantization:
By default, for weight-only quantization, we use quantization with [Automatic Mixed Precision](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html) inference ("--quant-with-amp") to get peak performance and fair accuracy.
For weight-only quantization with deepspeed, we quantize the model then run the benchmark. The quantized model won't be saved.
- Command:
```bash
deepspeed --bind_cores_to_rank run.py  --benchmark -m <MODEL_ID> --ipex --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp  --autotp --shard-model --output-dir "saved_results"
```
- An example of llama2 7b model:
```bash
deepspeed --bind_cores_to_rank  run.py --benchmark -m meta-llama/Llama-2-7b-hf --ipex --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --autotp --shard-model --output-dir "saved_results"
```

## Instructions for Running LLM with Intel® Xeon® CPU Max Series

Intel® Xeon® CPU Max Series are equipped with high bandwidth memory (HBM), which further accelerates LLM inference. For the common case that HBM and DDR are both installed in a Xeon® CPU Max Series server, the memory mode can be configured to Flat Mode or Cache Mode. Details about memory modes can be found at Section 3.1 in [the Xeon® CPU Max Series Configuration Guide](https://cdrdv2-public.intel.com/769060/354227-intel-xeon-cpu-max-series-configuration-and-tuning-guide.pdf).

### Single Instance Inference with Xeon® CPU Max Series

#### Cache Mode HBM

In cache mode, only DDR address space is visible to software and HBM functions as a transparent memory-side cache for DDR. Therefore the usage is the same with [the common usage](#single-instance-inference).

#### Flat Mode HBM

In flat mode, HBM and DDR are exposed to software as separate address spaces in this mode. Therefore we need to check the `HBM_NODE_INDEX` of interest with commands like `lscpu`, then the LLM inference invoking command would be like:

- Command:
```bash
OMP_NUM_THREADS=<HBM node cores num> numactl -m <HBM_NODE_INDEX> -C <HBM cores list> python run.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex
```
- An example of llama2 7b model with HBM numa node index being 2:
```bash
OMP_NUM_THREADS=56 numactl -m 2 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex
```

*Note:* For some very large models we may get an "OOM Error" due to HBM capacity limitations. In this case we can change `-m` argument for `numactl` to `-p` in the above command to enable the model inference with the larger DDR memory.

- Command:
```bash
OMP_NUM_THREADS=<HBM node cores num> numactl -p <HBM_NODE_INDEX> -C <HBM cores list> python run.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex
```
- An example of llama2 7b model with HBM numa node index being 2:
```bash
OMP_NUM_THREADS=56 numactl -p 2 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex
```

### Distributed Inference with Xeon® CPU Max Series

As HBM has memory capacity limitations, we need to shard the model in advance with DDR memory. Please follow [the example](#how-to-shard-model-for-distributed-tests-with-deepspeed-autotp).

Then we can invoke distributed inference with `deepspeed` command:

- Command:
```bash
deepspeed --bind_cores_to_rank  run.py --benchmark -m <SHARDED_MODEL_PATH> --dtype bfloat16 --ipex --autotp
```

As the model has been sharded, we specify `SHARDED_MODEL_PATH` for `-m` argument instead of original model name or path, and `--shard-model` argument is not needed.

## Performance Results

The performance results on AWS instances can be found [here](../../../../../docs/tutorials/performance.md#llm-performance).

# Advanced Usage

## Weight-only quantization with low precision checkpoint (Experimental)

Using INT4 weights can further improve performance by reducing memory bandwidth. However, direct per-channel quantization of weights to INT4 probably results in poor accuracy. Some algorithms can modify weights through calibration before quantizing weights to minimize accuracy drop. GPTQ is one of such algorithms. You may generate modified weights and quantization info (scales, zero points) for a certain model with a dataset by such algorithms. The low precision checkpoint is saved as a `state_dict` in a `.pt` file and can be loaded later for weight only quantization. We provide an example here to run GPTQ.

*Note:* Currently GPTQ API is verified on the following models: gpt-j, opt, llama, Llama-2, bloom, bloomz, dolly-v1, dolly-v2, gpt-neo, gpt-neox, mpt, falcon, starcoder. Some of them are not in the list of optimized models. Please use with care.

Here is how to use it:

```bash
# Step 1: Generate modified weights and quantization info and save as checkpoint
python utils/run_gptq.py --model <MODEL_ID> --output-dir ./saved_results
```
The dataset for calibration is `NeelNanda/pile-10k` by default. To use other dataset, such as lambada, you may use `--dataset <dataset id>` to specify. Group size is specified by `--group-size <group_size>` (default is 128). You can specify calibration sample size by modifying `--nsamples <int>` (default is 128); you can also choose whether or not to align calibration data to a fixed length by modifying `--use_max_length <bool>` and `--pad_max_length <int>`. For details please refer to [GPTQ](../../../../../intel_extension_for_pytorch/quantization/_GPTQ/README.md)

It may take a few hours to finish. Modified weights and their quantization info are stored in `gptq_checkpoint_g128.pt`, where g128 means group size for input channel is 128 by default. Group size controls the granularity of quantization of weight along input channel. 
Then generate model for weight only quantization with INT4 weights and run tasks.

```bash
# Step 2: Generate quantized model with INT4 weights
# Provide checkpoint file name by --low-precision-checkpoint <file name>
python single_instance/run_quantization.py --ipex-weight-only-quantization --quant-with-amp -m <MODEL_ID> --low-precision-checkpoint "saved_results/gptq_checkpoint_g128.pt" --output-dir "saved_results" 

# Step 3: Run quantized model for latency benchmark
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python single_instance/run_quantization.py -m <MODEL_ID> --benchmark --quant-with-amp --quantized-model-path "./saved_results/best_model.pt" 
# To run accuracy tests, please follow the instructions in the **Single Instance Accuracy** part
```

If the checkpoint is generated by some other methods and has different keys in the state_dict, you will need to specify the keys for weight, scales, zero points and bias. Bias is optional in the state_dict while others are required. Default keys are:

```python
{
    "weight_key": "packed_weight",
    "scale_key": "scale",
    "zero_point_key": "packed_zp",
    "bias_key": "bias",
}
```

You need to make a config dict like above and pass it to `ipex.llm.optimize` together with the state_dict from the checkpoint as a tuple `(state_dict, config_dict)`. You will need to modify the example script.

```python
low_precision_checkpoint = torch.load(args.low_precision_checkpoint)
config_dict = {
    "weight_key": "...",
    "scale_key": "...",
    "zero_point_key": "...",
    "bias_key": "...",
}
state_dict_and_config = (low_precision_checkpoint, config_dict)
user_model = ipex.llm.optimize(
    user_model.eval(),
    dtype=amp_dtype,
    quantization_config=qconfig,
    inplace=True,
    low_precision_checkpoint=state_dict_and_config,
    deployment_mode=False,
)
```

**Example**

Intel® Extension for PyTorch\* with INT4 weight only quantization has been used in latest MLPerf submission (August 2023) to fully maximize the power of Intel® Xeon®, and also shows good accuracy as comparing with FP32. This example is a simplified version of the MLPerf task. It will download a finetuned FP32 GPT-J model used for MLPerf submission, quantize the model to INT4 and run a text summarization task on the `cnn_dailymail` dataset. The example runs for 1000 samples, which is a good approximation of the results for the entire dataset and saves time.

```bash
pip install evaluate nltk absl-py rouge_score
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> bash single_instance/run_int4_gpt-j_on_cnndailymail.sh
```

Please note that 100 GB disk space, 100 GB memory and Internet access are needed to run this example. The example will run for a few hours depending on your hardware and network condition. The example is verified on the 4th Generation Intel® Xeon® Scalable (Sapphire Rapids) platform. You may get different results on older platforms as some new hardware features are unavailable.

**Checkpoint Requirements**

IPEX now only supports some certain cases. Weights must be N by K and asymmetrically quantized to UINT4 and then compressed along K axis to `torch.int32`. Data type of scales can be any floating point types. Shape of scales should be [N, number_of_groups] or with additional dimensions whose length is 1. Zero points should have the same shape as scales and stored as `torch.int32` but the true data type is UINT4. Bias is optional in the `state_dict` (checkpoint). If it is present, we read bias in the `state_dict`. Otherwise we read bias from the original model. Bias is `None` if it cannot be found in both cases.

## Accuracy test:

We leverage [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for the accuracy test.

We verify and recommend to test "lambada_openai" task, for more choice, see {TASK_NAME} in this [link](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md).

### Single Instance

```bash
cd ./single_instance
```
### FP32:

- Command:
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run_accuracy.py  -m <MODEL_ID> --dtype float32 --ipex --tasks {TASK_NAME}
```
- An example of llama2 7b model:
```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py  -m meta-llama/Llama-2-7b-hf --dtype float32 --ipex --tasks lambada_openai
```

### BF16:

- Command:
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run_accuracy.py  -m <MODEL_ID> --dtype bfloat16 --ipex --tasks {TASK_NAME}
```
- An example of llama2 7b model:
```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py  -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex --tasks lambada_openai
```

### Quantizations:

For the quantized models to be used in accuracy tests, we can reuse the model files that are named "best_model.pt" in the "--output-dir" path ([generated during inference performance tests](#generation_sq)).

- Command:
```bash
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_accuracy.py --model <MODEL ID> --quantized-model-path "./saved_results/best_model.pt" --dtype <int8 or int4> --tasks {TASK_NAME}
# Please add  "--quant-with-amp" if your model is quantized with this flag
```
- An example of llama2 7b model:
```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py -m meta-llama/Llama-2-7b-hf --quantized-model-path "./saved_results/best_model.pt" --dtype int8  --tasks lambada_openai
```

### Distributed with DeepSpeed (autoTP)

### Prepare:

```bash
# Run distributed accuracy with 2 ranks of one node
cd ./distributed
unset KMP_AFFINITY
```

### FP32:

- Command:
```bash
deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model <MODEL_ID> --dtype float32 --ipex --tasks <TASK_NAME> 
```
- An example of llama2 7b model:
```bash
deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model  meta-llama/Llama-2-7b-hf --dtype float32 --ipex --tasks lambada_openai 
```

### BF16:
- Command:
```bash
deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model <MODEL_ID> --dtype  bfloat16 -ipex --tasks <TASK_NAME> 
```
- An example of llama2 7b model:
```bash
deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model  meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex --tasks lambada_openai 
```

### Weight-only quantization (INT8):

- Command:
```bash
deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model <MODEL_ID> --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --ipex --tasks <TASK_NAME>  
```
- An example of llama2 7b model:
```bash
deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model  meta-llama/Llama-2-7b-hf --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --ipex --tasks <TASK_NAME>  
```

## How to Shard model for Distributed tests with DeepSpeed (autoTP)

To save memory usage, we could shard the model weights under the local path before we launch distributed tests with DeepSpeed.

```
cd ./utils
# general command:
python create_shard_model.py -m <MODEL ID>  --save-path <SHARD MODEL NEW PATH>
# After sharding the model, using -m <SHARD MODEL NEW PATH> in later tests

# An example of llama2 7b:
python create_shard_model.py meta-llama/Llama-2-7b-hf --save-path ./local_llama2_7b
```

# Miscellaneous Tips

(1) We can build up LLM services optimized by Intel® Extension for PyTorch\* with Triton Server. Please refer [here](../../../serving/triton/README.md) for best practice.

(2) The LLM inference methods introduced in this page can be well applied for AWS. We can just follow the above instructions and enjoy the boosted performance of LLM with Intel® Extension for PyTorch\* optimizations on the AWS instances.
