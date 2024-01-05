# Text Generation

We provide the inference benchmarking scripts for large language models text generation.<br/>
Support large language model families, including GPT-J, LLaMA, GPT-Neox, OPT, Falcon, Bloom, CodeGen, Baichuan, ChatGLM, GPTBigCode, T5, Mistral, MPT.<br/>
The scripts include both single instance and distributed (DeepSpeed) use cases.<br/>
The scripts cover model generation inference with low precions cases for different models with best perf and accuracy (bf16 AMP，static quantization and weight only quantization).<br/>

# Optimized Model List

| MODEL FAMILY | Verified <MODEL ID> (Huggingface hub)| FP32/BF16 | Weight only quantzation INT8 | Weight only quantization INT4| Static quantization INT8 |
|---|:---:|:---:|:---:|:---:|:---:|
|LLAMA| "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-70b-hf" | ✅ | ✅ | ✅ | ✅ | 
|GPT-J| "EleutherAI/gpt-j-6b" | ✅ | ✅ | ✅ | ✅ | 
|GPT-NEOX| "EleutherAI/gpt-neox-20b", "databricks/dolly-v2-12b" | ✅ | ✅ | ✅ | ❎ \*\* | 
|FALCON\*|"tiiuae/falcon-40b" | ✅ | ✅ |  ✅ | ❎ \*\*| 
|OPT|"facebook/opt-30b", "facebook/opt-1.3b"| ✅ | ✅ |  ✅ | ❎ \*\*| 
|Bloom|"bigscience/bloom", "bigscience/bloom-1b7"| ✅ | ✅ |  ✅ | ❎ \*\*|
|CodeGen|"Salesforce/codegen-2B-multi"| ✅ | ✅ |  ✅ | ❎ \*\*|
|Baichuan|"baichuan-inc/Baichuan2-13B-Chat", "baichuan-inc/Baichuan2-7B-Chat", "Baichuan-inc/Baichuan-13B-Chat"| ✅ | ✅ |  ✅ | ❎ \*\*|
|ChatGLM\*|"THUDM/chatglm3-6b", "THUDM/chatglm2-6b"| ✅ | ✅ |  ✅ | ❎ \*\*|
|GPTBigCode|"bigcode/starcoder"| ✅ | ✅ |  ✅ | ❎ \*\*|
|T5|"google/flan-t5-xl"| ✅ | ✅ |  ✅ | ❎ \*\*|
|Mistral|"mistralai/Mistral-7B-v0.1"| ✅ | ✅ |  ✅ | ❎ \*\*|
|MPT\*|"mosaicml/mpt-7b"| ✅ | ✅ |  ✅ | ❎ \*\*|

\* For Falcon models from remote hub, we need to modify the config.json to use the modeling_falcon.py in transformers. Therefore, in the following scripts, we need to pass an extra configuration file like "--config-file=model_config/tiiuae_falcon-40b_config.json". This is optional for FP32/BF16 but needed for quantizations.

\* For ChatGLM models, the default torch_dtype is float16 in config.json. We need to replace the "float16" with "float32" in config.json.

\* For MPT models from remote hub, we need to modify the config.json to use the modeling_mpt.py in transformers. Therefore, in the following scripts, we need to pass an extra configuration file like "--config-file=model_config/mosaicml_mpt-7b_config.json".

\*\* For GPT-NEOX/FALCON/OPT/Bloom/CodeGen/Baichuan/GPTBigCode/T5/Mistral/MPT models, the accuracy recipes of static quantization INT8 are not ready thus they will be skipped in our coverage.

*Note*: The above verified models (including other models in the same model family, like "codellama/CodeLlama-7b-hf" from LLAMA family) are well supported with all optimizations like indirect access KV cache, fused ROPE, and prepacked TPP Linear (fp32/bf16). For other LLM model families, we are working in progress to cover those optimizations, which will expand the model list above.


# Environment Setup

*Note*: The instructions in this section will setup an environment with a recent PyTorch\* nightly build and **a latest source build of IPEX**. If you like to use stable PyTorch\* and IPEX release versions, please refer the instructions [in the release branch](https://github.com/intel/intel-extension-for-pytorch/blob/v2.1.100%2Bcpu/examples/cpu/inference/python/llm/README.md#environment-setup), in which IPEX is installed via prebuilt wheels using `pip install` rather than source code building.

1\. Get the Intel® Extension for PyTorch\* source code

```bash
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git submodule sync
git submodule update --init --recursive
```

2\.a. It is highly recommended to build a Docker container from the provided `Dockerfile`.

```bash
# Build an image with the provided Dockerfile by compiling Intel® Extension for PyTorch\* from source
DOCKER_BUILDKIT=1 docker build -f examples/cpu/inference/python/llm/Dockerfile --build-arg COMPILE=ON -t ipex-llm:main .

# Build an image with the provided Dockerfile by installing from Intel® Extension for PyTorch\* prebuilt wheel files
DOCKER_BUILDKIT=1 docker build -f examples/cpu/inference/python/llm/Dockerfile -t ipex-llm:main .

# Run the container with command below
docker run --rm -it --privileged ipex-llm:main bash

# When the command prompt shows inside the docker container, enter llm examples directory
cd llm
```

2\.b. Alternatively, you can take advantage of a provided environment configuration script to setup an environment without using a docker container.

```bash
# GCC 12.3 is required. Installation can be taken care of by the environment configuration script.
# Create a conda environment
conda create -n llm python=3.10 -y
conda activate llm

# Setup the environment with the provided script
cd examples/cpu/inference/python/llm
bash ./tools/env_setup.sh
```

3\. Once an environment is configured with either method above, set necessary environment variables with an environment variables activation script and download the sample `prompt.json`.

```bash
# Activate environment variables
source ./tools/env_activate.sh
```

# Run Models Generations

| Benchmark mode | FP32/BF16 | Weight only quantzation INT8 | Weight only quantization INT4 | Static quantization INT8 |
|---|:---:|:---:|:---:|:---:|
|Single instance | ✅ | ✅ | ✅ | ✅ |
| Distributed (autotp) |  ✅ | ✅ | ❎ | ❎ |

You can run LLM with a one-click Python script "run.py" for all inference cases.
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

## Example usages of one-click Python script

### Quick start example commands for benchmarking LLaMA2-7b

```bash
# The following "OMP_NUM_THREADS" and "numactl" settings are based on the assumption that
# the target server has 56 physical cores per numa socket, and we benchmark with 1 socket.
# Please adjust the settings per your hardware.

# Running FP32 model
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype float32

# Running FP32 model with IPEX
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype float32 --ipex --deployment-mode

# Running BF16 model with IPEX
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex --deployment-mode

# INT8 static quantization
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-hf --ipex-smooth-quant --qconfig-summary-file <path to "llama-2-7b_qconfig.json"> --output-dir "saved_results" --int8-bf16-mixed

# INT8 weight-only quantization
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-hf --ipex-weight-only-quantization  --output-dir "saved_results" --int8-bf16-mixed

# INT4 weight-only quantization
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-hf --ipex-weight-only-quantization  --output-dir "saved_results" --int8-bf16-mixed --gptq

# Distributed inference in BF16
deepspeed --bind_cores_to_rank  run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex --deployment-mode --autotp --shard-model

# Distributed inference in weight-only quantization mode
deepspeed --bind_cores_to_rank  run.py --benchmark -m meta-llama/Llama-2-7b-hf --ipex --ipex-weight-only-quantization --output-dir "saved_results" --int8-bf16-mixed --autotp --shard-model --deployment-mode
```

### Quick start example commands for accuracy test with LLaMA2-7b

Check [Advanced Usage](#accuracy-test) for details.

For the quantized models used in accuracy tests below, we can reuse the model files that are named "best_model.pt" in the "--output-dir" path ([generated during inference performance tests above](#generation_sq)).

```bash
# The following "OMP_NUM_THREADS" and "numactl" settings are based on the assumption that
# the target server has 56 physical cores per numa socket, and we benchmark with 1 socket.
# Please adjust the settings per your hardware.

# run_accuracy.py script is inside single_instance directory.
cd single_instance

# Running FP32 model
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py --accuracy-only -m meta-llama/Llama-2-7b-hf --dtype float32 --ipex --jit --tasks lambada_openai

# Running BF16 model
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py --accuracy-only -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex --jit --tasks lambada_openai

# Quantization
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py -m meta-llama/Llama-2-7b-hf --quantized-model-path "./saved_results/best_model.pt" --dtype int8 --accuracy-only --jit --tasks lambada_openai --int8-bf16-mixed


# run_accuracy_with_deepspeed.py script is inside distributed directory.
cd distributed
unset KMP_AFFINITY

# Distributed inference in FP32
deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model  meta-llama/Llama-2-7b-hf --dtype float32 --ipex --jit --tasks lambada_openai --accuracy-only

# Distributed inference in BF16
deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model  meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex --jit --tasks lambada_openai --accuracy-only

# Distributed inference with Weight-Only Quantization
deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model  meta-llama/Llama-2-7b-hf --int8-bf16-mixed --ipex --jit --tasks lambada_openai --accuracy-only --ipex-weight-only-quantization
```

### Single Instance inference

#### FP32:

```bash
# general command:
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py --benchmark -m <MODEL_ID> --dtype float32 --ipex --deployment-mode

# An example of llama2 7b model:
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype float32 --ipex --deployment-mode
```

#### BF16:

```bash
# general command:
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex --deployment-mode

# An example of llama2 7b model:
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex --deployment-mode
```

#### Static quantization (int8):

```bash
# general command:
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py  --benchmark -m <MODEL_ID> --ipex-smooth-quant --qconfig-summary-file <path to the qconfig of the model_id> --output-dir "saved_results" --int8-bf16-mixed
# Note: by default, we use "--int8-bf16-mixed" to run int8 mixed bf16 inference with peak performance of static quantization, if you observe accuracy drops, please use "--int8" instead to run int8 mixed fp32.

# An example of llama2 7b model:
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-hf --ipex-smooth-quant --qconfig-summary-file <path to "llama-2-7b_qconfig.json"> --output-dir "saved_results" --int8-bf16-mixed
```

- We provide the downloading links of tuned static quantization qconfig summary files with good quality: ["meta-llama/Llama-2-7b-hf"](https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/llama-2-7b_qconfig.json), ["meta-llama/Llama-2-7b-chat-hf"](https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/llama-2-7b-chat_qconfig.json), ["meta-llama/Llama-2-13b-hf"](https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/llama-2-13b_qconfig.json) and ["EleutherAI/gpt-j-6b"](https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/gpt-j-6b_qconfig.json). We verify these qconfig files with the certain codebase of Transformers/PyTorch/IPEX to make sure their ease of use, while their compatibility may change due to the changes on codebases. If you meet any failure when you adopt these existing qconfig files, please refer to [Intel® Neural Compressor scripts](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/language-modeling/quantization/llm/ipex) to generate them based on your environment of codebases.

- For other models' qconfig recipes, you can just try to run your model_id and use IPEX default recipes by removing "--qconfig-summary-file <path to specific model qconfig>". If IPEX default recipes are not good enough for accuracy requirements, please refer to the [Intel® Neural Compressor tutorial](https://github.com/intel/neural-compressor/blob/master/docs/source/smooth_quant.md#validated-models) and [scripts](https://github.com/intel/neural-compressor/tree/master/examples/pytorch/nlp/huggingface_models/language-modeling/quantization/llm/ipex) for more tuned recipes.

#### Weight-only quantization:

```bash
# int8 general command:
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py  --benchmark -m <MODEL_ID> --ipex-weight-only-quantization  --output-dir "saved_results" --int8-bf16-mixed
# for GPT-NEOX Weight-only quantizations, using "--int8" instead of "--int8-bf16-mixed" for accuracy concerns.

# An example of llama2 7b model:
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-hf --ipex-weight-only-quantization  --output-dir "saved_results" --int8-bf16-mixed


# int4 general command:
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py  --benchmark -m <MODEL_ID> --ipex-weight-only-quantization --gptq --output-dir "saved_results" --int8-bf16-mixed
# for GPT-NEOX Weight-only quantizations, using "--int8" instead of "--int8-bf16-mixed" for accuracy concerns.

# An example of llama2 7b model:
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-hf --ipex-weight-only-quantization  --output-dir "saved_results" --int8-bf16-mixed --gptq
```

*Notes:*

(1) [_numactl_](https://linux.die.net/man/8/numactl) is used to specify memory and cores of your hardware to get better performance. _\<node N\>_ specifies the [numa](https://en.wikipedia.org/wiki/Non-uniform_memory_access) node id (e.g., 0 to use the memory from the first numa node). _\<physical cores list\>_ specifies phsysical cores which you are using from the _\<node N\>_ numa node (e.g., 0-56 from the first numa node). You can use [_lscpu_](https://man7.org/linux/man-pages/man1/lscpu.1.html) command in Linux to check the numa node information.

(2) The _\<MODEL_ID\>_ (e.g., "meta-llama/Llama-2-13b-hf") specifies the model you will run. we provide some _Verified \<MODEL ID\>_ in the [Optimized Model List](#optimized-model-list). You can also try other models from [HuggingFace Models](https://huggingface.co/models).

(3) <a name="generation_sq">for all quantization benchmarks</a>, both quantization and inference stages will be triggered by default. For quantization stage, it will auto-generate the quantized model named "best_model.pt" in the "--output-dir" path, and for inference stage, it will launch the inference with the quantized model "best_model.pt".  For inference-only benchmarks (avoid the repeating quantization stage), you can also reuse these quantized models for by adding "--quantized-model-path <output_dir + "best_model.pt">" . Besides, specific for static quantization, if not using "--qconfig-summary-file", a qconfig recipe will also be generated in the "--output-dir" path, which could be reused as well (to generate the quantized model "best_model.pt").

(4) for Falcon quantizations, "--config-file <CONFIG_FILE>" is needed in the command and the example of <CONFIG_FILE> is provided here: "utils/model_config/tiiuae_falcon-40b_config.json".

### Distributed inference with DeepSpeed (autoTP)

#### Prepare:

```bash
unset KMP_AFFINITY
```

In the DeepSpeed cases below, we recommend "--shard-model" to shard model weight sizes more even for better memory usage when running with DeepSpeed.

If using "--shard-model", it will save a copy of the shard model weights file in the path of "--output-dir" (default path is "./saved_results" if not provided).
If you have used "--shard-model" and generated such a shard model path, in further repeated benchmarks, please remove "--shard-model", and replace "-m <MODEL_ID>" with "-m <shard model path>" to skip the repeated shard steps.

Besides, the standalone shard model function/scripts are also provided in the [Advanced Usage](#how-to-shard-model-for-distributed-tests-with-deepspeed-autotp) section, in case you would like to generate the shard model weights files in advance before running distributed inference.

#### FP32:
```bash
# general command:
deepspeed --bind_cores_to_rank  run.py --benchmark -m <MODEL_ID> --dtype float32 --ipex --deployment-mode --autotp --shard-model

# An example of llama2 7b model:
deepspeed --bind_cores_to_rank  run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype float32 --ipex --deployment-mode --autotp --shard-model
```

*Note:* By default, we use "--shard-model" for better memory usage, if your model path is already sharded, please remove "--shard-model"

#### BF16:

```bash
# general command:
deepspeed --bind_cores_to_rank  run.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex --deployment-mode --autotp --shard-model

# An example of llama2 7b model:
deepspeed --bind_cores_to_rank  run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex --deployment-mode --autotp --shard-model
```

#### Weight-only quantization:

```bash
# int8 general command:
deepspeed --bind_cores_to_rank run.py  --benchmark -m <MODEL_ID> --ipex --ipex-weight-only-quantization --output-dir "./saved_results" --int8-bf16-mixed --autotp --shard-model --deployment-mode
# The command will shard the model, quantize the model by weight-only quantization then run the benchmark. The quantized model won't be saved.
# for Falcon quantizations, "--config-file <CONFIG_FILE>" is needed and example of <CONFIG_FILE>: "utils/model_config/tiiuae_falcon-40b_config.json".
# for GPT-NEOX weight-only quantizations, using "--int8" instead of "--int8-bf16-mixed", and add "--dtype float32" for accuracy concerns.

# An example of llama2 7b model:
deepspeed --bind_cores_to_rank  run.py --benchmark -m meta-llama/Llama-2-7b-hf --ipex --ipex-weight-only-quantization --output-dir "saved_results" --int8-bf16-mixed --autotp --shard-model --deployment-mode
```

## Instructions for Running LLM with Intel® Xeon® CPU Max Series

Intel® Xeon® CPU Max Series are equipped with high bandwidth memory (HBM), which further accelerates LLM inference. For the common case that HBM and DDR are both installed in a Xeon® CPU Max Series server, the memory mode can be configured to Flat Mode or Cache Mode. Details about memory modes can be found at Section 3.1 in [the Xeon® CPU Max Series Configuration Guide](https://cdrdv2-public.intel.com/769060/354227-intel-xeon-cpu-max-series-configuration-and-tuning-guide.pdf).

### Single Instance Inference with Xeon® CPU Max Series

#### Cache Mode HBM

In cache mode, only DDR address space is visible to software and HBM functions as a transparent memory-side cache for DDR. Therefore the usage is the same with [the common usage](#single-instance-inference).

#### Flat Mode HBM

In flat mode, HBM and DDR are exposed to software as separate address spaces in this mode. Therefore we need to check the `HBM_NODE_INDEX` of interest with commands like `lscpu`, then the LLM inference invoking command would be like:

```bash
# general command:
OMP_NUM_THREADS=<HBM node cores num> numactl -m <HBM_NODE_INDEX> -C <HBM cores list> python run.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex --deployment-mode

# An example of llama2 7b model with HBM numa node index being 2:
OMP_NUM_THREADS=56 numactl -m 2 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex --deployment-mode
```

*Note:* For some very large models we may get an "OOM Error" due to HBM capacity limitations. In this case we can change `-m` argument for `numactl` to `-p` in the above command to enable the model inference with the larger DDR memory.

```bash
# general command:
OMP_NUM_THREADS=<HBM node cores num> numactl -p <HBM_NODE_INDEX> -C <HBM cores list> python run.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex --deployment-mode

# An example of llama2 7b model with HBM numa node index being 2:
OMP_NUM_THREADS=56 numactl -p 2 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex --deployment-mode
```

### Distributed Inference with Xeon® CPU Max Series

As HBM has memory capacity limitations, we need to shard the model in advance with DDR memory. Please follow [the example](#how-to-shard-model-for-distributed-tests-with-deepspeed-autotp).

Then we can invoke distributed inference with `deepspeed` command:

```bash
# general command:
deepspeed --bind_cores_to_rank  run.py --benchmark -m <SHARDED_MODEL_PATH> --dtype bfloat16 --ipex --deployment-mode --autotp
```

As the model has been sharded, we specify `SHARDED_MODEL_PATH` for `-m` argument instead of original model name or path, and `--shard-model` argument is not needed.

## Performance Results

The performance results on AWS instances can be found [here](../../../../../docs/tutorials/performance.md#llm-performance).

# Advanced Usage

## Weight-only quantization with low precision checkpoint (Experimental)

Using INT4 weights can further improve performance by reducing memory bandwidth. However, direct per-channel quantization of weights to INT4 probably results in poor accuracy. Some algorithms can modify weights through calibration before quantizing weights to minimize accuracy drop. GPTQ is one of such algorithms. You may generate modified weights and quantization info (scales, zero points) for a certain model with a dataset by such algorithms. The low precision checkpoint is saved as a `state_dict` in a `.pt` file and can be loaded later for weight only quantization. We provide an example here to run GPTQ.

*Note:* Currently pytorch models validated by GPTQ include: gpt-j, opt, llama, Llama-2, bloom, bloomz,
dolly-v1, dolly-v2, gpt-neo, gpt-neox, mpt, falcon.

Here is how to use it:

```bash
# Step 1: Generate modified weights and quantization info and save as checkpoint
python utils/run_gptq.py --model <MODEL_ID> --output-dir ./saved_results
# Please note that tiiuae/falcon-40b is not supported yet
```
The dataset for calibration is NeelNanda/pile-10k by default. To use other dataset, such as lambada, you may use `--dataset <dataset id>` to specify. You can specify calibration sample size by modifying `--nsamples <int>` (default is 128); you can also choose whether or not to align calibration data to a fixed length by modifying `--use_max_length <bool>` and `--pad_max_length <int>`. For details please refer to [GPTQ](../../../../../intel_extension_for_pytorch/quantization/_GPTQ/README.md)

It may take a few hours to finish. Modified weights and their quantization info are stored in `gptq_checkpoint_g128.pt`, where g128 means group size for input channel is 128 by default. Group size controls the granularity of quantization of weight along input channel. 
Then generate model for weight only quantization with INT4 weights and run tasks.

```bash
# Step 2: Generate quantized model with INT4 weights
# Provide checkpoint file name by --low-precision-checkpoint <file name>
python single_instance/run_<MODEL_ID>_quantization.py --ipex-weight-only-quantization --output-dir "saved_results" --int8-bf16-mixed -m <MODEL_ID> --low-precision-checkpoint "saved_results/gptq_checkpoint_g128.pt"

# Step 3: Run quantized model for latency benchmark
# For GPT-NEOX, use --int8 instead of --int8-bf16-mixed
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python single_instance/run_<MODEL_ID>_quantization.py -m <MODEL_ID> --quantized-model-path "./saved_results/best_model.pt" --benchmark --int8-bf16-mixed
# To run accuracy tests, please follow the instructions in the **Single Instance Accuracy** part
```

If the checkpoint is generated by some other methods and have different keys in the state_dict, you will need to specify the keys for weight, scales, zero points and bias. Bias is optional in the state_dict while others are required. Default keys are:

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

```bash
# general command:
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run_accuracy.py --accuracy-only -m <MODEL_ID> --dtype float32 --ipex --jit --tasks {TASK_NAME}

# An example of llama2 7b model:
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py --accuracy-only -m meta-llama/Llama-2-7b-hf --dtype float32 --ipex --jit --tasks lambada_openai
```

### BF16:

```bash
# general command:
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run_accuracy.py --accuracy-only -m <MODEL_ID> --dtype bfloat16 --ipex --jit --tasks {TASK_NAME}

# An example of llama2 7b model:
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py --accuracy-only -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex --jit --tasks lambada_openai
```

### Quantizations:

For the quantized models to be used in accuracy tests, we can reuse the model files that are named "best_model.pt" in the "--output-dir" path ([generated during inference performance tests](#generation_sq)).

```bash
# general command:
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_accuracy.py --model <MODEL ID> --quantized-model-path "./saved_results/best_model.pt" --dtype int8 --accuracy-only --jit --tasks {TASK_NAME} --int8-bf16-mixed
# Please remove  "--int8-bf16-mixed" if your model is quantized without this flag

# An example of llama2 7b model:
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py -m meta-llama/Llama-2-7b-hf --quantized-model-path "./saved_results/best_model.pt" --dtype int8 --accuracy-only --jit --tasks lambada_openai --int8-bf16-mixed
```

### Distributed with DeepSpeed (autoTP)

### Prepare:

```bash
# Run distributed accuracy with 2 ranks of one node
cd ./distributed
unset KMP_AFFINITY
```

### FP32:

```bash
# general command:
deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model <MODEL_ID> --dtype float32 --ipex --jit --tasks <TASK_NAME> --accuracy-only

# An example of llama2 7b model:
deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model  meta-llama/Llama-2-7b-hf --dtype float32 --ipex --jit --tasks lambada_openai --accuracy-only
```

### BF16:

# Run GPTJ/LLAMA/OPT/Falcon/Bloom/CodeGen/Baichuan/GPTBigCode/T5/Mistral/MPT with bfloat16 DeepSpeed
deepspeed --bind_cores_to_rank run_generation_with_deepspeed.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex --deployment-mode

# An example of llama2 7b model:
deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model  meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex --jit --tasks lambada_openai --accuracy-only
```

### Weight-only quantization:

```bash
# general command:
deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model <MODEL_ID> --int8-bf16-mixed --ipex --jit --tasks <TASK_NAME> --accuracy-only --ipex-weight-only-quantization
# note that GPT-NEOX please remove "--int8-bf16-mixed" and add "--dtype float32" for accuracy concerns

# An example of llama2 7b model:
deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model  meta-llama/Llama-2-7b-hf --int8-bf16-mixed --ipex --jit --tasks <TASK_NAME> --accuracy-only --ipex-weight-only-quantization
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

(1) Please be aware that we need to set argument `torchscript=True` when loading model with `from_pretrained()` function if we are running the script in `deployment_mode`.

(2) We can build up LLM services optimized by Intel® Extension for PyTorch\* with Triton Server. Please refer [here](../../../serving/triton/README.md) for best practice.

(3) The LLM inference methods introduced in this page can be well applied for AWS. We can just follow the above instructions and enjoy the boosted performance of LLM with Intel® Extension for PyTorch\* optimizations on the AWS instances.
