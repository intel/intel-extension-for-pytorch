# Text Generation

We provide the inference benchmarking scripts for large language models text generation.<br/>
Support large language model families, including GPT-J, LLaMA, GPT-Neox, OPT, Falcon, CodeGen.<br/>
The scripts include both single instance and distributed (DeepSpeed) use cases.<br/>
The scripts cover model generation inference with low precions cases for different models with best perf and accuracy (bf16 AMP，static quantization and weight only quantization).<br/>

# Supported Model List

| MODEL FAMILY | Verified < MODEL ID > (Huggingface hub)| FP32/BF16 | Weight only quantzation INT8 | Weight only quantization INT4| Static quantization INT8 |
|---|:---:|:---:|:---:|:---:|:---:|
|LLAMA| "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-70b-hf" | ✅ | ✅ | ✅ | ✅ |
|GPT-J| "EleutherAI/gpt-j-6b" | ✅ | ✅ | ✅ | ✅ |
|GPT-NEOX| "EleutherAI/gpt-neox-20b" | ✅ | ✅ | ✅ | ❎ \*\* |
|FALCON\*|"tiiuae/falcon-40b" | ✅ | ✅ |  ✅ | ❎ \*\*|
|OPT|"facebook/opt-30b", "facebook/opt-1.3b"| ✅ | ✅ |  ✅ | ❎ \*\*|
|CodeGen|"Salesforce/codegen-2B-multi"| ✅ | ✅ |  ✅ | ❎ \*\*|

\* For Falcon models from remote hub, we need to modify the config.json to use the modeling_falcon.py in transformers. Therefore, in the following scripts, we need to pass an extra configuration file like "--config-file=model_config/tiiuae_falcon-40b_config.json". This is optional for FP32/BF16 but needed for quantizations.

\*\* For GPT-NEOX/FALCON/OPT/CodeGen models, the accuracy recipes of static quantization INT8 are not ready thus they will be skipped in our coverage.

*Note*: The above verified models (including other models in the same model family, like "codellama/CodeLlama-7b-hf" from LLAMA family) are well supported with all optimizations like indirect access KV cache, fused ROPE, and prepacked TPP Linear (fp32/bf16). For other LLM model families, we are working in progress to cover those optimizations, which will expand the model list above.

# Environment Setup

1. Get the Intel® Extension for PyTorch\* source code

```bash
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout v2.1.100+cpu
cd examples/cpu/inference/python/llm
```

2.a. It is highly recommended to build a Docker container from the provided `Dockerfile`.

```bash
# Build an image with the provided Dockerfile
docker build -t ipex-llm:2.1.100 .

# Run the container with command below
docker run --rm -it --privileged ipex-llm:2.1.100 bash

# When the command prompt shows inside the docker container, enter llm examples directory
cd llm
```

2.b. Alternatively, you can take advantage of a provided environment configuration script to setup an environment without using a docker container.

```bash
# GCC 12.3 is required. Installation can be taken care of by the environment configuration script.
# Create a conda environment
conda create -n llm python=3.9 -y
conda activate llm

# Setup the environment with the provided script
bash ./tools/env_setup.sh
```

3. Once an environment is configured with either method above, set necessary environment variables with an environment variables activation script and download the sample `prompt.json`.

```bash
# Activate environment variables
source ./tools/env_activate.sh

# Get the sample prompt.json
wget https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/prompt.json

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

## Example usages of one-click Python script
### Single Instance inference
#### Prepare:
```bash
# Get prompt file to the path of scripts
cp prompt.json ./single_instance
export WORK_DIR=./
```
#### BF16:
```bash
# general command:
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex --deployment-mode

# An example of llama2 7b model:
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex --deployment-mode
```
#### Weight-only quantization:
```bash
# int8 general command:
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py  --benchmark -m <MODEL_ID> --ipex-weight-only-quantization  --output-dir "saved_results" --int8-bf16-mixed

# An example of llama2 7b model:
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-hf --ipex-weight-only-quantization  --output-dir "saved_results" --int8-bf16-mixed


# int4 general command:
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py  --benchmark -m <MODEL_ID> --ipex-weight-only-quantization --gptq --output-dir "saved_results" --int8-bf16-mixed
# for GPT-NEOX Weight-only quantizations, using "--int8" instead of "--int8-bf16-mixed" for accuracy concerns.

# An example of llama2 7b model:
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-hf --ipex-weight-only-quantization  --output-dir "saved_results" --int8-bf16-mixed --gptq
```
#### Static quantization (int8):
```bash
# general command:
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py  --benchmark -m <MODEL_ID> --ipex-smooth-quant --alpha <Tuned alpha for specific models> --output-dir "saved_results" --int8
# For the best alpha values (range [0, 1.0], float) tuned for specific models, we verified good accuracy: "EleutherAI/gpt-j-6b" with alpha=1.0, "meta-llama/Llama-2-7b-chat-hf" with alpha=0.8.
# For more recipes, please refer to https://github.com/intel/neural-compressor/blob/master/docs/source/smooth_quant.md#validated-models
# Note: by default, we use "--int8" to run int8 mixed fp32 mode, while for peak performance of static quantization, please use "--int8-bf16-mixed" instead (may impact accuracy).

# An example of llama2 7b model:
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py  --benchmark -m meta-llama/Llama-2-7b-chat-hf --ipex-smooth-quant --alpha 0.8 --output-dir "saved_results" --int8
```
*Notes for all quantizations:*

(1) for quantization benchmarks, the first runs will auto-generate the quantized model named "best_model.pt" in the "--output-dir" path, you can reuse these quantized models for inference-only benchmarks by adding "--quantized-model-path <output_dir + "best_model.pt">".

(2) for Falcon quantizations, "--config-file <CONFIG_FILE>" is needed and example of <CONFIG_FILE>: "utils/model_config/tiiuae_falcon-40b_config.json".

### Distributed inference with DeepSpeed (autoTP)
#### Prepare:
```bash
# Get prompt file to the path of scripts
cp prompt.json ./distributed
export WORK_DIR=./
unset KMP_AFFINITY
# By default, we use "--shard-model" for better memory usage, if your model path is already sharded, please remove "--shard-model"
```
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
deepspeed --bind_cores_to_rank run.py  --benchmark -m <MODEL_ID> --ipex --ipex-weight-only-quantization --output-dir "saved_results" --int8-bf16-mixed --autotp --shard-model
# for Falcon quantizations, "--config-file <CONFIG_FILE>" is needed and example of <CONFIG_FILE>: "utils/model_config/tiiuae_falcon-40b_config.json".
# for GPT-NEOX weight-only quantizations, using "--int8" instead of "--int8-bf16-mixed", and add "--dtype float32" for accuracy concerns.

# An example of llama2 7b model:
deepspeed --bind_cores_to_rank  run.py --benchmark -m meta-llama/Llama-2-7b-hf --ipex --ipex-weight-only-quantization --output-dir "saved_results" --int8-bf16-mixed --autotp --shard-model
```

# Advanced Usage
## Weight-only quantization with low precision checkpoint (Experimental)
Using INT4 weights can further improve performance by reducing memory bandwidth. However, direct per-channel quantization of weights to INT4 probably results in poor accuracy. Some algorithms can modify weights through calibration before quantizing weights to minimize accuracy drop. GPTQ is one of such algorithms. You may generate modified weights and quantization info (scales, zero points) for a certain model with a some dataset by such algorithms. The results are saved as a `state_dict` in a `.pt` file. We provided a script here to run GPTQ (Intel(R) Neural Compressor 2.3.1 is required).

Here is how to use it:
```bash
# Step 1: Generate modified weights and quantization info
python utils/run_gptq.py --model <MODEL_ID> --output-dir ./saved_results
```
It may take a few hours to finish. Modified weights and their quantization info are stored in `gptq_checkpoint_g128.pt`, where g128 means group size for input channel is 128 by default.
Then generate model for weight only quantization with INT4 weights and run tasks.
```bash
# Step 2: Generate quantized model with INT4 weights
# Provide checkpoint file name by --low-precision-checkpoint <file name>
python single_instance/run_<MODEL_ID>_quantization.py --ipex-weight-only-quantization --output-dir "saved_results" --int8-bf16-mixed -m <MODEL_ID> --low-precision-checkpoint "saved_results/gptq_checkpoint.pt"

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
You need to make a config dict like above and pass it to `ipex.optimize_transformers` together with the state_dict from the checkpoint as a tuple `(state_dict, config_dict)`. You will need to modify the example script.
```python
low_precision_checkpoint = torch.load(args.low_precision_checkpoint)
config_dict = {
    "weight_key": "...",
    "scale_key": "...",
    "zero_point_key": "...",
    "bias_key": "...",
}
state_dict_and_config = (low_precision_checkpoint, config_dict)
user_model = ipex.optimize_transformers(
    user_model.eval(),
    dtype=amp_dtype,
    quantization_config=qconfig,
    inplace=True,
    low_precision_checkpoint=state_dict_and_config,
    deployment_mode=False,
)
```
**Example**

Intel(R) Extension for PyTorch* with INT4 weight only quantization has been used in latest MLPerf submission (August 2023) to fully maximize the power of Intel(R) Xeon((R), and also shows good accuracy as comparing with FP32. This example is a simplified version of the MLPerf task. It will download a finetuned FP32 GPT-J model used for MLPerf submission, quantize the model to INT4 and run a text summarization task on the `cnn_dailymail` dataset. The example runs for 1000 samples, which is a good approximation of the results for the entire dataset and saves time.
```sh
pip install evaluate nltk absl-py rouge_score
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> bash single_instance/run_int4_gpt-j_on_cnndailymail.sh
```
Please note that 100 GB disk space, 100 GB memory and Internet access are needed to run this example. The example will run for a few hours depending on your hardware and network condition. The example is verified on the 4th generation Intel(R) Xeon(R) Scalable (Sapphire Rapids) platform. You may get different results on older platforms as some new hardware features are unavailable.

**Checkpoint Requirements**

IPEX now only supports some certain cases. Weights must be N by K and per-channel asymmetrically quantized (group size = -1) to UINT4 and then compressed along K axis to `torch.int32`.
Data type of scales can be any floating point types. Shape of scales should be [N] or with additional dimensions whose length is 1, e.g., [N, 1] or [1, N]. Zero points should have the same shape as scales and stored as `torch.int32` but the true data type is UINT4. Bias is optional in the `state_dict` (checkpoint). If it is present, we read bias in the `state_dict`. Otherwise we read bias from the original model. Bias is `None` if it cannot be found in both cases.

## Accuracy test:
We leverage [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for the accuracy test.
By default we test "lambada_standard" task, for more choice, see {TASK_NAME} in this [link](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md), 
### Single Instance
```bash
cd ./single_instance
```
### BF16:
```bash
# general command:
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run_accuracy.py --accuracy-only -m <MODEL_ID> --dtype bfloat16 --ipex --jit --tasks {TASK_NAME}

# An example of llama2 7b model:
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py --accuracy-only -m meta-llama/Llama-2-7b-hf --dtype bfloat16 --ipex --jit --tasks lambada_openai
```
### Quantizations:
```bash
# general command:
# For the quantized models to be used in accuracy tests, we can reuse the model files that are named "best_model.pt" in the "--output-dir" path (generated during inference performance tests).
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_accuracy.py --model <MODEL ID> --quantized-model-path "./saved_results/best_model.pt" --dtype int8 --accuracy-only --jit --tasks {TASK_NAME}
# please also add  "--int8-bf16-mixed" if your model is quantized with this flag

# An example of llama2 7b model:
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run_accuracy.py -m meta-llama/Llama-2-7b-hf --quantized-model-path "./saved_results/best_model.pt" --dtype int8 --accuracy-only --jit --int8 --tasks lambada_openai
```
### Distributed with DeepSpeed (autoTP)
### Prepare:
```bash
# Run distributed accuracy with 2 ranks of one node
cd ./distributed
unset KMP_AFFINITY
```
### BF16:
```bash
# general command:
deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model <MODEL_ID> --dtype bfloat16 --ipex --jit --tasks <TASK_NAME> --accuracy-only

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
```
# For saving memory usage, we could shard the model weights under the local path before we launch distributed tests with DeepSpeed
export WORK_DIR=./
cd utils
# general command:
python create_shard_model.py -m <MODEL ID>  --save-path <SHARD MODEL NEW PATH>
# After sharding the model, using -m <SHARD MODEL NEW PATH> in later tests

# An example of llama2 7b:
python create_shard_model.py meta-llama/Llama-2-7b-hf --save-path ./local_llama2_7b
```
