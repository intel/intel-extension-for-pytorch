# Text Generation
We provide the inference benchmarking scripts for large language models text generation.<br/>
Support large language model families, including GPT-J, LLaMA, GPT-Neox, OPT, Falcon, Bloom, CodeGen, Baichuan.<br/>
The scripts include both single instance and distributed (DeepSpeed) use cases.<br/>
The scripts cover model generation inference with low precions cases for different models with best perf and accuracy (bf16 AMP，static quantization and weight only quantization).<br/>

# Setup
```bash
WORK_DIR=$PWD
# GCC 12.3 is required, please set it firstly
# Create environment (conda recommended)
conda create -n llm python=3.9 -y
# install deps
conda install cmake ninja mkl mkl-include -y
conda install gperftools -c conda-forge -y

# Install PyTorch 2.1 release
python -m pip install torch==2.1 --index-url https://download.pytorch.org/whl/cpu

# Install IPEX 2.1 release
python -m pip install intel_extension_for_pytorch

# Used for accuracy test only
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .

# Install transformers
pip install transformers==4.31.0
# Install others deps
pip install cpuid accelerate datasets sentencepiece protobuf==3.20.3

# Setup environment variables for performance on Xeon
export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so.6
export KMP_BLOCKTIME=INF
export KMP_TPAUSE=0
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_FORJOIN_BARRIER_PATTERN=dist,dist
export KMP_PLAIN_BARRIER_PATTERN=dist,dist
export KMP_REDUCTION_BARRIER_PATTERN=dist,dist
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so # Intel OpenMP
# Tcmalloc is a recommended malloc implementation that emphasizes fragmentation avoidance and scalable concurrency support.
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

# [Optional] install neural-compressor for GPT-J static quantization and running GPTQ (see below)
pip install neural-compressor==2.3.1

# [Optional] The following is only for DeepSpeed case
#Install oneccl-bind-pt(also named torch-ccl)
git clone https://github.com/intel/torch-ccl.git
cd torch-ccl && git checkout v2.1.0+cpu
git submodule sync && git submodule update --init --recursive
python setup.py install
cd ../
#Install DeepSpeed
git clone https://github.com/delock/DeepSpeedSYCLSupport
cd DeepSpeedSYCLSupport
git checkout gma/run-opt-branch
python -m pip install -r requirements/requirements.txt
python setup.py install
cd ../
#Install OneCCL
git clone https://github.com/oneapi-src/oneCCL.git
cd oneCCL
mkdir build
cd build
cmake ..
make -j install
source _install/env/setvars.sh
cd ../..

# Get the sample prompt.json
# Make sure the downloaded prompt.json file is under the same directory as that of the python scripts mentioned above.
wget https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/prompt.json

```

# Supported Model List

| MODEL FAMILY | Verified < MODEL ID > (Huggingface hub)| FP32/BF16 | Weight only quantzation INT8 | Weight only quantization INT4| Static quantization INT8 | 
|---|:---:|:---:|:---:|:---:|:---:|
|LLAMA| "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-70b-hf" | ✅ | ✅ | ✅ | ✅ | 
|GPT-J| "EleutherAI/gpt-j-6b" | ✅ | ✅ | ✅ | ✅ | 
|GPT-NEOX| "EleutherAI/gpt-neox-20b" | ✅ | ✅ | ✅ | ❎ ** | 
|FALCON*|"tiiuae/falcon-40b" | ✅ | ✅ |  ✅ | ❎ **| 
|OPT|"facebook/opt-30b", "facebook/opt-1.3b"| ✅ | ✅ |  ✅ | ❎ **| 
|Bloom|"bigscience/bloom", "bigscience/bloom-1b7"| ✅ | ✅ |  ✅ | ❎ **|
|CodeGen|"Salesforce/codegen-2B-multi"| ✅ | ✅ |  ✅ | ❎ **|
|Baichuan|"baichuan-inc/Baichuan2-13B-Chat", "baichuan-inc/Baichuan2-7B-Chat", "Baichuan-inc/Baichuan-13B-Chat"| ✅ | ✅ |  ✅ | ❎ **|

*For Falcon models from remote hub, we need to modify the config.json to use the modeling_falcon.py in transformers. Therefore, in the following scripts, we need to pass an extra configuration file like "--config-file=model_config/tiiuae_falcon-40b_config.json". This is optional for FP32/BF16 but needed for quantizations.

** For GPT-NEOX/FALCON/OPT/Bloom/CodeGen/Baichuan models, the accuracy recipes of static quantization INT8 are not ready thus they will be skipped in our coverage.

*Note*: The above verified models (including other models in the same model family, like "codellama/CodeLlama-7b-hf" from LLAMA family) are well supported with all optimizations like indirect access KV cache, fused ROPE, and prepacked TPP Linear (fp32/bf16). For other LLM model families, we are working in progress to cover those optimizations, which will expand the model list above.

# Run Models Generations

| Benchmark mode | FP32/BF16 | Weight only quantzation INT8 | Weight only quantization INT4 | Static quantization INT8 | 
|---|:---:|:---:|:---:|:---:|
|Single instance | ✅ | ✅ | ✅ | ✅ | 
| Distributed (autotp) |  ✅ | ✅ | ❎ | ❎ | 

You can run LLM with a one-click Python script "run.py" for all inference cases.
```
python run.py --help # for more detailed usages
```
## Example usages of one-click Python script
### Single Instance Performance
```bash
# Get prompt file to the path of scripts
mv PATH/TO/prompt.json ./single_instance
export WORK_DIR=./

# bf16 benchmark
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex --deployment-mode

# weight only quantization int8 benchmark
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py  --benchmark -m <MODEL_ID> --ipex-weight-only-quantization  --output-dir "saved_results" --int8-bf16-mixed

# weight only quantization int4 benchmark
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py  --benchmark -m <MODEL_ID> --ipex-weight-only-quantization --gptq --output-dir "saved_results" --int8-bf16-mixed

# static quantization int8 benchmark
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run.py  --benchmark -m <MODEL_ID> --ipex-smooth-quant --alpha <Tuned alpha for specific models> --output-dir "saved_results" --int8-bf16-mixed
# For the best alpha values (range [0, 1.0], float) tuned for specific models, we verified good accuracy: "EleutherAI/gpt-j-6b" with alpha=1.0, "meta-llama/Llama-2-7b-chat-hf" with alpha=0.8.
# For other variant models, suggest using default alpha=0.5, and could be further tuned in the range [0, 1.0]. (suggest step_size of 0.05)

Notes:
(1) for quantization benchmarks, the first runs will auto-generate the quantized model named "best_model.pt" in the "--output-dir" path, you can reuse these quantized models for inference-only benchmarks by using "--quantized-model-path <output_dir + "best_model.pt">".
(2) for Falcon quantizations, "--config-file <CONFIG_FILE>" is needed and example of <CONFIG_FILE>: "utils/model_config/tiiuae_falcon-40b_config.json".
(3) for GPT-NEOX quantizations, using "--int8" instead of "--int8-bf16-mixed" for accuracy concerns.
(4) By default, generations are based on "beam search", and beam size = 4. For beam size = 1, please add "--greedy"

```
### Distributed Performance with DeepSpeed (autoTP)
```bash
# Get prompt file to the path of scripts
mv PATH/TO/prompt.json ./distributed
export WORK_DIR=./
unset KMP_AFFINITY

# bf16 benchmark
deepspeed --bind_cores_to_rank  run.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex --deployment-mode --autotp --shard-model

# weight only quantization int8 benchmark
deepspeed --bind_cores_to_rank run.py  --benchmark -m <MODEL_ID> --ipex --ipex-weight-only-quantization --output-dir "saved_results" --int8-bf16-mixed --autotp --shard-model

Notes:
(1) for Falcon quantizations, "--config-file <CONFIG_FILE>" is needed and example of <CONFIG_FILE>: "utils/model_config/tiiuae_falcon-40b_config.json".
(2) for GPT-NEOX quantizations, using "--int8" instead of "--int8-bf16-mixed", and "--dtype float32" for accuracy concerns.
(3) by default, we use "--shard-model" for better memory usage, if your model is already sharded, please remove "--shard-model"
(4) By default, generations are based on "beam search", and beam size = 4. For beam size = 1, please add "--greedy"

```

# Advanced Usage
## Single Instance Performance
```bash
# Get prompt file to the path of scripts
export WORK_DIR=./
cd single_instance
mv PATH/TO/prompt.json ./
# bfloat16 benchmark
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run_generation.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex --deployment-mode

# quantization benchmark
#To run quantization performance, you need to firstly get the quantized model with the following step (1) and then run the performance benchmark with the following step (2)
## (1) Do quantization to get the quantized model 
## note: llama/gptj we have both IPEX smooth quant and weight-only-quantization, while for rest models, we recommend weight-only-quantization
mkdir saved_results
# Option a. For smooth quantiztion
python run_quantization.py --ipex-smooth-quant --alpha <Tuned alpha for specific models> --output-dir "saved_results" --int8-bf16-mixed -m <MODEL_ID>
# For the best alpha values (range [0, 1.0], float) tuned for specific models, we verified good accuracy: "EleutherAI/gpt-j-6b" with alpha=1.0, "meta-llama/Llama-2-7b-chat-hf" with alpha=0.8.
# For other variant models, suggest using default alpha=0.5, and could be further tuned in the range [0, 1.0]. (suggest step_size of 0.05)

# Option b. For weight only quantization
python run_quantization.py --ipex-weight-only-quantization --output-dir "saved_results" --int8-bf16-mixed -m <MODEL_ID>

## (2) Run quantization performance test
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_quantization.py -m <MODEL_ID> --quantized-model-path "./saved_results/best_model.pt" --benchmark --int8-bf16-mixed

# Notes:
# (1) for Falcon quantizations, "--config-file <CONFIG_FILE>" is needed and example of <CONFIG_FILE>: "utils/model_config/tiiuae_falcon-40b_config.json".
# (2) for GPT-NEOX quantizations, using "--int8" instead of "--int8-bf16-mixed", and "--dtype float32" for accuracy concerns.
```

## Weight only quantization with low precision checkpoint (Experimental)
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
python single_instance/run_quantization.py --ipex-weight-only-quantization --output-dir "saved_results" --int8-bf16-mixed -m <MODEL_ID> --low-precision-checkpoint "saved_results/gptq_checkpoint.pt"

# Step 3: Run quantized model for latency benchmark
# For GPT-NEOX, use --int8 instead of --int8-bf16-mixed
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python single_instance/run_quantization.py -m <MODEL_ID> --quantized-model-path "./saved_results/best_model.pt" --benchmark --int8-bf16-mixed
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

## Single Instance Accuracy
```bash
Accuracy test {TASK_NAME}, choice in this [link](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md), by default we use "lambada_standard"

# bfloat16
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run_accuracy.py --accuracy-only -m <MODEL_ID> --dtype bfloat16 --ipex --jit --tasks {TASK_NAME}

# Quantization as a performance part
# (1) Do quantization to get the quantized model as mentioned above
# (2) Run int8 accuracy test (note that GPT-NEOX please remove --int8-bf16-mixed)
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_accuracy.py --model <MODEL ID> --quantized-model-path "./saved_results/best_model.pt" --dtype int8 --accuracy-only --jit --int8-bf16-mixed --tasks {TASK_NAME}
```
## Shard model for Distributed Performance
```
# We need to make sure the model is well shard before we test Distributed Performance with DeepSpeed (saving memory usage purpose)
export WORK_DIR=./
cd utils
python create_shard_model.py -m <MODEL ID>  --save-path <SHARD MODEL NEW PATH>
# After sharding the model, using -m <SHARD MODEL NEW PATH> in later tests.
```
## Distributed Performance with DeepSpeed (autoTP)
```bash
unset KMP_AFFINITY

# Get prompt file to the path of scripts
export WORK_DIR=./
cd distributed
mv PATH/TO/prompt.json ./

# Run GPTJ/LLAMA/OPT/Falcon/Bloom/CodeGen/Baichuan with bfloat16 DeepSpeed
deepspeed --bind_cores_to_rank run_generation_with_deepspeed.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex --deployment-mode

# Run GPT-NeoX with ipex weight only quantization
deepspeed --bind_cores_to_rank run_generation_with_deepspeed.py --benchmark -m EleutherAI/gpt-neox-20b --dtype float32 --ipex --ipex-weight-only-quantization --deployment-mode
```

## Distributed Accuracy with DeepSpeed (autoTP)
```bash
# Run distributed accuracy with 2 ranks of one node for bfloat16 with ipex and jit 
source ${ONECCL_DIR}/build/_install/env/setvars.sh

export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so:${CONDA_PREFIX}/lib/libtcmalloc.so
export LD_LIBRARY_PATH=${ONECCL_DIR}/lib:$LD_LIBRARY_PATH
unset KMP_AFFINITY

deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model <MODEL_ID> --dtype bfloat16 --ipex --jit --tasks <TASK_NAME> --accuracy-only

# with weight only quantization

deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py  --model <MODEL_ID> --int8-bf16-mixed --ipex --jit --tasks <TASK_NAME> --accuracy-only --ipex-weight-only-quantization

```
