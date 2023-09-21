# Text Generation
We provide the inference benchmarking scripts for large language models text generation.<br/>
Support large language models, such as GPT-J, LLaMA, GPT-Neox, OPT, Falcon.<br/>
The scripts include both single instance and distributed (DeepSpeed) use cases.<br/>
The scripts cover model generation inference with low precions cases for different models with best perf and accuracy (bf16 AMP，static quantization and weight only quantization).<br/>

## Setup
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

# [Optional] install neural-compressor for GPT-J static quantization only
pip install neural-compressor==2.2

# [Optional] The following is only for DeepSpeed case
#Install oneccl-bind-pt(also named torch-ccl)
git clone https://github.com/intel/torch-ccl.git
cd torch-ccl && git checkout ccl_torch_dev_0905
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

## Supported Model List
```
<MODEL ID> in
(1) "EleutherAI/gpt-j-6b" (model id from transformers Hub)
(2) "EleutherAI/gpt-neox-20b" (model id from transformers Hub)
(3) Llama 2 Model directory path
(4) "facebook/opt-30b" (model id from transformers Hub)
(5) "tiiuae/falcon-40b" (model id from transformers Hub)
Note: Above models are well supported with all optimizations like indirect access KV cache, fused ROPE, and prepacked TPP Linear (fp32/bf16). For other LLM models, we could still run with this BKC, and may get parts of optimizations like prepacked TPP Linear (fp32/bf16), and we are working in progress to cover all optimizations to these other LLM models, which will expand the model list above.
```
* Llama 2 model conversion steps:
    1) [transformers conversion tool](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py) (Verified [meta-llama/Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat) and [meta-llama/Llama-2-13b-chat](https://huggingface.co/meta-llama/Llama-2-13b-chat)).
    2) Follow [instructions](https://github.com/facebookresearch/llama#access-on-hugging-face) to download model files for conversion.
    3) Decompress the downloaded model file.
    4) Follow [instructions](https://github.com/facebookresearch/llama-recipes#model-conversion-to-hugging-face) to convert the model.
    5) Launch example scripts with the place holder <MODEL_ID> substituted by the --output_dir argument value of the conversion script.
* For Falcon model from remote hub, we need to modify the config.json to use the modeling_falcon.py in transformers. Therefore, in the following scripts, we need to pass an extra configuration file like "--config-file=model_config/tiiuae_falcon-40b_config.json". This is optional for BF16, but must for quantization benchmark.

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

## GPT-J quantization
python run_gpt-j_quantization.py --ipex-smooth-quant --output-dir "saved_results" --int8-bf16-mixed -m <GPTJ MODEL_ID>
## Llama 2 quantization
python run_llama_quantization.py --ipex-smooth-quant --output-dir "saved_results" --int8-bf16-mixed -m <LLAMA MODEL_ID>
## GPT-NEOX quantization
python run_gpt-neox_quantization.py --ipex-weight-only-quantization --output-dir "saved_results" --int8 -m <GPT-NEOX MODEL_ID>
## Falcon quantization (example of config-file: utils/model_config/tiiuae_falcon-40b_config.json)
python run_falcon_quantization.py --ipex-weight-only-quantization --output-dir "saved_results"  --int8-bf16-mixed -m <FALCON MODEL_ID> --config-file <CONFIG_FILE>
## OPT quantization
python run_opt_quantization.py --ipex-weight-only-quantization --output-dir "saved_results"  --int8-bf16-mixed -m <OPT MODEL_ID> 

## (2) Run quantization performance test (note that GPT-NEOX uses --int8 instead of --int8-bf16-mixed)
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_<MODEL>_quantization.py -m <MODEL_ID> --quantized-model-path "./saved_results/best_model.pt" --benchmark --int8-bf16-mixed

```
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

# Run GPTJ/LLAMA/OPT/Falcon with bfloat16 DeepSpeed
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
