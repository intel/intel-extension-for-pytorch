# Text Generation
We provide the inference benchmarking script `run_generation.py` for large language models text generation.<br/>
Support large language models, such as GPT-J, LLaMA, GPT-Neox.<br/>
And script `run_generation_with_deepspeed.py` for distributed with DeepSpeed.<br/>
And script `run_model_int8.py` for int8.<br/>

## Setup
```bash
WORK_DIR=$PWD
# GCC 12.3 is required, please set it firstly
# Create environment (conda recommended)
conda create -n llm python=3.9 -y
# install deps
conda install gcc=12.3 gxx=12.3 cxx-compiler -c conda-forge -y
conda install cmake ninja mkl mkl-include -y
conda install gperftools -c conda-forge -y

# Install PyTorch
python -m pip install torch==2.1.0.dev20230711+cpu torchvision==0.16.0.dev20230711+cpu torchaudio==2.1.0.dev20230711+cpu --index-url https://download.pytorch.org/whl/nightly/cpu

# Install IPEX with semi-compiler, require gcc 12.3
rm -rf llvm-project && mkdir llvm-project && cd llvm-project
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/cmake-16.0.6.src.tar.xz
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/llvm-16.0.6.src.tar.xz
tar -xf cmake-16.0.6.src.tar.xz && mv cmake-16.0.6.src cmake
tar -xf llvm-16.0.6.src.tar.xz && mv llvm-16.0.6.src llvm
mkdir build && cd build
cmake ../llvm -DCMAKE_INSTALL_PREFIX=${PWD}/_install/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
make install -j$(nproc)
ln -s ${PWD}/_install/llvm/bin/llvm-config ${CONDA_PREFIX}/bin/llvm-config-13
cd ../../

git clone --branch llm_feature_branch https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu
cd frameworks.ai.pytorch.ipex-cpu
git submodule sync && git submodule update --init --recursive
export DNNL_GRAPH_BUILD_COMPILER_BACKEND=1
export CXXFLAGS="${CXXFLAGS} -D__STDC_FORMAT_MACROS"
python setup.py install
cd ../

# Install transformers
pip install transformers==4.28.1
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

# [Optional] install neural-compressor for GPT-J INT8 only
pip install neural-compressor==2.2

# [Optional] The following is only for DeepSpeed case
git clone https://github.com/delock/DeepSpeedSYCLSupport
cd DeepSpeedSYCLSupport
git checkout gma/run-opt-branch
python -m pip install -r requirements/requirements.txt
python setup.py install
cd ../
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
Note: Above models are well supported with all optimizations like indirect access KV cache, fused ROPE, and prepacked TPP Linear (fp32/bf16). For other LLM models (like OPT, Bloom...), we could still run with this BKC, and may get parts of optimizations like prepacked TPP Linear (fp32/bf16), and we are working in progress to cover all optimizations to these other LLM models, which will expand the model list above.
```
* Llama 2 model conversion steps:
    1) [transformers conversion tool](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py) (Verified [meta-llama/Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat) and [meta-llama/Llama-2-13b-chat](https://huggingface.co/meta-llama/Llama-2-13b-chat)).
    2) Follow [instructions](https://github.com/facebookresearch/llama#access-on-hugging-face) to download model files for conversion.
    3) Decompress the downloaded model file.
    4) Follow [instructions](https://github.com/facebookresearch/llama-recipes#model-conversion-to-hugging-face) to convert the model.
    5) Launch example scripts with the place holder <MODEL_ID> substituted by the --output_dir argument value of the conversion script.


## Single Instance Performance
```bash
# Get prompt file to the path of scripts
mv PATH/TO/prompt.json WORK_DIR

# bfloat16 benchmark
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run_generation.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex --jit

# int8 benchmark
## (1) Do quantization to get the quantized model
mkdir saved_results

## GPT-J quantization
python run_gpt-j_int8.py --ipex-smooth-quant --lambada --output-dir "saved_results" --jit --int8-bf16-mixed -m <GPTJ MODEL_ID>
## Llama 2 quantization
python run_llama_int8.py --ipex-smooth-quant --lambada --output-dir "saved_results" --jit --int8-bf16-mixed -m <LLAMA MODEL_ID>
## GPT-NEOX quantization
python run_gpt-neox_int8.py --ipex-weight-only-quantization --lambada --output-dir "saved_results" --jit --int8 -m <GPT-NEOX MODEL_ID>

## (2) Run int8 performance test (note that GPT-NEOX uses --int8 instead of --int8-bf16-mixed)
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_<MODEL>_int8.py -m <MODEL_ID> --quantized-model-path "./saved_results/best_model.pt" --benchmark --jit --int8-bf16-mixed
```
## Single Instance Accuracy
```bash
Accuracy test {TASK_NAME}, choice in this [link](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md), by default we use "lambada_openai"

# bfloat16
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run_accuracy.py --accuracy-only -m <MODEL_ID> --dtype bfloat16 --ipex --jit --tasks {TASK_NAME}

# Quantization as a performance part
# (1) Do quantization to get the quantized model as mentioned above
# (2) Run int8 accuracy test (note that GPT-NEOX please remove --int8-bf16-mixed)
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_accuracy.py -m <MODEL ID> --quantized-model-path "./saved_results/best_model.pt" --dtype int8 --accuracy-only --jit --int8-bf16-mixed --tasks {TASK_NAME}
```

## Distributed Performance with DeepSpeed (autoTP)
```bash
export DS_SHM_ALLREDUCE=1
unset KMP_AFFINITY

# Get prompt file to the path of scripts
mv PATH/TO/prompt.json WORK_DIR

# Run GPTJ/LLAMA with bfloat16  DeepSpeed
deepspeed --bind_cores_to_rank run_generation_with_deepspeed.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex --jit

# Run GPT-NeoX with ipex weight only quantization
deepspeed --bind_cores_to_rank run_generation_with_deepspeed.py --benchmark -m EleutherAI/gpt-neox-20b --dtype float32 --ipex --jit --ipex-weight-only-quantization
```
