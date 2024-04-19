# Intel® Extension for PyTorch* Large Language Model (LLM) Feature Get Started For Llama 3 models

Intel® Extension for PyTorch* provides dedicated optimization for running Llama 3 models on Intel® Core™ Ultra Processors with Intel® Arc™ Graphics, including weight-only quantization (WOQ), Rotary Position Embedding fusion, etc. You are welcomed to have a try with these optimizations on Intel® Core™ Ultra Processors with Intel® Arc™ Graphics. This document shows how to run Llama 3 with a preview version of Intel® Extension for PyTorch*.

# 1. Environment Setup

## 1.1 Conda-based environment setup with pre-built wheels on Windows 11 Home

```bash
# Install Visual Studio 2022
https://visualstudio.microsoft.com/zh-hans/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2022&source=VSLandingPage&cid=2030&passive=false

# Install Intel® oneAPI Base Toolkit 2024.1
https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=window

# Install Intel® Core™ Ultra Processors with Intel® Arc™ Graphics driver
https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html

# Create a conda environment (pre-built wheel only available with python=3.9)
conda create -n llm python=3.9 -y
conda activate llm
conda install libuv

# Set environment variable
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

# Install PyTorch*
pip install https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_dev/xpu/torch-2.1.0a0%2Bgit04048c2-cp39-cp39-win_amd64.whl

# Install Intel® Extension for PyTorch*
pip install https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_dev/xpu/intel_extension_for_pytorch-2.1.30%2Bgit03c5535-cp39-cp39-win_amd64.whl

# Install Intel® Extension for Transformers*
git clone https://github.com/intel/intel-extension-for-transformers.git intel-extension-for-transformers -b xpu_lm_head 
cd intel-extension-for-transformers 
pip install -v .

# Install dependencies
pip install transformers==4.35
pip install huggingface_hub==0.22
pip install lm_eval==0.4.2 --no-deps
pip install accelerate datasets diffusers
```


# 2. How To Run Llama 3

**Intel® Extension for PyTorch\* provides a single script `run_generation_gpu_woq_for_llama.py` to facilitate running generation tasks as below:**

```
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout dev/llama-int4
cd examples/gpu/inference/python/llm
```

| Key args of run_generation_gpu_woq_for_llama.py | Notes |
|---|---|
| model id | "--model" or "-m" to specify the <LLAMA3_MODEL_ID_OR_LOCAL_PATH>, it is model id from Huggingface or downloaded local path |
| benchmark | "--benchmark" to specify whether to generate sentences using model.generate |
| accuracy | "--accuracy" to specify whether to use the dataset to detect accuracy |
| output tokens | default: 32, use "--max-new-tokens" to choose any other size |
| token latency |  enable "--profile_token_latency" to print out the first or next token latency |
| generation iterations |  use "--iters" and "--num-warmup" to control the repeated iterations of generation, default: 10-iter/3-warmup |

## 2.1 Usage of running Llama 3 models

### 2.1.1 INT4 WOQ Model
LLM quantization procedure is heavily constrained by client memory and computation capabilities. If you plan to create an INT4 WOQ model by your own, please use a powerful machine, for example, Intel® Xeon® Server, then execute the following steps. Otherwise, it is highly recommended waiting for INT4 model available in HuggingFace Model Hub.

Environment installation:
```
cd ${YourWorkSpace}
git clone https://github.com/intel/neural-compressor.git neural-compressor
cd neural-compressor
git checkout xpu_export
python setup.py develop
cd ..
git clone https://github.com/intel/intel-extension-for-transformers.git intel-extension-for-transformers
cd intel-extension-for-transformers
git checkout xpu_int4
python setup.py develop
cd ..
git clone https://github.com/intel/auto-round.git auto-round
cd auto-round
git checkout lm-head-quant
python setup.py develop
cd ..
pip install schema==0.7.5
```

Command to quantize:

```
cd ${YourWorkSpace}/intel-extension-for-transformers/examples/huggingface/pytorch/text-generation/quantization

python run_generation_gpu_woq.py \
    --model $model_path \
    --woq --woq_algo AutoRound  \
    --use_quant_input \
    --calib_iters 200  \
    --lr 5e-3 \
    --minmax_lr 1e-2 \
    --output_dir llama3_all_int4  \
    --nsamples 512
```

The int4 model is saved in folder ~/llama3_all_int4.

### 2.1.2 Measure Llama 3 WOQ INT4 Performance on Windows 11 Home

- Command:
```bash
unset LLM_ACC_TEST
python run_generation_gpu_woq_for_llama.py --model ${PATH/TO/MODEL} --benchmark --profile_token_latency
*Note:* replace ${PATH/TO/MODEL} with actual Llama 3 INT4 model local path
```

### 2.1.3 Validate Llama 3 WOQ INT4 Accuracy on Windows 11 Home

- Command:
```bash
set LLM_ACC_TEST=1 
python run_generation_gpu_woq_for_llama.py --model ${PATH/TO/MODEL} --accuracy --task "piqa"
*Note:* replace ${PATH/TO/MODEL} with actual Llama 3 INT4 model local path
```

## Miscellaneous Tips
Intel® Extension for PyTorch* also provides dedicated optimization for many other Large Language Models (LLM), which covers a set of data types for supporting various scenarios. For more details, please check [Large Language Models (LLM) Optimizations Overview](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/llm.html). To replicate Llama 3 performance numbers on Intel ARC A770, please take advantage of [IPEX-LLM](https://github.com/intel-analytics/ipex-llm).
