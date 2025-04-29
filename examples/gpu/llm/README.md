# LLM Optimization Overview

Here you can find examples for large language models (LLM) text generation. These scripts:

> [!NOTE]  
> New Llama models like Llama3.2-1B, Llama3.2-3B and Llama3.3-7B are also supported from release v2.7.10+xpu.

- Include both inference/finetuning(lora)/bitsandbytes(qlora-finetuning).
- Include both single instance and distributed (DeepSpeed) use cases for FP16 optimization.
- Support Llama, GPT-J, Qwen, OPT, Bloom model families and some other models such as Baichuan2-13B and Phi3-mini. 
- Cover model generation inference with low precision cases for different models with best performance and accuracy (fp16 AMP and weight only quantization)

## Environment Setup

### [Recommended] Docker-based environment setup with prebuilt wheel files

```bash
# Get the Intel® Extension for PyTorch* source code
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout release/xpu/2.7.10
git submodule sync
git submodule update --init --recursive

# Build an image with the provided Dockerfile by installing Intel® Extension for PyTorch* with prebuilt wheels
docker build -f examples/gpu/llm/Dockerfile -t ipex-llm:xpu .

# Run the container with command below
docker run -it --rm --privileged -v /dev/dri/by-path:/dev/dri/by-path ipex-llm:xpu bash

# When the command prompt shows inside the docker container, enter llm examples directory
cd llm

# Activate environment variables
source ./tools/env_activate.sh  [inference|fine-tuning|bitsandbytes]
# on Windows, use env_activate.bat instead
call .\tools\env_activate.bat [inference|fine-tuning|bitsandbytes]
```
### Conda-based environment setup with prebuilt wheel files

Make sure the driver packages are installed. Refer to [Installation Guide](https://intel.github.io/intel-extension-for-pytorch/#installation?platform=gpu&version=v2.7.10%2Bxpu&os=linux%2Fwsl2&package=pip).

```bash

# Get the Intel® Extension for PyTorch* source code
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout release/xpu/2.7.10
git submodule sync
git submodule update --init --recursive

# Make sure you have GCC >= 11 is installed on your system.
# Create a conda environment
conda create -n llm python=3.10 -y
conda activate llm
# Setup the environment with the provided script
cd examples/gpu/llm
# If you want to install Intel® Extension for PyTorch\* with prebuilt wheels, use the commands below:
python ./tools/env_setup.py --setup --deploy
conda deactivate
conda activate llm
source ./tools/env_activate.sh  [inference|fine-tuning|bitsandbytes]
# on Windows, use env_activate.bat instead
call .\tools\env_activate.bat [inference|fine-tuning|bitsandbytes]
```

### Docker-based environment setup with compilation from source

```bash
# Get the Intel® Extension for PyTorch* source code
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout release/xpu/2.7.10
git submodule sync
git submodule update --init --recursive

# Build an image with the provided Dockerfile by compiling Intel® Extension for PyTorch* from source
docker build -f examples/gpu/llm/Dockerfile --build-arg COMPILE=ON -t ipex-llm:xpu .

# Run the container with command below
docker run -it --rm --privileged -v /dev/dri/by-path:/dev/dri/by-path ipex-llm:xpu bash

# When the command prompt shows inside the docker container, enter llm examples directory
cd llm

# Activate environment variables
source ./tools/env_activate.sh [inference|fine-tuning|bitsandbytes]
# on Windows, use env_activate.bat instead
call .\tools\env_activate.bat [inference|fine-tuning|bitsandbytes]
```

### Conda-based environment setup with compilation from source

Make sure the driver and Base Toolkit are installed. Refer to [Installation Guide](https://intel.github.io/intel-extension-for-pytorch/#installation?platform=gpu&version=v2.3.110%2Bxpu&os=linux%2Fwsl2&package=source).

```bash

# Get the Intel® Extension for PyTorch* source code
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout release/xpu/2.7.10
git submodule sync
git submodule update --init --recursive

# Make sure you have GCC >= 11 is installed on your system.
# Create a conda environment
conda create -n llm python=3.10 -y
conda activate llm
# Setup the environment with the provided script
cd examples/gpu/llm
# If you want to install Intel® Extension for PyTorch\* from source, use the commands below:
# e.g. python ./tools/env_setup.py --setup --install-pytorch compile --aot pvc --oneapi-root-dir /opt/intel/oneapi --deploy
python ./tools/env_setup.py --setup --install-pytorch compile --aot <AOT> --oneapi-root-dir <ONEAPI_ROOT_DIR> --deploy

conda deactivate
conda activate llm
source ./tools/env_activate.sh [inference|fine-tuning|bitsandbytes]
# on Windows, use env_activate.bat instead
call .\tools\env_activate.bat [inference|fine-tuning|bitsandbytes]
```

where <br />
- `AOT` is a text string to enable `Ahead-Of-Time` compilation for specific GPU models. For example 'pvc,ats-m150' for the Platform Intel® Data Center GPU Max Series, Intel® Data Center GPU Flex Series and Intel® Arc™ A-Series Graphics (A770). Check [tutorial](../../../docs/tutorials/technical_details/AOT.md) for details.<br />


<br />
 
## How To Run LLM with ipex.llm

Inference and fine-tuning are supported in individual directories.

For inference example scripts, visit the [inference](./inference/) directory.

For fine-tuning example scripts, visit the [fine-tuning](./fine-tuning/) directory.

For fine-tuning with quantized model, visit the [bitsandbytes](./bitsandbytes/) directory.

