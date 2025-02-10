# 1. LLM Optimization Overview

`ipex.llm` provides dedicated optimization for running Large Language Models (LLM) faster, including technical points like paged attention, ROPE fusion, etc.
And a set of data types are supported for various scenarios, including FP32, BF16, Smooth Quantization INT8, Weight Only Quantization INT8/INT4 (prototype).

<br>

# 2. Environment Setup

**Note**: Please be aware that in order to enable the latest optimizations for MoE models (DeepSeek, Mixtral, etc.) in `DeepSpeed`,
we are setting a different argument for `env_setup.sh` in IPEX v2.6.0+cpu comparing with previous versions,
in order to build `DeepSpeed` from source code with a recent commit.

## 2.1 [RECOMMENDED] Docker-based environment setup with pre-built wheels

```bash
# Get the Intel® Extension for PyTorch\* source code
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout v2.6.0+cpu
git submodule sync
git submodule update --init --recursive

# Build an image with the provided Dockerfile by installing from Intel® Extension for PyTorch\* prebuilt wheel files
# To have a custom ssh server port for multi-nodes run, please add --build-arg PORT_SSH=<CUSTOM_PORT> ex: 2345, otherwise use the default 22 SSH port
DOCKER_BUILDKIT=1 docker build -f examples/cpu/llm/Dockerfile --build-arg PORT_SSH=2345 -t ipex-llm:2.6.0 .

# Run the container with command below
docker run --rm -it --privileged -v /dev/shm:/dev/shm ipex-llm:2.6.0 bash

# When the command prompt shows inside the docker container, enter llm examples directory
cd llm

# Activate environment variables
# set bash script argument to "inference" or "fine-tuning" for different usages
source ./tools/env_activate.sh [inference|fine-tuning]
```

## 2.2 Conda-based environment setup with pre-built wheels

```bash
# Get the Intel® Extension for PyTorch\* source code
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout v2.6.0+cpu
git submodule sync
git submodule update --init --recursive

# GCC 12.3 is required. Installation can be taken care of by the environment configuration script.
# Create a conda environment
conda create -n llm python=3.10 -y
conda activate llm

# Setup the environment with the provided script
cd examples/cpu/llm
bash ./tools/env_setup.sh 15

# Activate environment variables
# set bash script argument to "inference" or "fine-tuning" for different usages
source ./tools/env_activate.sh [inference|fine-tuning]
```

## 2.3 Docker-based environment setup with compilation from source

```bash
# Get the Intel® Extension for PyTorch\* source code
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout v2.6.0+cpu
git submodule sync
git submodule update --init --recursive

# Build an image with the provided Dockerfile by compiling Intel® Extension for PyTorch\* from source
# To have a custom ssh server port for multi-nodes run, please add --build-arg PORT_SSH=<CUSTOM_PORT> ex: 2345, otherwise use the default 22 SSH port
docker build -f examples/cpu/llm/Dockerfile --build-arg COMPILE=ON --build-arg PORT_SSH=2345 -t ipex-llm:2.6.0 .

# Run the container with command below
docker run --rm -it --net host --privileged -v /dev/shm:/dev/shm ipex-llm:2.6.0 bash

# When the command prompt shows inside the docker container, enter llm examples directory
cd llm

# Activate environment variables
# set bash script argument to "inference" or "fine-tuning" for different usages
source ./tools/env_activate.sh [inference|fine-tuning]
```

## 2.4 Conda-based environment setup with compilation from source

```bash
# Get the Intel® Extension for PyTorch\* source code
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout v2.6.0+cpu
git submodule sync
git submodule update --init --recursive

# GCC 12.3 is required. Installation can be taken care of by the environment configuration script.
# Create a conda environment
conda create -n llm python=3.10 -y
conda activate llm

# Setup the environment with the provided script
cd examples/cpu/llm
bash ./tools/env_setup.sh 8

# Activate environment variables
# set bash script argument to "inference" or "fine-tuning" for different usages
source ./tools/env_activate.sh [inference|fine-tuning]
```

## 2.3 [Optional] Setup for Running Jupyter Notebooks

After setting up your docker or conda environment, you may follow these additional steps to setup and run Jupyter Notebooks. The port number can be changed.

### 2.3.1 Jupyter Notebooks for Docker-based Environments

```bash
# Install dependencies
pip install notebook matplotlib

# Launch Jupyter Notebook
jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root
```

1. Open up a web browser with the given URL and token.
2. Open the notebook.
3. Run all cells. 

### 2.3.2 Jupyter Notebooks for Conda-based Environments

```bash
# Install dependencies
pip install notebook ipykernel matplotlib

# Register ipykernel with Conda
python -m ipykernel install --user --name=IPEX-LLM

# Launch Jupyter Notebook
jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root
```

1. Open up a web browser with the given URL and token.
2. Open the notebook.
3. Change your Jupyter Notebook kernel to IPEX-LLM.
4. Run all cells. 

<br>

*Note*: In `env_setup.sh` script a `prompt.json` file is downloaded, which provides prompt samples with pre-defined input token lengths for benchmarking.
For **Llama-3 models** benchmarking, the users need to download a specific `prompt.json` file, overwriting the original one.

```bash
wget -O prompt.json https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/prompt-3.json
```

The original `prompt.json` file can be restored from the repository if needed.

```bash
wget https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/prompt.json
```

<br>

# 3. How To Run LLM with ipex.llm

Inference and fine-tuning are supported in individual directories.

For inference example scripts, visit the [inference](./inference/) directory.

For fine-tuning example scripts, visit the [fine-tuning](./fine-tuning/) directory.
