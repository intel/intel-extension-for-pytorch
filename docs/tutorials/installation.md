Installation Guide
==================

## System Requirements

### Hardware Requirement

Verified Hardware Platforms:
 - Intel® Data Center GPU Flex Series 170
 - Intel® Data Center GPU Max Series
 - Intel® Arc™ A-Series Graphics (with limited validation)

### Software Requirements

- Ubuntu 22.04 (64-bit)
- Intel GPU Drivers 
  - Intel® Data Center GPU Flex Series [TBD](https://dgpu-docs.intel.com/releases/stable_419_40_20220914.html)
  - Intel® Data Center GPU Max Series [TBD](https://dgpu-docs.intel.com/releases/stable_419_40_20220914.html)
  - Intel® Arc™ A-Series Graphics [TBD](https://dgpu-docs.intel.com/releases/stable_419_40_20220914.html)
- Intel® oneAPI Base Toolkit 2023.0
- Python 3.7-3.10

## PyTorch-Intel® Extension for PyTorch\* Version Mapping

Intel® Extension for PyTorch\* has to work with a corresponding version of PyTorch. Here are the PyTorch versions that we support and the mapping relationship:

|PyTorch Version|Extension Version|
|--|--|
|[v1.13.\*](https://github.com/pytorch/pytorch/tree/v1.13.0) (patches needed)|[v1.13.\*](https://github.com/intel/intel-extension-for-pytorch/tree/v1.13.10+xpu)|
|[v1.10.\*](https://github.com/pytorch/pytorch/tree/v1.10.0) (patches needed)|[v1.10.\*](https://github.com/intel/intel-extension-for-pytorch/tree/v1.10.200+gpu)|

## Preparations

### Install Intel GPU Driver

|Release|OS|Intel GPU|Install Intel GPU Driver|
|-|-|-|-|
|v1.0.0|Ubuntu 20.04|Intel® Data Center GPU Flex Series| Refer to the [Installation Guides](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-focal-dc.html) for the latest driver installation. If installing the verified Intel® Data Center GPU Flex Series [419.40](https://dgpu-docs.intel.com/releases/stable_419_40_20220914.html), use a specific version for component package names, such as `sudo apt-get install intel-opencl-icd=22.28.23726.1+i419~u20.04`|

### Install oneAPI Base Toolkit

Please refer to [Install oneAPI Base Toolkit Packages](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#base-kit).

Need to install components of Intel® oneAPI Base Toolkit:
 - Intel® oneAPI DPC++ Compiler
 - Intel® oneAPI Math Kernel Library (oneMKL)

Default installation location *{ONEAPI_ROOT}* is `/opt/intel/oneapi` for root account, `${HOME}/intel/oneapi` for other accounts.

**_NOTE:_** You need to activate oneAPI environment when using Intel® Extension for PyTorch\* on Intel GPU.

```bash
source {ONEAPI_ROOT}/setvars.sh
```

## Install via wheel files

Prebuilt wheel files availability matrix for Python versions:

| Extension Version | Python 3.6 | Python 3.7 | Python 3.8 | Python 3.9 | Python 3.10 |
| :--: | :--: | :--: | :--: | :--: | :--: |
| 1.13.10+xpu |  | ✔️ | ✔️ | ✔️ | ✔️ |
| 1.10.200+gpu | ✔️ | ✔️ | ✔️ | ✔️ |  |

### Install PyTorch and TorchVision

```bash
python -m pip install torch==1.13.0a0 torchvision==0.14.0 -f https://developer.intel.com/ipex-whl-stable-xpu
```

**Note:** Installation of TorchVision is optional.

### Install Numpy

Numpy is required to work with PyTorch\*. Verified numpy versions differ according to python versions.

```bash
python -m pip install numpy==1.19.5  # for Python 3.6
python -m pip install numpy==1.21.6  # for Python 3.7
python -m pip install numpy==1.23.4  # for Python 3.8 and 3.9
```

### Install torchaudio (Optional)

Intel® Extension for PyTorch\* doesn't depend on torchaudio. If you need TorchAudio, please follow the [instructions](https://github.com/pytorch/audio/tree/v0.13.0#from-source) to compile it from source. According to torchaudio-pytorch dependency table, torchaudio 0.13.0 is recommended.

### Install Intel® Extension for PyTorch\*

```bash
python -m pip install intel_extension_for_pytorch==1.13.10+xpu -f https://developer.intel.com/ipex-whl-stable-xpu
```

## Install via compiling from source

### Download source code of PyTorch and Intel® Extension for PyTorch\*:

```bash
$ git clone https://github.com/pytorch/pytorch.git
$ cd pytorch
$ git checkout v1.13.0

$ git clone https://github.com/intel/intel-extension-for-pytorch.git 
$ cd intel-extension-for-pytorch
$ git checkout v1.13.10+xpu
```

### Install PyTorch:

```bash
$ cd pytorch
$ git apply ${intel_extension_for_pytorch_directory}/torch_patches/*.patch 
$ git submodule sync
$ git submodule update --init --recursive
$ pip install -r requirements.txt
$ source {ONEAPI_ROOT}/setvars.sh
$ python setup.py bdist_wheel
$ pip install dist/*.whl
```

### Configure the AOT (Optional)

Please refer to [AOT documentation](./AOT.md) for how to configure `USE_AOT_DEVLIST`.

```bash
$ export USE_AOT_DEVLIST='ats-m150,pvc'
```

### Install Intel® Extension for PyTorch\*:

```bash
$ cd intel-extension-for-pytorch
$ git submodule sync
$ git submodule update --init --recursive
$ pip install -r requirements.txt
$ source {ONEAPI_ROOT}/setvars.sh # If you have sourced the oneAPI environment when compiling PyTorch, please skip this step.
$ python setup.py bdist_wheel
$ pip install dist/*.whl
```
