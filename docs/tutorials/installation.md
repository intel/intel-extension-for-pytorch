Installation Guide
==================

## System Requirements

### Hardware Requirement

Verified Hardware Platforms:
 - Intel® Data Center GPU Flex Series 170
 - Intel® Data Center GPU Max Series
 - Intel® Arc™ A-Series GPUs (Experimental support)

### Software Requirements

- OS & Intel GPU Drivers

|Hardware|OS|Driver|
|-|-|-|
|Intel® Data Center GPU Flex Series|Ubuntu 22.04, Red Hat 8.6|[Stable 540](https://dgpu-docs.intel.com/releases/stable_540_20221205.html)|
|Intel® Data Center GPU Max Series|Red Hat 8.6, Sles 15sp3/sp4|[Stable 540](https://dgpu-docs.intel.com/releases/stable_540_20221205.html)|
|Intel® Arc™ A-Series Graphics|Ubuntu 22.04|[Stable 540](https://dgpu-docs.intel.com/releases/stable_540_20221205.html)|
|Intel® Arc™ A-Series Graphics|Windows 11 or Windows 10 21H2 (via WSL2)|[for Windows 11 or Windows 10 21H2](https://www.intel.com/content/www/us/en/download/726609/intel-arc-graphics-windows-dch-driver.html)|

- Intel® oneAPI Base Toolkit 2023.0
- Python 3.7-3.10
- Verified with GNU GCC 11

## Preparations

### Install Intel GPU Driver

|OS|Instructions for installing Intel GPU Driver|
|-|-|
|Linux\*|Refer to the [Installation Guides](https://dgpu-docs.intel.com/installation-guides/index.html) for the latest driver installation for individual Linux\* distributions. When installing the verified [Stable 540](https://dgpu-docs.intel.com/releases/stable_540_20221205.html) driver, use a specific version for component package names, such as `sudo apt-get install intel-opencl-icd=22.43.24595.35`|
|Windows 11 or Windows 10 21H2 (via WSL2)|Please download drivers for Intel® Arc™ A-Series [for Windows 11 or Windows 10 21H2](https://www.intel.com/content/www/us/en/download/726609/intel-arc-graphics-windows-dch-driver.html). Please note that you would have to follow the rest of the steps in WSL2, but the drivers should be installed on Windows|

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

## PyTorch-Intel® Extension for PyTorch\* Version Mapping

Intel® Extension for PyTorch\* has to work with a corresponding version of PyTorch. Here are the PyTorch versions that we support and the mapping relationship:

|PyTorch Version|Extension Version|
|--|--|
|[v1.13.\*](https://github.com/pytorch/pytorch/tree/v1.13.0) (patches needed)|[v1.13.\*](https://github.com/intel/intel-extension-for-pytorch/tree/v1.13.10+xpu)|
|[v1.10.\*](https://github.com/pytorch/pytorch/tree/v1.10.0) (patches needed)|[v1.10.\*](https://github.com/intel/intel-extension-for-pytorch/tree/v1.10.200+gpu)|

## Install via wheel files

Prebuilt wheel files availability matrix for Python versions:

| Extension Version | Python 3.6 | Python 3.7 | Python 3.8 | Python 3.9 | Python 3.10 |
| :--: | :--: | :--: | :--: | :--: | :--: |
| 1.13.10+xpu |  | ✔️ | ✔️ | ✔️ | ✔️ |
| 1.10.200+gpu | ✔️ | ✔️ | ✔️ | ✔️ |  |

**Note:** Wheel files for Intel® Distribution for Python\* only supports Python 3.9.

**Note:** Wheel files supporting Intel® Distribution for Python\* starts from 1.13.

### Repositories for prebuilt wheel files

```
# Stock PyTorch
REPO_URL: https://developer.intel.com/ipex-whl-stable-xpu

# Intel® Distribution for Python*
REPO_URL: https://developer.intel.com/ipex-whl-stable-xpu-idp
```

### Install PyTorch and TorchVision

```bash
python -m pip install torch==1.13.0a0 torchvision==0.14.1a0 -f <REPO_URL>
```

**Note:** Installation of TorchVision is optional.

**Note:** Please install Numpy 1.22.3 under Intel® Distribution for Python\*.

### Install torchaudio (Optional)

Intel® Extension for PyTorch\* doesn't depend on torchaudio. If you need TorchAudio, please follow the [instructions](https://github.com/pytorch/audio/tree/v0.13.0#from-source) to compile it from source. According to torchaudio-pytorch dependency table, torchaudio 0.13.0 is recommended.

### Install Intel® Extension for PyTorch\*

```bash
python -m pip install intel_extension_for_pytorch==1.13.10+xpu -f <REPO_URL>
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

Please refer to [AOT documentation](./AOT.md) for how to configure `USE_AOT_DEVLIST`. Without configuring AOT, the start-up time for processes using Intel® Extension for PyTorch\* will be high, so this step is important.

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


## Solutions to potential issues on WSL2

|Issue|Explanation|
|-|-|
|Building from source for Intel® Arc™ A-Series GPUs failed on WSL2 without any error thrown|Your system probably does not have enough RAM, so Linux kernel's Out-of-memory killer got invoked. You can verify it by running `dmesg` on bash (WSL2 terminal). If the OOM killer had indeed killed the build process, then you can try increasing the swap-size of WSL2, and/or decreasing the number of parallel build jobs with the environment variable `MAX_JOBS` (by default, it's equal to the number of logical CPU cores. So, setting `MAX_JOBS` to 1 is a very conservative approach, which would slow things down a lot).|
|On WSL2, some workloads terminate with an error `CL_DEVICE_NOT_FOUND` after some time | This is due to the [TDR feature](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys#tdrdelay) in Windows. You can try increasing TDRDelay in your Windows Registry to a large value, such as 20 (it is 2 seconds, by default), and reboot.|
