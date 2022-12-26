Installation Guide
==================

## System Requirements

### Hardware Requirement

Verified Hardware Platforms:
 - Intel® Data Center GPU Flex Series 170
 - Intel® Arc™ series GPUs (Experimental support)

### Operating system Requirements

|Intel GPU|Verified Operating System Platform|
|-|-|
|Intel® Data Center GPU Flex Series|Ubuntu 20.04 (64-bit)|
|Intel® Arc™ series GPUs|  native Ubuntu 20.04 (64-bit)<br />WSL2 Ubuntu 20.04 on Windows 11 or Windows 10 21H2|

## Drivers & Software package Requirements

- Intel GPU Drivers 
- Intel® oneAPI Base Toolkit 2022.3
- Python 3.6-3.9

## Preparations

### PyTorch-Intel® Extension for PyTorch\* Version Mapping

Intel® Extension for PyTorch\* has to work with a corresponding version of PyTorch. Here are the PyTorch versions that we support and the mapping relationship:

|PyTorch Version|Extension Version|
|--|--|
|[v1.13.0](https://github.com/pytorch/pytorch/tree/v1.13.0) (patches needed)|[v1.13.\*](https://github.com/intel/intel-extension-for-pytorch/tree/v1.13.200+gpu)|


### Instructions for installing drivers

#### Instructions for Intel® Data Center GPU Flex Series

|Release|OS|Instructions for installing Intel GPU Driver|
|-|-|-|
|v1.0.0|Ubuntu 20.04 (native)| Please refer to the [Installation Guides](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-focal-dc.html) for latest driver installation. If you are installing the verified Intel® Data Center GPU Flex Series drivers - viz. [419.40](https://dgpu-docs.intel.com/releases/stable_419_40_20220914.html), please append the specific version after components, such as `sudo apt-get install intel-opencl-icd=22.28.23726.1+i419~u20.04`|

#### Instructions for Intel® Arc™ A-Series GPUs

|Release|OS|Instructions for installing Intel GPU Driver|
|-|-|-|
|v1.0.0|Ubuntu 20.04 (native)| Please refer to the [Installation Guides](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-focal-dc.html). When installing the Intel® Arc™ A-Series GPU Drivers [419.40](https://dgpu-docs.intel.com/releases/stable_419_40_20220914.html), please be sure to append the specific version after components, such as `sudo apt-get install intel-opencl-icd=22.28.23726.1+i419~u20.04`|
|v1.0.0|WSL2 Ubuntu 20.04 on Windows 11 or Windows 10 21H2|Please download drivers for Intel® Arc™ series [for Windows 11 or Windows 10 21H2](https://www.intel.com/content/www/us/en/download/726609/intel-arc-graphics-windows-dch-driver.html). Please note that you would have to follow the rest of the steps in WSL2, but the drivers should be installed on Windows|

### Instructions for installing required Packages for WSL2 Ubuntu 20.04 (only for Intel® Arc™ series GPUs)

Please skip this step on native Ubuntu.
The steps to install the runtime components in WSL2 Ubuntu 20.04 are:

#### Add the repositories.intel.com/graphics package repository to your Ubuntu installation:

```bash
sudo apt-get install -y gpg-agent wget
wget -qO - https://repositories.intel.com/graphics/intel-graphics.key |
sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu focal main' | \
    sudo tee  /etc/apt/sources.list.d/intel.gpu.focal.list
sudo apt-get update
```

#### Install the necessary runtime packages:

```bash
sudo apt-get install \
intel-opencl-icd=22.28.23726.1+i419~u20.04 \
intel-level-zero-gpu=1.3.23726.1+i419~u20.04 \
level-zero=1.8.1+i419~u20.04
```

#### Add the Intel® oneAPI library repositories to your Ubuntu installation:
```bash
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB |
    sudo gpg --dearmor --output /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | \
    sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt-get update
```

#### Install the necessary Intel® oneAPI library runtime packages:
```bash
sudo apt-get install \
    intel-oneapi-runtime-dpcpp-cpp=2022.2.0-8734 \
    intel-oneapi-runtime-mkl=2022.2.0-8748
```

The above commands install only runtime libraries for Intel® oneAPI which are used by the Intel® Extension for PyTorch*.

```bash
source {ONEAPI_ROOT}/setvars.sh
```

## Install via wheel files

If you want to build from source instead, please jump to [these instructions](#building--installing-from-source).

Prebuilt wheel files availability matrix for Python versions:

| Extension Version | Python 3.6 | Python 3.7 | Python 3.8 | Python 3.9 | Python 3.10 |
| :--: | :--: | :--: | :--: | :--: | :--: |
| 1.13.200+gpu | ✔️ | ✔️ | ✔️ | ✔️ |  |

### Install PyTorch

```bash
python -m pip install torch==1.13.0a0 -f https://developer.intel.com/ipex-whl-stable-xpu
```

### Install Numpy

Numpy is required to work with PyTorch\*. Verified numpy versions differ according to python versions.

```bash
python -m pip install numpy==1.19.5  # for Python 3.6
python -m pip install numpy==1.21.6  # for Python 3.7
python -m pip install numpy==1.23.4  # for Python 3.8 and 3.9
```

### Install torchvision and torchaudio (Optional)

Intel® Extension for PyTorch\* doesn't depend on torchvision or torchaudio.

You can install torchvision via the following command.

```bash
python -m pip install torchvision==0.14.0+cpu --no-deps -f https://download.pytorch.org/whl/torch_stable.html
```

For torchaudio installation, please follow the [instructions](https://github.com/pytorch/audio/tree/v0.13.0#from-source) to compile it from source. According to torchaudio-pytorch dependency table, torchaudio 0.13.0 is recommended.

### Install Intel® Extension for PyTorch\*

```bash
python -m pip install intel_extension_for_pytorch==1.13.200+gpu -f https://developer.intel.com/ipex-whl-stable-xpu
```


## Building & installing from source

### Install oneAPI Base Toolkit

Please refer to [Install oneAPI Base Toolkit Packages](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#base-kit).

Need to install components of Intel® oneAPI Base Toolkit:
 - Intel® oneAPI DPC++ Compiler
 - Intel® oneAPI Math Kernel Library (oneMKL)

Default installation location *{ONEAPI_ROOT}* is `/opt/intel/oneapi` for root account, `${HOME}/intel/oneapi` for other accounts.

**_NOTE:_** You need to activate oneAPI environment when building Intel® Extension for PyTorch\* on Intel GPU.
For running workloads, runtime packages of DPCPP & MKL will suffice.

### Download source code of PyTorch and Intel® Extension for PyTorch\*:

Make sure PyTorch is installed so that the extension will work properly. For each PyTorch release, we have a corresponding release of the extension. Here are the PyTorch versions that we support and the mapping relationship:

|PyTorch Version|Intel® Extension for PyTorch\* Version|
|--|--|
|[v1.13.\*](https://github.com/pytorch/pytorch/tree/v1.13.0 "v1.13.0")|[v1.13.\*](https://github.com/intel/intel-extension-for-pytorch/tree/v1.13.200)|

```bash
$ git clone https://github.com/pytorch/pytorch.git
$ cd pytorch
$ git checkout v1.13.0

$ git clone https://github.com/intel/intel-extension-for-pytorch.git 
$ cd intel-extension-for-pytorch
$ git checkout v1.13.200+gpu
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

#### Configure the AOT

Please refer to [AOT documentation](./AOT.md) for how to configure [AOT](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html).
Without configuring AOT, the start-up time for processes using Intel® Extension for PyTorch* will be high, so this step is important.  

#### Build and Install Intel® Extension for PyTorch\*:

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
|Building from source for Intel® Arc™ series GPUs failed on WSL2 without any error thrown|Your system probably does not have enough RAM, so Linux kernel's Out-of-memory killer got invoked. You can verify it by running `dmesg` on bash (WSL2 terminal). If the OOM killer had indeed killed the build process, then you can try increasing the swap-size of WSL2, and/or decreasing the number of parallel build jobs with the environment variable `MAX_JOBS` (by default, it's equal to the number of logical CPU cores. So, setting `MAX_JOBS` to 1 is a very conservative approach, which would slow things down a lot).|
|On WSL2, some workloads terminate with an error `CL_DEVICE_NOT_FOUND` after some time | This is due to the [TDR feature](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys#tdrdelay) in Windows. You can try increasing TDRDelay in your Windows Registry to a large value, such as 20 (it is 2 seconds, by default), and reboot.|
