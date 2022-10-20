# Build and Install from Source Code

This is guide to build an Intel® Extension for PyTorch* PyPI package from source and install it in Linux.


## Prepare

### Hardware Requirement

Verified Hardware Platforms:
 - Intel® Data Center GPU Flex Series 170

### Software Requirements

- Ubuntu 20.04 (64-bit)
- Intel GPU Drivers 
  - Intel® Data Center GPU Flex Series [419.40](https://dgpu-docs.intel.com/releases/stable_419_40_20220914.html)
- Intel® oneAPI Base Toolkit 2022.3
- Python 3.7-3.10

### Install Intel GPU Driver

|Release|OS|Intel GPU|Install Intel GPU Driver|
|-|-|-|-|
|v1.0.0|Ubuntu 20.04|Intel® Data Center GPU Flex Series| Refer to the [Installation Guides](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-focal-dc.html) for latest driver installation. If install the verified Intel® Data Center GPU Flex Series [419.40](https://dgpu-docs.intel.com/releases/stable_419_40_20220914.html), please append the specific version after components, such as `sudo apt-get install intel-opencl-icd=22.28.23726.1+i419~u20.04`|

### Install oneAPI Base Toolkit

Please refer to [Install oneAPI Base Toolkit Packages](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#base-kit)

Need to install components of Intel® oneAPI Base Toolkit:
 - Intel® oneAPI DPC++ Compiler
 - Intel® oneAPI Math Kernel Library (oneMKL)

Default installation location is /opt/intel/oneapi for root account, ${HOME}/intel/oneapi for other accounts.

### Configure the AOT

Please refer to [AOT documentation](./AOT.md) for how to configure AOT.

### Build and Install from Source Code

Make sure PyTorch is installed so that the extension will work properly. For each PyTorch release, we have a corresponding release of the extension. Here are the PyTorch versions that we support and the mapping relationship:

|PyTorch Version|Intel® Extension for PyTorch* Version|
|--|--|
|[v1.10.\*](https://github.com/pytorch/pytorch/tree/v1.10.0 "v1.10.0")|[v1.10.\*](https://github.com/intel/intel-extension-for-pytorch/tree/v1.10.200)|


Build and Install PyTorch:

```bash
$ git clone https://github.com/pytorch/pytorch.git
$ cd pytorch
# checkout to specific release branch if in need
$ git checkout ${PYTORCH_RELEASE_BRANCH_NAME}
# apply git patch to pytorch code, e.g., apply patch for pytorch v1.10.
$ git apply ${intel_extension_for_pytorch_directory}/torch_patches/{xpu-1.10}.patch 
$ git submodule update --init --recursive
$ pip install -r requirements.txt
# configure MKL env to enable MKL features
$ source ${oneAPI_HOME}/mkl/latest/env/vars.sh
# build pypi package and install it locally
$ python setup.py bdist_wheel
$ pip install dist/*.whl
```

Build and Install Intel® Extension for PyTorch*:

```bash
$ git clone -b xpu-master https://github.com/intel/intel-extension-for-pytorch.git 
$ cd intel-extension-for-pytorch
# checkout to specific release branch if in need
$ git checkout ${IPEX_RELEASE_BRANCH_NAME}
$ git submodule update --init --recursive
$ pip install -r requirements.txt
# configure dpcpp compiler env
$ source ${oneAPI_HOME}/compiler/latest/env/vars.sh
# configure MKL env to enable MKL features
$ source ${oneAPI_HOME}/mkl/latest/env/vars.sh
# build pypi package and install it locally
$ ${USE_AOT_DEVLIST} python setup.py bdist_wheel
$ pip install dist/*.whl
```
