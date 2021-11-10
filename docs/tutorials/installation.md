Installation Guide
==================

## System Requirements

|Category|Content|
|--|--|
|Compiler|GCC version greater than 7|
|Operator System|CentOS 7, RHEL 8, Ubuntu newer than 18.04|
|Python|3.6, 3.7, 3.8, 3.9|

## Install PyTorch (Optional)

 |Extension Version|PyTorch Version|
 |--|--|
 |[v1.9.0](https://github.com/intel/intel-extension-for-pytorch/tree/v1.9.0)|[v1.9.0](https://github.com/pytorch/pytorch/tree/v1.9.0 "v1.9.0")|
 |[v1.8.0](https://github.com/intel/intel-extension-for-pytorch/tree/v1.8.0)|[v1.8.0](https://github.com/pytorch/pytorch/tree/v1.8.0 "v1.8.0")|
 |[v1.2.0](https://github.com/intel/intel-extension-for-pytorch/tree/v1.2.0)|[v1.7.0](https://github.com/pytorch/pytorch/tree/v1.7.0 "v1.7.0")|
 |[v1.1.0](https://github.com/intel/intel-extension-for-pytorch/tree/v1.1.0)|[v1.5.0-rc3](https://github.com/pytorch/pytorch/tree/v1.5.0-rc3 "v1.5.0-rc3")|
 |[v1.0.2](https://github.com/intel/intel-extension-for-pytorch/tree/v1.0.2)|[v1.5.0-rc3](https://github.com/pytorch/pytorch/tree/v1.5.0-rc3 "v1.5.0-rc3")|
 |[v1.0.1](https://github.com/intel/intel-extension-for-pytorch/tree/v1.0.1)|[v1.5.0-rc3](https://github.com/pytorch/pytorch/tree/v1.5.0-rc3 "v1.5.0-rc3")|
 |[v1.0.0](https://github.com/intel/intel-extension-for-pytorch/tree/v1.0.0)|[v1.5.0-rc3](https://github.com/pytorch/pytorch/tree/v1.5.0-rc3 "v1.5.0-rc3")|

For the extension version earlier than 1.8.0, a patch has to be manually applied to PyTorch source code. Please check previous installation guide.

From 1.8.0, compiling PyTorch from source is not required. If you still want to compile PyTorch, please follow instructions [here](https://github.com/pytorch/pytorch#installation). Please make sure to checkout the correct PyTorch version according to the table above.

**Note:** Installing the extension will automatically invoke installation of the corresponding version of PyTorch.

## Install via wheel file

```
python -m pip install torch_ipex==1.9.0 -f https://software.intel.com/ipex-whl-stable
```

**Note:** Wheel files availability for Python versions

| Extension Version | Python 3.6 | Python 3.7 | Python 3.8 | Python 3.9 |
| :--: | :--: | :--: | :--: | :--: |
| 1.9.0 | ✔️ | ✔️ | ✔️ | ✔️ |
| 1.8.0 |  | ✔️ |  |  |

**Note:** The wheel files released are compiled with AVX-512 instruction set support only. They cannot be running on hardware platforms that don't support AVX-512 instruction set. Please compile from source with AVX2 support in this case.

## Install Extension by compiling from source

```bash
git clone --recursive https://github.com/intel/intel-extension-for-pytorch
cd intel-extension-for-pytorch

# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive

# run setup.py to compile and install the binaries
# if you need to compile from source with AVX2 support, please uncomment the following line.
# export AVX2=1
python setup.py install
```
