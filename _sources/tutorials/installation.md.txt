Installation Guide
==================

## System Requirements

|Category|Content|
|--|--|
|Compiler|Recommend to use GCC 9|
|Operating System|CentOS 7, RHEL 8, Rocky Linux 8.5, Ubuntu newer than 18.04|
|Python|See prebuilt wheel files availability matrix below|

## Install PyTorch

You need to make sure PyTorch is installed in order to get the extension working properly. For each PyTorch release, we have a corresponding release of the extension. Here is the PyTorch versions that we support and the mapping relationship:

|PyTorch Version|Extension Version|
|--|--|
|[v1.11.\*](https://github.com/pytorch/pytorch/tree/v1.11.0 "v1.11.0")|[v1.11.\*](https://github.com/intel/intel-extension-for-pytorch/tree/v1.11.0)|
|[v1.10.\*](https://github.com/pytorch/pytorch/tree/v1.10.0 "v1.10.0")|[v1.10.\*](https://github.com/intel/intel-extension-for-pytorch/tree/v1.10.100)|
|[v1.9.0](https://github.com/pytorch/pytorch/tree/v1.9.0 "v1.9.0")|[v1.9.0](https://github.com/intel/intel-extension-for-pytorch/tree/v1.9.0)|
|[v1.8.0](https://github.com/pytorch/pytorch/tree/v1.8.0 "v1.8.0")|[v1.8.0](https://github.com/intel/intel-extension-for-pytorch/tree/v1.8.0)|
|[v1.7.0](https://github.com/pytorch/pytorch/tree/v1.7.0 "v1.7.0")|[v1.2.0](https://github.com/intel/intel-extension-for-pytorch/tree/v1.2.0)|
|[v1.5.0-rc3](https://github.com/pytorch/pytorch/tree/v1.5.0-rc3 "v1.5.0-rc3")|[v1.1.0](https://github.com/intel/intel-extension-for-pytorch/tree/v1.1.0)|
|[v1.5.0-rc3](https://github.com/pytorch/pytorch/tree/v1.5.0-rc3 "v1.5.0-rc3")|[v1.0.2](https://github.com/intel/intel-extension-for-pytorch/tree/v1.0.2)|
|[v1.5.0-rc3](https://github.com/pytorch/pytorch/tree/v1.5.0-rc3 "v1.5.0-rc3")|[v1.0.1](https://github.com/intel/intel-extension-for-pytorch/tree/v1.0.1)|
|[v1.5.0-rc3](https://github.com/pytorch/pytorch/tree/v1.5.0-rc3 "v1.5.0-rc3")|[v1.0.0](https://github.com/intel/intel-extension-for-pytorch/tree/v1.0.0)|

Here is an example showing how to install PyTorch. For more details, please refer to [pytorch.org](https://pytorch.org/get-started/locally/)

---

**Note:**

For the extension version earlier than 1.8.0, a patch has to be manually applied to PyTorch source code. Please check previous installation guide.

From 1.8.0, compiling PyTorch from source is not required. If you still want to compile PyTorch, please follow instructions [here](https://github.com/pytorch/pytorch#installation). Please make sure to checkout the correct PyTorch version according to the table above.

---

## Install via wheel file

Prebuilt wheel files availability matrix for Python versions

| Extension Version | Python 3.6 | Python 3.7 | Python 3.8 | Python 3.9 | Python 3.10 |
| :--: | :--: | :--: | :--: | :--: | :--: |
| 1.11.0 |  | ✔️ | ✔️ | ✔️ | ✔️ |
| 1.10.100 | ✔️ | ✔️ | ✔️ | ✔️ |  |
| 1.10.0 | ✔️ | ✔️ | ✔️ | ✔️ |  |
| 1.9.0 | ✔️ | ✔️ | ✔️ | ✔️ |  |
| 1.8.0 |  | ✔️ |  |  |  |

**Note:** Intel® Extension for PyTorch\* has PyTorch version requirement. Please check the mapping table above.

Starting from 1.11.0, you can use normal pip command to install the package.

```
python -m pip install intel_extension_for_pytorch
```

Alternatively, you can also install the latest version with the following commands:

```
python -m pip install intel_extension_for_pytorch -f https://software.intel.com/ipex-whl-stable
```

**Note:** For version prior to 1.10.0, please use package name `torch_ipex`, rather than `intel_extension_for_pytorch`.

**Note:** To install a package with a specific version, please run with the following command.

```
python -m pip install <package_name>==<version_name> -f https://software.intel.com/ipex-whl-stable
```

## Install via source compilation

```bash
git clone --recursive https://github.com/intel/intel-extension-for-pytorch
cd intel-extension-for-pytorch
git checkout v1.11.0

# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive

python setup.py install
```

## Install C++ SDK

|Version|Pre-cxx11 ABI|cxx11 ABI|
|--|--|--|
| 1.11.0 | [libintel-ext-pt-1.11.0+cpu.run](http://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libtorch_zip/libintel-ext-pt-1.11.0%2Bcpu.run) | [libintel-ext-pt-cxx11-abi-1.11.0+cpu.run](http://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libtorch_zip/libintel-ext-pt-cxx11-abi-1.11.0%2Bcpu.run) |
| 1.10.100 | [libtorch-shared-with-deps-1.10.0%2Bcpu-intel-ext-pt-cpu-1.10.100.zip](http://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/wheels/v1.10/libtorch-shared-with-deps-1.10.0%2Bcpu-intel-ext-pt-cpu-1.10.100.zip) | [libtorch-cxx11-abi-shared-with-deps-1.10.0%2Bcpu-intel-ext-pt-cpu-1.10.100.zip](http://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/wheels/v1.10/libtorch-cxx11-abi-shared-with-deps-1.10.0%2Bcpu-intel-ext-pt-cpu-1.10.100.zip) |
| 1.10.0 | [intel-ext-pt-cpu-libtorch-shared-with-deps-1.10.0+cpu.zip](https://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/wheels/v1.10/intel-ext-pt-cpu-libtorch-shared-with-deps-1.10.0%2Bcpu.zip) | [intel-ext-pt-cpu-libtorch-cxx11-abi-shared-with-deps-1.10.0+cpu.zip](https://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/wheels/v1.10/intel-ext-pt-cpu-libtorch-cxx11-abi-shared-with-deps-1.10.0%2Bcpu.zip) |

**Usage:** For version newer than 1.11.0, donwload one run file above according to your scenario, run the following command to install it and follow the [C++ example](./examples.html#c).
```
bash <libintel-ext-pt-name>.run install <libtorch_path>
```

You can get full usage help message by running the run file alone, as the following command.

```
bash <libintel-ext-pt-name>.run
```

**Usage:** For version prior to 1.11.0, donwload one zip file above according to your scenario, unzip it and follow the [C++ example](./examples.html#c).
