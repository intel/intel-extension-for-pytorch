Installation Guide
==================

## System Requirements

|Category|Content|
|--|--|
|Compiler|Recommend using GCC 11|
|Operating System|CentOS 7, RHEL 8, Rocky Linux 8.5, Ubuntu newer than 18.04|
|Python|See prebuilt wheel files availability matrix below|

* Intel® Extension for PyTorch\* is functional on systems with AVX2 instruction set support (such as Intel® Core™ Processor Family and Intel® Xeon® Processor formerly Broadwell). However, it is highly recommended to run on systems with AVX-512 and above instructions support for optimal performance (such as Intel® Xeon® Scalable Processors).

## Install PyTorch

Make sure PyTorch is installed so that the extension will work properly. For each PyTorch release, we have a corresponding release of the extension. Here are the PyTorch versions that we support and the mapping relationship:

|PyTorch Version|Extension Version|
|--|--|
|[v1.13.\*](https://github.com/pytorch/pytorch/tree/v1.13.0 "v1.13.0")|[v1.13.\*](https://github.com/intel/intel-extension-for-pytorch/tree/v1.13.100+cpu)|
|[v1.12.\*](https://github.com/pytorch/pytorch/tree/v1.12.0 "v1.12.0")|[v1.12.\*](https://github.com/intel/intel-extension-for-pytorch/tree/v1.12.300)|
|[v1.11.\*](https://github.com/pytorch/pytorch/tree/v1.11.0 "v1.11.0")|[v1.11.\*](https://github.com/intel/intel-extension-for-pytorch/tree/v1.11.200)|
|[v1.10.\*](https://github.com/pytorch/pytorch/tree/v1.10.0 "v1.10.0")|[v1.10.\*](https://github.com/intel/intel-extension-for-pytorch/tree/v1.10.100)|
|[v1.9.0](https://github.com/pytorch/pytorch/tree/v1.9.0 "v1.9.0")|[v1.9.0](https://github.com/intel/intel-extension-for-pytorch/tree/v1.9.0)|
|[v1.8.0](https://github.com/pytorch/pytorch/tree/v1.8.0 "v1.8.0")|[v1.8.0](https://github.com/intel/intel-extension-for-pytorch/tree/v1.8.0)|
|[v1.7.0](https://github.com/pytorch/pytorch/tree/v1.7.0 "v1.7.0")|[v1.2.0](https://github.com/intel/intel-extension-for-pytorch/tree/v1.2.0)|
|[v1.5.0-rc3](https://github.com/pytorch/pytorch/tree/v1.5.0-rc3 "v1.5.0-rc3")|[v1.1.0](https://github.com/intel/intel-extension-for-pytorch/tree/v1.1.0)|
|[v1.5.0-rc3](https://github.com/pytorch/pytorch/tree/v1.5.0-rc3 "v1.5.0-rc3")|[v1.0.2](https://github.com/intel/intel-extension-for-pytorch/tree/v1.0.2)|
|[v1.5.0-rc3](https://github.com/pytorch/pytorch/tree/v1.5.0-rc3 "v1.5.0-rc3")|[v1.0.1](https://github.com/intel/intel-extension-for-pytorch/tree/v1.0.1)|
|[v1.5.0-rc3](https://github.com/pytorch/pytorch/tree/v1.5.0-rc3 "v1.5.0-rc3")|[v1.0.0](https://github.com/intel/intel-extension-for-pytorch/tree/v1.0.0)|

Please install CPU version of PyTorch through its official channel. For more details, refer to [pytorch.org](https://pytorch.org/get-started/locally/).

---

**Note:**

For the extension version earlier than 1.8.0, a patch has to be manually applied to PyTorch source code. Check that version's installation guide.

From 1.8.0, compiling PyTorch from source is not required. If you still want to compile PyTorch, follow these [installation instructions](https://github.com/pytorch/pytorch#installation). Make sure to check out the correct PyTorch version according to the table above.

---

## Install via wheel file

Prebuilt wheel files availability matrix for Python versions

| Extension Version | Python 3.6 | Python 3.7 | Python 3.8 | Python 3.9 | Python 3.10 |
| :--: | :--: | :--: | :--: | :--: | :--: |
| 1.13.100 |  | ✔️ | ✔️ | ✔️ | ✔️ |
| 1.13.0 |  | ✔️ | ✔️ | ✔️ | ✔️ |
| 1.12.300 |  | ✔️ | ✔️ | ✔️ | ✔️ |
| 1.12.100 |  | ✔️ | ✔️ | ✔️ | ✔️ |
| 1.12.0 |  | ✔️ | ✔️ | ✔️ | ✔️ |
| 1.11.200 |  | ✔️ | ✔️ | ✔️ | ✔️ |
| 1.11.0 |  | ✔️ | ✔️ | ✔️ | ✔️ |
| 1.10.100 | ✔️ | ✔️ | ✔️ | ✔️ |  |
| 1.10.0 | ✔️ | ✔️ | ✔️ | ✔️ |  |
| 1.9.0 | ✔️ | ✔️ | ✔️ | ✔️ |  |
| 1.8.0 |  | ✔️ |  |  |  |

**Note:** Intel® Extension for PyTorch\* has PyTorch version requirement. Check the mapping table above.

Starting from 1.11.0, you can use normal pip command to install the package with the latest version.

```
python -m pip install intel_extension_for_pytorch
```

Alternatively, you can also install the latest version with the following commands:

```
python -m pip install intel_extension_for_pytorch -f https://developer.intel.com/ipex-whl-stable-cpu
```

**Note:** For versions before 1.10.0, use package name `torch_ipex`, rather than `intel_extension_for_pytorch`.

**Note:** To install a history package with a specific version, run with the following command:

```
python -m pip install <package_name>==<version_name> -f https://developer.intel.com/ipex-whl-stable-cpu
```

## Install via source compilation

To ensure a smooth compilation, a script is provided in the Github repo. If you would like to compile the binaries from source, it is highly recommended to utilize this script.

```bash
$ wget https://raw.githubusercontent.com/intel/intel-extension-for-pytorch/master/scripts/compile_bundle.sh
$ bash compile_bundle.sh
```

**Note:** Recommend to use the `compile_bundle.sh` script in a clean docker container.

**Note:** Use the `compile_bundle.sh` script under a `conda` environment.

**Note:** Depends on what applications are available on your OS, you probably need to install some Linux commands, like `git`, etc. Installation of these Linux commands are not included in this script.

**Note:** The `compile_bundle.sh` script downloads source code of llvm-project and Intel® Extension for PyTorch\* into individual folders in its directory. You can consider to create a specific folder to use this script. Wheel files will be generated under `dist` folder of each source code directory. Besides, compilation progress is dumped into a log file `build.log` in source code directory. The log file is helpful to identify errors occurred during compilation. Should any failure happened, after addressing the issue, you can simply run the `compile_bundle.sh` script again with the same command.

```bash
$ mkdir ipex_bundle
$ cd ipex_bundle
$ wget .../compile_bundle.sh
$ bash compile_bundle.sh
$ ls
compile_bundle.sh  intel_extension_for_pytorch  llvm-project
$ tree -L 3 .
.
├── intel_extension_for_pytorch
│   ├── dist
│   │   └── intel_extension_for_pytorch-....whl
│   ├ build.log
│   └ ...
└── llvm-project
    └ ...
```

## Install via Docker container

### Build Docker container from Dockerfile

Run the following commands to build the `pip` based deployment container:

```console
$ cd docker
$ DOCKER_BUILDKIT=1 docker build -f Dockerfile.pip -t intel-extension-for-pytorch:pip .
$ docker run --rm intel-extension-for-pytorch:pip python -c "import torch; import intel_extension_for_pytorch as ipex; print('torch:', torch.__version__,' ipex:',ipex.__version__)"
```

Run the following commands to build the `conda` based development container:

```console
$ cd docker
$ DOCKER_BUILDKIT=1 docker build -f Dockerfile.conda -t intel-extension-for-pytorch:conda .
$ docker run --rm intel-extension-for-pytorch:conda python -c "import torch; import intel_extension_for_pytorch as ipex; print('torch:', torch.__version__,' ipex:',ipex.__version__)"
```

### Get docker container from dockerhub

Pre-built docker images are available at [DockerHub](https://hub.docker.com/r/intel/intel-optimized-pytorch/tags).

Run the following command to pull the image to your local machine.

```console
docker pull intel/intel-optimized-pytorch:latest
```

## Install C++ SDK

|Version|Pre-cxx11 ABI|cxx11 ABI|
|--|--|--|
| 1.13.100 | [libintel-ext-pt-1.13.100+cpu.run](http://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libipex/cpu/libintel-ext-pt-1.13.100%2Bcpu.run) | [libintel-ext-pt-cxx11-abi-1.13.100+cpu.run](http://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libipex/cpu/libintel-ext-pt-cxx11-abi-1.13.100%2Bcpu.run) |
| 1.13.0 | [libintel-ext-pt-1.13.0+cpu.run](http://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libipex/cpu/libintel-ext-pt-1.13.0%2Bcpu.run) | [libintel-ext-pt-cxx11-abi-1.13.0+cpu.run](http://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libipex/cpu/libintel-ext-pt-cxx11-abi-1.13.0%2Bcpu.run) |
| 1.12.300 | [libintel-ext-pt-1.12.300+cpu.run](http://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libipex/cpu/libintel-ext-pt-1.12.300%2Bcpu.run) | [libintel-ext-pt-cxx11-abi-1.12.300+cpu.run](http://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libipex/cpu/libintel-ext-pt-cxx11-abi-1.12.300%2Bcpu.run) |
| 1.12.100 | [libintel-ext-pt-1.12.100+cpu.run](http://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libipex/cpu/libintel-ext-pt-1.12.100%2Bcpu.run) | [libintel-ext-pt-cxx11-abi-1.12.100+cpu.run](http://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libipex/cpu/libintel-ext-pt-cxx11-abi-1.12.100%2Bcpu.run) |
| 1.12.0 | [libintel-ext-pt-1.12.0+cpu.run](http://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libipex/cpu/libintel-ext-pt-1.12.0%2Bcpu.run) | [libintel-ext-pt-cxx11-abi-1.12.0+cpu.run](http://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libipex/cpu/libintel-ext-pt-cxx11-abi-1.12.0%2Bcpu.run) |
| 1.11.200 | [libintel-ext-pt-1.11.200+cpu.run](http://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libipex/cpu/libintel-ext-pt-shared-with-deps-1.11.200%2Bcpu.run) | [libintel-ext-pt-cxx11-abi-1.11.200+cpu.run](http://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libipex/cpu/libintel-ext-pt-cxx11-abi-shared-with-deps-1.11.200%2Bcpu.run) |
| 1.11.0 | [libintel-ext-pt-1.11.0+cpu.run](http://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libipex/cpu/libintel-ext-pt-1.11.0%2Bcpu.run) | [libintel-ext-pt-cxx11-abi-1.11.0+cpu.run](http://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libipex/cpu/libintel-ext-pt-cxx11-abi-1.11.0%2Bcpu.run) |
| 1.10.100 | [libtorch-shared-with-deps-1.10.0%2Bcpu-intel-ext-pt-cpu-1.10.100.zip](http://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libipex/cpu/libtorch-shared-with-deps-1.10.0%2Bcpu-intel-ext-pt-cpu-1.10.100.zip) | [libtorch-cxx11-abi-shared-with-deps-1.10.0%2Bcpu-intel-ext-pt-cpu-1.10.100.zip](http://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libipex/cpu/libtorch-cxx11-abi-shared-with-deps-1.10.0%2Bcpu-intel-ext-pt-cpu-1.10.100.zip) |
| 1.10.0 | [intel-ext-pt-cpu-libtorch-shared-with-deps-1.10.0+cpu.zip](http://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libipex/cpu/intel-ext-pt-cpu-libtorch-shared-with-deps-1.10.0%2Bcpu.zip) | [intel-ext-pt-cpu-libtorch-cxx11-abi-shared-with-deps-1.10.0+cpu.zip](http://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/libipex/cpu/intel-ext-pt-cpu-libtorch-cxx11-abi-shared-with-deps-1.10.0%2Bcpu.zip) |

**Usage:** For version newer than 1.11.0, download one run file above according to your scenario, run the following command to install it and follow the [C++ example](./examples.md#c).
```
bash <libintel-ext-pt-name>.run install <libtorch_path>
```

You can get full usage help message by running the run file alone, as the following command.

```
bash <libintel-ext-pt-name>.run
```

**Usage:** For version before 1.11.0, download one zip file above according to your scenario, unzip it and follow the [C++ example](./examples.md#c).
