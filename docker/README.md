# Intel® Extension for PyTorch\* Container

## Description

This document has instruction for running Intel® Extension for PyTorch\* for
GPU in a container.

Assumptions:
* Host machine has the Intel® Data Center GPU 
* Host machine has the [Intel® Data Center GPU Ubuntu driver](https://dgpu-docs.intel.com/releases/index.html)
* Host machine has Docker installed

## Docker

### Build or Pull Container:

Run the following commands to build a docker image by compiling from source.

```
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout xpu-main
git submodule sync
git submodule update --init --recursive
docker build -f docker/Dockerfile.compile -t intel/intel-extension-for-pytorch:xpu .
```

Alternatively, `./build.sh` script has docker build command to install prebuilt wheel files, update all the relevant build arguments and execute the script. Run the command below in current directory.

```bash
./build.sh 
```
To pull docker images use the following command:

```bash
docker pull intel/intel-extension-for-pytorch:2.3.110-xpu
```

### Running container:

Run the following commands to start Intel® Extension for PyTorch\* GPU container. You can use `-v` option to mount your
local directory into the container. The `-v` argument can be omitted if you do not need
access to a local directory in the container. Pass the video and render groups to your
docker container so that the GPU is accessible.

```bash
IMAGE_NAME=intel/intel-extension-for-pytorch:2.3.110-xpu
```

```bash
docker run --rm \
    -v <your-local-dir>:/workspace \
    --device=/dev/dri \
    --ipc=host \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    -e no_proxy=$no_proxy \
    -it $IMAGE_NAME bash
```

#### Verify if XPU is accessible from PyTorch:
You are inside the container now. Run the following command to verify XPU is visible to PyTorch:

```bash
python -c "import torch;print(torch.device('xpu'))"
```

Sample output looks like below:

```bash
xpu
```

Then, verify that the XPU device is available to Intel® Extension for PyTorch\*:

```bash
python -c "import torch;import intel_extension_for_pytorch as ipex;print(torch.xpu.has_xpu())"
```

Sample output looks like below:

```bash
True
```

Use the following command to check whether MKL is enabled as default:

```bash
python -c "import torch;import intel_extension_for_pytorch as ipex;print(torch.xpu.has_onemkl())"
```

Sample output looks like below:

```bash
True
```

Finally, use the following command to show detailed info of detected device:

```bash
python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__); [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];"
```

Sample output looks like below:

```bash
2.3.1+cxx11.abi
2.3.110+xpu
[0]: _XpuDeviceProperties(name='Intel(R) Data Center GPU Max 1550', platform_name='Intel(R) Level-Zero', type='gpu', driver_version='1.3.30049', total_memory=65536MB, max_compute_units=448, gpu_eu_count=448, gpu_subslice_count=56, max_work_group_size=1024, max_num_sub_groups=64, sub_group_sizes=[16 32], has_fp16=1, has_fp64=1, has_atomic64=1)
[1]: _XpuDeviceProperties(name='Intel(R) Data Center GPU Max 1550', platform_name='Intel(R) Level-Zero', type='gpu', driver_version='1.3.30049', total_memory=65536MB, max_compute_units=448, gpu_eu_count=448, gpu_subslice_count=56, max_work_group_size=1024, max_num_sub_groups=64, sub_group_sizes=[16 32], has_fp16=1, has_fp64=1, has_atomic64=1)
```

#### Running your own script

Now you are inside container with Python 3.10, PyTorch, and Intel® Extension for PyTorch\* preinstalled. You can run your own script
to run on Intel GPU.
