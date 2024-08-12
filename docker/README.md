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
git checkout release/xpu/2.1.40
git submodule sync
git submodule update --init --recursive
docker build -f docker/Dockerfile.compile --build-arg GID_RENDER=$(getent group render | sed -E 's,^render:[^:]*:([^:]*):.*$,\1,') -t intel/intel-extension-for-pytorch:2.1.40-xpu .
```

Alternatively, `./build.sh` script has docker build command to install prebuilt wheel files, update all the relevant build arguments and execute the script. Run the command below in current directory.

```bash
export IMAGE_TYPE="xpu"
./build.sh 
```
To pull docker images use the following command:

```bash
docker pull intel/intel-extension-for-pytorch:2.1.40-xpu
```
### Running container:

Run the following commands to start Intel® Extension for PyTorch\* GPU container. You can use `-v` option to mount your
local directory into the container. The `-v` argument can be omitted if you do not need
access to a local directory in the container. 

```bash
IMAGE_NAME=intel/intel-extension-for-pytorch:2.1.40-xpu
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
```
xpu
```
Then, verify that the XPU device is available to Intel® Extension for PyTorch\*:
```bash
python -c "import intel_extension_for_pytorch as ipex;print(ipex.xpu.is_available())"
```
Sample output looks like below:
```
True
```
Use the following command to check whether MKL is enabled as default:
```bash
python -c "import intel_extension_for_pytorch as ipex;print(ipex.xpu.has_onemkl())"
```
Sample output looks like below:
```
True
```
Finally, use the following command to show detailed info of detected device:
```bash
python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__); [print(f'[{i}]: {ipex.xpu.get_device_properties(i)}') for i in range(ipex.xpu.device_count())];"
```

Sample output looks like below:
```bash
2.1.0.post2+cxx11.abi
2.1.40+xpu
[0]: _DeviceProperties(name='Intel(R) Data Center GPU Max 1550', platform_name='Intel(R) Level-Zero', dev_type='gpu', driver_version='1.3.27642', has_fp64=1, total_memory=65536MB, max_compute_units=448, gpu_eu_count=448)
[1]: _DeviceProperties(name='Intel(R) Data Center GPU Max 1550', platform_name='Intel(R) Level-Zero', dev_type='gpu', driver_version='1.3.27642', has_fp64=1, total_memory=65536MB, max_compute_units=448, gpu_eu_count=448)
```

#### Running your own script

Now you are inside container with Python 3.10, PyTorch, and Intel® Extension for PyTorch\* preinstalled. You can run your own script
to run on Intel GPU.
