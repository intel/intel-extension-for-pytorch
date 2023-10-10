# Intel® Extension for PyTorch\* Container

## Description

This document has instruction for running Intel® Extension for PyTorch\* for
GPU in a container.

Assumptions:
* Host machine has the Intel® Data Center GPU 
* Host machine has the Intel® Data Center GPU Ubuntu driver
* Host machine has Docker installed

## Docker

### Build or Pull Container:

`./build.sh` script has docker build command, update all the relevant build arguments and execute the script.

```bash
./build.sh 
```
To pull docker images use the following command:

```bash
docker pull intel/intel-extension-for-pytorch:xpu
```
### Running container:

Run the following commands to start Intel® Extension for PyTorch\* GPU container. You can use `-v` option to mount your
local directory into the container. The `-v` argument can be omitted if you do not need
access to a local directory in the container. Pass the video and render groups to your
docker container so that the GPU is accessible.

```
IMAGE_NAME=intel/intel-extension-for-pytorch:xpu
```
```bash

VIDEO=$(getent group video | sed -E 's,^video:[^:]*:([^:]*):.*$,\1,')
RENDER=$(getent group render | sed -E 's,^render:[^:]*:([^:]*):.*$,\1,')

test -z "$RENDER" || RENDER_GROUP="--group-add ${RENDER}"

docker run --rm \
    -v <your-local-dir>:/workspace \
    --group-add ${VIDEO} \
    ${RENDER_GROUP} \
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
```
1.13.0a0+gitb1dde16 
1.13.10+xpu 
[0]: _DeviceProperties(name='Intel(R) Graphics [0x0bd5]', platform_name='Intel(R) Level-Zero', dev_type='gpu, support_fp64=1, total_memory=62244MB, max_compute_units=512) 
[1]: _DeviceProperties(name='Intel(R) Graphics [0x0bd5]', platform_name='Intel(R) Level-Zero', dev_type='gpu, support_fp64=1, total_memory=62244MB, max_compute_units=512)
```
#### Running your own script

Now you are inside container with Python 3.10, PyTorch, and Intel® Extension for PyTorch\* preinstalled. You can run your own script
to run on Intel GPU.
