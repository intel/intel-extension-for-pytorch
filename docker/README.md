# Intel® Extension for PyTorch\* Container

## Description

This document has instruction for running Intel® Extension for PyTorch\* for
GPU in a container.

Assumptions:
* Host machine has the Intel® Data Center GPU Flex Series.
* Host machine has the Intel® Data Center GPU Flex Series Ubuntu driver.
* Host machine has Docker installed

## Docker

### Build the container:

`./build.sh` script has docker build command, update all the relevant build arguments and execute the script.

```bash
./build.sh
```

### Running container:

Run the following commands to start Intel® Extension for PyTorch\* GPU tools container. You can use `-v` option to mount your
local directory into the container. The `-v` argument can be omitted if you do not need
access to a local directory in the container. Pass the video and render groups to your
docker container so that the GPU is accessible.

```bash
IMAGE_NAME=intel-extension-for-pytorch:gpu

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
Finally, use the following command to check whether MKL is enabled as default:
```bash
python -c "import intel_extension_for_pytorch as ipex;print(ipex.xpu.has_onemkl())"
```
Sample output looks like below:
```
True
```

#### Running your own script

Now you are inside container with Python 3.9, PyTorch, and Intel® Extension for PyTorch\* preinstalled. You can run your own script
to run on Intel GPU.
