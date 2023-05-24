
Memory Check Tool

This tool aims to help collect GPU memory information, including reserved memory on XPU device, and GPU device memory footprint.

### Dependency
The sysmon tool is needed to fetch GPU device memory
```console
git clone -b master --depth=1 https://github.com/intel/pti-gpu 
cd pti-gpu/tools/sysmon 
mkdir build && cd build 
cmake .. 
make -j
```

### Install
```console
$ python setup.py install
```

### Usage Example
To show allocated memory on XPU device
```python
import torch
import intel_extension_for_pytorch
import memory_check
memory_check.get_xpu_allocated_mem("xpu:0")
```
Output example:
```console
Rank xpu:0: memory allocated 450MB, reserved 13262MB
```

Set sysmon path in SYSMON_PATH env variable to fetch GPU device memory information
```console
export SYSMON_PATH=${YOUR_PATH}/pti-gpu/tools/sysmon/build/sysmon
```
```python
import torch
import intel_extension_for_pytorch
import memory_check
memory_check.get_gpu_device_mem()
```
Output example:
```console
GPU 0: device memory 13399.5MB, shared Memory 0.0MB
```

Put all above together
```console
export SYSMON_PATH=${YOUR_PATH}/pti-gpu/tools/sysmon/build/sysmon
```
```python
import torch
import intel_extension_for_pytorch
import memory_check
memory_check.display_mem("xpu:0")
```
Output example:
```console
Rank xpu:0: memory allocated 452MB, reserved 13264MB
GPU 0: device memory 13438.6MB, shared Memory 0.0MB
```

### BKC For Adding In Model Script
Please add memory_check.display_mem("xpu:0") as below example:
NOTICE:
1. To avoid introducing extra overhead, call memory_chech at outside of performance measurement scope
2. Call memory_check at the beginning of next iteration
3. Call memory_check after model loading/conversion, JIT enabling, ipex.optimize

```python
try:
    import memory_check
    memory_check.display_mem("xpu:0")
except:
    pass
```
