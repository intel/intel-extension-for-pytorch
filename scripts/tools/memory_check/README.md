
Memory Check Tool

This tool aims to help collect GPU memory information, including reserved memory on XPU device, and GPU device memory footprint.

### Dependency 
To get GPU device memory, need install sysmon tool first
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
To print XPU allocated memory
```console
import torch
import intel_extension_for_pytorch
import memory_check
memory_check.get_xpu_allocated_mem("xpu:0")
```
Output,for example:
```console
Rank xpu:0: memory allocated 450MB, reserved 13262MB
```

To print GPU device memory, please set SYSMON_PATH first
```console
export SYSMON_PATH=${YOUR_PATH}/pti-gpu/tools/sysmon/build/sysmon
import torch
import intel_extension_for_pytorch
import memory_check
memory_check.get_gpu_device_mem()
```
Output,for example:
```console
GPU 0: device memory 13399.5MB, shared Memory 0.0MB
```

To print both
```console
export SYSMON_PATH=${YOUR_PATH}/pti-gpu/tools/sysmon/build/sysmon
import torch
import intel_extension_for_pytorch
import memory_check
memory_check.display_mem("xpu:0")
```
Output,for example:
```console
Rank xpu:0: memory allocated 452MB, reserved 13264MB
GPU 0: device memory 13438.6MB, shared Memory 0.0MB
```

### BKC For Adding In Model Script
Please add memory_check.display_mem("xpu:0") at below check point:
1. For each iter, add print before next iter begin
2. After Model loading, if has
3. After JIT enabling, if has
4. After optimize, if has

```console
memory_check_flag = False
try:
    import memory_check
    memory_check_flag = True
except:
    pass

if memory_check_flag:
    memory_check.display_mem("xpu:0")
```
