# Goal and limitations
* The goal of compatible_mode is, to migrate existing CUDA model scripts to target device (for example, XPU), much smoothly and simple.
* The compatible_mode only provides apple-to-apple API conversion as further as possible. It will NOT transform original model script to adopt aggressive APIs for performance improvement.

# API definition
```python
def compatible_mode(target_device="xpu", dist_backend="ccl", compile_backend="inductor"):
```
* target_device: target device where should be converted from CUDA, currently only support XPU device type within IPEX.
* dist_backend: target distribution backend where should be adopted on the target device, currently only support ccl backend
* compile_backend: target backend for torch.compile path, default is inductor

# Usage 
* Only single line is needed to add into model script, as following,
```python
import torch
import intel_extension_for_pytorch as ipex
 
ipex.compatible_mode(...)
```
* Please be noticed, the ipex.compatible_mode(...) must be invoked 
    - at the very beginning of the workload.
    - only once on the global scale of the workload.
    - immediately after import the torch and IPEX.
* With above, run model script as usual without other changes in model script. The workload will run on XPU device along with original CUDA path.
