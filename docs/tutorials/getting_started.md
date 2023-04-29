# Getting Started

## Installation

Prebuilt wheel files are released for multiple Python versions. You can install them simply with the following pip command.

```bash
python -m pip install torch==1.13.0a0+git6c9b55e torchvision==0.14.1a0 intel_extension_for_pytorch==1.13.120+xpu -f https://developer.intel.com/ipex-whl-stable-xpu
```

You can run a simple sanity test to double confirm if the correct version is installed, and if the software stack can get correct hardware information onboard your system.

```bash
python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__); [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];"
```

More detailed instructions can be found at [Installation Guide](./installation.md).


## Coding

Intel® Extension for PyTorch\* doesn't require complex code changes to get it working. Usage is as simple as several-line code change.

In general, APIs invocation should follow orders below.

1. `import intel_extension_for_pytorch as ipex`
2. Move model and data to GPU with `to('xpu')`, if you want to run on GPU.
3. Invoke `optimize()` function to apply optimizations.
4. For Torchscript, invoke `torch.jit.trace()` and `torch.jit.freeze()`.

**Note:** It is highly recommended to `import intel_extension_for_pytorch` right after `import torch`, prior to importing other packages.

```python
import torch
import intel_extension_for_pytorch as ipex

model = Model()
model.eval() # Set the model to evaluation mode for inference, as required by ipex.optimize() function.
data = ...
dtype=torch.float32 # torch.bfloat16, torch.float16 (float16 only works on GPU)

##### Run on GPU ######
model = model.to('xpu')
data = data.to('xpu')
#######################

model = ipex.optimize(model, dtype=dtype)

########## FP32 ############
with torch.no_grad():
####### BF16 on CPU ########
with torch.no_grad(), with torch.cpu.amp.autocast():
##### BF16/FP16 on GPU #####
with torch.no_grad(), with torch.xpu.amp.autocast(enabled=True, dtype=dtype, cache_enabled=False):
############################
  ###### Torchscript #######
  model = torch.jit.trace(model, data)
  model = torch.jit.freeze(model)
  ###### Torchscript #######

  model(data)
```

More examples, including training and usage of low precision data types are available at [Examples](./examples.md).


## Execution

Execution requires an active Intel® oneAPI environment. Suppose you have the Intel® oneAPI Base Toolkit installed in `/opt/intel/oneapi` directory, activating the environment is as simple as sourcing its environment activation bash scripts.

There are some environment variables in runtime that can be used to configure executions on GPU. Please check [Advanced Configuration](./features/advanced_configuration.html#runtime-configuration) for more detailed information.

```bash
source /opt/intel/oneapi/compiler/latest/env/vars.sh
source /opt/intel/oneapi/mkl/latest/env/vars.sh
python <script>
```
