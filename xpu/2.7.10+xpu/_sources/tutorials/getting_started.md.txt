# Quick Start

The following instructions assume you have installed the Intel® Extension for PyTorch\*. For installation instructions, refer to [Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.1.30%2bxpu).

To start using the Intel® Extension for PyTorch\* in your code, you need to make the following changes:

1. Import the extension with `import intel_extension_for_pytorch as ipex`.
2. Move model and data to GPU with `to('xpu')`, if you want to run on GPU.
3. Invoke the `optimize()` function to apply optimizations.
3. For TorchScript, invoke `torch.jit.trace()` and `torch.jit.freeze()`.

**Important:** It is highly recommended to `import intel_extension_for_pytorch` right after `import torch`, prior to importing other packages.

The example below demostrates how to use the Intel® Extension for PyTorch\*:

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
with torch.no_grad(), torch.cpu.amp.autocast():
##### BF16/FP16 on GPU #####
with torch.no_grad(), torch.xpu.amp.autocast(enabled=True, dtype=dtype, cache_enabled=False):
############################
  ###### Torchscript #######
  model = torch.jit.trace(model, data)
  model = torch.jit.freeze(model)
  ###### Torchscript #######

  model(data)
```

More examples, including training and usage of low precision data types are available at [Examples](./examples.md).


## Execution

There are some environment variables in runtime that can be used to configure executions on GPU. Please check [Advanced Configuration](./features/advanced_configuration.html#runtime-configuration) for more detailed information.

