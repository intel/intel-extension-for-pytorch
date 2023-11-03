# Quick Start

The following instructions assume you have installed the Intel® Extension for PyTorch\*. For installation instructions, refer to [Installation](../../../index.html#installation?platform=cpu&version=v2.1.0%2Bcpu).

To start using the Intel® Extension for PyTorch\* in your code, you need to make the following changes:

1. Import the extension with `import intel_extension_for_pytorch as ipex`.
2. Invoke the `optimize()` function to apply optimizations.
3. Convert the imperative model to a graph model.
    - For TorchScript, invoke `torch.jit.trace()` and `torch.jit.freeze()`
    - For TorchDynamo, invoke `torch.compile(model, backend="ipex")`(*Experimental feature*)

**Important:** It is highly recommended to `import intel_extension_for_pytorch` right after `import torch`, prior to importing other packages.

The example below demostrates how to use the Intel® Extension for PyTorch\*:


```python
import torch
############## import ipex ###############
import intel_extension_for_pytorch as ipex
##########################################

model = Model()
model.eval()
data = ...

############## TorchScript ###############
model = ipex.optimize(model, dtype=torch.bfloat16)

with torch.no_grad(), torch.cpu.amp.autocast():
  model = torch.jit.trace(model, data)
  model = torch.jit.freeze(model)
  model(data)
##########################################

############## TorchDynamo ###############
model = ipex.optimize(model, weights_prepack=False)

model = torch.compile(model, backend="ipex")
with torch.no_grad():
  model(data)
##########################################
```

More examples, including training and usage of low precision data types are available in the [Examples](./examples.md) section.

In [Cheat Sheet](cheat_sheet.md), you can find more commands that can help you start using the Intel® Extension for PyTorch\*.
