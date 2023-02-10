# Getting Started

## Installation

Prebuilt wheel files are released for multiple Python versions. You can install them simply with the following pip command.

```bash
python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install intel_extension_for_pytorch
```

You can run a simple sanity test to double confirm if the correct version is installed, and if the software stack can get correct hardware information onboard your system.

```bash
python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__);"
```

More detailed instructions can be found at [Installation Guide](./installation.md).


## Coding

Intel® Extension for PyTorch\* doesn't require complex code changes to get it working. Usage is as simple as several-line code change.

In general, APIs invocation should follow orders below.

1. `import intel_extension_for_pytorch as ipex`
2. Invoke `optimize()` function to apply optimizations.
3. For Torchscript, invoke `torch.jit.trace()` and `torch.jit.freeze()`.

**Note:** It is highly recommended to `import intel_extension_for_pytorch` right after `import torch`, prior to importing other packages.

```python
import torch
####### import ipex ########
import intel_extension_for_pytorch as ipex
############################

model = Model()
model.eval()
data = ...
dtype=torch.float32 # torch.bfloat16

##### ipex.optimize() ######
model = ipex.optimize(model, dtype=dtype)
############################

########## FP32 ############
with torch.no_grad():
####### BF16 on CPU ########
with torch.no_grad(), with torch.cpu.amp.autocast():
############################
  ###### Torchscript #######
  model = torch.jit.trace(model, data)
  model = torch.jit.freeze(model)
  ###### Torchscript #######

  model(data)
```

More examples, including training and usage of low precision data types are available at [Examples](./examples.md).
