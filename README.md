# Intel® Extension for PyTorch\*

Intel® Extension for PyTorch\* extends PyTorch with up-to-date features optimizations for an extra performance boost on Intel hardware. Example optimizations use AVX-512 Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX). Over time, most of these optimizations will be included directly into stock PyTorch releases. More importantly, Intel® Extension for PyTorch\* provides easy GPU acceleration for Intel® discrete graphics cards with PyTorch\*.

Intel® Extension for PyTorch\* provides optimizations for both eager mode and graph mode, however, compared to eager mode, graph mode in PyTorch normally yields better performance from optimization techniques such as operation fusion, and Intel® Extension for PyTorch\* amplified them with more comprehensive graph optimizations. Therefore we recommended you to take advantage of Intel® Extension for PyTorch\* with [TorchScript](https://pytorch.org/docs/stable/jit.html) whenever your workload supports it. You could choose to run with `torch.jit.trace()` function or `torch.jit.script()` function, but based on our evaluation, `torch.jit.trace()` supports more workloads so we recommend you to use `torch.jit.trace()` as your first choice. On Intel® graphics cards, through registering feature implementations into PyTorch\* as torch.xpu, PyTorch\* scripts work on Intel® discrete graphics cards.

The extension can be loaded as a Python module for Python programs or linked as a C++ library for C++ programs. In Python scripts users can enable it dynamically by importing `intel_extension_for_pytorch`.

More detailed tutorials are available at **Intel® Extension for PyTorch\* online document website**. Both [CPU version](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/) and [XPU/GPU version](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/) are available.

## Installation

### CPU version

You can use either of the following 2 commands to install Intel® Extension for PyTorch\* CPU version.

```python
python -m pip install intel_extension_for_pytorch
```

```python
python -m pip install intel_extension_for_pytorch -f https://software.intel.com/ipex-whl-stable-cpu
```

**Note:** Intel® Extension for PyTorch\* has PyTorch version requirement. Please check more detailed information via the URL below.

More installation methods can be found at [CPU Installation Guide](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/installation.html)

### XPU/GPU version

You can install Intel® Extension for PyTorch\* for XPU/GPU via command below.

```python
python -m pip install torch==1.10.0a0 -f https://developer.intel.com/ipex-whl-stable-xpu
python -m pip install intel_extension_for_pytorch==1.10.200+gpu -f https://software.intel.com/ipex-whl-stable-xpu
```

**Note:** The patched PyTorch 1.10.0a0 is required to work with Intel® Extension for PyTorch\* on Intel® graphics card for now.

More installation methods can be found at [XPU/GPU Installation Guide](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installation.html)

## Getting Started

Minor code changes are required for users to get start with Intel® Extension for PyTorch\*. Both PyTorch imperative mode and TorchScript mode are supported. You just need to import Intel® Extension for PyTorch\* package and apply its optimize function against the model object. If it is a training workload, the optimize function also needs to be applied against the optimizer object.

The following code snippet shows an inference code with FP32 data type. More examples on CPU, including training and C++ examples, are available at [CPU Example page](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/examples.html). More examples on XPU/GPU are available at [XPU/GPU Example page](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/examples.html).

### Inference on CPU

```python
import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)
model.eval()
data = torch.rand(1, 3, 224, 224)

import intel_extension_for_pytorch as ipex
model = ipex.optimize(model)

with torch.no_grad():
  model(data)
```

### Inference on GPU

```python
import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)
model.eval()
data = torch.rand(1, 3, 224, 224)

import intel_extension_for_pytorch as ipex
model.to('xpu')
data.to('xpu')
model = ipex.optimize(model)

with torch.no_grad():
  model(data)
```

## Model Zoo

Use cases that had already been optimized by Intel engineers are available at [Model Zoo for Intel® Architecture](https://github.com/IntelAI/models/tree/pytorch-r1.13-models). A bunch of PyTorch use cases for benchmarking are also available on the [Github page](https://github.com/IntelAI/models/tree/pytorch-r1.13-models/benchmarks#pytorch-use-cases). You can get performance benefits out-of-box by simply running scipts in the Model Zoo.

## License

_Apache License_, Version _2.0_. As found in [LICENSE](https://github.com/intel/intel-extension-for-pytorch/blob/master/LICENSE.txt) file.

## Security

See Intel's [Security Center](https://www.intel.com/content/www/us/en/security-center/default.html)
for information on how to report a potential security issue or vulnerability.

See also: [Security Policy](SECURITY.md)

