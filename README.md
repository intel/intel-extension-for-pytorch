# Intel® Extension for PyTorch\*

Intel® Extension for PyTorch\* extends PyTorch with optimizations for extra performance boost on Intel hardware. Most of the optimizations will be included in stock PyTorch releases eventually, and the intention of the extension is to deliver up-to-date features and optimizations for PyTorch on Intel hardware, examples include AVX-512 Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX).

Intel® Extension for PyTorch\* is loaded as a Python module for Python programs or linked as a C++ library for C++ programs. Users can enable it dynamically in script by importing `intel_extension_for_pytorch`. It covers optimizations for both imperative mode and graph mode. Optimized operators and kernels are registered through PyTorch dispatching mechanism. These operators and kernels are accelerated from native vectorization feature and matrix calculation feature of Intel hardware. During execution, Intel® Extension for PyTorch\* intercepts invocation of ATen operators, and replace the original ones with these optimized ones. In graph mode, further operator fusions are applied manually by Intel engineers or through a tool named *oneDNN Graph* to reduce operator/kernel invocation overheads, and thus increase performance.

More detailed tutorials are available at [**Intel® Extension for PyTorch\* online document website**](https://intel.github.io/intel-extension-for-pytorch/).

## Installation

You can use either of the following 2 commands to install Intel® Extension for PyTorch\*.

```python
python -m pip install intel_extension_for_pytorch
```

```python
python -m pip install intel_extension_for_pytorch -f https://software.intel.com/ipex-whl-stable
```

**Note:** Intel® Extension for PyTorch\* has PyTorch version requirement. Please check more detailed information via the URL below.

More installation methods can be found at [Installation Guide](https://intel.github.io/intel-extension-for-pytorch/tutorials/installation.html)

## Getting Started

Minor code changes are required for users to get start with Intel® Extension for PyTorch\*. Both PyTorch imperative mode and TorchScript mode are supported. You just need to import Intel® Extension for PyTorch\* package and apply its optimize function against the model object. If it is a training workload, the optimize function also needs to be applied against the optimizer object.

The following code snippet shows an inference code with FP32 data type. More examples, including training and C++ examples, are available at [Example page](https://intel.github.io/intel-extension-for-pytorch/tutorials/examples.html).

```python
import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)
model.eval()
data = torch.rand(1, 3, 224, 224)

import intel_extension_for_pytorch as ipex
model = model.to(memory_format=torch.channels_last)
model = ipex.optimize(model)
data = data.to(memory_format=torch.channels_last)

with torch.no_grad():
  model(data)
```

## Model Zoo

Use cases that had already been optimized by Intel engineers are available at [Model Zoo for Intel® Architecture](https://github.com/IntelAI/models/tree/pytorch-r1.10-models). A bunch of PyTorch use cases for benchmarking are also available on the [Github page](https://github.com/IntelAI/models/tree/pytorch-r1.10-models/benchmarks#pytorch-use-cases). You can get performance benefits out-of-box by simply running scipts in the Model Zoo.

## License

_Apache License_, Version _2.0_. As found in [LICENSE](https://github.com/intel/intel-extension-for-pytorch/blob/master/LICENSE.txt) file.

## Security

See Intel's [Security Center](https://www.intel.com/content/www/us/en/security-center/default.html)
for information on how to report a potential security issue or vulnerability.

See also: [Security Policy](SECURITY.md)

