<div align="center">
  
Intel® Extension for Pytorch*
===========================

[💻Examples](./docs/tutorials/examples.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[📖CPU Documentations](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[📖GPU Documentations](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/)
</div>



Intel® Extension for PyTorch\* extends PyTorch\* with up-to-date features optimizations for an extra performance boost on Intel hardware. Optimizations take advantage of AVX-512 Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX) on Intel CPUs as well as Intel X<sup>e</sup> Matrix Extensions (XMX) AI engines on Intel discrete GPUs. Moreover, through PyTorch\* `xpu` device, Intel® Extension for PyTorch\* provides easy GPU acceleration for Intel discrete GPUs with PyTorch\*.

Intel® Extension for PyTorch\* provides optimizations for both eager mode and graph mode, however, compared to eager mode, graph mode in PyTorch\* normally yields better performance from optimization techniques, such as operation fusion. Intel® Extension for PyTorch\* amplifies them with more comprehensive graph optimizations. Therefore we recommend you to take advantage of Intel® Extension for PyTorch\* with [TorchScript](https://pytorch.org/docs/stable/jit.html) whenever your workload supports it. You could choose to run with `torch.jit.trace()` function or `torch.jit.script()` function, but based on our evaluation, `torch.jit.trace()` supports more workloads so we recommend you to use `torch.jit.trace()` as your first choice.

The extension can be loaded as a Python module for Python programs or linked as a C++ library for C++ programs. In Python scripts users can enable it dynamically by importing `intel_extension_for_pytorch`.

* Check [CPU tutorial](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/) for detailed information of Intel® Extension for PyTorch\* for Intel® CPUs. Source code is available at the [main branch](https://github.com/intel/intel-extension-for-pytorch/tree/main).
* Check [GPU tutorial](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/) for detailed information of Intel® Extension for PyTorch\* for Intel® GPUs. Source code is available at the [xpu-main branch](https://github.com/intel/intel-extension-for-pytorch/tree/xpu-main).



## Large Language Models (LLMs) Optimization

In the current technological landscape, Generative AI (GenAI) workloads and models have gained widespread attention and popularity. Large Language Models (LLMs) have emerged as the dominant models driving these GenAI applications. Starting from 2.1.0, specific optimizations for certain LLM models are introduced in the Intel® Extension for PyTorch\*. Check [LLM optimizations CPU](./examples/cpu/inference/python/llm) and [LLM optimizations GPU](./examples/gpu/inference/python/llm) for details.


## Installation

### CPU version

You can use either of the following 2 commands to install Intel® Extension for PyTorch\* CPU version.

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install intel-extension-for-pytorch --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
# for PRC user, you can check with the following link
python -m pip install intel-extension-for-pytorch --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/cn/
```

**Note:** Intel® Extension for PyTorch\* has PyTorch version requirement. Please check more detailed information via the URL below.

More installation methods can be found at [CPU Installation Guide](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/installation.html).

Compilation instruction of the latest CPU code base `main` branch can be found in the session Package `source` at [CPU Installation Guide](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/installation.html).

### GPU version

You can install Intel® Extension for PyTorch\* for GPU via command below.

```bash
python -m pip install torch==2.1.0.post2 torchvision==0.16.0.post2 torchaudio==2.1.0.post2 intel-extension-for-pytorch==2.1.30+xpu oneccl_bind_pt==2.1.300+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ 
# for PRC user, you can check with the following link
python -m pip install torch==2.1.0.post2 torchvision==0.16.0.post2 torchaudio==2.1.0.post2 intel-extension-for-pytorch==2.1.30+xpu oneccl_bind_pt==2.1.300+xpu  --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/

```

**Note:** The patched PyTorch 2.1.0 is required to work with Intel® Extension for PyTorch\* on Intel® graphics card for now.

More installation methods can be found at [GPU Installation Guide](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installation.html).

Compilation instruction of the latest GPU code base `xpu-main` branch can be found in the session Package `source` at [GPU Installation Guide](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installation.html).

## Getting Started

Minor code changes are required for users to get start with Intel® Extension for PyTorch\*. Both PyTorch imperative mode and TorchScript mode are supported. You just need to import Intel® Extension for PyTorch\* package and apply its optimize function against the model object. If it is a training workload, the optimize function also needs to be applied against the optimizer object.

The following code snippet shows an inference code with FP32 data type. More examples on CPU, including training and C++ examples, are available at [CPU Example page](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/examples.html). More examples on GPU are available at [GPU Example page](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/examples.html).

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
model = model.to('xpu')
data = data.to('xpu')
model = ipex.optimize(model)

with torch.no_grad():
  model(data)
```

## License

_Apache License_, Version _2.0_. As found in [LICENSE](https://github.com/intel/intel-extension-for-pytorch/blob/main/LICENSE) file.

## Security

See Intel's [Security Center](https://www.intel.com/content/www/us/en/security-center/default.html)
for information on how to report a potential security issue or vulnerability.

See also: [Security Policy](SECURITY.md)


