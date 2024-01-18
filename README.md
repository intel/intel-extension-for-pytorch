<div align="center">
  
Intel® Extension for Pytorch*
===========================

[💻Examples](./docs/tutorials/examples.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[📖CPU Documentations](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[📖GPU Documentations](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/)
</div>



Intel® Extension for PyTorch\* extends PyTorch\* with up-to-date features optimizations for an extra performance boost on Intel hardware. Optimizations take advantage of Intel® Advanced Vector Extensions 512 (Intel® AVX-512) Vector Neural Network Instructions (VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX) on Intel CPUs as well as Intel X<sup>e</sup> Matrix Extensions (XMX) AI engines on Intel discrete GPUs. Moreover, Intel® Extension for PyTorch* provides easy GPU acceleration for Intel discrete GPUs through the PyTorch* xpu device.

Intel® Extension for PyTorch\* provides optimizations both for eager and graph modes. However,  compared to the eager mode, the graph mode in PyTorch* normally yields better performance from the optimization techniques like operation fusion. Intel® Entension for PyTorch* amplifies them with more comprehensive graph optimizations. Both PyTorch `Torchscript` and `TorchDynamo` graph modes are supported. With `Torchscript`, we recommend using `torch.jit.trace()` as your preferred option, as it generally supports a wider range of workloads compared to `torch.jit.script()`.

The extension can be loaded as a Python module for Python programs or linked as a C++ library for C++ programs. In Python scripts, you can enable it dynamically by importing `intel_extension_for_pytorch`.

* **CPU**: [main branch](https://github.com/intel/intel-extension-for-pytorch/tree/main) | [Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=cpu&version=v2.1.0%2Bcpu) | [Quick Start](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/getting_started.html) | [Documentation](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/).
* **XPU**: [xpu-main branch](https://github.com/intel/intel-extension-for-pytorch/tree/xpu-main) | [Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.1.0%2Bxpu>) | [Quick Start](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/getting_started.html) | [Documentation](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/).

## Large Language Models (LLMs) Optimization

In the current technological landscape, Generative AI (GenAI) workloads and models have gained widespread attention and popularity. Large Language Models (LLMs) have emerged as the dominant models driving these GenAI applications. Starting from 2.1.0, specific optimizations for certain LLM models are introduced in the Intel® Extension for PyTorch\*. Check [LLM optimizations CPU](./examples/cpu/inference/python/llm) and [LLM optimizations GPU](./examples/gpu/inference/python/llm) for details.


## Installation

### CPU version

Use one of the following commands to install the CPU version of Intel® Extension for PyTorch\*.

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install intel-extension-for-pytorch --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
# for PRC user, you can check with the following link
python -m pip install intel-extension-for-pytorch --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/cn/
```

**Note:** Intel® Extension for PyTorch\* has PyTorch version requirement. Intel® Extension for PyTorch* [v2.1.100+cpu](https://github.com/intel/intel-extension-for-pytorch/tree/v2.1.100%2Bcpu) requires PyTorch*/libtorch [v2.1.*](https://github.com/pytorch/pytorch/tree/v2.1.1) to be installed.

For more installation methods and installation guidance for previous versions, refer to [Installation](https://intel.github.io/intel-extension-for-pytorch/#installation).

### GPU version

Use the command below to install Intel® Extension for PyTorch\* for GPU:

```python
python -m pip install torch==2.1.0a0 torchvision==0.16.0a0 torchaudio==2.1.0a0 intel-extension-for-pytorch==2.1.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

**Note:** Intel® Extension for PyTorch* v2.1.10+xpu requires PyTorch*/libtorch [v2.1.*](https://github.com/pytorch/pytorch/tree/v2.1.0) (patches needed) to be installed.

For more installation methods and installation guidance for previous versions, refer to [Installation](https://intel.github.io/intel-extension-for-pytorch/#installation).


## Getting Started

The following resources will help you get started with the Intel® Extension for PyTorch*:

* **CPU**: [Quick Start](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/getting_started.html) | [Examples](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/examples.html) 
* **XPU**: [Quick Start](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/getting_started.html) | [Examples](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/examples.html)

## Intel® AI Reference Models

Use cases that have already been optimized by Intel engineers are available at [Intel® AI Reference Models](https://github.com/IntelAI/models/tree/pytorch-r2.1-models). A bunch of PyTorch use cases for benchmarking are also available on the [Github page](https://github.com/IntelAI/models/tree/pytorch-r2.1-models/benchmarks#pytorch-use-cases). You can get performance benefits out-of-box by simply running scripts in the Intel® AI Reference Models.

## Support

The team tracks bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-pytorch/issues/). Before submitting a suggestion or bug report, search the existing GitHub issues to see if your issue has already been reported.

## License

_Apache License_, Version _2.0_. As found in [LICENSE](https://github.com/intel/intel-extension-for-pytorch/blob/main/LICENSE) file.

## Security

See Intel's [Security Center](https://www.intel.com/content/www/us/en/security-center/default.html)
for information on how to report a potential security issue or vulnerability.

See also: [Security Policy](SECURITY.md)

