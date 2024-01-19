<div align="center">
  
Intel® Extension for Pytorch*
===========================

</div>

**CPU** [💻main branch](https://github.com/intel/intel-extension-for-pytorch/tree/main)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[🌱Quick Start](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/getting_started.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[📖Documentations](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[🏃Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=cpu&version=v2.2.0%2Bcpu)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[💻LLM Example](https://github.com/intel/intel-extension-for-pytorch/tree/main/examples/cpu/inference/python/llm) <br>
**GPU** [💻main branch](https://github.com/intel/intel-extension-for-pytorch/tree/xpu-main)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[🌱Quick Start](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/getting_started.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[📖Documentations](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[🏃Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.1.10%2Bxpu)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[💻LLM Example](https://github.com/intel/intel-extension-for-pytorch/tree/xpu-main/examples/gpu/inference/python/llm)<br>  



Intel® Extension for PyTorch\* extends PyTorch\* with up-to-date features optimizations for an extra performance boost on Intel hardware. Optimizations take advantage of Intel® Advanced Vector Extensions 512 (Intel® AVX-512) Vector Neural Network Instructions (VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX) on Intel CPUs as well as Intel X<sup>e</sup> Matrix Extensions (XMX) AI engines on Intel discrete GPUs. Moreover, Intel® Extension for PyTorch\* provides easy GPU acceleration for Intel discrete GPUs through the PyTorch\* xpu device.




## ipex.llm - Large Language Models (LLMs) Optimization

In the current technological landscape, Generative AI (GenAI) workloads and models have gained widespread attention and popularity. Large Language Models (LLMs) have emerged as the dominant models driving these GenAI applications. Starting from 2.1.0, specific optimizations for certain LLM models are introduced in the Intel® Extension for PyTorch\*. Check [**LLM optimizations**](./examples/cpu/inference/python/llm) for details.


## Support

The team tracks bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-pytorch/issues/). Before submitting a suggestion or bug report, search the existing GitHub issues to see if your issue has already been reported.

## Intel® AI Reference Models

Use cases that had already been optimized by Intel engineers are available at [Intel® AI Reference Models](https://github.com/IntelAI/models/tree/pytorch-r2.2.0-models) (former Model Zoo). A bunch of PyTorch use cases for benchmarking are also available on the [Github page](https://github.com/IntelAI/models/tree/pytorch-r2.2.0-models/benchmarks#pytorch-use-cases). You can get performance benefits out-of-box by simply running the scripts in the Reference Models.

## License

_Apache License_, Version _2.0_. As found in [LICENSE](https://github.com/intel/intel-extension-for-pytorch/blob/main/LICENSE) file.

## Security

See Intel's [Security Center](https://www.intel.com/content/www/us/en/security-center/default.html)
for information on how to report a potential security issue or vulnerability.

See also: [Security Policy](SECURITY.md)

