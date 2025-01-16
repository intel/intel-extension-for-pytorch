<div align="center">
  
Intel® Extension for PyTorch*
=============================

</div>

**CPU** [💻main branch](https://github.com/intel/intel-extension-for-pytorch/tree/main)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[🌱Quick Start](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/getting_started.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[📖Documentations](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[🏃Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=cpu)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[💻LLM Example](https://github.com/intel/intel-extension-for-pytorch/tree/main/examples/cpu/llm) <br>
**GPU** [💻main branch](https://github.com/intel/intel-extension-for-pytorch/tree/xpu-main)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[🌱Quick Start](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/getting_started.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[📖Documentations](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[🏃Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[💻LLM Example](https://github.com/intel/intel-extension-for-pytorch/tree/xpu-main/examples/gpu/llm)<br>  

Intel® Extension for PyTorch\* extends PyTorch\* with up-to-date features optimizations for an extra performance boost on Intel hardware. Optimizations take advantage of AVX-512 Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX) on Intel CPUs as well as Intel X<sup>e</sup> Matrix Extensions (XMX) AI engines on Intel discrete GPUs. Moreover, through PyTorch\* `xpu` device, Intel® Extension for PyTorch\* provides easy GPU acceleration for Intel discrete GPUs with PyTorch\*.

Intel® Extension for PyTorch\* provides optimizations for both eager mode and graph mode, however, compared to eager mode, graph mode in PyTorch\* normally yields better performance from optimization techniques, such as operation fusion. Intel® Extension for PyTorch\* amplifies them with more comprehensive graph optimizations. Therefore we recommend you to take advantage of Intel® Extension for PyTorch\* with [TorchScript](https://pytorch.org/docs/stable/jit.html) whenever your workload supports it. You could choose to run with `torch.jit.trace()` function or `torch.jit.script()` function, but based on our evaluation, `torch.jit.trace()` supports more workloads so we recommend you to use `torch.jit.trace()` as your first choice.

The extension can be loaded as a Python module for Python programs or linked as a C++ library for C++ programs. In Python scripts users can enable it dynamically by importing `intel_extension_for_pytorch`.

* Check [CPU tutorial](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/) for detailed information of Intel® Extension for PyTorch\* for Intel® CPUs. Source code is available at the [main branch](https://github.com/intel/intel-extension-for-pytorch/tree/main).
* Check [GPU tutorial](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/) for detailed information of Intel® Extension for PyTorch\* for Intel® GPUs. Source code is available at the [xpu-main branch](https://github.com/intel/intel-extension-for-pytorch/tree/xpu-main).



## Large Language Models (LLMs) Optimization

In the current technological landscape, Generative AI (GenAI) workloads and models have gained widespread attention and popularity. Large Language Models (LLMs) have emerged as the dominant models driving these GenAI applications. Starting from 2.1.0, specific optimizations for certain LLM models are introduced in the Intel® Extension for PyTorch\*. Check [LLM optimizations CPU](./examples/cpu/llm) and [LLM optimizations GPU](./examples/gpu/llm) for details.

### Optimized Model List 

#### LLM Inference

| MODEL FAMILY | Verified < MODEL ID > (Huggingface hub)| FP16 | Weight only quantization INT4 | Optimized on Intel® Data Center GPU Max Series (1550/1100) | Optimized on Intel® Arc™ A-Series Graphics (A770) | Optimized on Intel® Arc™ B-Series Graphics (B580) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
|Llama 2| "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-70b-hf" |✅| ✅|✅|✅|$✅^1$|
|Llama 3| "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B" |✅| ✅|✅|✅|$✅^2$|
|Phi-3 mini| "microsoft/Phi-3-mini-128k-instruct", "microsoft/Phi-3-mini-4k-instruct" |✅| ✅|✅|✅|$✅^3$|
|GPT-J| "EleutherAI/gpt-j-6b" | ✅ | ✅ |✅ | ✅||
|Qwen|"Qwen/Qwen2-7B"|✅ | ✅ |✅ | ✅||
|Qwen|"Qwen/Qwen2-7B-Instruct"| | | | |✅|
|OPT|"facebook/opt-6.7b", "facebook/opt-30b"| ✅ |  |✅| ||
|Bloom|"bigscience/bloom-7b1", "bigscience/bloom"| ✅ |  |✅ |  ||
|ChatGLM3-6B|"THUDM/chatglm3-6b"| ✅ |  |✅ |  ||
|Baichuan2-13B|"baichuan-inc/Baichuan2-13B-Chat"| ✅ |  |✅|  ||

| Benchmark mode | FP16 | Weight only quantization INT4 |
|---|:---:|:---:|
|Single instance | ✅ | ✅ |
| Distributed (autotp) |  ✅ |  |


#### LLM fine-tuning

 **Note**: 
 Intel® Data Center Max 1550 GPU: support all the models in the model list above. Intel® Core™ Ultra Processors with Intel® Arc™ Graphics: support Llama 2 7B, Llama 3 8B and Phi-3-Mini 3.8B.

| MODEL FAMILY | Verified < MODEL ID > (Hugging Face hub)| Mixed Precision (BF16+FP32) | Full fine-tuning | LoRA | Intel® Data Center Max 1550 GPU | Intel® Core™ Ultra Processors with Intel® Arc™ Graphics |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
|Llama 2 7B| "meta-llama/Llama-2-7b-hf" | ✅ | ✅ | ✅ | ✅ | ✅ |
|Llama 2 70B| "meta-llama/Llama-2-70b-hf" | ✅ |  |✅ | ✅ |  |
|Llama 3 8B| "meta-llama/Meta-Llama-3-8B" | ✅ | ✅ |✅ | ✅ | ✅ |
|Qwen 7B|"Qwen/Qwen-7B"| ✅ | ✅ |✅ | ✅| |
|Phi-3-mini 3.8B|"Phi-3-mini-4k-instruct"| ✅ | ✅ |✅ |  | ✅ |



| Benchmark mode | Full fine-tuning | LoRA |
|---|:---:|:---:|
|Single-GPU |  | ✅ |
|Multi-GPU (FSDP) |  ✅ | ✅ |

- ✅ signifies that it is supported.

- A blank signifies that it is not supported yet.
  
- 1: signifies that Llama-2-7b-hf is verified.

- 2: signifies that Meta-Llama-3-8B is verified.
  
- 3: signifies that Phi-3-mini-4k-instruct is verified.

## Support

The team tracks bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-pytorch/issues/). Before submitting a suggestion or bug report, search the existing GitHub issues to see if your issue has already been reported.

## License

_Apache License_, Version _2.0_. As found in [LICENSE](https://github.com/intel/intel-extension-for-pytorch/blob/main/LICENSE) file.

## Security

See Intel's [Security Center](https://www.intel.com/content/www/us/en/security-center/default.html)
for information on how to report a potential security issue or vulnerability.

See also: [Security Policy](SECURITY.md)



