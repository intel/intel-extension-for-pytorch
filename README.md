<div align="center">
  
Intel® Extension for PyTorch\*
==============================

</div>

**CPU** [💻main branch](https://github.com/intel/intel-extension-for-pytorch/tree/main)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[🌱Quick Start](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/getting_started.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[📖Documentations](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[🏃Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=cpu&version=v2.2.0%2Bcpu)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[💻LLM Example](https://github.com/intel/intel-extension-for-pytorch/tree/main/examples/cpu/inference/python/llm) <br>
**GPU** [💻main branch](https://github.com/intel/intel-extension-for-pytorch/tree/xpu-main)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[🌱Quick Start](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/getting_started.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[📖Documentations](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[🏃Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.1.10%2Bxpu)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[💻LLM Example](https://github.com/intel/intel-extension-for-pytorch/tree/xpu-main/examples/gpu/inference/python/llm)<br>  


Intel® Extension for PyTorch\* extends PyTorch\* with up-to-date features optimizations for an extra performance boost on Intel hardware. Optimizations take advantage of Intel® Advanced Vector Extensions 512 (Intel® AVX-512) Vector Neural Network Instructions (VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX) on Intel CPUs as well as Intel X<sup>e</sup> Matrix Extensions (XMX) AI engines on Intel discrete GPUs. Moreover, Intel® Extension for PyTorch\* provides easy GPU acceleration for Intel discrete GPUs through the PyTorch\* xpu device.

Intel® Extension for PyTorch\* provides optimizations both for eager and graph modes. However,  compared to the eager mode, the graph mode in PyTorch\* normally yields better performance from the optimization techniques like operation fusion. Intel® Extension for PyTorch\* amplifies them with more comprehensive graph optimizations. Both PyTorch `Torchscript` and `TorchDynamo` graph modes are supported. With `Torchscript`, we recommend using `torch.jit.trace()` as your preferred option, as it generally supports a wider range of workloads compared to `torch.jit.script()`.

## ipex.llm - Large Language Models (LLMs) Optimization

In the current technological landscape, Generative AI (GenAI) workloads and models have gained widespread attention and popularity. Large Language Models (LLMs) have emerged as the dominant models driving these GenAI applications. Starting from 2.1.0, specific optimizations for certain LLM models are introduced in the Intel® Extension for PyTorch\*. Check [**LLM optimizations**](./examples/cpu/inference/python/llm) for details.

### Optimized Model List

| MODEL FAMILY | MODEL NAME (Huggingface hub) | FP32 | BF16 | Static quantization INT8 | Weight only quantization INT8 | Weight only quantization INT4 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|LLAMA| meta-llama/Llama-2-7b-hf | ✅ | ✅ | ✅ | ✅ | ☑️ | 
|LLAMA| meta-llama/Llama-2-13b-hf | ✅ | ✅ | ✅ | ✅ | ☑️ | 
|LLAMA| meta-llama/Llama-2-70b-hf | ✅ | ✅ | ✅ | ✅ | ☑️ | 
|GPT-J| EleutherAI/gpt-j-6b | ✅ | ✅ | ✅ | ✅ | ✅ | 
|GPT-NEOX| EleutherAI/gpt-neox-20b | ✅ | ✅ | ☑️ | ✅ | ☑️ | 
|DOLLY| databricks/dolly-v2-12b | ✅ | ✅ | ☑️ | ☑️ | ☑️ | 
|FALCON| tiiuae/falcon-40b | ✅ | ✅ | ✅ |  ✅ | ✅ | 
|OPT| facebook/opt-30b | ✅ | ✅ | ✅ |    | ☑️ | 
|OPT| facebook/opt-1.3b | ✅ | ✅ | ✅ |  ✅ | ☑️ | 
|Bloom| bigscience/bloom-1b7 | ✅ | ☑️ | ✅ |    | ☑️ |
|CodeGen| Salesforce/codegen-2B-multi | ✅ | ✅ | ☑️ |  ✅ | ✅ |
|Baichuan| baichuan-inc/Baichuan2-7B-Chat | ✅ | ✅ | ✅ | ✅  |    |
|Baichuan| baichuan-inc/Baichuan2-13B-Chat | ✅ | ✅ |    |  ✅ |    |
|Baichuan| baichuan-inc/Baichuan-13B-Chat | ✅ | ☑️ | ✅ |    |    |
|ChatGLM| THUDM/chatglm3-6b | ✅ | ✅ | ☑️ |  ✅ |    |
|ChatGLM| THUDM/chatglm2-6b | ✅ | ☑️ | ☑️ |  ☑️ |    |
|GPTBigCode| bigcode/starcoder | ✅ | ✅ | ☑️ |  ✅ | ☑️ |
|T5| google/flan-t5-xl | ✅ | ✅ | ☑️ |  ✅ |    |
|Mistral| mistralai/Mistral-7B-v0.1 | ✅ | ✅ | ☑️ |  ✅ | ☑️ |
|MPT| mosaicml/mpt-7b | ✅ | ✅ | ☑️ |  ✅ | ✅ |

*Note*: All above models have undergone thorough optimization and verification processes for both performance and accuracy. In the context of the optimized model list table above, the symbol ✅ signifies that the model can achieve an accuracy drop of less than 1% when using a specific data type compared to FP32, whereas the accuracy drop may exceed 1% for ☑️ marked ones. We are working in progress to better support the models in the table with various data types. In addition, more models will be optimized, which will expand the table.

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

