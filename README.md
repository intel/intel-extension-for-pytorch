<div align="center">
  
Intel® Extension for PyTorch\*
===========================

</div>

**CPU** [💻main branch](https://github.com/intel/intel-extension-for-pytorch/tree/main)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[🌱Quick Start](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/getting_started.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[📖Documentations](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[🏃Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=cpu&version=v2.2.0%2Bcpu)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[💻LLM Example](https://github.com/intel/intel-extension-for-pytorch/tree/main/examples/cpu/inference/python/llm) <br>
**GPU** [💻main branch](https://github.com/intel/intel-extension-for-pytorch/tree/xpu-main)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[🌱Quick Start](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/getting_started.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[📖Documentations](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[🏃Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.1.10%2Bxpu)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[💻LLM Example](https://github.com/intel/intel-extension-for-pytorch/tree/xpu-main/examples/gpu/inference/python/llm)<br>  

Intel® Extension for PyTorch\* extends PyTorch\* with up-to-date features optimizations for an extra performance boost on Intel hardware. Optimizations take advantage of Intel® Advanced Vector Extensions 512 (Intel® AVX-512) Vector Neural Network Instructions (VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX) on Intel CPUs as well as Intel X<sup>e</sup> Matrix Extensions (XMX) AI engines on Intel discrete GPUs. Moreover, Intel® Extension for PyTorch* provides easy GPU acceleration for Intel discrete GPUs through the PyTorch* xpu device.

## ipex.llm - Large Language Models (LLMs) Optimization

In the current technological landscape, Generative AI (GenAI) workloads and models have gained widespread attention and popularity. Large Language Models (LLMs) have emerged as the dominant models driving these GenAI applications. Starting from 2.1.0, specific optimizations for certain LLM models are introduced in the Intel® Extension for PyTorch\*. Check [**LLM optimizations**](./examples/cpu/inference/python/llm) for details.

### Optimized Model List

| MODEL FAMILY | Verified <MODEL ID> (Huggingface hub)| FP32/BF16 | Weight only quantzation INT8 | Weight only quantization INT4| Static quantization INT8 |
|---|:---:|:---:|:---:|:---:|:---:|
|LLAMA| "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-70b-hf" | ✅ | ✅ | ✅ | ✅ | 
|GPT-J| "EleutherAI/gpt-j-6b" | ✅ | ✅ | ✅ | ✅ | 
|GPT-NEOX| "EleutherAI/gpt-neox-20b", "databricks/dolly-v2-12b" | ✅ | ✅ | ✅ | ✅ | 
|FALCON|"tiiuae/falcon-40b" | ✅ | ✅ |  ✅ | ✅ | 
|OPT|"facebook/opt-30b", "facebook/opt-1.3b"| ✅ | ✅ |  ✅ | ✅ | 
|Bloom|"bigscience/bloom", "bigscience/bloom-1b7"| ✅ | ✅ |  ✅ | ✅ |
|CodeGen|"Salesforce/codegen-2B-multi"| ✅ | ✅ |  ✅ | ✅ |
|Baichuan|"baichuan-inc/Baichuan2-13B-Chat", "baichuan-inc/Baichuan2-7B-Chat", "baichuan-inc/Baichuan-13B-Chat"| ✅ | ✅ |  ✅ | ✅ |
|ChatGLM|"THUDM/chatglm3-6b", "THUDM/chatglm2-6b"| ✅ | ✅ |  ✅ | ✅ |
|GPTBigCode|"bigcode/starcoder"| ✅ | ✅ |  ✅ | ✅ |
|T5|"google/flan-t5-xl"| ✅ | ✅ |  ✅ | ✅ |
|Mistral|"mistralai/Mistral-7B-v0.1"| ✅ | ✅ |  ✅ | ✅ |
|MPT|"mosaicml/mpt-7b"| ✅ | ✅ |  ✅ | ✅ |

*Note*: The above verified models (including other models in the same model family, like "codellama/CodeLlama-7b-hf" from LLAMA family) are well supported with all optimizations like indirect access KV cache, fused ROPE, and prepacked TPP Linear (fp32/bf16). For other LLM model families, we are working in progress to cover those optimizations, which will expand the model list above.

## Support

The team tracks bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-pytorch/issues/). Before submitting a suggestion or bug report, search the existing GitHub issues to see if your issue has already been reported.

## License

_Apache License_, Version _2.0_. As found in [LICENSE](https://github.com/intel/intel-extension-for-pytorch/blob/main/LICENSE) file.

## Security

See Intel's [Security Center](https://www.intel.com/content/www/us/en/security-center/default.html)
for information on how to report a potential security issue or vulnerability.

See also: [Security Policy](SECURITY.md)

