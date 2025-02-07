<div align="center">
  
Intel® Extension for PyTorch\*
===========================

</div>

**CPU** [💻main branch](https://github.com/intel/intel-extension-for-pytorch/tree/main)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[🌱Quick Start](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/getting_started.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[📖Documentations](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[🏃Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=cpu&version=v2.6.0%2Bcpu)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[💻LLM Example](https://github.com/intel/intel-extension-for-pytorch/tree/release/2.6/examples/cpu/llm) <br>
**GPU** [💻main branch](https://github.com/intel/intel-extension-for-pytorch/tree/xpu-main)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[🌱Quick Start](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/getting_started.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[📖Documentations](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[🏃Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[💻LLM Example](https://github.com/intel/intel-extension-for-pytorch/tree/xpu-main/examples/gpu/llm)<br>  

Intel® Extension for PyTorch\* extends PyTorch\* with up-to-date features optimizations for an extra performance boost on Intel hardware. Optimizations take advantage of Intel® Advanced Vector Extensions 512 (Intel® AVX-512) Vector Neural Network Instructions (VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX) on Intel CPUs as well as Intel X<sup>e</sup> Matrix Extensions (XMX) AI engines on Intel discrete GPUs. Moreover, Intel® Extension for PyTorch* provides easy GPU acceleration for Intel discrete GPUs through the PyTorch* xpu device.

## ipex.llm - Large Language Models (LLMs) Optimization

In the current technological landscape, Generative AI (GenAI) workloads and models have gained widespread attention and popularity. Large Language Models (LLMs) have emerged as the dominant models driving these GenAI applications. Starting from 2.1.0, specific optimizations for certain LLM models are introduced in the Intel® Extension for PyTorch\*. Check [**LLM optimizations**](./examples/cpu/llm) for details.

### Optimized Model List

| MODEL FAMILY | MODEL NAME (Huggingface hub) | FP32 | BF16 | Static quantization INT8 | Weight only quantization INT8 | Weight only quantization INT4 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|LLAMA| meta-llama/Llama-2-7b-hf | ✅ | ✅ | ✅ | ✅ | ✅ |
|LLAMA| meta-llama/Llama-2-13b-hf | ✅ | ✅ | ✅ | ✅ | ✅ |
|LLAMA| meta-llama/Llama-2-70b-hf | ✅ | ✅ | ✅ | ✅ | ✅ |
|LLAMA| meta-llama/Meta-Llama-3-8B | ✅ | ✅ | ✅ | ✅ | ✅ |
|LLAMA| meta-llama/Meta-Llama-3-70B | ✅ | ✅ | ✅ | ✅ | ✅ |
|LLAMA| meta-llama/Meta-Llama-3.1-8B-Instruct | ✅ | ✅ | ✅ | ✅ | ✅ |
|LLAMA| meta-llama/Llama-3.2-3B-Instruct | ✅ | ✅ | ✅ | ✅ | ✅ |
|LLAMA| meta-llama/Llama-3.2-11B-Vision-Instruct | ✅ | ✅ |   | ✅ | ✅ |
|GPT-J| EleutherAI/gpt-j-6b | ✅ | ✅ | ✅ | ✅ | ✅ |
|GPT-NEOX| EleutherAI/gpt-neox-20b | ✅ | ✅ | ✅ | ✅ | ✅ |
|DOLLY| databricks/dolly-v2-12b | ✅ | ✅ | ✅ | ✅ | ✅ |
|FALCON| tiiuae/falcon-7b  | ✅ | ✅ | ✅ | ✅ | ✅ |
|FALCON| tiiuae/falcon-11b | ✅ | ✅ | ✅ | ✅ | ✅ |
|FALCON| tiiuae/falcon-40b | ✅ | ✅ | ✅ | ✅ | ✅ |
|OPT| facebook/opt-30b | ✅ | ✅ | ✅ | ✅ | ✅ |
|OPT| facebook/opt-1.3b | ✅ | ✅ | ✅ | ✅ | ✅ |
|Bloom| bigscience/bloom-1b7 | ✅ | ✅ | ✅ | ✅ | ✅ |
|CodeGen| Salesforce/codegen-2B-multi | ✅ | ✅ | ✅ | ✅ | ✅ |
|Baichuan| baichuan-inc/Baichuan2-7B-Chat | ✅ | ✅ | ✅ | ✅ | ✅ |
|Baichuan| baichuan-inc/Baichuan2-13B-Chat | ✅ | ✅ | ✅ | ✅ | ✅ |
|Baichuan| baichuan-inc/Baichuan-13B-Chat | ✅ | ✅ | ✅ | ✅ | ✅ |
|ChatGLM| THUDM/chatglm3-6b | ✅ | ✅ | ✅ | ✅ | ✅ |
|ChatGLM| THUDM/chatglm2-6b | ✅ | ✅ | ✅ | ✅ | ✅ |
|GPTBigCode| bigcode/starcoder | ✅ | ✅ | ✅ | ✅ | ✅ |
|T5| google/flan-t5-xl | ✅ | ✅ | ✅ | ✅ | ✅ |
|MPT| mosaicml/mpt-7b | ✅ | ✅ | ✅ | ✅ | ✅ |
|Mistral| mistralai/Mistral-7B-v0.1 | ✅ | ✅ | ✅ | ✅ | ✅ |
|Mixtral| mistralai/Mixtral-8x7B-v0.1 | ✅ | ✅ |   | ✅ | ✅ |
|Stablelm| stabilityai/stablelm-2-1_6b | ✅ | ✅ | ✅ | ✅ | ✅ |
|Qwen| Qwen/Qwen-7B-Chat | ✅ | ✅ | ✅ | ✅ | ✅ |
|Qwen| Qwen/Qwen2-7B | ✅ | ✅ | ✅ | ✅ | ✅ |
|LLaVA| liuhaotian/llava-v1.5-7b | ✅ | ✅ |   | ✅ | ✅ |
|GIT| microsoft/git-base | ✅ | ✅ |   | ✅ | ✅ |
|Yuan| IEITYuan/Yuan2-102B-hf | ✅ | ✅ |   | ✅ |   |
|Phi| microsoft/phi-2 | ✅ | ✅ | ✅ | ✅ | ✅ |
|Phi| microsoft/Phi-3-mini-4k-instruct | ✅ | ✅ | ✅ | ✅ | ✅ |
|Phi| microsoft/Phi-3-mini-128k-instruct | ✅ | ✅ | ✅ | ✅ | ✅ |
|Phi| microsoft/Phi-3-medium-4k-instruct | ✅ | ✅ | ✅ | ✅ | ✅ |
|Phi| microsoft/Phi-3-medium-128k-instruct | ✅ | ✅ | ✅ | ✅ | ✅ |
|Whisper| openai/whisper-large-v2 | ✅ | ✅ | ✅ | ✅ | ✅ |
|Maira| microsoft/maira-2 | ✅ | ✅ |   | ✅ | ✅ |
|Jamba| ai21labs/Jamba-v0.1 | ✅ | ✅ |   | ✅ | ✅ |
|DeepSeek| deepseek-ai/DeepSeek-V2.5-1210 | ✅ | ✅ |   | ✅ | ✅ |

*Note*: The above verified models (including other models in the same model family, like "codellama/CodeLlama-7b-hf" from LLAMA family) are well supported with all optimizations like indirect access KV cache, fused ROPE, and customized linear kernels.
We are working in progress to better support the models in the tables with various data types. In addition, more models will be optimized in the future.

In addition, Intel® Extension for PyTorch* introduces module level optimization APIs (prototype feature) since release 2.3.0.
The feature provides optimized alternatives for several commonly used LLM modules and functionalities for the optimizations of the niche or customized LLMs.
Please read [**LLM module level optimization practice**](./examples/cpu/inference/python/llm-modeling) to better understand how to optimize your own LLM and achieve better performance.

## Support

The team tracks bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-pytorch/issues/). Before submitting a suggestion or bug report, search the existing GitHub issues to see if your issue has already been reported.

## License

_Apache License_, Version _2.0_. As found in [LICENSE](https://github.com/intel/intel-extension-for-pytorch/blob/main/LICENSE) file.

## Security

See Intel's [Security Center](https://www.intel.com/content/www/us/en/security-center/default.html)
for information on how to report a potential security issue or vulnerability.

See also: [Security Policy](SECURITY.md)

