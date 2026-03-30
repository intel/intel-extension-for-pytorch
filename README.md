## THIS PROJECT IS ARCHIVED 

Intel will not provide or guarantee development of or support for this project, including but not limited to, maintenance, bug fixes, new releases or updates. 

Patches to this project are no longer accepted by Intel.   

This project has been identified as having known security issues. 

Contact: webadmin@linux.intel.com
<div align="center">
  
Intel® Extension for PyTorch\*
===========================

</div>

## Retirement Plan

You may already be aware that we plan to retire Intel® Extension for PyTorch\* soon. This was announced in the Intel® Extension for PyTorch\* 2.8 release notes and also highlighted in community ticket [#867](https://github.com/intel/intel-extension-for-pytorch/issues/867).

We launched the Intel® Extension for PyTorch\* in 2020 with the goal of extending the official PyTorch\* to simplify achieving high performance on Intel® CPU and GPU platforms. Over the years, we have successfully upstreamed most of our features and optimizations for Intel® platforms into PyTorch* itself. As a result, we have discontinued active development of the Intel® Extension for PyTorch\* and ceased official quarterly releases following the 2.8 release. We strongly recommend using PyTorch\* directly going forward, as we remain committed to delivering robust support and performance with PyTorch* for Intel® CPU and GPU platforms.

We will continue to provide critical bug fixes and security patches for two additional quarters to ensure a smooth transition for our partners and the broader community. After that, we plan to mark the project End-of-Life unless there is a solid need to continue maintenance. Concretely, this means:
- We will continue to provide critical bug fixes and security patches in the main branches of Intel® Extension for PyTorch\*:  CPU ([main](https://github.com/intel/intel-extension-for-pytorch/tree/main)) and GPU ([xpu-main](https://github.com/intel/intel-extension-for-pytorch/tree/xpu-main)).
- We have stopped official quarterly releases. We will not create release branches or publish official binary wheels for Intel® Extension for PyTorch\*.
- We will maintain Intel® Extension for PyTorch\* as an open source project until the end of March 2026, to allow projects which depend on Intel® Extension for PyTorch\* to completely remove the dependency.

Thank you all for your continued support! Let’s keep the momentum going together!

## Introduction

**CPU** [💻main branch](https://github.com/intel/intel-extension-for-pytorch/tree/main)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[🌱Quick Start](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/getting_started.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[📖Documentations](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[🏃Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=cpu&version=v2.8.0%2Bcpu)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[💻LLM Example](https://github.com/intel/intel-extension-for-pytorch/tree/main/examples/cpu/llm) <br>

**GPU** [💻main branch](https://github.com/intel/intel-extension-for-pytorch/tree/xpu-main)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[🌱Quick Start](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/getting_started.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[📖Documentations](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[🏃Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[💻LLM Example](https://github.com/intel/intel-extension-for-pytorch/tree/xpu-main/examples/gpu/llm)<br>  

Intel® Extension for PyTorch\* extends PyTorch\* with up-to-date features optimizations for an extra performance boost on Intel hardware. Optimizations take advantage of Intel® Advanced Vector Extensions 512 (Intel® AVX-512) Vector Neural Network Instructions (VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX) on Intel CPUs as well as Intel X<sup>e</sup> Matrix Extensions (XMX) AI engines on Intel discrete GPUs. Moreover, Intel® Extension for PyTorch* provides easy GPU acceleration for Intel discrete GPUs through the PyTorch* xpu device.

## ipex.llm - Large Language Models (LLMs) Optimization

In the current technological landscape, Generative AI (GenAI) workloads and models have gained widespread attention and popularity. Large Language Models (LLMs) have emerged as the dominant models driving these GenAI applications. Starting from 2.1.0, specific optimizations for certain LLM models are introduced in the Intel® Extension for PyTorch\*. Check [**LLM optimizations**](./examples/cpu/llm) for details.

### Optimized Model List

We have supported a long list of LLMs, including the most notable open-source models
like Llama series, Qwen series, Phi-3/Phi-4 series,
and the phenomenal high-quality reasoning model DeepSeek-R1.

| MODEL FAMILY | MODEL NAME (Huggingface hub) | FP32 | BF16 | Weight only quantization INT8 | Weight only quantization INT4 |
|:---:|:---:|:---:|:---:|:---:|:---:|
|LLAMA| meta-llama/Llama-2-7b-hf | ✅ | ✅ | ✅ | ✅ |
|LLAMA| meta-llama/Llama-2-13b-hf | ✅ | ✅ | ✅ | ✅ |
|LLAMA| meta-llama/Llama-2-70b-hf | ✅ | ✅ | ✅ | ✅ |
|LLAMA| meta-llama/Meta-Llama-3-8B | ✅ | ✅ | ✅ | ✅ |
|LLAMA| meta-llama/Meta-Llama-3-70B | ✅ | ✅ | ✅ | ✅ |
|LLAMA| meta-llama/Meta-Llama-3.1-8B-Instruct | ✅ | ✅ | ✅ | ✅ |
|LLAMA| meta-llama/Llama-3.2-3B-Instruct | ✅ | ✅ | ✅ | ✅ |
|LLAMA| meta-llama/Llama-3.2-11B-Vision-Instruct | ✅ | ✅ | ✅ | ✅ |
|GPT-J| EleutherAI/gpt-j-6b | ✅ | ✅ | ✅ | ✅ |
|GPT-NEOX| EleutherAI/gpt-neox-20b | ✅ | ✅ | ✅ | ✅ |
|DOLLY| databricks/dolly-v2-12b | ✅ | ✅ | ✅ | ✅ |
|FALCON| tiiuae/falcon-7b  | ✅ | ✅ | ✅ | ✅ |
|FALCON| tiiuae/falcon-11b | ✅ | ✅ | ✅ | ✅ |
|FALCON| tiiuae/falcon-40b | ✅ | ✅ | ✅ | ✅ |
|FALCON| tiiuae/Falcon3-7B-Instruct | ✅ | ✅ | ✅ | ✅ |
|OPT| facebook/opt-30b | ✅ | ✅ | ✅ | ✅ |
|OPT| facebook/opt-1.3b | ✅ | ✅ | ✅ | ✅ |
|Bloom| bigscience/bloom-1b7 | ✅ | ✅ | ✅ | ✅ |
|CodeGen| Salesforce/codegen-2B-multi | ✅ | ✅ | ✅ | ✅ |
|Baichuan| baichuan-inc/Baichuan2-7B-Chat | ✅ | ✅ | ✅ | ✅ |
|Baichuan| baichuan-inc/Baichuan2-13B-Chat | ✅ | ✅ | ✅ | ✅ |
|Baichuan| baichuan-inc/Baichuan-13B-Chat | ✅ | ✅ | ✅ | ✅ |
|ChatGLM| THUDM/chatglm3-6b | ✅ | ✅ | ✅ | ✅ |
|ChatGLM| THUDM/chatglm2-6b | ✅ | ✅ | ✅ | ✅ |
|GPTBigCode| bigcode/starcoder | ✅ | ✅ | ✅ | ✅ |
|T5| google/flan-t5-xl | ✅ | ✅ | ✅ | ✅ |
|MPT| mosaicml/mpt-7b | ✅ | ✅ | ✅ | ✅ |
|Mistral| mistralai/Mistral-7B-v0.1 | ✅ | ✅ | ✅ | ✅ |
|Mixtral| mistralai/Mixtral-8x7B-v0.1 | ✅ | ✅ | ✅ | ✅ |
|Stablelm| stabilityai/stablelm-2-1_6b | ✅ | ✅ | ✅ | ✅ |
|Qwen| Qwen/Qwen-7B-Chat | ✅ | ✅ | ✅ | ✅ |
|Qwen| Qwen/Qwen2-7B | ✅ | ✅ | ✅ | ✅ |
|Qwen| Qwen/Qwen2.5-7B-Instruct | ✅ | ✅ | ✅ | ✅ |
|Qwen| Qwen/Qwen3-14B | ✅ | ✅ | ✅ |   |
|Qwen| Qwen/Qwen3-30B-A3B | ✅ | ✅ | ✅ | ✅ |
|LLaVA| liuhaotian/llava-v1.5-7b | ✅ | ✅ | ✅ | ✅ |
|GIT| microsoft/git-base | ✅ | ✅ | ✅ | ✅ |
|Yuan| IEITYuan/Yuan2-102B-hf | ✅ | ✅ | ✅ |   |
|Phi| microsoft/phi-2 | ✅ | ✅ | ✅ | ✅ |
|Phi| microsoft/Phi-3-mini-4k-instruct | ✅ | ✅ | ✅ | ✅ |
|Phi| microsoft/Phi-3-mini-128k-instruct | ✅ | ✅ | ✅ | ✅ |
|Phi| microsoft/Phi-3-medium-4k-instruct | ✅ | ✅ | ✅ | ✅ |
|Phi| microsoft/Phi-3-medium-128k-instruct | ✅ | ✅ | ✅ | ✅ |
|Phi| microsoft/Phi-4-mini-instruct | ✅ | ✅ | ✅ |   |
|Phi| microsoft/Phi-4-multimodal-instruct | ✅ | ✅ | ✅ |   |
|Whisper| openai/whisper-large-v2 | ✅ | ✅ | ✅ | ✅ |
|Whisper| openai/whisper-large-v3 | ✅ | ✅ | ✅ |   |
|Maira| microsoft/maira-2 | ✅ | ✅ | ✅ | ✅ |
|Jamba| ai21labs/Jamba-v0.1 | ✅ | ✅ | ✅ | ✅ |
|DeepSeek| deepseek-ai/DeepSeek-V2.5-1210 | ✅ | ✅ | ✅ | ✅ |
|DeepSeek| meituan/DeepSeek-R1-Channel-INT8 |   |   | ✅ |   |

*Note*: The above verified models (including other models in the same model family, like "codellama/CodeLlama-7b-hf" from LLAMA family) are well supported with all optimizations like indirect access KV cache, fused ROPE, and customized linear kernels.
We are working in progress to better support the models in the tables with various data types. In addition, more models will be optimized in the future.

In addition, Intel® Extension for PyTorch* introduces module level optimization APIs (prototype feature) since release 2.3.0.
The feature provides optimized alternatives for several commonly used LLM modules and functionalities for the optimizations of the niche or customized LLMs.
Please read [**LLM module level optimization practice**](./examples/cpu/inference/python/llm-modeling) to better understand how to optimize your own LLM and achieve better performance.

Above models are intended to allow users to examine and evaluate models and the associated performance of Intel technology solutions.
The accuracy of computer models is a function of the relation between the data used to train them and the data that the models encounter after deployment.
Models have been tested using datasets that may or may not be sufficient for use in production applications.
Accordingly, while the model may serve as a strong foundation, Intel recommends and requests that those models be tested against data the models are likely to encounter in specific deployments.

Intel is committed to respecting human rights and avoiding causing or contributing to adverse impacts on human rights. 
See [Intel’s Global Human Rights Principles](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/policy-human-rights.pdf). 
Intel’s products and software are intended only to be used in applications that do not cause or contribute to adverse impacts on human rights.

## Support

The team tracks bugs and enhancement requests using [GitHub issues](https://github.com/intel/intel-extension-for-pytorch/issues/). Before submitting a suggestion or bug report, search the existing GitHub issues to see if your issue has already been reported.

## License

_Apache License_, Version _2.0_. As found in [LICENSE](https://github.com/intel/intel-extension-for-pytorch/blob/main/LICENSE) file.

## Security

See Intel's [Security Center](https://www.intel.com/content/www/us/en/security-center/default.html)
for information on how to report a potential security issue or vulnerability.

See also: [Security Policy](SECURITY.md)
