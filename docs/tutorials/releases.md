Releases
=============

We launched Intel® Extension for PyTorch\* in 2020 with the goal of extending the official PyTorch\* to simplify achieving high performance on Intel® CPU and GPU platforms. Over the years, we have successfully upstreamed most of our features and optimizations for Intel® platforms into PyTorch\*. Moving forward, our strategy is to focus on developing new features and supporting upcoming platform launches directly within PyTorch\*. We are discontinuing active development on Intel® Extension for PyTorch\*, effective immediately after 2.8 release. We will continue to provide critical bug fixes and security patches throughout the PyTorch\* 2.9 timeframe to ensure a smooth transition for our partners and the community.

## 2.8.10+xpu

Intel® Extension for PyTorch\* v2.8.10+xpu is the new release which supports Intel® GPU platforms (Intel® Arc™ Graphics family, Intel® Core™ Ultra Processors with Intel® Arc™ Graphics, Intel® Core™ Ultra Series 2 with Intel® Arc™ Graphics, Intel® Core™ Ultra Series 2 Mobile Processors and Intel® Data Center GPU Max Series) based on PyTorch\* 2.8.0.

### Highlights

- Intel® oneDNN v3.8.1 integration
- Intel® Deep Learning Essentials 2025.1.3 compatibility
- Large Language Model (LLM) optimization

   Intel® Extension for PyTorch\* optimizes the performance of Qwen3, along with other typical LLM models on Intel® GPU platforms，with the supported transformer version upgraded to [4.51.3](https://github.com/huggingface/transformers/releases/tag/v4.51.3). A full list of optimized LLM models is available in the [LLM Optimizations Overview](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/llm.html). Intel® Extension for PyTorch\* also adds the support for more custom kernels, such as `selective_scan_fn`, `causal_conv1d_fn` and `causal_conv1d_update`, for the functionality support of [Jamba](https://arxiv.org/abs/2403.19887) model.

- PyTorch\* XCCL adoption for distributed scenarios

  Intel® Extension for PyTorch\* adopts the PyTorch\* XCCL backend for distrubuted scenarios on the Intel® GPU platform. We observed that the scaling performance using PyTorch\* XCCL is on par with OneCCL Bindings for PyTorch\* (torch-ccl) for validated AI workloads. As a result, we will discontinue active development of torch-ccl immediately after the 2.8 release.

  A pseudocode example illustrating the transition from torch-ccl to PyTorch\* XCCL at the model script level is shown below:

    ```
    import torch

    if torch.distributed.is_xccl_available():
      torch.distributed.init_process_group(backend='xccl')
    else:
      import oneccl_bindings_for_pytorch
      torch.distributed.init_process_group(backend='ccl')      
    ```

- Redundant code removal

  Intel® Extension for PyTorch\* no longer overrides the device allocator. It is recommended to use the allocator provided by PyTorch\* instead. Intel® Extension for PyTorch\* also removes all overridden oneMKL and oneDNN related operators except GEMM and SDPA.

### Known Issues

Please refer to [Known Issues webpage](./known_issues.md).

## 2.7.10+xpu

Intel® Extension for PyTorch\* v2.7.10+xpu is the new release which supports Intel® GPU platforms (Intel® Arc™ Graphics family, Intel® Core™ Ultra Processors with Intel® Arc™ Graphics, Intel® Core™ Ultra Series 2 with Intel® Arc™ Graphics, Intel® Core™ Ultra Series 2 Mobile Processors and Intel® Data Center GPU Max Series) based on PyTorch\* 2.7.0.

### Highlights

- Intel® oneDNN v3.7.1 integration
  
- Large Language Model (LLM) optimization

  Intel® Extension for PyTorch* optimizes typical LLM models like Llama 2, Llama 3, Phi-3-mini, Qwen2, and GLM-4 on the Intel® Arc™ Graphics family. Moreover, new LLM inference models such as Llama 3.3, Phi-3.5-mini, Qwen2.5, and Mistral-7B are also optimized on Intel® Data Center GPU Max Series platforms compared to the previous release. A full list of optimized models can be found in the [LLM Optimizations Overview](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/llm.html), with supported transformer version updates to [4.48.3](https://github.com/huggingface/transformers/releases/tag/v4.48.3).

- Serving framework support

  Intel® Extension for PyTorch\* offers extensive support for various ecosystems, including [vLLM](https://github.com/vllm-project/vllm) and [TGI](https://github.com/huggingface/text-generation-inference), with the goal of enhancing performance and flexibility for LLM workloads on Intel® GPU platforms (intensively verified on Intel® Data Center GPU Max Series and Intel® Arc™ B-Series graphics on Linux). The vLLM/TGI features, such as chunked prefill and MoE (Mixture of Experts), are supported by the backend kernels provided in Intel® Extension for PyTorch*. In this release, Intel® Extension for PyTorch\* adds sliding windows support in `ipex.llm.modules.PagedAttention.flash_attn_varlen_func` to meet the need of models like Phi3, and Mistral, which enable sliding window support by default.

- [Prototype] QLoRA/LoRA finetuning using BitsAndBytes

  Intel® Extension for PyTorch* supports QLoRA/LoRA finetuning with [BitsAndBytes](https://github.com/bitsandbytes-foundation/bitsandbytes) on Intel® GPU platforms. This release includes several enhancements for better performance and functionality:
  - The performance of the NF4 dequantize kernel has been improved by approximately 4.4× to 5.6× across different shapes compared to the previous release.
  - `_int_mm` support in INT8 has been added to enable INT8 LoRA finetuning in PEFT (with float optimizers like `adamw_torch`).

- Codegen support removal
  
  Removes codegen support from Intel® Extension for PyTorch\* and reuses the codegen capability from [Torch XPU Operators](https://github.com/intel/torch-xpu-ops), to ensure interoperability of code change in codegen with usages in Intel® Extension for PyTorch\*.

- [Prototype] Python 3.13t support

  Adds prototype support for Python 3.13t and provides prebuilt binaries on the [download server](https://pytorch-extension.intel.com/release-whl/stable/xpu/us/).

### Known Issues

Please refer to [Known Issues webpage](./known_issues.md).


## 2.6.10+xpu

Intel® Extension for PyTorch\* v2.6.10+xpu is the new release which supports Intel® GPU platforms (Intel® Data Center GPU Max Series, Intel® Arc™ Graphics family, Intel® Core™ Ultra Processors with Intel® Arc™ Graphics, Intel® Core™ Ultra Series 2 with Intel® Arc™ Graphics, Intel® Core™ Ultra Series 2 Mobile Processors and Intel® Data Center GPU Flex Series) based on PyTorch* 2.6.0.

### Highlights

- Intel® oneDNN v3.7 integration
- Official PyTorch 2.6 prebuilt binaries support

  Starting this release, Intel® Extension for PyTorch\* supports official PyTorch prebuilt binaries, as they are built with `_GLIBCXX_USE_CXX11_ABI=1` since PyTorch\* 2.6 and hence ABI compatible with Intel® Extension for PyTorch\* prebuilt binaries which are always built with `_GLIBCXX_USE_CXX11_ABI=1`.
  
- Large Language Model (LLM) optimization

  Intel® Extension for PyTorch\* provides support for a variety of custom kernels, which include commonly used kernel fusion techniques, such as `rms_norm` and `rotary_embedding`, as well as attention-related kernels like `paged_attention` and `chunked_prefill`, and `punica` kernel for serving multiple LoRA finetuned LLM. It also provides the MoE (Mixture of Experts) custom kernels including `topk_softmax`, `moe_gemm`, `moe_scatter`, `moe_gather`, etc. These optimizations enhance the functionality and efficiency of the ecosystem on Intel® GPU platform by improving the execution of key operations.

  Besides that, Intel® Extension for PyTorch\* optimizes more LLM models for inference and finetuning, such as Phi3-vision-128k, phi3-small-128k, llama3.2-11B-vision, etc. A full list of optimized models can be found at [LLM Optimizations Overview](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/llm.html).

- Serving framework support
  
  Intel® Extension for PyTorch\* offers extensive support for various ecosystems, including [vLLM](https://github.com/vllm-project/vllm) and [TGI](https://github.com/huggingface/text-generation-inference), with the goal of enhancing performance and flexibility for LLM workloads on Intel® GPU platforms (intensively verified on Intel® Data Center GPU Max Series and Intel® Arc™ B-Series graphics on Linux). The vLLM/TGI features like chunked prefill, MoE (Mixture of Experts) etc. are supported by the backend kernels provided in Intel® Extension for PyTorch*. The support to low precision such as Weight Only Quantization (WOQ) INT4 is also enhanced in this release:
  -  The performance of INT4 GEMM kernel based on Generalized Post-Training Quantization (GPTQ) algorithm has been improved by approximately 1.3× compared with previous release. During the prefill stage, it achieves similar performance to FP16, while in the decode stage, it outperforms FP16 by approximately 1.5×.
  -  The support of Activation-aware Weight Quantization (AWQ) algorithm is added and the performance is on par with GPTQ without g_idx.
  
- [Prototype] NF4 QLoRA finetuning using BitsAndBytes

  Intel® Extension for PyTorch\* now supports QLoRA finetuning with BitsAndBytes on Intel® GPU platforms. It enables efficient adaptation of LLMs using NF4 4-bit quantization with LoRA, reducing memory usage while maintaining accuracy.

- [Beta] Intel® Core™ Ultra Series 2 Mobile Processors support on Windows

  Intel® Extension for PyTorch\* provides beta quality support of Intel® Core™ Ultra Series 2 Mobile Processors (codename Arrow Lake-H) on Windows in this release, based on redistributed PyTorch 2.6 prebuilt binaries with additional AOT compilation target for Arrow Lake-H in the [download server](https://pytorch-extension.intel.com/release-whl/stable/xpu/us/).
  
- Hybrid ATen operator implementation
  
  Intel® Extension for PyTorch\* uses ATen operators available in [Torch XPU Operators](https://github.com/intel/torch-xpu-ops) as much as possible and overrides very limited operators for better performance and broad data type support.

### Breaking Changes

- Intel® Data Center GPU Flex Series support is being deprecated and will no longer be available starting from the release after v2.6.10+xpu.
- Channels Last 1D support on XPU is being deprecated and will no longer be available starting from the release after v2.6.10+xpu.

### Known Issues

Please refer to [Known Issues webpage](./known_issues.md).

## 2.5.10+xpu

Intel® Extension for PyTorch\* v2.5.10+xpu is the new release which supports Intel® GPU platforms (Intel® Data Center GPU Max Series, Intel® Arc™ Graphics family, Intel® Core™ Ultra Processors with Intel® Arc™ Graphics, Intel® Core™ Ultra Series 2 with Intel® Arc™ Graphics and Intel® Data Center GPU Flex Series) based on PyTorch* 2.5.1.

### Highlights

- Intel® oneDNN v3.6 integration
- Intel® oneAPI Base Toolkit 2025.0.1 compatibility
- Intel® Arc™ B-series Graphics support on Windows (prototype)
- Large Language Model (LLM) optimization
  
  Intel® Extension for PyTorch\* enhances KV Cache management to cover both Dynamic Cache and Static Cache methods defined by Hugging Face, which helps reduce computation time and improve response rates so as to optimize the performance of models in various generative tasks. Intel® Extension for PyTorch\* also supports new LLM features including speculative decoding which optimizes inference by making educated guesses about future tokens while generating the current token, sliding window attention which uses a fixed-size window to limit the attention span of each token thus significantly improves processing speed and efficiency for long documents, and multi-round conversations for supporting a natural human conversation where information is exchanged in multiple turns back and forth.

  Besides that, Intel® Extension for PyTorch\* optimizes more LLM models for inference and finetuning. A full list of optimized models can be found at [LLM Optimizations Overview](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/llm.html).

- Serving framework support
  
  Typical LLM serving frameworks including [vLLM](https://github.com/vllm-project/vllm) and [TGI](https://github.com/huggingface/text-generation-inference) can co-work with Intel® Extension for PyTorch\* on Intel® GPU platforms on Linux (intensively verified on Intel® Data Center GPU Max Series). The support to low precision such as INT4 Weight Only Quantization, which based on Generalized Post-Training Quantization (GPTQ) algorithm, is enhanced in this release.

- Beta support of full fine-tuning and LoRA PEFT with mixed precision

  Intel® Extension for PyTorch\* enhances this feature for optimizing typical LLM models and makes it reach Beta quality.

- Kineto Profiler Support
  
  Intel® Extension for PyTorch\* removes this redundant feature as the support of Kineto Profiler based on [PTI](https://github.com/intel/pti-gpu) on Intel® GPU platforms is available in PyTorch\* 2.5.
  
- Hybrid ATen operator implementation
  
  Intel® Extension for PyTorch\* uses ATen operators available in [Torch XPU Operators](https://github.com/intel/torch-xpu-ops) as much as possible and overrides very limited operators for better performance and broad data type support.

### Breaking Changes

- Block format support: oneDNN Block format integration support has been removed since v2.5.10+xpu.

### Known Issues

Please refer to [Known Issues webpage](./known_issues.md).

## 2.3.110+xpu

Intel® Extension for PyTorch\* v2.3.110+xpu is the new release which supports Intel® GPU platforms (Intel® Data Center GPU Flex Series, Intel® Data Center GPU Max Series and Intel® Arc™ A-Series Graphics) based on PyTorch\* 2.3.1.

### Highlights

- Intel® oneDNN v3.5.3 integration
- Intel® oneAPI Base Toolkit 2024.2.1 compatibility
- Large Language Model (LLM) optimization
  
  Intel® Extension for PyTorch\* provides a new dedicated module, `ipex.llm`, to host for Large Language Models (LLMs) specific APIs. With `ipex.llm`, Intel® Extension for PyTorch\* provides comprehensive LLM optimization on FP16 and INT4 datatypes. Specifically for low precision, Weight-Only Quantization is supported for various scenarios. And user can also run Intel® Extension for PyTorch\* with Tensor Parallel to fit in the multiple ranks or multiple nodes scenarios to get even better performance.

  A typical API under this new module is `ipex.llm.optimize`, which is designed to optimize transformer-based models within frontend Python modules, with a particular focus on Large Language Models (LLMs). It provides optimizations for both model-wise and content-generation-wise. `ipex.llm.optimize` is an upgrade API to replace previous `ipex.optimize_transformers`, which will bring you more consistent LLM experience and performance. Below shows a simple example of `ipex.llm.optimize` for fp16 inference:

  ```python
    import torch
    import intel_extension_for_pytorch as ipex
    import transformers

    model= transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path).eval()

    dtype = torch.float16
    model = ipex.llm.optimize(model, dtype=dtype, device="xpu")

    model.generate(YOUR_GENERATION_PARAMS)
  ```
    
  More examples of this API can be found at [LLM optimization API](https://intel.github.io/intel-extension-for-pytorch/xpu/2.3.110+xpu/tutorials/api_doc.html#ipex.llm.optimize).

  Besides that, we optimized more LLM inference models. A full list of optimized models can be found at [LLM Optimizations Overview](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/llm.html).

- Serving framework support
  
  Typical LLM serving frameworks including [vLLM](https://github.com/vllm-project/vllm) and [TGI](https://huggingface.co/text-generation-inference) can co-work with Intel® Extension for PyTorch\* on Intel® GPU platforms (Intel® Data Center GPU Max 1550 and Intel® Arc™ A-Series Graphics). Besides the integration of LLM serving frameworks with `ipex.llm` module level APIs, we enhanced the performance and quality of underneath Intel® Extension for PyTorch\* operators such as paged attention and flash attention for better end to end model performance.

- Prototype support of full fine-tuning and LoRA PEFT with mixed precision

  Intel® Extension for PyTorch\* also provides new capability for supporting popular recipes with both full fine-tuning and [LoRA PEFT](https://github.com/huggingface/peft)  for mixed precision with BF16 and FP32. We optimized many typical LLM models including Llama 2 (7B and 70B), Llama 3 8B, Phi-3-Mini 3.8B model families and Chinese model Qwen-7B, on both single GPU and Multi-GPU (distributed fine-tuning based on PyTorch FSDP) use cases.

### Breaking Changes

- Block format support: oneDNN Block format integration support is being deprecated and will no longer be available starting from the release after v2.3.110+xpu.

### Known Issues

Please refer to [Known Issues webpage](./known_issues.md).


## 2.1.40+xpu

Intel® Extension for PyTorch\* v2.1.40+xpu is a minor release which supports Intel® GPU platforms (Intel® Data Center GPU Flex Series, Intel® Data Center GPU Max Series，Intel® Arc™ A-Series Graphics and Intel® Core™ Ultra Processors with Intel® Arc™ Graphics) based on PyTorch\* 2.1.0.

### Highlights

- Intel® oneAPI Base Toolkit 2024.2.1 compatibility
- Intel® oneDNN v3.5 integration
- Intel® oneCCL 2021.13.1 integration
- Intel® Core™ Ultra Processors with Intel® Arc™ Graphics (MTL-H) support on Windows (Prototype)
- Bug fixing and other optimization
  - Fix host memory leak [#4280](https://github.com/intel/intel-extension-for-pytorch/commit/5c252a1e34ccecc8e2e5d10ccc67f410ac7b87e2)
  - Fix LayerNorm issue for undefined grad_input [#4317](https://github.com/intel/intel-extension-for-pytorch/commit/619cd9f5c300a876455411bcacc470bd94c923be)
  - Replace FP64 device check method [#4354](https://github.com/intel/intel-extension-for-pytorch/commit/d60d45187b1dd891ec8aa2abc42eca8eda5cb242)
  - Fix online doc search issue [#4358](https://github.com/intel/intel-extension-for-pytorch/commit/2e957315fdad776617e24a3222afa55f54b51507)
  - Fix pdist unit test failure on client GPUs [#4361](https://github.com/intel/intel-extension-for-pytorch/commit/00f94497a94cf6d69ebba33ff95d8ab39113ecf4)
  - Remove primitive cache from conv fwd [#4429](https://github.com/intel/intel-extension-for-pytorch/commit/bb1c6e92d4d11faac5b6fc01b226d27950b86579)
  - Fix sdp bwd page fault with no grad bias [#4439](https://github.com/intel/intel-extension-for-pytorch/commit/d015f00011ad426af33bb970451331321417bcdb)  
  - Fix implicit data conversion [#4463](https://github.com/intel/intel-extension-for-pytorch/commit/d6987649e58af0da4964175aed3286aef16c78c9)
  - Fix compiler version parsing issue [#4468](https://github.com/intel/intel-extension-for-pytorch/commit/50b2b5933b6df6632a18d76bdec46b638750dc48)  
  - Fix irfft invalid descriptor [#4480](https://github.com/intel/intel-extension-for-pytorch/commit/3e60e87cf011b643cc0e72d82c10b28417061d97)
  - Change condition order to fix out-of-bound access in index [#4495](https://github.com/intel/intel-extension-for-pytorch/commit/8b74d6c5371ed0bd442279be42b0d454cb2b31b3)
  - Add parameter check in embedding bag [#4504](https://github.com/intel/intel-extension-for-pytorch/commit/57174797bab9de2647abb8fdbcda638b0c694e01)
  - Add the backward implementation for rms norm [#4527](https://github.com/intel/intel-extension-for-pytorch/commit/e4938e0a9cee15ffe2f8d205e0228c1842a5735c)
  - Fix attn_mask for sdpa beam_search [#4557](https://github.com/intel/intel-extension-for-pytorch/commit/80ed47655b003fa132ac264b3d3008c298865473)
  - Use data_ptr template instead of force data conversion [#4558](https://github.com/intel/intel-extension-for-pytorch/commit/eeb92d2f4c34f143fc76e409987543d42e68d065)
  - Workaround windows AOT image size over 2GB issue on Intel® Core™ Ultra Processors with Intel® Arc™ Graphics [#4407](https://github.com/intel/intel-extension-for-pytorch/commit/d7ebba7c94374bdd12883ffd45d6670b96029d11) [#4450](https://github.com/intel/intel-extension-for-pytorch/commit/550fd767b723bd9a1a799b05be5d8ce073e6faf7)
  
### Known Issues

Please refer to [Known Issues webpage](./known_issues.md).


## 2.1.30+xpu

Intel® Extension for PyTorch\* v2.1.30+xpu is an update release which supports Intel® GPU platforms (Intel® Data Center GPU Flex Series, Intel® Data Center GPU Max Series and Intel® Arc™ A-Series Graphics) based on PyTorch\* 2.1.0.

### Highlights

- Intel® oneDNN v3.4.1 integration
- Intel® oneAPI Base Toolkit 2024.1 compatibility
- Large Language Model (LLM) optimizations for FP16 inference on Intel® Data Center GPU Max Series (Beta): Intel® Extension for PyTorch* provides a lot of specific optimizations for LLM workloads in this release on Intel® Data Center GPU Max Series.  In operator level, we provide highly efficient GEMM kernel to speed up Linear layer and customized fused operators to reduce HBM access/kernel launch overhead. To reduce memory footprint, we define a segment KV Cache policy to save device memory and improve the throughput. Such optimizations are added in this release to enhance existing optimized LLM FP16 models and more Chinese LLM models such as Baichuan2-13B, ChatGLM3-6B and Qwen-7B.

- LLM optimizations for INT4 inference on Intel® Data Center GPU Max Series and Intel® Arc™ A-Series Graphics (Prototype): Intel® Extension for PyTorch* shows remarkable performance when executing LLM models on Intel® GPU. However, deploying such models on GPUs with limited resources is challenging due to their high computational and memory requirements. To achieve a better trade-off, a low-precision solution, e.g., weight-only-quantization for INT4 is enabled to allow Llama 2-7B, GPT-J-6B and Qwen-7B to be executed efficiently on Intel® Arc™ A-Series Graphics. The same optimization makes INT4 models achieve 1.5x speeded up in total latency performance compared with FP16 models with the same configuration and parameters on Intel® Data Center GPU Max Series.

- Opt-in collective performance optimization with oneCCL Bindings for Pytorch*: This opt-in feature can be enabled by setting `TORCH_LLM_ALLREDUCE=1` to provide better scale-up performance by enabling optimized collectives such as `allreduce`, `allgather`, `reducescatter` algorithms in Intel® oneCCL. This feature requires XeLink enabled for cross-cards communication.

### Known Issues

Please refer to [Known Issues webpage](./known_issues.md).

## 2.1.20+xpu

Intel® Extension for PyTorch\* v2.1.20+xpu is a minor release which supports Intel® GPU platforms (Intel® Data Center GPU Flex Series, Intel® Data Center GPU Max Series and Intel® Arc™ A-Series Graphics) based on PyTorch\* 2.1.0.

### Highlights

- Intel® oneAPI Base Toolkit 2024.1 compatibility
- Intel® oneDNN v3.4 integration
- LLM inference scaling optimization based on Intel® oneCCL 2021.12 (Prototype)
- Bug fixing and other optimization
  - Uplift XeTLA to v0.3.4.1 [#3696](https://github.com/intel/intel-extension-for-pytorch/commit/dc0f6d39739404d38226ccf444c421706f14f2de)
  - [SDP] Fallback unsupported bias size to native impl [#3706](https://github.com/intel/intel-extension-for-pytorch/commit/d897ebd585da05a90295165584efc448e265a38d)
  - Error handling enhancement [#3788](https://github.com/intel/intel-extension-for-pytorch/commit/bd034e7a37822f84706f0068ec85d989fb766529), [#3841](https://github.com/intel/intel-extension-for-pytorch/commit/7d4f297ecb4c076586a22908ecadf4689cb2d5ef)
  - Fix beam search accuracy issue in workgroup reduce [#3796](https://github.com/intel/intel-extension-for-pytorch/commit/f2f20a523ee85ed1f44c7fa6465b8e5e1e2edfea)
  - Support int32 index tensor in index operator [#3808](https://github.com/intel/intel-extension-for-pytorch/commit/f7bb4873c0416a9f56d1f7ecfbcdbe7ad58b47cd)
  - Add deepspeed in LLM dockerfile [#3829](https://github.com/intel/intel-extension-for-pytorch/commit/6266f89833f8010d6c683f9b45cfb2031575ad92)
  - Fix batch norm accuracy issue [#3882](https://github.com/intel/intel-extension-for-pytorch/commit/a1e2271717ff61dc3ea7d8d471c2356b3e469b93)
  - Prebuilt wheel dockerfile update [#3887](https://github.com/intel/intel-extension-for-pytorch/commit/8d5d71522910c1f622dac6a52cb0025e469774b2#diff-022fb5910f470cc5c44ab38cb20586d014f37c06ac8f3378e146ed35ee202a46), [#3970](https://github.com/intel/intel-extension-for-pytorch/commit/54b8171940cd694ba91c928c99acc440c9993881)
  - Fix windows build failure with Intel® oneMKL 2024.1 in torch_patches [#18](https://github.com/intel/intel-extension-for-pytorch/blob/release/xpu/2.1.20/torch_patches/0018-use-ONEMKL_LIBRARIES-for-mkl-libs-in-torch-to-not-ov.patch)
  - Fix FFT core dump issue with Intel® oneMKL 2024.1 in torch_patches [#20](https://github.com/intel/intel-extension-for-pytorch/blob/release/xpu/2.1.20/torch_patches/0020-Hide-MKL-symbols-211-212.patch), [#21](https://github.com/intel/intel-extension-for-pytorch/blob/release/xpu/2.1.20/torch_patches/0021-Fix-Windows-Build-214-215.patch)

### Known Issues

Please refer to [Known Issues webpage](./known_issues.md).

## 2.1.10+xpu

Intel® Extension for PyTorch\* v2.1.10+xpu is the new Intel® Extension for PyTorch\* release supports both CPU platforms and GPU platforms (Intel® Data Center GPU Flex Series, Intel® Data Center GPU Max Series and Intel® Arc™ A-Series Graphics) based on PyTorch\* 2.1.0. It extends PyTorch\* 2.1.0 with up-to-date features and optimizations on `xpu` for an extra performance boost on Intel hardware. Optimizations take advantage of AVX-512 Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX) on Intel CPUs as well as Intel Xe Matrix Extensions (XMX) AI engines on Intel discrete GPUs. Moreover, through PyTorch* `xpu` device, Intel® Extension for PyTorch* provides easy GPU acceleration for Intel discrete GPUs with PyTorch*.

### Highlights

This release provides the following features:

- Large Language Model (LLM) optimizations for FP16 inference on Intel® Data Center GPU Max Series (Prototype): Intel® Extension for PyTorch* provides a lot of specific optimizations for LLM workloads on Intel® Data Center GPU Max Series in this release. In operator level, we provide highly efficient GEMM kernel to speedup Linear layer and customized fused operators to reduce HBM access and kernel launch overhead. To reduce memory footprint, we define a segment KV Cache policy to save device memory and improve the throughput. To better trade-off the performance and accuracy, low-precision solution e.g., weight-only-quantization for INT4 is enabled. Besides, tensor parallel can also be adopted to get lower latency for LLMs.

  - A new API function, `ipex.optimize_transformers`, is designed to optimize transformer-based models within frontend Python modules, with a particular focus on LLMs. It provides optimizations for both model-wise and content-generation-wise. You just need to invoke the `ipex.optimize_transformers` API instead of the `ipex.optimize` API to apply all optimizations transparently. More detailed information can be found at [Large Language Model optimizations overview](https://intel.github.io/intel-extension-for-pytorch/xpu/2.1.10+xpu/tutorials/llm.html).
  - A typical usage of this new feature is quite simple as below:
    ```            
    import torch
    import intel_extension_for_pytorch as ipex
    ...
    model = ipex.optimize_transformers(model, dtype=dtype)
    ```

- `Torch.compile` functionality on Intel® Data Center GPU Max Series (Beta): Extends Intel® Extension for PyTorch* capabilities to support [torch.compile](https://pytorch.org/docs/stable/generated/torch.compile.html#torch-compile) APIs on Intel® Data Center GPU Max Series. And provides Intel GPU support on top of [Triton*](https://github.com/openai/triton) compiler to reach competitive performance speed-up over eager mode by default "inductor" backend of Intel® Extension for PyTorch*.
  
- Intel® Arc™ A-Series Graphics on WSL2, native Windows and native Linux are officially supported in this release. Intel® Arc™ A770 Graphic card has been used as primary verification vehicle for product level test.
  
- Other features are listed as following, more detailed information can be found in [public documentation](https://intel.github.io/intel-extension-for-pytorch/xpu/2.1.10+xpu/):
  - FP8 datatype support (Prototype): Add basic data type and FP8 Linear operator support based on emulation kernel.
  - Kineto Profiling (Prototype): An extension of PyTorch* profiler for profiling operators on Intel® GPU devices.
  - Fully Sharded Data Parallel (FSDP):  Support new PyTorch* [FSDP](https://pytorch.org/docs/stable/fsdp.html) API which provides an industry-grade solution for large-scale model training.
  - Asymmetric INT8 quantization: Support asymmetric quantization to align with stock PyTorch* and provide better accuracy in INT8.

- CPU support has been merged in this release. CPU features and optimizations are equivalent to what has been released in [Intel® Extension for PyTorch* v2.1.0+cpu release](https://github.com/intel/intel-extension-for-pytorch/releases/tag/v2.1.0+cpu) that was made publicly available in Oct 2023. For customers who would like to evaluate workloads on both GPU and CPU, they can use this package. For customers who are focusing on CPU only, we still recommend them to use Intel® Extension for PyTorch* v2.1.0+cpu release for smaller footprint, less dependencies and broader OS support.

### Known Issues

Please refer to [Known Issues webpage](./known_issues.md).

## 2.0.110+xpu

Intel® Extension for PyTorch\* v2.0.110+xpu is the new Intel® Extension for PyTorch\* release supports both CPU platforms and GPU platforms (Intel® Data Center GPU Flex Series and Intel® Data Center GPU Max Series) based on PyTorch\* 2.0.1. It extends PyTorch\* 2.0.1 with up-to-date features and optimizations on `xpu` for an extra performance boost on Intel hardware. Optimizations take advantage of AVX-512 Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX) on Intel CPUs as well as Intel Xe Matrix Extensions (XMX) AI engines on Intel discrete GPUs. Moreover, through PyTorch* `xpu` device, Intel® Extension for PyTorch* provides easy GPU acceleration for Intel discrete GPUs with PyTorch*.

### Highlights

This release introduces specific XPU solution optimizations on Intel discrete GPUs which include Intel® Data Center GPU Flex Series and Intel® Data Center GPU Max Series. Optimized operators and kernels are implemented and registered through PyTorch\* dispatching mechanism for the `xpu` device. These operators and kernels are accelerated on Intel GPU hardware from the corresponding native vectorization and matrix calculation features. In graph mode, additional operator fusions are supported to reduce operator/kernel invocation overheads, and thus increase performance.

This release provides the following features:
- oneDNN 3.3 API integration and adoption
- Libtorch support
- ARC support on Windows, WSL2 and Ubuntu (Prototype)
- OOB models improvement
  - More fusion patterns enabled for optimizing OOB models
- CPU support is merged in this release:
  - CPU features and optimizations are equivalent to what has been released in Intel® Extension for PyTorch* v2.0.100+cpu release that was made publicly available in May 2023. For customers who would like to evaluate workloads on both GPU and CPU, they can use this package. For customers who are focusing on CPU only, we still recommend them to use Intel® Extension for PyTorch* v2.0.100+cpu release for smaller footprint, less dependencies and broader OS support.

This release adds the following fusion patterns in PyTorch\* JIT mode for Intel GPU:
- `add` + `softmax`
- `add` + `view` + `softmax`

### Known Issues

Please refer to [Known Issues webpage](./known_issues.md).

## 1.13.120+xpu

Intel® Extension for PyTorch\* v1.13.120+xpu is the updated Intel® Extension for PyTorch\* release supports both CPU platforms and GPU platforms (Intel® Data Center GPU Flex Series and Intel® Data Center GPU Max Series) based on PyTorch\* 1.13.1. It extends PyTorch\* 1.13.1 with up-to-date features and optimizations on `xpu` for an extra performance boost on Intel hardware. Optimizations take advantage of AVX-512 Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX) on Intel CPUs as well as Intel Xe Matrix Extensions (XMX) AI engines on Intel discrete GPUs. Moreover, through PyTorch* `xpu` device, Intel® Extension for PyTorch* provides easy GPU acceleration for Intel discrete GPUs with PyTorch*.

### Highlights

This release introduces specific XPU solution optimizations on Intel discrete GPUs which include Intel® Data Center GPU Flex Series and Intel® Data Center GPU Max Series. Optimized operators and kernels are implemented and registered through PyTorch\* dispatching mechanism for the `xpu` device. These operators and kernels are accelerated on Intel GPU hardware from the corresponding native vectorization and matrix calculation features. In graph mode, additional operator fusions are supported to reduce operator/kernel invocation overheads, and thus increase performance.

This release provides the following features:
- oneDNN 3.1 API integration and adoption
- OOB models improvement
  - More fusion patterns enabled for optimizing OOB models
- CPU support is merged in this release:
  - CPU features and optimizations are equivalent to what has been released in Intel® Extension for PyTorch* v1.13.100+cpu release that was made publicly available in Feb 2023. For customers who would like to evaluate workloads on both GPU and CPU, they can use this package. For customers who are focusing on CPU only, we still recommend them to use Intel® Extension for PyTorch* v1.13.100+cpu release for smaller footprint, less dependencies and broader OS support.

This release adds the following fusion patterns in PyTorch\* JIT mode for Intel GPU:
- `Matmul` + UnaryOp(`abs`, `sqrt`, `square`, `exp`, `log`, `round`, `Log_Sigmoid`, `Hardswish`, `HardSigmoid`, `Pow`, `ELU`, `SiLU`, `hardtanh`, `Leaky_relu`)
- `Conv2d` + BinaryOp(`add`, `sub`, `mul`, `div`, `max`, `min`, `eq`, `ne`, `ge`, `gt`, `le`, `lt`)
- `Linear` + BinaryOp(`add`, `sub`, `mul`, `div`, `max`, `min`)
- `Conv2d` + `mul` + `add`
- `Conv2d` + `mul` + `add` + `relu`
- `Conv2d` + `sigmoid` + `mul` + `add`
- `Conv2d` + `sigmoid` + `mul` + `add` + `relu`

### Known Issues

Please refer to [Known Issues webpage](./known_issues.md).

## 1.13.10+xpu

Intel® Extension for PyTorch\* v1.13.10+xpu is the first Intel® Extension for PyTorch\* release supports both CPU platforms and GPU platforms (Intel® Data Center GPU Flex Series and Intel® Data Center GPU Max Series) based on PyTorch\* 1.13. It extends PyTorch\* 1.13 with up-to-date features and optimizations on `xpu` for an extra performance boost on Intel hardware. Optimizations take advantage of AVX-512 Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX) on Intel CPUs as well as Intel Xe Matrix Extensions (XMX) AI engines on Intel discrete GPUs. Moreover, through PyTorch* `xpu` device, Intel® Extension for PyTorch* provides easy GPU acceleration for Intel discrete GPUs with PyTorch*.

### Highlights

This release introduces specific XPU solution optimizations on Intel discrete GPUs which include Intel® Data Center GPU Flex Series and Intel® Data Center GPU Max Series. Optimized operators and kernels are implemented and registered through PyTorch\* dispatching mechanism for the `xpu` device. These operators and kernels are accelerated on Intel GPU hardware from the corresponding native vectorization and matrix calculation features. In graph mode, additional operator fusions are supported to reduce operator/kernel invocation overheads, and thus increase performance.

This release provides the following features:
- Distributed Training on GPU:
  - support of distributed training with DistributedDataParallel (DDP) on Intel GPU hardware
  - support of distributed training with Horovod (prototype feature) on Intel GPU hardware
- Automatic channels last format conversion on GPU:
  - Automatic channels last format conversion is enabled. Models using `torch.xpu.optimize` API running on Intel® Data Center GPU Max Series will be converted to channels last memory format, while models running on Intel® Data Center GPU Flex Series will choose oneDNN block format.
- CPU support is merged in this release:
  - CPU features and optimizations are equivalent to what has been released in Intel® Extension for PyTorch* v1.13.0+cpu release that was made publicly available in Nov 2022. For customers who would like to evaluate workloads on both GPU and CPU, they can use this package. For customers who are focusing on CPU only, we still recommend them to use Intel® Extension for PyTorch* v1.13.0+cpu release for smaller footprint, less dependencies and broader OS support.

This release adds the following fusion patterns in PyTorch\* JIT mode for Intel GPU:
- `Conv2D` + UnaryOp(`abs`, `sqrt`, `square`, `exp`, `log`, `round`, `GeLU`, `Log_Sigmoid`, `Hardswish`, `Mish`, `HardSigmoid`, `Tanh`, `Pow`, `ELU`, `hardtanh`)
- `Linear` + UnaryOp(`abs`, `sqrt`, `square`, `exp`, `log`, `round`, `Log_Sigmoid`, `Hardswish`, `HardSigmoid`, `Pow`, `ELU`, `SiLU`, `hardtanh`, `Leaky_relu`)

### Known Issues

Please refer to [Known Issues webpage](./known_issues.md).

## 1.10.200+gpu

Intel® Extension for PyTorch\* v1.10.200+gpu extends PyTorch\* 1.10 with up-to-date features and optimizations on XPU for an extra performance boost on Intel Graphics cards. XPU is a user visible device that is a counterpart of the well-known CPU and CUDA in the PyTorch\* community. XPU represents an Intel-specific kernel and graph optimizations for various “concrete” devices. The XPU runtime will choose the actual device when executing AI workloads on the XPU device. The default selected device is Intel GPU. XPU kernels from Intel® Extension for PyTorch\* are written in [DPC++](https://github.com/intel/llvm#oneapi-dpc-compiler) that supports [SYCL language](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html) and also a number of [DPC++ extensions](https://github.com/intel/llvm/tree/sycl/sycl/doc/extensions).

### Highlights

This release introduces specific XPU solution optimizations on Intel® Data Center GPU Flex Series 170. Optimized operators and kernels are implemented and registered through PyTorch\* dispatching mechanism for the XPU device. These operators and kernels are accelerated on Intel GPU hardware from the corresponding native vectorization and matrix calculation features. In graph mode, additional operator fusions are supported to reduce operator/kernel invocation overheads, and thus increase performance.

This release provides the following features:
- Auto Mixed Precision (AMP)
  - support of AMP with BFloat16 and Float16 optimization of GPU operators
- Channels Last
  - support of channels\_last (NHWC) memory format for most key GPU operators
- DPC++ Extension
  - mechanism to create PyTorch\* operators with custom DPC++ kernels running on the XPU device
- Optimized Fusion
  - support of SGD/AdamW fusion for both FP32 and BF16 precision

This release supports the following fusion patterns in PyTorch\* JIT mode:

- Conv2D + ReLU
- Conv2D + Sum
- Conv2D + Sum + ReLU
- Pad + Conv2d
- Conv2D + SiLu
- Permute + Contiguous
- Conv3D + ReLU
- Conv3D + Sum
- Conv3D + Sum + ReLU
- Linear + ReLU
- Linear + Sigmoid
- Linear + Div(scalar)
- Linear + GeLu
- Linear + GeLu\_
- T + Addmm
- T + Addmm + ReLu
- T + Addmm + Sigmoid
- T + Addmm + Dropout
- T + Matmul
- T + Matmul + Add
- T + Matmul + Add + GeLu
- T + Matmul + Add + Dropout
- Transpose + Matmul
- Transpose + Matmul + Div
- Transpose + Matmul + Div + Add
- MatMul + Add
- MatMul + Div
- Dequantize + PixelShuffle
- Dequantize + PixelShuffle + Quantize
- Mul + Add
- Add + ReLU
- Conv2D + Leaky\_relu
- Conv2D + Leaky\_relu\_
- Conv2D + Sigmoid
- Conv2D + Dequantize
- Softplus + Tanh
- Softplus + Tanh + Mul
- Conv2D + Dequantize + Softplus + Tanh + Mul
- Conv2D + Dequantize + Softplus + Tanh + Mul + Quantize
- Conv2D + Dequantize + Softplus + Tanh + Mul + Quantize + Add

### Known Issues

- [FATAL ERROR] Kernel 'XXX' removed due to usage of FP64 instructions unsupported by the targeted hardware

    FP64 is not natively supported by the [Intel® Data Center GPU Flex Series](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/data-center-gpu/flex-series/overview.html) platform. If you run any AI workload on that platform and receive this error message, it means a kernel requiring FP64 instructions is removed and not executed, hence the accuracy of the whole workload is wrong.

- symbol undefined caused by \_GLIBCXX\_USE\_CXX11\_ABI

    ```bash
    ImportError: undefined symbol: _ZNK5torch8autograd4Node4nameB5cxx11Ev
    ```
    
    DPC++ does not support \_GLIBCXX\_USE\_CXX11\_ABI=0, Intel® Extension for PyTorch\* is always compiled with \_GLIBCXX\_USE\_CXX11\_ABI=1. This symbol undefined issue appears when PyTorch\* is compiled with \_GLIBCXX\_USE\_CXX11\_ABI=0. Update PyTorch\* CMAKE file to set \_GLIBCXX\_USE\_CXX11\_ABI=1 and compile PyTorch\* with particular compiler which supports \_GLIBCXX\_USE\_CXX11\_ABI=1. We recommend to use gcc version 9.4.0 on ubuntu 20.04.

- Can't find oneMKL library when build Intel® Extension for PyTorch\* without oneMKL

    ```bash
    /usr/bin/ld: cannot find -lmkl_sycl
    /usr/bin/ld: cannot find -lmkl_intel_ilp64
    /usr/bin/ld: cannot find -lmkl_core
    /usr/bin/ld: cannot find -lmkl_tbb_thread
    dpcpp: error: linker command failed with exit code 1 (use -v to see invocation)
    ```
    
    When PyTorch\* is built with oneMKL library and Intel® Extension for PyTorch\* is built without oneMKL library, this linker issue may occur. Resolve it by setting:
    
    ```bash
    export USE_ONEMKL=OFF
    export MKL_DPCPP_ROOT=${PATH_To_Your_oneMKL}/__release_lnx/mkl
    ```
    
    Then clean build Intel® Extension for PyTorch\*.

- undefined symbol: mkl\_lapack\_dspevd. Intel MKL FATAL ERROR: cannot load libmkl\_vml\_avx512.so.2 or libmkl\_vml\_def.so.2

    This issue may occur when Intel® Extension for PyTorch\* is built with oneMKL library and PyTorch\* is not build with any MKL library. The oneMKL kernel may run into CPU backend incorrectly and trigger this issue. Resolve it by installing MKL library from conda:
    
    ```bash
    conda install mkl
    conda install mkl-include
    ```
    
    then clean build PyTorch\*.

- OSError: libmkl\_intel\_lp64.so.1: cannot open shared object file: No such file or directory

    Wrong MKL library is used when multiple MKL libraries exist in system. Preload oneMKL by:
    
    ```bash
    export LD_PRELOAD=${MKL_DPCPP_ROOT}/lib/intel64/libmkl_intel_lp64.so.1:${MKL_DPCPP_ROOT}/lib/intel64/libmkl_intel_ilp64.so.1:${MKL_DPCPP_ROOT}/lib/intel64/libmkl_sequential.so.1:${MKL_DPCPP_ROOT}/lib/intel64/libmkl_core.so.1:${MKL_DPCPP_ROOT}/lib/intel64/libmkl_sycl.so.1
    ```
    
    If you continue seeing similar issues for other shared object files, add the corresponding files under ${MKL\_DPCPP\_ROOT}/lib/intel64/ by `LD_PRELOAD`. Note that the suffix of the libraries may change (e.g. from .1 to .2), if more than one oneMKL library is installed on the system.

