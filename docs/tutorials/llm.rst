Large Language Models (LLM) Optimizations Overview
==================================================

In the current technological landscape, Generative AI (GenAI) workloads and models have gained widespread attention and popularity. LLMs have emerged as the dominant models driving these GenAI applications. Most of LLMs are GPT-like architectures that consist of multiple Decoder layers.
The MultiHeadAttention and FeedForward layer are two key components of every Decoder layer. The generation task is memory bound because iterative decode and kv_cache require special management to reduce memory overheads. Intel® Extension for PyTorch* provides a lot of specific optimizations for these LLMs.
On the operator level, the extension provides highly efficient GEMM kernel to speed up Linear layer and customized operators to reduce the memory footprint. To better trade-off the performance and accuracy, different low-precision solutions e.g., smoothQuant is enabled. Besides, tensor parallel can also adopt to get lower latency for LLMs.

These LLM-specific optimizations can be automatically applied with a single frontend API function in Python interface, `ipex.llm.optimize()`. Check `ipex.llm.optimize <./llm/llm_optimize_transformers.md>`_ for more details.

.. toctree::
   :hidden:
   :maxdepth: 1

   llm/llm_optimize_transformers

Validated Models List
---------------------

LLM Inference
~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Model Family
     - Verified models from Huggingface hub
     - Dynamic KV-Cache
     - Static KV-Cache
     - FP16
     - INT4 WoQ
   * - Llama2
     - meta-llama/Llama-2-7b-hf, meta-llama/Llama-2-13b-hf, meta-llama/Llama-2-70b-hf
     - ✅
     - ✅
     - ✅
     - ✅
   * - Llama3
     - meta-llama/Meta-Llama-3-8B-Instruct, meta-llama/Meta-Llama-3-70B-Instruct, meta-llama/Llama-3.2-1B, meta-llama/Llama-3.2-3B,meta-llama/Llama-3.3-70B-Instruct
     - ✅
     - ✅
     - ✅
     - ✅
   * - Phi-3 mini
     - microsoft/Phi-3-mini-4k-instruct, microsoft/Phi-3-mini-128k-instruct, microsoft/Phi-3.5-mini-instruct
     - ✅
     - ✅
     - ✅
     - ✅
   * - GPT-J
     - EleutherAI/gpt-j-6b
     - ✅
     - ✅
     - ✅
     - ✅
   * - Qwen
     - Qwen/Qwen2-VL-7B-Instruct, Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen3-4B, Qwen/Qwen3-8B, thewimo/Qwen3-4B-AWQ, AlphaGaO/Qwen3-4B-GPTQ, Qwen/Qwen3-8B-AWQ
     - ✅
     - ✅
     - ✅
     - ✅
   * - GLM-Chat
     - THUDM/glm-4-9b-chat
     - ✅
     - ✅
     - ✅
     - ✅
   * - Bloom
     - bigscience/bloom-7b1
     - ✅
     - ✅
     - ✅
     -
   * - Baichuan2
     - baichuan-inc/Baichuan2-13B-Chat
     - ✅
     - ✅
     - ✅
     -
   * - OPT
     - facebook/opt-6.7b, facebook/opt-30b
     - ✅
     -
     - ✅
     -
   * - Mixtral
     - mistralai/Mistral-7B-Instruct-v0.2
     - ✅
     - ✅
     - ✅
     - ✅

Platforms
~~~~~~~~~~~~~
All above workloads are validated on Intel® Data Center Max 1550 GPU.
The WoQ (Weight Only Quantization) INT4 workloads are also partially validated on Intel® Core™ Ultra series (Arrow Lake-H, Lunar Lake) with Intel® Arc™ Graphics and Intel® Arc™ B-Series GPUs (code-named Battlemage). Refer to Weight Only Quantization INT4 section.

*Note*: The above verified models (including other models in the same model family, like "meta-llama/Llama-2-7b-hf" from Llama family) are well supported with all optimizations like indirect access KV cache, fused ROPE, and prepacked TPP Linear (fp16). For other LLMs families, we are working in progress to cover those optimizations, which will expand the model list above.

LLM fine-tuning on Intel® Data Center Max 1550 GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Model Family
     - Verified models from Huggingface hub
     - Mixed Precision (BF16+FP32)
     - Full fine-tuning
     - LoRA
   * - Llama2
     - meta-llama/Llama-2-7b-hf
     - ✅
     - ✅
     - ✅
   * - Llama2
     - meta-llama/Llama-2-70b-hf
     - ✅
     -
     - ✅
   * - Llama3
     - meta-llama/Meta-Llama-3-8B
     - ✅
     - ✅
     - ✅
   * - Qwen
     - Qwen/Qwen-1.5B
     - ✅
     - ✅
     - ✅
   * - Phi-3-mini 3.8B
     - Phi-3-mini-4k-instruct
     - ✅
     - ✅
     - ✅


Check `LLM best known practice <https://github.com/intel/intel-extension-for-pytorch/tree/release/xpu/2.8.10/examples/gpu/llm>`_ for instructions to install/setup environment and example scripts..

Optimization Methodologies
--------------------------

The brief introduction of these optimizations are as following:

Linear Operator Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LLM inference is Linear weight memory bound task. There are three backends to speedup linear GEMM kernels in Intel® Extension for PyTorch*. They are Intel® oneDNN, Intel® Xe Templates for Linear Algebra (XeLTA) and customized linear kernels for weight only quantization.

Deep Fusion Policy
~~~~~~~~~~~~~~~~~~

Operators fusion is a general approach to reduce the memory access and kernel launch overhead. Except for linear post ops fusion, e.g, linear + activation function, a lot of customized operators are also provided in Intel® Extension for PyTorch\* for further performance improvement, for example, Rotary Position Embedding (RoPE) and Root Mean Square Layer Normalization (RMSNorm).

Segment KV Cache
~~~~~~~~~~~~~~~~

KV Cache is used to reduce computation for decoder layer but it also brings memory overheads, for example, when we use beam search, the KV Cache should be reordered according to latest beam idx and the current key/value should also be concatenated with KV Cache in the attention layer to get entire context to do scale dot product attention. When the sequence is very long, memory overheads caused by the reorder_cache and concatenate will be performance bottleneck. Moreover, in standard
implementation, prompt and response key/value will be kept in contiguous
KV Cache buffers for attention context computation, making it memory
wasting to extend the prompt key/value with Beam Width times. Segment KV
Cache is provided to reduce these overheads. Firstly, prompt key/value
will be computed at Prefill phase and kept on device during the decoding
phase, the shapes of which will not be influenced by the Beam Width
value. At decoding phase, we firstly pre-allocate buffers (key and value
use different buffers) to store the response key/value hidden states and
beam index information, then use beam index history which is shown in
the following left figure to decide which beam should be used by a
timestamp and this information will generate an offset to access the KV
Cache buffer which means that the reorder_cache and concat overheads
will be eliminated by this way. The SDPA kernel based on Segment KV
cache policy is shown as the following right figure.


.. image:: ../images/llm/llm_iakv_2.png
  :width: 400
  :alt: The beam idx trace for every step


.. image:: ../images/llm/llm_kvcache.png
  :width: 800
  :alt: The beam idx trace for every step

Distributed Inference
~~~~~~~~~~~~~~~~~~~~~

All above optimizations already help you to get very good performance
with single GPU card/tile. To further reduce the inference latency and
improve throughput, tensor parallel is also enabled in our solution. You
can firstly use DeepSpeed to auto shard the model and then apply above
optimizations with the frontend API function provided by Intel®
Extension for PyTorch*.

Low Precision Data Types
~~~~~~~~~~~~~~~~~~~~~~~~

While Generative AI (GenAI) workloads and models are getting more and
more popular, large language models (LLM) used in these workloads are
getting more and more parameters. The increasing size of LLM models
enhances workload accuracies; however, it also leads to significantly
heavier computations and places higher requirements to the underlying
hardware. Given that, quantization becomes a more important methodology
for inference workloads.


Weight Only Quantization INT4
-----------------------------

Large Language Models (LLMs) have shown remarkable performance in various natural language processing tasks.

However, deploying them on devices with limited resources is challenging due to their high computational and memory requirements.

To overcome this issue, we propose quantization methods that reduce the size and complexity of LLMs. Unlike `normal quantization <https://github.com/intel/neural-compressor/blob/master/docs/source/quantization.md>`_, such as w8a8, that quantizes both weights and activations, we focus on Weight-Only Quantization (WoQ), which only quantizes the weights statically. WoQ is a better trade-off between efficiency and accuracy, as the main bottleneck of deploying LLMs is the memory bandwidth and WoQ usually preserves more accuracy. Experiments on Qwen-7B, a large-scale LLM, show that we can obtain accurate quantized models with minimal loss of quality.

For more detailed information, check `WoQ INT4 <llm/int4_weight_only_quantization.html>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   llm/int4_weight_only_quantization



