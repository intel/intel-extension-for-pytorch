Large Language Models (LLM) Optimizations Overview
==================================================

In the current technological landscape, Generative AI (GenAI) workloads and models have gained widespread attention and popularity. LLMs have emerged as the dominant models driving these GenAI applications. Most of LLMs are GPT-like architectures that consist of multiple Decoder layers. 
The MultiHeadAttention and FeedForward layer are two key components of every Decoder layer. The generation task is memory bound because iterative decode and kv_cache require special management to reduce memory overheads. Intel® Extension for PyTorch* provides a lot of specific optimizations for these LLMs. 
On the operator level, the extension provides highly efficient GEMM kernel to speed up Linear layer and customized operators to reduce the memory footprint. To better trade-off the performance and accuracy, different low-precision solutions e.g., smoothQuant is enabled. Besides, tensor parallel can also adopt to get lower latency for LLMs.

These LLM-specific optimizations can be automatically applied with a single frontend API function in Python interface, `ipex.optimize_transformers()`. Check `optimize_transformers <./llm/llm_optimize_transformers.md>`_ for more details.

.. toctree::
   :hidden:
   :maxdepth: 1

   llm/llm_optimize_transformers

Optimized Models
----------------

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Model Family
     - LLAMA
     - GPT-J
     - OPT
     - BLOOM
   * - Verified < MODEL ID > (Huggingface hub)
     - "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-70b-hf"
     - "EleutherAI/gpt-j-6b"
     - "facebook/opt-30b", "facebook/opt-1.3b"
     - "bigscience/bloom-7b1", "bigscience/bloom"
   * - FP16
     - ✅
     - ✅
     - ✅
     - ✅

*Note*: The above verified models (including other models in the same model family, like "codellama/CodeLlama-7b-hf" from LLAMA family) are well supported with all optimizations like indirect access KV cache, fused ROPE, and prepacked TPP Linear (fp16). For other LLMs families, we are working in progress to cover those optimizations, which will expand the model list above.

Check `LLM best known practice <https://github.com/intel/intel-extension-for-pytorch/tree/v2.1.20%2Bxpu/examples/gpu/inference/python/llm>`_ for instructions to install/setup environment and example scripts..

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

