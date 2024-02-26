Large Language Models (LLM) Optimization Overview
==================================================

In the current technological landscape, Generative AI (GenAI) workloads and models have gained widespread attention and popularity. Large Language Models (LLMs) have emerged as the dominant models driving these GenAI applications. Most of LLMs are GPT-like architectures that consist of multiple Decoder layers. 
The MultiHeadAttention and FeedForward layer are two key components of every Decoder layer. The generation task is memory bound because iterative decode and kv_cache require special management to reduce memory overheads. Intel® Extension for PyTorch* provides a lot of specific optimizations for these LLMs. 
On the operator level, the extension provides highly efficient GEMM kernel to speed up Linear layer and customized operators to reduce the memory footprint. To better trade-off the performance and accuracy, different low-precision solutions e.g., smoothQuant and weight-only-quantization are also enabled. Besides, tensor parallel can also adopt to get lower latency for LLMs.

These LLM-specific optimizations can be automatically applied with a single frontend API function in Python interface, `ipex.llm.optimize()`. Check `llm.optimize <./llm/llm_optimize.md>`_ for more details.

.. toctree::
   :hidden:
   :maxdepth: 1

   llm/llm_optimize

`ipex.llm` Optimized Model List
-------------------------------

Verified for single instance mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html
    :file: ../_static/htmls/tbl_single.html

Verified for distributed inference mode via DeepSpeed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html
    :file: ../_static/htmls/tbl_deepspeed.html

*Note*: The above verified models (including other models in the same model family, like "codellama/CodeLlama-7b-hf" from LLAMA family) are well supported with all optimizations like indirect access KV cache, fused ROPE, and prepacked TPP Linear (fp32/bf16). We are working in progress to better support the models in the tables with various data types. In addition, more models will be optimized in the future.

*Note*: The accuracy drop issue in distributed inference mode for "tiiuae/falcon-40b" has been fixed by DeepSpeed in a recent patch release `v0.13.1 <https://github.com/microsoft/DeepSpeed/tree/v0.13.1>`_.

Please check `LLM best known practice <../../examples/cpu/inference/python/llm>`_ for instructions to install/setup environment and example scripts.


Demos
-----

Intel® Extension for PyTorch* LLM optimizations can be integrated into a typical LLM Q&A web service.

.. list-table::

   * - .. image:: ../../images/llm/GenAI-bf16.gif
          :width: 500
          :alt: UI with BF16

     - .. image:: ../../images/llm/GenAI-int8.gif
          :width: 500
          :alt: UI with INT8

Following figures show demos with Llama 2 model and GPT-J model with single inference and distributed inference with deepspeed with lower precision data types.

.. list-table::

   * - .. figure:: ../../images/llm/bf16_llama.gif
          :width: 300
          :alt: Llama 2 with BF16

          a

     - .. figure:: ../../images/llm/smoothquant_int8_llama.gif
          :width: 300
          :alt: Llama 2 with INT8 Quantization with SmoothQuant

          b

     - .. figure:: ../../images/llm/woq_int8_llama.gif
          :width: 300
          :alt: Weight Only Quantization with INT8 for Llama 2

          c

   * - .. figure:: ../../images/llm/woq_int4_gptj.gif
          :width: 300
          :alt: Weight Only Quantization with INT4 for GPT-J

          d

     - .. figure:: ../../images/llm/autotp_bf16_llama.gif
          :width: 300
          :alt: Distributed Inference with DeepSpeed with BF16 on Llama 2 with AutoTP feature

          e

     - .. figure:: ../../images/llm/autotp_woq_int8_llama.gif
          :width: 300
          :alt: Distributed Inference with DeepSpeed with Weight Only Quantization INT8 on Llama 2 with AutoTP feature

          f

Figure Legends:

a. Llama 2 model with BF16
b. Llama 2 model with INT8 Quantization with SmoothQuant technique
c. Llama 2 model with INT8 Weight Only Quantization
d. GPT-J model with INT4 Weight Only Quantization
e. Llama 2 model Distributed Inference with DeepSpeed with AutoTP feature on BF16
f. Llama 2 model Distributed Inference with DeepSpeed with AutoTP feature with Weight Only Quantization INT8

Optimization Methodologies
--------------------------

The section below provides a brief introduction to LLM optimization methodologies:

Linear Operator Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Linear operator is the most obvious hotspot in LLMs inference. Intel® Extension for PyTorch* provides dedicated optimization to speedup linear GEMM kernels, through oneDNN, customized linear kernels for weight only quantization, and some other specific tuning. All of them use specific block format to utilize hardware resources in a highly efficient way. 

Low Precision Data Types
~~~~~~~~~~~~~~~~~~~~~~~~

While Generative AI (GenAI) workloads and models are getting more and more popular, LLMs used in these workloads are getting more and more parameters. The increasing size of LLMs enhances workload accuracies; however, it also leads to significantly heavier computations and places higher requirements to the underlying hardware. Given that, quantization becomes a more important methodology for inference workloads.

Quantization with shorter data types benefits from its nature to improve memory IO throughputs and amount of computations on CPU. Moreover, shorter data types make it possible to keep more data in CPU cache, thus reducing memory access occurrences. Comparing to cache access, memory access is much more time costing. Specifically from computation perspective, AVX-512 Vector Neural Network Instructions (VNNI) instruction set shipped with the 2nd Generation Intel® Xeon® Scalable Processors and newer, as well as Intel® Advanced Matrix Extensions (Intel® AMX) instruction set shipped with the 4th Generation Intel® Xeon® Scalable Processors, provide instruction level accelerations to INT8 computations.

Except for the mixed-precision and INT8 native quantization solution, e.g., post-training static quantization and dynamic quantization in Pytorch, `SmoothQuant <https://arxiv.org/abs/2211.10438>`_ and weight only quantization (both INT8 weight and INT4 weight are supported) are also enabled in Intel® Extension for PyTorch* to get beeter accuracy and performance compared with native solution.

Intel® Extension for PyTorch* speeds up INT8 computations by leveraging oneDNN and oneDNN graph as the backend. Intel® Extension for PyTorch* static quantization provides a default recipe to automatically decide which operators to quantize. Its backend oneDNN graph brings matrix-multiplication-based fusions for common seen operator patterns and other common fusions like quantization + data type casting. These fusions help achieve best computation cache locality and efficiency, and thus reduce INT8 quantization overhead significantly.       

Intel® Extension for PyTorch* also delivers INT4 optimizations via 4-bit weight-only quantization (WOQ). As the name indicates, WOQ quantizes only weights to 4-bit integers to further improve the computation efficiency via saved memory bandwidth utilization. This technique reduces text generation latency especially from the second token. AMX INT8 instructions and fusions are also applied for these performant computations.

Indirect Access KV Cache 
~~~~~~~~~~~~~~~~~~~~~~~~

kv_cache is used to reduce computation for decoder layer but it also brings memory overheads. For example, when we use beam search, the kv_cache should be reordered according to latest beam idx and the current key/value should also be concat with kv_cache in the attention layer to get entire context to do scale dot product. When the sequence is very long, memory overheads caused by the reorder_cache and concat will be performance bottleneck. Indirect Access KV_cache (IAKV) is provided to reduce these overheads. Firstly, IAKV pre-allocates buffers (key and value use different buffer) to store all key/value hidden states and beam index information, the data format is shown in the following left figure (beam_width=4 in this case) and token state of key (value) in every timestamp will be store in this pre-allocated buffer. Secondly, we can use beam index history which is shown in the following right figure to decide which beam should be used by a timestamp and this information will generate a offset to access the kv_cache buffer which means that the reorder_cache and concat overheads will be eliminated by this way.


.. image:: ../../images/llm/llm_iakv_1.png
  :width: 400
  :alt: The key/value cache data format


.. image:: ../../images/llm/llm_iakv_2.png
  :width: 400
  :alt: The beam idx trace for every step

Graph Optimization
~~~~~~~~~~~~~~~~~~

Operators fusion is generally used to enable sub-graph fusion to reduce the memory footprint. Except for linear post ops fusion, e.g, linear + activation function, a lot of customized operators are also provided in Intel® Extension for PyTorch* for further performance improvement. For example, Rotary Position Embedding (ROPE) and Root Mean Square Layer Normalization (RMSNorm).

Distributed Inference
~~~~~~~~~~~~~~~~~~~~~

All above optimizations already help you to get very good performance with single instance. To furthly reduce the inference latency and improve throughput, tensor parallel is also enabled in our soluction. You can firstly use DeepSpeed to auto shard the model and then apply above optimizations with the frontend API function provided by Intel® Extension for PyTorch.
