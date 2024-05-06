API Documentation
#################

General
*******

`ipex.optimize` is generally used for generic PyTorch models.

.. automodule:: intel_extension_for_pytorch
.. autofunction:: optimize


`ipex.llm.optimize` is used for Large Language Models (LLM).

.. automodule:: intel_extension_for_pytorch.llm
.. autofunction:: optimize

.. currentmodule:: intel_extension_for_pytorch
.. autoclass:: verbose

LLM Module Level Optimizations (Prototype)
******************************************

Module level optimization APIs are provided for optimizing customized LLMs.

.. automodule:: intel_extension_for_pytorch.llm.modules
.. autoclass:: LinearSilu

.. currentmodule:: intel_extension_for_pytorch.llm.modules
.. autoclass:: LinearSiluMul

.. currentmodule:: intel_extension_for_pytorch.llm.modules
.. autoclass:: Linear2SiluMul

.. currentmodule:: intel_extension_for_pytorch.llm.modules
.. autoclass:: LinearRelu

.. currentmodule:: intel_extension_for_pytorch.llm.modules
.. autoclass:: LinearNewGelu

.. currentmodule:: intel_extension_for_pytorch.llm.modules
.. autoclass:: LinearGelu

.. currentmodule:: intel_extension_for_pytorch.llm.modules
.. autoclass:: LinearMul

.. currentmodule:: intel_extension_for_pytorch.llm.modules
.. autoclass:: LinearAdd

.. currentmodule:: intel_extension_for_pytorch.llm.modules
.. autoclass:: LinearAddAdd

.. currentmodule:: intel_extension_for_pytorch.llm.modules
.. autoclass:: RotaryEmbedding

.. currentmodule:: intel_extension_for_pytorch.llm.modules
.. autoclass:: RMSNorm

.. currentmodule:: intel_extension_for_pytorch.llm.modules
.. autoclass:: FastLayerNorm

.. currentmodule:: intel_extension_for_pytorch.llm.modules
.. autoclass:: IndirectAccessKVCacheAttention

.. currentmodule:: intel_extension_for_pytorch.llm.modules
.. autoclass:: PagedAttention

.. currentmodule:: intel_extension_for_pytorch.llm.modules
.. autoclass:: VarlenAttention

.. automodule:: intel_extension_for_pytorch.llm.functional
.. autofunction:: rotary_embedding

.. currentmodule:: intel_extension_for_pytorch.llm.functional
.. autofunction:: rms_norm

.. currentmodule:: intel_extension_for_pytorch.llm.functional
.. autofunction:: fast_layer_norm

.. currentmodule:: intel_extension_for_pytorch.llm.functional
.. autofunction:: indirect_access_kv_cache_attention

.. currentmodule:: intel_extension_for_pytorch.llm.functional
.. autofunction:: varlen_attention

Fast Bert (Prototype)
************************

.. currentmodule:: intel_extension_for_pytorch
.. autofunction:: fast_bert

Graph Optimization
******************

.. currentmodule:: intel_extension_for_pytorch
.. autofunction:: enable_onednn_fusion

Quantization
************

.. automodule:: intel_extension_for_pytorch.quantization
.. autofunction:: get_smooth_quant_qconfig_mapping
.. autofunction:: prepare
.. autofunction:: convert

Prototype API, introduction is avaiable at `feature page <./features/int8_recipe_tuning_api.md>`_.

.. autofunction:: autotune

CPU Runtime
***********

.. automodule:: intel_extension_for_pytorch.cpu.runtime
.. autofunction:: is_runtime_ext_enabled
.. autoclass:: CPUPool
.. autoclass:: pin
.. autoclass:: MultiStreamModuleHint
.. autoclass:: MultiStreamModule
.. autoclass:: Task
.. autofunction:: get_core_list_of_node_id

.. .. automodule:: intel_extension_for_pytorch.quantization
..    :members:
