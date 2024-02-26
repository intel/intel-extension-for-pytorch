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

.. automodule:: intel_extension_for_pytorch
.. autoclass:: verbose



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
