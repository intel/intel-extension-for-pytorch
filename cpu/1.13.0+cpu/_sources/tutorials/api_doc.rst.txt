API Documentation
#################

General
*******

.. currentmodule:: intel_extension_for_pytorch
.. autofunction:: optimize
.. autofunction:: enable_onednn_fusion
.. autoclass:: verbose

Quantization
************

.. automodule:: intel_extension_for_pytorch.quantization
.. autofunction:: prepare
.. autofunction:: convert

Experimental API, introduction is avaiable at `feature page <./features/int8_recipe_tuning_api.md>`_.

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
