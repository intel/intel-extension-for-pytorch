API Documentation
#################

General
=======

.. currentmodule:: intel_extension_for_pytorch
.. autofunction:: optimize
.. currentmodule:: intel_extension_for_pytorch.llm
.. autofunction:: optimize
.. currentmodule:: intel_extension_for_pytorch
.. autofunction:: get_fp32_math_mode
.. autofunction:: set_fp32_math_mode

Memory management
=================

.. currentmodule:: intel_extension_for_pytorch.xpu
.. autofunction:: empty_cache
.. list_gpu_processes
.. autofunction:: mem_get_info
.. autofunction:: memory_stats
.. autofunction:: memory_summary
.. autofunction:: memory_snapshot
.. autofunction:: memory_allocated
.. autofunction:: max_memory_allocated
.. reset_max_memory_allocated
.. autofunction:: memory_reserved
.. autofunction:: max_memory_reserved
.. set_per_process_memory_fraction
.. memory_cached
.. max_memory_cached
.. reset_max_memory_cached
.. autofunction:: reset_peak_memory_stats
.. caching_allocator_alloc
.. caching_allocator_delete

.. autofunction:: memory_stats_as_nested_dict
.. autofunction:: reset_accumulated_memory_stats


Quantization
============

.. currentmodule:: intel_extension_for_pytorch.quantization.fp8
.. autofunction:: fp8_autocast


C++ API
=======

.. doxygenenum:: torch_ipex::xpu::FP32_MATH_MODE

.. doxygenfunction:: torch_ipex::xpu::set_fp32_math_mode
