API Documentation
#################

General
*******

.. currentmodule:: intel_extension_for_pytorch
.. autofunction:: optimize



    `torch.xpu.optimize` is an alternative of optimize API in Intel® Extension for PyTorch*, to provide identical usage for XPU device only.
    The motivation of adding this alias is to unify the coding style in user scripts base on torch.xpu modular.

    .. code-block:: python
        >>> # bfloat16 inference case.
        >>> model = ...
        >>> model.load_state_dict(torch.load(PATH))
        >>> model.eval()
        >>> optimized_model = torch.xpu.optimize(model, dtype=torch.bfloat16)
        >>> # running evaluation step.
        >>> # bfloat16 training case.
        >>> optimizer = ...
        >>> model.train()
        >>> optimized_model, optimized_optimizer = torch.xpu.optimize(model, dtype=torch.bfloat16, optimizer=optimizer)
        >>> # running training step.



.. currentmodule:: intel_extension_for_pytorch.xpu
.. StreamContext
.. can_device_access_peer
.. current_blas_handle
.. autofunction:: current_device
.. autofunction:: current_stream
.. default_stream
.. autoclass:: device
.. autofunction:: device_count
.. autoclass:: device_of
.. autofunction:: getDeviceIdListForCard
.. autofunction:: get_device_name
.. autofunction:: get_device_properties
.. get_gencode_flags
.. get_sync_debug_mode
.. autofunction:: init
.. ipc_collect
.. autofunction:: is_available
.. autofunction:: is_initialized
.. memory_usage
.. autofunction:: set_device
.. set_stream
.. autofunction:: stream
.. autofunction:: synchronize



Random Number Generator
***********************

.. currentmodule:: intel_extension_for_pytorch.xpu
.. autofunction:: get_rng_state
.. autofunction:: get_rng_state_all
.. autofunction:: set_rng_state
.. autofunction:: set_rng_state_all
.. autofunction:: manual_seed
.. autofunction:: manual_seed_all
.. autofunction:: seed
.. autofunction:: seed_all
.. autofunction:: initial_seed



Streams and events
******************

.. currentmodule:: intel_extension_for_pytorch.xpu
.. autoclass:: Stream
    :members: 
.. ExternalStream
.. autoclass:: Event
    :members: 

Memory management
*****************

.. currentmodule:: intel_extension_for_pytorch.xpu
.. autofunction:: empty_cache
.. list_gpu_processes
.. mem_get_info
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

Other
*****

.. currentmodule:: intel_extension_for_pytorch.xpu
.. autofunction:: get_fp32_math_mode
.. autofunction:: set_fp32_math_mode

    
.. .. automodule:: intel_extension_for_pytorch.quantization
..    :members:

C++ API
*******

.. doxygenenum:: xpu::FP32_MATH_MODE

.. doxygenfunction:: xpu::set_fp32_math_mode

.. doxygenfunction:: xpu::get_queue_from_stream
