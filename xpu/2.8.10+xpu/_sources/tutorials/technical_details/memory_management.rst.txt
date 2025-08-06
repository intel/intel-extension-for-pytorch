Memory Management
=================

You can use :meth:`~torch.xpu.memory_allocated` and
:meth:`~torch.xpu.max_memory_allocated` to monitor memory occupied by
tensors, and use :meth:`~torch.xpu.memory_reserved` and
:meth:`~torch.xpu.max_memory_reserved` to monitor the total amount of memory
managed by the caching allocator. Calling :meth:`~torch.xpu.empty_cache`
releases all **unused** cached memory from PyTorch so that those can be used
by other GPU applications. However, the occupied GPU memory by tensors will not
be freed so it can not increase the amount of GPU memory available for PyTorch.
We provide an API :meth:`~intel_extension_for_pytorch.xpu.mem_get_info`, which
can help you estimate the free memory. Please note that it is overestimated. 

For more advanced users, we offer more comprehensive memory benchmarking via
:meth:`~torch.xpu.memory_stats`. We also offer the capability to capture a
complete snapshot of the memory allocator state via
:meth:`~torch.xpu.memory_snapshot`, which can help you understand the
underlying allocation patterns produced by your code.
