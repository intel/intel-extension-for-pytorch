Technical Details
=================

Optimizer Optimization
----------------------

Optimizers are a key part of the training workloads. Operator fusion for the computation in the optimizers is a good method to get performance boost.


.. toctree::
   :hidden:
   :maxdepth: 1

   technical_details/optimizer_fusion_gpu

For more detailed information, check `Optimizer Fusion on GPU <technical_details/optimizer_fusion_gpu.md>`_.

Ahead of Time Compilation (AOT)
-------------------------------
 
`AOT Compilation <https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html>`_ is a helpful feature for development lifecycle or distribution time, when you know beforehand what your target device is going to be at application execution time. When AOT compilation is enabled, no additional compilation time is needed when running application. It also benifits the product quality since no just-in-time (JIT) bugs encountered as JIT is skipped and final code executing on the target device can be tested as-is before delivery to end-users. The disadvantage of this feature is that the final distributed binary size will be increased a lot (e.g. from 500MB to 2.5GB for Intel® Extension for PyTorch\*).

.. toctree::
   :hidden:
   :maxdepth: 1

   technical_details/AOT

.. _xpu-memory-management:

Memory Management
-----------------

Intel® Extension for PyTorch* uses a caching memory allocator to speed up memory allocations. This allows fast memory deallocation without any overhead.
Allocations are associated with a sycl device. The allocator attempts to find the smallest cached block that will fit the requested size from the reserved block pool.
If it unable to find a appropriate memory block inside of already allocated ares, the allocator will delegate to allocate a new block memory.

For more detailed information, check `Memory Management <technical_details/memory_management.html>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   technical_details/memory_management
