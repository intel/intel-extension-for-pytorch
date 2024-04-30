Technical Details
=================


Optimizer Optimization [GPU]
---------------------------------

Optimizers are a key part of the training workloads. Intel® Extension for PyTorch* brings two types of optimizations to optimizers:

1. Operator fusion for the computation in the optimizers. **[GPU]**


.. toctree::
   :hidden:
   :maxdepth: 1

   technical_details/optimizer_fusion_gpu
 

For more detailed information, check `Optimizer Fusion on CPU <technical_details/optimizer_fusion_cpu.md>`_, `Optimizer Fusion on GPU <technical_details/optimizer_fusion_gpu.md>`_ and `Split SGD <technical_details/split_sgd.html>`_.

Ahead of Time Compilation (AOT) [GPU]
-------------------------------------
 
`AOT Compilation <https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html>`_ is a helpful feature for development lifecycle or distribution time, when you know beforehand what your target device is going to be at application execution time. When AOT compilation is enabled, no additional compilation time is needed when running application. It also benifits the product quality since no just-in-time (JIT) bugs encountered as JIT is skipped and final code executing on the target device can be tested as-is before delivery to end-users. The disadvantage of this feature is that the final distributed binary size will be increased a lot (e.g. from 500MB to 2.5GB for Intel® Extension for PyTorch\*).

.. toctree::
   :hidden:
   :maxdepth: 1

   technical_details/AOT

.. _xpu-memory-management:

Memory Management [GPU]
-----------------------

Intel® Extension for PyTorch* uses a caching memory allocator to speed up memory allocations. This allows fast memory deallocation without any overhead.
Allocations are associated with a sycl device. The allocator attempts to find the smallest cached block that will fit the requested size from the reserved block pool.
If it unable to find a appropriate memory block inside of already allocated ares, the allocator will delegate to allocate a new block memory.

For more detailed information, check `Memory Management <technical_details/memory_management.html>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   technical_details/memory_management



``ipex.optimize`` [GPU]
-----------------------

The ``ipex.optimize`` API is designed to optimize PyTorch\* modules
(``nn.modules``) and specific optimizers within Python modules. Its
optimization options for Intel® GPU device include:

-  Automatic Channels Last
-  Fusing Convolutional Layers with Batch Normalization
-  Fusing Linear Layers with Batch Normalization
-  Replacing Dropout with Identity
-  Splitting Master Weights
-  Fusing Optimizer Update Step

For more detailed information, check `ipex.optimize <technical_details/ipex_optimize.html>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   technical_details/ipex_optimize
