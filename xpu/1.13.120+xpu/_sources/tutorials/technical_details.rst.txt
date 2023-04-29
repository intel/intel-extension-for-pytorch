Technical Details
=================

ISA Dynamic Dispatching [CPU]
-----------------------------

Intel® Extension for PyTorch\* features dynamic dispatching functionality to automatically adapt execution binaries to the most advanced instruction set available on your machine.

For more detailed information, check `ISA Dynamic Dispatching <technical_details/isa_dynamic_dispatch.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   technical_details/isa_dynamic_dispatch


Graph Optimization [CPU]
------------------------

To further optimize TorchScript performance, Intel® Extension for PyTorch\* supports transparent fusion of frequently used operator patterns such as Conv2D+ReLU and Linear+ReLU.
For more detailed information, check `Graph Optimization <technical_details/graph_optimization.md>`_.

Compared to eager mode, graph mode in PyTorch normally yields better performance from optimization methodologies such as operator fusion. Intel® Extension for PyTorch* provides further optimizations in graph mode. We recommend you take advantage of Intel® Extension for PyTorch* with `TorchScript <https://pytorch.org/docs/stable/jit.html>`_. You may wish to run with the `torch.jit.trace()` function first, since it generally works better with Intel® Extension for PyTorch* than using the `torch.jit.script()` function. More detailed information can be found at the `pytorch.org website <https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#tracing-modules>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   technical_details/graph_optimization


Optimizer Optimization [CPU, GPU]
---------------------------------

Optimizers are a key part of the training workloads. Intel® Extension for PyTorch* brings two types of optimizations to optimizers:

1. Operator fusion for the computation in the optimizers. **[CPU, GPU]**
2. SplitSGD for BF16 training, which reduces the memory footprint of the master weights by half. **[CPU]**


.. toctree::
   :hidden:
   :maxdepth: 1

   technical_details/optimizer_fusion_cpu
   technical_details/optimizer_fusion_gpu
   technical_details/split_sgd

For more detailed information, check `Optimizer Fusion on CPU <technical_details/optimizer_fusion_cpu.md>`_, `Optimizer Fusion on GPU <technical_details/optimizer_fusion_gpu.md>`_ and `Split SGD <technical_details/split_sgd.html>`_.

.. _xpu-memory-management:

Memory Management [GPU]
---------------------------------

Intel® Extension for PyTorch* uses a caching memory allocator to speed up memory allocations. This allows fast memory deallocation without any overhead.
Allocations are associated with a sycl device. The allocator attempts to find the smallest cached block that will fit the requested size from the reserved block pool.
If it unable to find a appropriate memory block inside of already allocated ares, the allocator will delegate to allocate a new block memory.

For more detailed information, check `Memory Management <technical_details/memory_management.html>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   technical_details/memory_management

