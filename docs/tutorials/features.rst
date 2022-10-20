Features
========

Ease-of-use Python API
----------------------

Intel® Extension for PyTorch\* provides simple frontend Python APIs and utilities to get performance optimizations such as operator optimization.

Check the `API Documentation <api_doc.html>`_ for details of API functions and `Examples <examples.md>`_ for helpful usage tips.

DPC++ Extension
---------------

Intel® Extension for PyTorch\* provides C++ APIs to get DPCPP queue and configure floating-point math mode.

Check the `API Documentation`_ for the details of API functions. `DPC++ Extension <features/DPC++_Extension.md>`_ describes how to write customized DPC++ kernels with a practical example and build it with setuptools and CMake.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/DPC++_Extension

Here are detailed discussions of specific feature topics, summarized in the rest of this document:


Channels Last
-------------

Compared with the default NCHW memory format, using channels_last (NHWC) memory format can further accelerate convolutional neural networks. In Intel® Extension for PyTorch\*, NHWC memory format has been enabled for most key GPU operators.

For more detailed information, check `Channels Last <features/nhwc.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/nhwc

Auto Mixed Precision (AMP)
--------------------------

The support of Auto Mixed Precision (AMP) with BFloat16 and Float16 optimization of operators has been enabled in Intel® Extension for PyTorch\*. BFloat16 is the default low precision floating data type when AMP is enabled. We suggest use AMP for accelerating convolutional and matmul based neural networks.

For more detailed information, check `Auto Mixed Precision (AMP) <features/amp.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/amp

Advanced Configuration
----------------------

The default settings for Intel® Extension for PyTorch* are sufficient for most use cases. However, if users want to customize Intel® Extension for PyTorch*, advanced configuration is available at build time and runtime.

For more detailed information, check `Advanced Configuration <features/advanced_configuration.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/advanced_configuration

Optimizer Optimization
----------------------

Optimizers are a key part of the training workloads. Intel® Extension for PyTorch\* supports operator fusion for computation in the optimizers.

For more detailed information, check `Optimizer Fusion <features/optimizer_fusion.md>`_.

.. toctree::
    :hidden:
    :maxdepth: 1

    features/optimizer_fusion
   
Simple Trace Tool
-----------------

Simple Trace is a built-in debugging tool that lets you control printing out the call stack for a piece of code. Once enabled, it can automatically print out verbose messages of called operators in a stack format with indenting to distinguish the context. 

For more detailed information, check `Simple Trace Tool <features/simple_trace.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/simple_trace
