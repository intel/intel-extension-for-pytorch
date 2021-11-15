Features
========

Ease-of-use Python API
----------------------

Intel® Extension for PyTorch\* provides simple frontend Python APIs and utilities for users to get performance optimizations such as graph optimization and operator optimization with minor code changes. Typically, only two to three clauses are required to be added to the original code.

Please check `API Documentation <api_doc.html>`_ page for details of API functions. Examples are available in `Examples <examples.html>`_ page.

.. note::

   Please check the following table for package name of Intel® Extension for PyTorch\* from version to version when you do the package importing in Python scripts.

   .. list-table::
      :widths: auto
      :align: center
      :header-rows: 1
   
      * - version
        - package name
      * - 1.2.0
        - intel_pytorch_extension
      * - 1.8.0
        - intel_pytorch_extension
      * - 1.9.0
        - intel_pytorch_extension
      * - 1.10.0
        - intel_extension_for_pytorch

Channels Last
-------------

Comparing to the default NCHW memory format, channels_last (NHWC) memory format could further accelerate convolutional neural networks. In Intel® Extension for PyTorch\*, NHWC memory format has been enabled for most key CPU operators, though not all of them have been merged to PyTorch master branch yet. They are expected to be fully landed in PyTorch upstream soon.

Check more detailed information for `Channels Last <features/nhwc.html>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/nhwc

Auto Mixed Precision (AMP)
--------------------------

Low precision data type BFloat16 has been natively supported on the 3rd Generation Xeon®  Scalable Processors (aka Cooper Lake) with AVX512 instruction set and will be supported on the next generation of Intel® Xeon® Scalable Processors with Intel® Advanced Matrix Extensions (Intel® AMX) instruction set with further boosted performance. The support of Auto Mixed Precision (AMP) with BFloat16 for CPU and BFloat16 optimization of operators have been massively enabled in Intel® Extension for PyTorch\*, and partially upstreamed to PyTorch master branch. Most of these optimizations will be landed in PyTorch master through PRs that are being submitted and reviewed.

Check more detailed information for `Auto Mixed Precision (AMP) <features/amp.html>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/amp

Graph Optimization
------------------

To optimize performance further with torchscript, Intel® Extension for PyTorch\* supports fusion of frequently used operator patterns, like Conv2D+ReLU, Linear+ReLU, etc.  The benefit of the fusions are delivered to users in a transparant fashion.

Check more detailed information for `Graph Optimization <features/graph_optimization.html>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/graph_optimization

Operator Optimization
---------------------

Intel® Extension for PyTorch\* also optimizes operators and implements several customized operators for performance boost. A few  ATen operators are replaced by their optimized counterparts in Intel® Extension for PyTorch\* via ATen registration mechanism. Moreover, some customized operators are implemented for several popular topologies. For instance, ROIAlign and NMS are defined in Mask R-CNN. To improve performance of these topologies, Intel® Extension for PyTorch\* also optimized these customized operators.

.. currentmodule:: intel_extension_for_pytorch.nn
.. autoclass:: FrozenBatchNorm2d

.. currentmodule:: intel_extension_for_pytorch.nn.functional
.. autofunction:: interaction

Train Optimizer
---------------

Not only optimizations for inference workloads are Intel's focus, training workloads are also within Intel's optimization scope. As part of it, optimizations for train optimizer functions are an important perspective. The optimizations as implemented as a mechanism called **Split SGD**, taking advantage of BFloat16 data type and operator fusion. Optimizer **adagrad**, **lamb** and **sgd** are supported.

Check more detailed information for `Split SGD <features/split_sgd.html>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/split_sgd

Runtime Extension (Experimental)
--------------------------------

Intel® Extension for PyTorch* Runtime Extension provides a runtime CPU pool API to bind threads to cores. It also features async tasks. Please **note**: Intel® Extension for PyTorch* Runtime extension is still in the **POC** stage. The API is subject to change. More detailed descriptions are available at `API Documentation page <../api_doc.html>`_.

Check more detailed information for `Runtime Extension <features/runtime_extension.html>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/runtime_extension

INT8 Quantization (Experimental)
--------------------------------

The quantization in Intel® Extension for PyTorch\* integrates `oneDNN graph API <https://spec.oneapi.io/onednn-graph/latest/introduction.html>`_ in the TorchScript graph of PyTorch.

Check more detailed information for `INT8 <features/int8.html>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/int8
