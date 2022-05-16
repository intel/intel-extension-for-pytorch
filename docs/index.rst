.. Sphinx Test documentation master file, created by
   sphinx-quickstart on Sun Sep 19 17:30:45 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Intel® Extension for PyTorch* documentation!
#######################################################

Intel® Extension for PyTorch* extends PyTorch with optimizations for extra performance boost on Intel hardware. Most of the optimizations will be included in stock PyTorch releases eventually, and the intention of the extension is to deliver up-to-date features and optimizations for PyTorch on Intel hardware, examples include AVX-512 Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX).

Intel® Extension for PyTorch* is loaded as a Python module for Python programs or linked as a C++ library for C++ programs. Users can enable it dynamically in script by importing `intel_extension_for_pytorch`.

Comparing to eager mode, graph mode in PyTorch normally yields better performance from optimization methodologies like operator fusion. Intel® Extension for PyTorch* provides further optimizations in graph mode. It is highly recommended for users to take advantage of Intel® Extension for PyTorch* with `TorchScript <https://pytorch.org/docs/stable/jit.html>`_. Users may wish to run with `torch.jit.trace()` function first, since it works with Intel® Extension for PyTorch* better than `torch.jit.script()` function in general. More detailed information can be found at `pytorch.org website <https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#tracing-modules>`_.

It is structured as the following figure. PyTorch components are depicted with white boxes while Intel Extensions are with blue boxes. Intel® Extension for PyTorch* covers optimizations for both eager mode and graph mode. Extra performance of the extension is delivered via both custom addons and overriding existing PyTorch components. In eager mode, the PyTorch frontend is extended with custom Python modules (such as fusion modules), optimal optimizers and INT8 quantization API. Further performance boost is available by converting the eager-mode model into the graph mode via the extended graph fusion passes. Intel® Extension for PyTorch* dispatches the operators into their underlying kernels automatically based on ISA that it detects and leverages vectorization and matrix acceleration units available in Intel hardware as much as possible. oneDNN library is used for computation intensive operations. Intel Extension for PyTorch runtime extension brings better efficiency with finer-grained thread runtime control and weight sharing.

.. image:: ../images/intel_extension_for_pytorch_structure.png
  :width: 800
  :align: center
  :alt: Structure of Intel® Extension for PyTorch*

|

Intel® Extension for PyTorch* has been released as an open–source project at `Github <https://github.com/intel/intel-extension-for-pytorch>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   tutorials/features
   tutorials/releases
   tutorials/installation
   tutorials/examples
   tutorials/performance
   tutorials/api_doc
   tutorials/performance_tuning
   tutorials/blogs_publications
   tutorials/contribution
   tutorials/license
