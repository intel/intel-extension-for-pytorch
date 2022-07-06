.. Sphinx Test documentation master file, created by
   sphinx-quickstart on Sun Sep 19 17:30:45 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Intel® Extension for PyTorch* Documentation
######################################################

Intel® Extension for PyTorch* extends PyTorch with up-to-date features optimizations for an extra performance boost on Intel hardware. Example optimizations use AVX-512 Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX). Over time, most of these optimizations will be included directly into stock PyTorch releases.

Intel® Extension for PyTorch* provides optimizations for both eager mode and graph mode, however, compared to eager mode, graph mode in PyTorch normally yields better performance from optimization techniques such as operation fusion, and Intel® Extension for PyTorch* amplified them with more comprehensive graph optimizations. Therefore we recommended you to take advantage of Intel® Extension for PyTorch* with `TorchScript <https://pytorch.org/docs/stable/jit.html>`_ whenever your workload supports it. You could choose to run with `torch.jit.trace()` function or `torch.jit.script()` function, but based on our evaluation, `torch.jit.trace()` supports more workloads so we recommend you to use `torch.jit.trace()` as your first choice. More detailed information can be found at `pytorch.org website <https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#tracing-modules>`_.

The extension can be loaded as a Python module for Python programs or linked as a C++ library for C++ programs. In Python scripts users can enable it dynamically by importing `intel_extension_for_pytorch`.

Intel® Extension for PyTorch* is structured as shown in the following figure:

.. figure:: ../images/intel_extension_for_pytorch_structure.png
  :width: 800
  :align: center
  :alt: Structure of Intel® Extension for PyTorch*


PyTorch components are depicted with white boxes while Intel Extensions are with blue boxes. Extra performance of the extension is delivered via both custom addons and overriding existing PyTorch components. In eager mode, the PyTorch frontend is extended with custom Python modules (such as fusion modules), optimal optimizers and INT8 quantization API. Further performance boosting is available by converting the eager-mode model into graph mode via the extended graph fusion passes. Intel® Extension for PyTorch* dispatches the operators into their underlying kernels automatically based on ISA that it detects and leverages vectorization and matrix acceleration units available in Intel hardware, as much as possible. oneDNN library is used for computation intensive operations. Intel Extension for PyTorch runtime extension brings better efficiency with finer-grained thread runtime control and weight sharing.

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
