
Welcome to Intel® Extension for PyTorch* Documentation
######################################################

Intel® Extension for PyTorch* extends `PyTorch\* <https://github.com/pytorch/pytorch>`_ with up-to-date features and optimizations for an extra performance boost on Intel hardware. It is a heterogeneous, high-performance deep-learning implementation for both CPU and XPU. XPU is a user visible device that is a counterpart of the well-known CPU and CUDA in the PyTorch* community. XPU represents an Intel-specific kernel and graph optimizations for various “concrete” devices. The XPU runtime will choose the actual device when executing AI workloads on the XPU device. The default selected device is Intel GPU. This release introduces specific XPU solution optimizations and gives PyTorch end-users up-to-date features and optimizations on Intel Graphics cards.

Intel® Extension for PyTorch* provides aggressive optimizations for both eager mode and graph mode. Graph mode in PyTorch* normally yields better performance from optimization techniques such as operation fusion, and Intel® Extension for PyTorch* amplifies them with more comprehensive graph optimizations. This extension can be loaded as a Python module for Python programs or linked as a C++ library for C++ programs. In Python scripts users can enable it dynamically by ``import intel_extension_for_pytorch``. To execute AI workloads on XPU, the input tensors and models must be converted to XPU beforehand by ``input = input.to("xpu")`` and ``model = model.to("xpu")``.

Intel® Extension for PyTorch* is structured as shown in the following figure:

.. figure:: images/Intel_Extension_for_PyTorch_Architecture.svg
  :width: 800
  :align: center
  :alt: Architecture of Intel® Extension for PyTorch*

PyTorch components are depicted with white boxes and Intel extensions are with blue boxes. Extra performance of the extension comes from optimizations for both eager mode and graph mode. In eager mode, the PyTorch frontend is extended with custom Python modules (such as fusion modules), optimal optimizers, and INT8 quantization API. Further performance boosting is available by converting the eager-mode model into graph mode via extended graph fusion passes. For the XPU device, optimized operators and kernels are implemented and registered through PyTorch dispatching mechanism. These operators and kernels are accelerated from native vectorization feature and matrix calculation feature of Intel GPU hardware. In graph mode, further operator fusions are supported to reduce operator/kernel invocation overheads, and thus increase performance.

Intel® Extension for PyTorch* utilizes the `DPC++ <https://github.com/intel/llvm#oneapi-dpc-compiler>`_ compiler that supports the latest `SYCL* <https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html>`_ standard and also a number of extensions to the SYCL* standard, which can be found in the `sycl/doc/extensions <https://github.com/intel/llvm/tree/sycl/sycl/doc/extensions>`_ directory. Intel® Extension for PyTorch* also integrates `oneDNN <https://github.com/oneapi-src/oneDNN>`_ and `oneMKL <https://github.com/oneapi-src/oneMKL>`_ libraries and provides kernels based on that. The oneDNN library is used for computation intensive operations. The oneMKL library is used for fundamental mathematical operations.

Intel® Extension for PyTorch* has been released as an open–source project on `GitHub <https://github.com/intel/intel-extension-for-pytorch>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   tutorials/features
   tutorials/releases
   tutorials/installation
   tutorials/examples
   tutorials/api_doc
   tutorials/contribution
   tutorials/license

