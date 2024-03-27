Features
========

Device-Agnostic
***************

Easy-to-use Python API
----------------------

Intel® Extension for PyTorch\* provides simple frontend Python APIs and utilities to get performance optimizations such as operator optimization.

Check the `API Documentation <api_doc.html>`_ for API functions description and `Examples <examples.md>`_ for usage guidance.

Channels Last
-------------

Compared with the default NCHW memory format, using channels_last (NHWC) memory format can further accelerate convolutional neural networks. In Intel® Extension for PyTorch\*, NHWC memory format has been enabled for most key CPU and GPU operators. More detailed information is available at `Channels Last <features/nhwc.md>`_.

Intel® Extension for PyTorch* automatically converts a model to channels last memory format when users optimize the model with ``ipex.optimize(model)``. With this feature, users do not need to manually apply ``model=model.to(memory_format=torch.channels_last)`` anymore. However, models running on Intel® Data Center GPU Flex Series will choose oneDNN layout, so users still need to manually convert the model and data to channels last format. More detailed information is available at `Auto Channels Last <features/auto_channels_last.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/nhwc
   features/auto_channels_last


Auto Mixed Precision (AMP)
--------------------------

Benefiting from less memory usage and computation, low precision data types typically speed up both training and inference workloads. 
On GPU side, support of BFloat16 and Float16 are both available in Intel® Extension for PyTorch\*. BFloat16 is the default low precision floating data type when AMP is enabled.

Detailed information of AMP for GPU are available at `Auto Mixed Precision (AMP) on GPU <features/amp_gpu.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/amp_gpu


Quantization
------------

Intel® Extension for PyTorch* currently supports imperative mode and TorchScript mode for post-training static quantization on GPU. This section illustrates the quantization workflow on Intel GPUs.

Check more detailed information for `INT8 Quantization [XPU] <features/int8_overview_xpu.md>`_. 

On Intel® GPUs, Intel® Extension for PyTorch* also provides INT4 and FP8 Quantization.  Check more detailed information for `FP8 Quantization <./features/float8.md>`_ and `INT4 Quantization <./features/int4.md>`_ 

.. toctree::
   :hidden:
   :maxdepth: 1

   features/int8_overview_xpu
   features/int4
   features/float8


Distributed Training
--------------------

To meet demands of large scale model training over multiple devices, distributed training on Intel® GPUs is supported. Two alternative methodologies are available. Users can choose either to use PyTorch native distributed training module, `Distributed Data Parallel (DDP) <https://pytorch.org/docs/stable/notes/ddp.html>`_, with `Intel® oneAPI Collective Communications Library (oneCCL) <https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneccl.html>`_ support via `Intel® oneCCL Bindings for PyTorch (formerly known as torch_ccl) <https://github.com/intel/torch-ccl>`_ or use Horovod with `Intel® oneAPI Collective Communications Library (oneCCL) <https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneccl.html>`_ support (Prototype).

For more detailed information, check `DDP <features/DDP.md>`_ and `Horovod (Prototype) <features/horovod.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/DDP
   features/horovod


GPU-Specific
************

DLPack Solution
---------------

DLPack defines a stable in-memory data structure for sharing tensors among frameworks. It enables sharing of tensor data without copying when interoparating with other libraries. Intel® Extension for PyTorch* extends DLPack support in PyTorch* for XPU device particularly.

For more detailed information, check `DLPack Solution <features/DLPack.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/DLPack

DPC++ Extension
---------------

Intel® Extension for PyTorch\* provides C++ APIs to get SYCL queue and configure floating-point math mode.

Check the `API Documentation`_ for the details of API functions. `DPC++ Extension <features/DPC++_Extension.md>`_ describes how to write customized DPC++ kernels with a practical example and build it with setuptools and CMake.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/DPC++_Extension

Advanced Configuration
----------------------

The default settings for Intel® Extension for PyTorch* are sufficient for most use cases. However, if you need to customize Intel® Extension for PyTorch*, advanced configuration is available at build time and runtime.

For more detailed information, check `Advanced Configuration <features/advanced_configuration.md>`_.

A driver environment variable `ZE_FLAT_DEVICE_HIERARCHY` is currently used to select the device hierarchy model with which the underlying hardware is exposed. By default, each GPU tile is used as a device. Check the `Level Zero Specification Documentation <https://spec.oneapi.io/level-zero/latest/core/PROG.html#environment-variables>`_ for more details.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/advanced_configuration

Fully Sharded Data Parallel (FSDP)
----------------------------------

`Fully Sharded Data Parallel (FSDP)` is a PyTorch\* module that provides industry-grade solution for large model training. FSDP is a type of data parallel training, unlike DDP, where each process/worker maintains a replica of the model, FSDP shards model parameters, optimizer states and gradients across DDP ranks to reduce the GPU memory footprint used in training. This makes the training of some large-scale models feasible.

For more detailed information, check `FSDP <features/FSDP.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/FSDP

Inductor
--------
Intel® Extension for PyTorch\* now empowers users to seamlessly harness graph compilation capabilities for optimal PyTorch model performance on Intel GPU via the flagship `torch.compile <https://pytorch.org/docs/stable/generated/torch.compile.html#torch-compile>`_ API through the default "inductor" backend (`TorchInductor <https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747/1>`_ ). 

For more detailed information, check `Inductor <features/torch_compile_gpu.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/torch_compile_gpu

Legacy Profiler Tool (Prototype)
-----------------------------------

The legacy profiler tool is an extension of PyTorch* legacy profiler for profiling operators' overhead on XPU devices. With this tool, you can get the information in many fields of the run models or code scripts. Build Intel® Extension for PyTorch* with profiler support as default and enable this tool by adding a `with` statement before the code segment.

For more detailed information, check `Legacy Profiler Tool <features/profiler_legacy.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/profiler_legacy

Simple Trace Tool (Prototype)
--------------------------------

Simple Trace is a built-in debugging tool that lets you control printing out the call stack for a piece of code. Once enabled, it can automatically print out verbose messages of called operators in a stack format with indenting to distinguish the context. 

For more detailed information, check `Simple Trace Tool <features/simple_trace.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/simple_trace

Kineto Supported Profiler Tool (Prototype)
---------------------------------------------

The Kineto supported profiler tool is an extension of PyTorch\* profiler for profiling operators' executing time cost on GPU devices. With this tool, you can get information in many fields of the run models or code scripts. Build Intel® Extension for PyTorch\* with Kineto support as default and enable this tool using the `with` statement before the code segment.

For more detailed information, check `Profiler Kineto <features/profiler_kineto.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/profiler_kineto


Compute Engine (Prototype feature for debug)
-----------------------------------------------

Compute engine is a prototype feature which provides the capacity to choose specific backend for operators with multiple implementations.

For more detailed information, check `Compute Engine <features/compute_engine.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/compute_engine


