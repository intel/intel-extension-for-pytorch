Features
========

Ease-of-use Python API
----------------------

With only two or three clauses added to your original code, Intel® Extension for PyTorch\* provides simple frontend Python APIs and utilities to get performance optimizations such as graph optimization and operator optimization.

Check the `API Documentation`_ for details of API functions. `Examples <examples.md>`_ are also available.

.. note::

  The package name used when you import Intel® Extension for PyTorch\* changed
  from ``intel_pytorch_extension`` (for versions 1.2.0 through 1.9.0) to
  ``intel_extension_for_pytorch`` (for versions 1.10.0 and later). Use the
  correct package name depending on the version you are using.

Here are detailed discussions of specific feature topics, summarized in the rest
of this document:

ISA Dynamic Dispatching
-----------------------

Intel® Extension for PyTorch\* features dynamic dispatching functionality to automatically adapt execution binaries to the most advanced instruction set available on your machine.

For more detailed information, check `ISA Dynamic Dispatching <features/isa_dynamic_dispatch.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/isa_dynamic_dispatch

Auto Channels Last
------------------

Comparing to the default NCHW memory format, using channels_last (NHWC) memory format could further accelerate convolutional neural networks. In Intel® Extension for PyTorch*, NHWC memory format has been enabled for most key CPU operators. More detailed information is available at `Channels Last <features/nhwc.md>`_.

Intel® Extension for PyTorch* automatically converts a model to channels last memory format when users optimize the model with `ipex.optimize(model)`. With this feature users won't need to manually apply `model=model.to(memory_format=torch.channels_last)` any more. More detailed information is available at `Auto Channels Last <features/auto_channels_last.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/nhwc
   features/auto_channels_last

Auto Mixed Precision (AMP)
--------------------------

Low precision data type BFloat16 has been natively supported on 3rd Generation Xeon® Scalable Processors (aka Cooper Lake) with AVX512 instruction set. It will also be supported on the next generation of Intel® Xeon® Scalable Processors with Intel® Advanced Matrix Extensions (Intel® AMX) instruction set providing further boosted performance. The support of Auto Mixed Precision (AMP) with BFloat16 for CPU and BFloat16 optimization of operators has been enabled in Intel® Extension for PyTorch\*, and partially upstreamed to PyTorch master branch. These optimizations will be landed in PyTorch master through PRs that are being submitted and reviewed.

For more detailed information, check `Auto Mixed Precision (AMP) <features/amp.md>`_.

Bfloat16 computation can be conducted on platforms with AVX512 instruction set. On platforms with `AVX512 BFloat16 instruction <https://www.intel.com/content/www/us/en/developer/articles/technical/intel-deep-learning-boost-new-instruction-bfloat16.html>`_, there will be an additional performance boost.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/amp

Graph Optimization
------------------

To further optimize TorchScript performance, Intel® Extension for PyTorch\* supports transparent fusion of frequently used operator patterns such as Conv2D+ReLU and Linear+ReLU.
For more detailed information, check `Graph Optimization <features/graph_optimization.md>`_.

Compared to eager mode, graph mode in PyTorch normally yields better performance from optimization methodologies such as operator fusion. Intel® Extension for PyTorch* provides further optimizations in graph mode. We recommend you take advantage of Intel® Extension for PyTorch* with `TorchScript <https://pytorch.org/docs/stable/jit.html>`_. You may wish to run with the `torch.jit.trace()` function first, since it generally works better with Intel® Extension for PyTorch* than using the `torch.jit.script()` function. More detailed information can be found at the `pytorch.org website <https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#tracing-modules>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/graph_optimization

Operator Optimization
---------------------

Intel® Extension for PyTorch* also optimizes operators and implements several customized operators for performance boosts. A few ATen operators are replaced by their optimized counterparts in Intel® Extension for PyTorch* via the ATen registration mechanism. Some customized operators are implemented for several popular topologies. For instance, ROIAlign and NMS are defined in Mask R-CNN. To improve performance of these topologies, Intel® Extension for PyTorch* also optimized these customized operators.

.. currentmodule:: intel_extension_for_pytorch.nn
.. autoclass:: FrozenBatchNorm2d

.. currentmodule:: intel_extension_for_pytorch.nn.functional
.. autofunction:: interaction

**Auto kernel selection** is a feature that enables users to tune for better performance with GEMM operations. It is provided as parameter –auto_kernel_selection, with boolean value, of the ipex.optimize() function. By default, the GEMM kernel is computed with oneMKL primitives. However, under certain circumstances oneDNN primitives run faster. Users are able to set –auto_kernel_selection to True to run GEMM kernels with oneDNN primitives.” -> "We aims to provide good default performance by leveraging the best of math libraries and enabled weights_prepack, and it has been verified with broad set of models. If you would like to try other alternatives, you can use auto_kernel_selection toggle in ipex.optimize to switch, and you can diesable weights_preack in ipex.optimize if you are concerning the memory footprint more than performance gain. However in majority cases, keeping default is what we recommend.

Optimizer Optimization
----------------------

Optimizers are one of key parts of the training workloads. Intel® Extension for PyTorch* brings two types of optimizations to optimizers:

1.	Operator fusion for the computation in the optimizers.
2.	SplitSGD for BF16 training, which reduces the memory footprint of the master weights by half.


For more detailed information, check `Optimizer Fusion <features/optimizer_fusion.md>`_ and `Split SGD <features/split_sgd.html>`_ 

.. toctree::
   :hidden:
   :maxdepth: 1

   features/optimizer_fusion
   features/split_sgd

Runtime Extension
-----------------

Intel® Extension for PyTorch* Runtime Extension provides PyTorch frontend APIs for users to get finer-grained control of the thread runtime and provides:

- Multi-stream inference via the Python frontend module MultiStreamModule.
- Spawn asynchronous tasks from both Python and C++ frontend.
- Program core bindings for OpenMP threads from both Python and C++ frontend.

.. note:: Intel® Extension for PyTorch* Runtime extension is still in the experimental stage. The API is subject to change. More detailed descriptions are available in the `API Documentation <api_doc.html>`_.

For more detailed information, check `Runtime Extension <features/runtime_extension.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/runtime_extension

INT8 Quantization
-----------------

Intel® Extension for PyTorch* provides built-in quantization recipes to deliver good statistical accuracy for most popular DL workloads including CNN, NLP and recommendation models. On top of that, if users would like to tune for a higher accuracy than what the default recipe provides, a recipe tuning API powered by Intel® Neural Compressor is provided for users to try.

Check more detailed information for `INT8 Quantization <features/int8_overview.md>`_ and `INT8 recipe tuning API guide (Experimental, *NEW feature in 1.13.0*) <features/int8_recipe_tuning_api.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/int8_overview
   features/int8_recipe_tuning_api

Codeless Optimization (Experimental, *NEW feature in 1.13.0*)
-------------------------------------------------------------

This feature enables users to get performance benefits from Intel® Extension for PyTorch* without changing Python scripts. It hopefully eases the usage and has been verified working well with broad scope of models, though in few cases there could be small overhead comparing to applying optimizations with Intel® Extension for PyTorch* APIs.

For more detailed information, check `Codeless Optimization <features/codeless_optimization.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/codeless_optimization.md

Graph Capture (Experimental, *NEW feature in 1.13.0*)
-----------------------------------------------------

Since graph mode is key for deployment performance, this feature automatically captures graphs based on set of technologies that PyTorch supports, such as TorchScript and TorchDynamo. Users won't need to learn and try different PyTorch APIs to capture graphs, instead, they can turn on a new boolean flag `--graph_mode` (default off) in `ipex.optimize` to get the best of graph optimization.

For more detailed information, check `Graph Capture <features/graph_capture.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/graph_capture

HyperTune (Experimental, *NEW feature in 1.13.0*)
-------------------------------------------------

HyperTune is an experimental feature to perform hyperparameter/execution configuration searching. The searching is used in various areas such as optimization of hyperparameters of deep learning models. The searching is extremely useful in real situations when the number of hyperparameters, including configuration of script execution, and their search spaces are huge that manually tuning these hyperparameters/configuration is impractical and time consuming. Hypertune automates this process of execution configuration searching for the `launcher <performance_tuning/launch_script.md>`_ and Intel® Extension for PyTorch*.

For more detailed information, check `HyperTune <features/hypertune.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/hypertune
