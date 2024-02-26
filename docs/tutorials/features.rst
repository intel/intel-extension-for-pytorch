Features
========

This section provides a detailed overview of supported features.

Easy-to-use Python API
----------------------

With only two or three clauses added to your original code, Intel® Extension for PyTorch\* provides simple frontend Python APIs and utilities to get performance optimizations such as graph optimization and operator optimization.

Check the `API Documentation`_ for API functions description and `Examples <examples.md>`_ for usage guidance.

.. note::

  The package name used when you import Intel® Extension for PyTorch\* changed
  from ``intel_pytorch_extension`` (for versions 1.2.0 through 1.9.0) to
  ``intel_extension_for_pytorch`` (for versions 1.10.0 and later). Use the
  correct package name depending on the version you are using.


Large Language Models (LLM, *NEW feature from 2.1.0*)
-----------------------------------------------------

In the current technological landscape, Generative AI (GenAI) workloads and models have gained widespread attention and popularity. Large Language Models (LLMs) have emerged as the dominant models driving these GenAI applications. Starting from 2.1.0, specific optimizations for certain LLM models are 
introduced in the Intel® Extension for PyTorch*.

For more detailed information, check `LLM Optimizations Overview <./llm.html>`_.

torch.compile (Beta, *NEW feature from 2.0.0*)
------------------------------------------------------

PyTorch* 2.0 introduces a new feature ``torch.compile`` to speed up PyTorch* code. It makes PyTorch code run faster by JIT-compiling of PyTorch code into optimized kernels. Intel® Extension for PyTorch\* enables a backend, ``ipex``, in the ``torch.compile`` to optimize generation of the graph model.

To use the feature, import the Intel® Extension for PyTorch* and set the backend parameter of the ``torch.compile`` to ``ipex``.

With ``torch.compile`` backend set to ``ipex``, the following will happen:

1. Register Intel® Extension for PyTorch\* operators to Inductor.
2. Custom fusions at FX graph level, e.g., the migration of existing TorchScript-based fusion kernels in IPEX to inductor, pattern-based fusions to achieve peak performance.

While optimizations with ``torch.compile`` apply to backend, invocation of the ``ipex.optimize`` function is highly recommended as well to apply optimizations in frontend.

.. code-block:: python

   import torch
   import intel_extension_for_pytorch as ipex
   ...
   model = ipex.optimize(model, weights_prepack=False)
   model = torch.compile(model, backend='ipex')
   ...

ISA Dynamic Dispatching
-----------------------

Intel® Extension for PyTorch\* features dynamic dispatching functionality to automatically adapt execution binaries to the most advanced instruction set available on your machine.

For details, refer to `ISA Dynamic Dispatching <features/isa_dynamic_dispatch.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/isa_dynamic_dispatch

Auto Channels Last
------------------

Comparing to the default NCHW memory format, using channels_last (NHWC) memory format could further accelerate convolutional neural networks. In Intel® Extension for PyTorch*, NHWC memory format has been enabled for most key CPU operators. More detailed information is available at `Channels Last <features/nhwc.md>`_.

Intel® Extension for PyTorch* automatically converts a model to channels last memory format when users optimize the model with ``ipex.optimize(model)``. With this feature, there is no need to manually apply ``model=model.to(memory_format=torch.channels_last)`` anymore. More detailed information is available at `Auto Channels Last <features/auto_channels_last.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/nhwc
   features/auto_channels_last

Auto Mixed Precision (AMP)
--------------------------

Low precision data type BFloat16 has been natively supported on 3rd Generation Xeon® Scalable Processors (aka Cooper Lake) with AVX512 instruction set. It will also be supported on the next generation of Intel® Xeon® Scalable Processors with Intel® Advanced Matrix Extensions (Intel® AMX) instruction set providing further boosted performance. The support of Auto Mixed Precision (AMP) with BFloat16 for CPU and BFloat16 optimization of operators has been enabled in Intel® Extension for PyTorch\*, and partially upstreamed to PyTorch master branch. These optimizations will be landed in PyTorch master through PRs that are being submitted and reviewed.

Prefer to use `torch.cpu.amp.autocast()` instead of `torch.autocast(device_name="cpu")`.

For details, refer to `Auto Mixed Precision (AMP) <features/amp.md>`_.

Bfloat16 computation can be conducted on platforms with AVX512 instruction set. On platforms with `AVX512 BFloat16 instruction <https://www.intel.com/content/www/us/en/developer/articles/technical/intel-deep-learning-boost-new-instruction-bfloat16.html>`_, there will be an additional performance boost.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/amp

Graph Optimization
------------------

To further optimize TorchScript performance, Intel® Extension for PyTorch\* supports transparent fusion of frequently used operator patterns such as Conv2D+ReLU and Linear+ReLU.
For more detailed information, check `Graph Optimization <features/graph_optimization.md>`_.

Compared to eager mode, graph mode in PyTorch normally yields better performance from optimization methodologies such as operator fusion. Intel® Extension for PyTorch* provides further optimizations in graph mode. We recommend taking advantage of Intel® Extension for PyTorch* with `TorchScript <https://pytorch.org/docs/stable/jit.html>`_. You may wish to run with the ``torch.jit.trace()`` function first, since it generally works better with Intel® Extension for PyTorch* than using the ``torch.jit.script()`` function. More detailed information can be found at the `pytorch.org website <https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#tracing-modules>`_.

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

.. currentmodule:: intel_extension_for_pytorch.nn.modules
.. autoclass:: MergedEmbeddingBag
.. autoclass:: MergedEmbeddingBagWithSGD

**Auto kernel selection** is a feature that enables users to tune for better performance with GEMM operations. We aim to provide good default performance by leveraging the best of math libraries and enabling `weights_prepack`. The feature was tested with broad set of models. If you want to try other options, you can use `auto_kernel_selection` toggle in `ipex.optimize()` to switch, and you can disable `weights_prepack` in `ipex.optimize()` if you are more concerned about the memory footprint than performance gain. However, in most cases, we recommend sticking with the default settings for the best experience.


Optimizer Optimization
----------------------

Optimizers are one of key parts of the training workloads. Intel® Extension for PyTorch* brings two types of optimizations to optimizers:

1. Operator fusion for the computation in the optimizers.
2. SplitSGD for BF16 training, which reduces the memory footprint of the master weights by half.


For details, refer to `Optimizer Fusion <features/optimizer_fusion.md>`_ and `Split SGD <features/split_sgd.html>`_ 

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

.. note:: Intel® Extension for PyTorch* Runtime extension is still in the prototype stage. The API is subject to change. More detailed descriptions are available in the `API Documentation <api_doc.html>`_.

For more detailed information, check `Runtime Extension <features/runtime_extension.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/runtime_extension

INT8 Quantization
-----------------

Intel® Extension for PyTorch* provides built-in quantization recipes to deliver good statistical accuracy for most popular DL workloads including CNN, NLP and recommendation models.

Users are always recommended to try quantization with the built-in quantization recipe first with Intel® Extension for PyTorch* quantization APIs. For even higher accuracy demandings, users can try with separate `recipe tuning APIs <features/int8_recipe_tuning_api.md>`_. The APIs are powered by Intel® Neural Compressor to take advantage of its tuning feature.

Check more detailed information for `INT8 Quantization <features/int8_overview.md>`_ and `INT8 recipe tuning API guide (Prototype, *NEW feature in 1.13.0*) <features/int8_recipe_tuning_api.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/int8_overview
   features/int8_recipe_tuning_api

Codeless Optimization (Prototype, *NEW feature from 1.13.0*)
---------------------------------------------------------------

This feature enables users to get performance benefits from Intel® Extension for PyTorch* without changing Python scripts. It hopefully eases the usage and has been verified working well with broad scope of models, though in few cases there could be small overhead comparing to applying optimizations with Intel® Extension for PyTorch* APIs.

For more detailed information, check `Codeless Optimization <features/codeless_optimization.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/codeless_optimization.md

Graph Capture (Prototype, *NEW feature from 1.13.0*)
-------------------------------------------------------

Since graph mode is key for deployment performance, this feature automatically captures graphs based on set of technologies that PyTorch supports, such as TorchScript and TorchDynamo. Users won't need to learn and try different PyTorch APIs to capture graphs, instead, they can turn on a new boolean flag `--graph_mode` (default off) in `ipex.optimize()` to get the best of graph optimization.

For more detailed information, check `Graph Capture <features/graph_capture.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/graph_capture

HyperTune (Prototype, *NEW feature from 1.13.0*)
---------------------------------------------------

HyperTune is an prototype feature to perform hyperparameter/execution configuration searching. The searching is used in various areas such as optimization of hyperparameters of deep learning models. The searching is extremely useful in real situations when the number of hyperparameters, including configuration of script execution, and their search spaces are huge that manually tuning these hyperparameters/configuration is impractical and time consuming. Hypertune automates this process of execution configuration searching for the `launcher <performance_tuning/launch_script.md>`_ and Intel® Extension for PyTorch*.

For more detailed information, check `HyperTune <features/hypertune.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/hypertune

Fast BERT Optimization (Prototype, *NEW feature from 2.0.0*)
---------------------------------------------------------------

Intel proposed a technique to speed up BERT workloads. Implementation is integrated into Intel® Extension for PyTorch\*. An API `ipex.fast_bert()` is provided for a simple usage.

Currently `ipex.fast_bert` API is well optimized for training tasks. It works for inference tasks, though, please use the `ipex.optimize` API with graph mode to achieve the peak performance.

For more detailed information, check `Fast BERT <features/fast_bert.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/fast_bert
