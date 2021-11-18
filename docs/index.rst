.. Sphinx Test documentation master file, created by
   sphinx-quickstart on Sun Sep 19 17:30:45 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Intel® Extension for PyTorch* documentation!
#######################################################

Intel® Extension for PyTorch* extends PyTorch with optimizations for extra performance boost on Intel hardware. Most of the optimizations will be included in stock PyTorch releases eventually, and the intention of the extension is to deliver up-to-date features and optimizations for PyTorch on Intel hardware, examples include AVX-512 Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX).

Intel® Extension for PyTorch* is structured as the following figure. It is a runtime extension. Users can enable it dynamically in script by importing `intel_extension_for_pytorch`. It covers optimizations for both imperative mode and graph mode. Optimized operators and kernels are registered through PyTorch dispatching mechanism. These operators and kernels are accelerated from native vectorization feature and matrix calculation feature of Intel hardware. During execution, Intel® Extension for PyTorch* intercepts invocation of ATen operators, and replace the original ones with these optimized ones. In graph mode, further operator fusions are applied manually by Intel engineers or through a tool named *oneDNN Graph* to reduce operator/kernel invocation overheads, and thus increase performance.

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
   tutorials/notices
   tutorials/release_notes
   tutorials/installation
   tutorials/examples
   tutorials/api_doc
   tutorials/performance_tuning
   tutorials/blogs_publications
   tutorials/contribution
   tutorials/license
