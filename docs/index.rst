.. meta::
   :description: This website introduces Intel® Extension for PyTorch*
   :keywords: Intel optimization, PyTorch, Intel® Extension for PyTorch*, LLM

============================================================================
Intel® Extension for PyTorch* Large Language Model (LLM) Feature Get Started
============================================================================

Intel® Extension for PyTorch* extends optimizations to large language models (LLM). Optimizations are at development and experimental phase at this moment. You are welcomed to have a try with these optimizations on Intel® Data Center GPU Max Series.

System Requirements
===================

.. list-table::
   :widths: auto
   :header-rows: 0
   :stub-columns: 1

   * - Hardware
     - Intel® Data Center GPU Max Series
   * - OS
     - Ubuntu 22.04
   * - Linux Kernel
     - 5.15.0-76-generic
   * - Intel® oneAPI Base Toolkit
     - 2023.2
   * - Driver
     - `647.21 <https://dgpu-docs.intel.com/releases/stable_647_21_20230714.html>`_

Installation
============

No prebuilt wheel files are provided. Please follow instructions below to compile from source.

Compile From Source
-------------------

Replace the place holders *<DPCPP_ROOT_PATH>* and *<ONEMKL_ROOT_PATH>* to path to DPC++ compiler and oneMKL in oneAPI installation directory.

For example, if oneAPI is installed in */opt/intel*, *<DPCPP_ROOT_PATH>* is */opt/intel/oneapi/compiler/2023.2.0/*, *<ONEMKL_ROOT_PATH>* is */opt/intel/oneapi/mkl/2023.2.0*.

.. code:: shell

  wget https://github.com/intel/intel-extension-for-pytorch/raw/dev/LLM-xpu/scripts/compile_bundle.sh
  bash compile_bundle.sh <DPCPP_ROOT_PATH> <ONEMKL_ROOT_PATH> pvc

Launch Examples
===============

Supported Models
----------------

The following models are supported.

.. list-table::
   :widths: auto
   :header-rows: 0
   :stub-columns: 1

   * - Llama 2
     - Model directory path output from the `transformers conversion tool <https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py>`_.* Verified `meta-llama/Llama-2-7b-chat <https://huggingface.co/meta-llama/Llama-2-7b-chat>`_ and `meta-llama/Llama-2-13b-chat <https://huggingface.co/meta-llama/Llama-2-13b-chat>`_.

\* Llama 2 model conversion steps:

  1. Follow `instructions <https://github.com/facebookresearch/llama#access-on-hugging-face>`_ to download model files for conversion.
  2. Decompress the downloaded model file.
  3. Follow `instructions <https://github.com/facebookresearch/llama-recipes#model-conversion-to-hugging-face>`_ to convert the model.
  4. Launch example scripts with the place holder *<MODEL_ID>* substituted by the *\-\-output_dir* argument value of the conversion script.

Install Dependencies
--------------------

.. code:: shell

  conda install -y libstdcxx-ng=12 -c conda-forge
  conda install gperftools -c conda-forge
  conda install intel-openmp
  python -m pip install cpuid accelerate datasets sentencepiece protobuf==3.20.3


Run Examples
------------

The following python scripts are provided in Github repo `example directory <https://github.com/intel/intel-extension-for-pytorch/tree/dev/llm/examples/gpu/inference/python/llm/text-generation/>`_ to launch inference workloads with supported models. 

- run_llama.py

Preparations
^^^^^^^^^^^^

A separate *prompt.json* file is required to run performance benchmarks. You can use the command below to download a sample file. For simple testing, an argument *\-\-prompt* is provided by the scripts to take a text for processing.

To get these Python scripts, you can either get the entire Github repository down with git command, or use the following wget commands to get individual scripts.

.. code:: shell

  # Get the example scripts with git command
  $ git clone https://github.com/intel/intel-extension-for-pytorch.git
  $ cd intel-extension-for-pytorch
  $ git checkout dev/LLM-xpu
  $ cd examples/gpu/inference/python/llm/text-generation

  # Get individual example scripts
  $ wget https://github.com/intel/intel-extension-for-pytorch/raw/dev/llm/examples/gpu/inference/python/llm/text-generation/run_llama.py  

  # Get the sample prompt.json
  # Make sure the downloaded prompt.json file is under the same directory as that of the python scripts mentioned above.
  $ wget https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/prompt.json

The following environment variables are required to run llama on Intel® Data Center GPU Max Series.

.. code:: shell

  export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so.6
  #  default installation location {ONEAPI_ROOT} is /opt/intel/oneapi for root account, ${HOME}/intel/oneapi for other accounts.
  source ${ONEAPI_ROOT}/setvars.sh
  export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 ENABLE_SDP_FUSION=1

Single Instance Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: shell

   # Get prompt file to the path of scripts
   mv PATH/TO/prompt.json WORK_DIR

   # Run model 

   # 7B 32 in 32 out
   python -u run_llama.py --device xpu --model-dir "/path/to/output/7B-path" --dtype float16 --ipex --greedy

   # 7B 1024 in 128 out
   python -u run_llama.py --device xpu --model-dir "/path/to/output/7B-path" --dtype float16 --ipex --greedy  --input-tokens 1024 --max-new-tokens 128

   # 13B 32 in 32 out
   python -u run_llama.py --device xpu --model-dir "/path/to/output/13B-path" --dtype float16 --ipex --greedy

   # 13B 1024 in 128 out
   python -u run_llama.py --device xpu --model-dir "/path/to/output/13B-path" --dtype float16 --ipex --greedy  --input-tokens 1024 --max-new-tokens 128



Single Instance Accuracy
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: shell

   cd examples/gpu/inference/python/llm/casual-language-modeling/

   LLM_ACC_TEST=1 python -u run_llama.py --device xpu --dtype float16 --accuracy_only --lambada --model "/path/to/output/7B-path" --greedy
   
   LLM_ACC_TEST=1 python -u run_llama.py --device xpu --dtype float16 --accuracy_only --lambada --model "/path/to/output/13B-path" --greedy
