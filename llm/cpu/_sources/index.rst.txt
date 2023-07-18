.. meta::
   :description: This website introduces Intel® Extension for PyTorch*
   :keywords: Intel optimization, PyTorch, Intel® Extension for PyTorch*, LLM

============================================================================
Intel® Extension for PyTorch* Large Language Model (LLM) Feature Get Started
============================================================================

Intel® Extension for PyTorch* extends optimizations to large language models (LLM). Optimizations are at development and experimental phase at this moment. You are welcomed to have a try with these optimizations on 4th Gen Intel® Xeon® Scalable processors.

System Requirements
===================

.. list-table::
   :widths: auto
   :header-rows: 0
   :stub-columns: 1

   * - Hardware
     - 4th Gen Intel® Xeon® Scalable processors
   * - OS
     - CentOS/RHEL 8
   * - Linux Kernel
     - Intel® 4th Gen Xeon® Platinum: 5.15.0; Intel® 4th Gen Xeon® Max: 5.19.0
   * - Python
     - 3.9, conda is required.
   * - Compiler
     - Preset in the compilation script below, if compile from source

Installation
============

Prebuilt wheel file are available for Python 3.9. Alternatively, a script is provided to compile from source.

Install From Prebuilt Wheel Files
---------------------------------

.. code:: shell

  python -m pip install torch==2.1.0.dev20230711+cpu torchvision==0.16.0.dev20230711+cpu torchaudio==2.1.0.dev20230711+cpu --index-url https://download.pytorch.org/whl/nightly/cpu
  python -m pip install https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_dev/cpu/intel_extension_for_pytorch-2.1.0.dev0%2Bcpu.llm-cp39-cp39-linux_x86_64.whl
  conda install -y libstdcxx-ng=12 -c conda-forge

Compile From Source
-------------------

.. code:: shell

  wget https://github.com/intel/intel-extension-for-pytorch/raw/v2.1.0.dev%2Bcpu.llm/scripts/compile_bundle.sh
  bash compile_bundle.sh

Launch Examples
===============

Supported Models
----------------

The following 3 models are supported. When running the example scripts, it is needed to replace the place holder *<MODEL_ID>* in example launch commands with:

.. list-table::
   :widths: auto
   :header-rows: 0
   :stub-columns: 1

   * - GPT-J
     - "EleutherAI/gpt-j-6b"
   * - GPT-Neox
     - "EleutherAI/gpt-neox-20b"
   * - LLaMA
     - Model directory path output from the `transformers conversion tool <https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py>`_.* Verified 7B and 13B.

\* LLaMA model conversion steps:

  1. Request access to the model at `Meta website <https://ai.meta.com/blog/large-language-model-llama-meta-ai/>`_.
  2. Decompress the downloaded model file.
  3. Convert the LLaMA model with the conversion script.
  4. Launch example scripts with the place holder *<MODEL_ID>* substituted with the output_dir *<OUTPUT_DIR>* of the conversion script.

.. code:: shell

  $ wget https://github.com/huggingface/transformers/raw/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
  $ python convert_llama_weights_to_hf.py --help
  usage: convert_llama_weights_to_hf.py [-h] [--input_dir INPUT_DIR]
                                        [--model_size {7B,13B,30B,65B,tokenizer_only}]
                                        [--output_dir OUTPUT_DIR]

  options:
    -h, --help            show this help message and exit
    --input_dir INPUT_DIR
                          Location of LLaMA weights, which contains tokenizer.model and model
                          folders
    --model_size {7B,13B,30B,65B,tokenizer_only}
    --output_dir OUTPUT_DIR
                          Location to write HF model and tokenizer

Install Dependencies
--------------------

.. code:: shell

  conda install gperftools -c conda-forge
  conda install intel-openmp
  python -m pip install transformers==4.28.1 cpuid accelerate datasets sentencepiece protobuf==3.20.3

  # [Optional] install neural-compressor for GPT-J INT8 only
  python -m pip install neural-compressor==2.2

  # [Optional] The following is only for DeepSpeed case
  git clone https://github.com/delock/DeepSpeedSYCLSupport
  cd DeepSpeedSYCLSupport
  git checkout gma/run-opt-branch
  python -m pip install -r requirements/requirements.txt
  python setup.py install
  cd ../
  git clone https://github.com/oneapi-src/oneCCL.git
  cd oneCCL
  mkdir build
  cd build
  cmake ..
  make -j install
  source _install/env/setvars.sh
  cd ../..

.. note::

  If an error complaining *ninja* is not found when compiling deepspeed, please use conda and pip command to uninstall all ninja packages, and reinstall it with pip.

Run Examples
------------

The following 5 python scripts are provided in Github repo `example directory <https://github.com/intel/intel-extension-for-pytorch/tree/v2.1.0.dev%2Bcpu.llm/examples/cpu/inference/python/llm/>`_ to launch inference workloads with supported models.

- run_generation.py
- run_generation_with_deepspeed.py
- run_gpt-j_int8.py
- run_gpt-neox_int8.py
- run_llama_int8.py

Preparations
^^^^^^^^^^^^

A separate *prompt.json* file is required to run performance benchmarks. You can use the command below to download a sample file. For simple testing, an argument *\-\-prompt* is provided by the scripts to take a text for processing.

To get these Python scripts, you can either get the entire Github repository down with git command, or use the following wget commands to get individual scripts.

.. code:: shell

  # Get the example scripts with git command
  $ git clone https://github.com/intel/intel-extension-for-pytorch.git
  $ cd intel-extension-for-pytorch
  $ git checkout v2.1.0.dev+cpu.llm
  $ cd examples/cpu/inference/python/llm

  # Alternatively, get individual example scripts
  $ wget https://github.com/intel/intel-extension-for-pytorch/raw/v2.1.0.dev%2Bcpu.llm/examples/cpu/inference/python/llm/run_generation.py
  $ wget https://github.com/intel/intel-extension-for-pytorch/raw/v2.1.0.dev%2Bcpu.llm/examples/cpu/inference/python/llm/run_generation_with_deepspeed.py
  $ wget https://github.com/intel/intel-extension-for-pytorch/raw/v2.1.0.dev%2Bcpu.llm/examples/cpu/inference/python/llm/run_gpt-j_int8.py
  $ wget https://github.com/intel/intel-extension-for-pytorch/raw/v2.1.0.dev%2Bcpu.llm/examples/cpu/inference/python/llm/run_gpt-neox_int8.py
  $ wget https://github.com/intel/intel-extension-for-pytorch/raw/v2.1.0.dev%2Bcpu.llm/examples/cpu/inference/python/llm/run_llama_int8.py

  # Get the sample prompt.json
  # Make sure the downloaded prompt.json file is under the same directory as that of the python scripts mentioned above.
  $ wget https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/prompt.json

The following environment variables are required to achieve a good performance on 4th Gen Intel® Xeon® Scalable processors.

.. code:: shell

  export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so.6

  # Setup environment variables for performance on Xeon
  export KMP_BLOCKTIME=INF
  export KMP_TPAUSE=0
  export KMP_SETTINGS=1
  export KMP_AFFINITY=granularity=fine,compact,1,0
  export KMP_FORJOIN_BARRIER_PATTERN=dist,dist
  export KMP_PLAIN_BARRIER_PATTERN=dist,dist
  export KMP_REDUCTION_BARRIER_PATTERN=dist,dist
  export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so # Intel OpenMP

  # Tcmalloc is a recommended malloc implementation that emphasizes fragmentation avoidance and scalable concurrency support.
  export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

Single Instance Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: shell

  # Get prompt file to the path of scripts
  mv PATH/TO/prompt.json WORK_DIR

  # bfloat16 benchmark
  OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run_generation.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex --jit

  # int8 benchmark
  ## (1) Do quantization to get the quantized model
  mkdir saved_results

  ## GPT-J quantization
  python run_gpt-j_int8.py --ipex-smooth-quant --lambada --output-dir "saved_results" --jit --int8-bf16-mixed -m <GPTJ MODEL_ID>
  ## LLaMA quantization
  python run_llama_int8.py --ipex-smooth-quant --lambada --output-dir "saved_results" --jit --int8-bf16-mixed -m <LLAMA MODEL_ID>
  ## GPT-NEOX quantization
  python run_gpt-neox_int8.py --ipex-weight-only-quantization --lambada --output-dir "saved_results" --jit --int8 -m <GPT-NEOX MODEL_ID>

  ## (2) Run int8 performance test
  OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_<MODEL>_int8.py -m <MODEL_ID> --quantized-model-path "./saved_results/best_model.pt" --benchmark --jit --int8-bf16-mixed

Single Instance Accuracy
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: shell

  # bfloat16
  OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run_generation.py --accuracy-only -m <MODEL_ID> --dtype bfloat16 --ipex --jit

  # Quantization as a performance part
  ## (1) Do quantization to get the quantized model as mentioned above
  ## (2) Run int8 accuracy test
  OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_<MODEL>_int8.py -m <MODEL_ID> --quantized-model-path "./saved_results/best_model.pt" --accuracy-only --jit --int8-bf16-mixed

Distributed Performance with DeepSpeed (autoTP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: shell

  export DS_SHM_ALLREDUCE=1
  unset KMP_AFFINITY

  # Get prompt file to the path of scripts
  mv PATH/TO/prompt.json WORK_DIR

  # Run GPTJ/LLAMA with bfloat16  DeepSpeed
  deepspeed --bind_cores_to_rank run_generation_with_deepspeed.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex --jit

  # Run GPT-NeoX with ipex weight only quantization
  deepspeed --bind_cores_to_rank run_generation_with_deepspeed.py --benchmark -m EleutherAI/gpt-neox-20b --dtype float32 --ipex --jit --ipex-weight-only-quantization
