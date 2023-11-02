.. meta::
   :description: This website introduces Intel® Extension for PyTorch*
   :keywords: Intel optimization, PyTorch, Intel® Extension for PyTorch*, LLM

==========================================================
Intel® Extension for PyTorch* - Large Language Model (LLM)
==========================================================

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
     - 3.8-3.11, conda is required.
   * - Compiler
     - Preset in the compilation script below, if compile from source

Installation
============

Prebuilt wheel file are available for Python 3.9. Alternatively, a script is provided to compile from source.

Install From Prebuilt Wheel Files
---------------------------------

.. code:: shell

  python -m pip install torch==2.2.0.dev20231006+cpu torchvision==0.17.0.dev20231006+cpu torchaudio==2.2.0.dev20231006+cpu --index-url https://download.pytorch.org/whl/nightly/cpu
  python -m pip install https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_dev/cpu/intel_extension_for_pytorch-2.2.0.dev0%2Bcpu.llm-cp39-cp39-linux_x86_64.whl
  conda install -y libstdcxx-ng=12 -c conda-forge

Compile From Source
-------------------

.. code:: shell

  wget https://github.com/intel/intel-extension-for-pytorch/raw/llm_feature_branch/scripts/compile_bundle.sh
  bash compile_bundle.sh

It is recommended to build a docker image with `dockerfile <https://github.com/intel/intel-extension-for-pytorch/tree/llm_feature_branch/docker/>`_.

Launch Examples
===============

Supported Models
----------------

The following models are supported. When running the example scripts, it is needed to replace the place holder *<MODEL_ID>* in example launch commands with:

- EleutherAI/gpt-j-6b
- EleutherAI/gpt-neox-20b
- meta-llama/Llama-2-7b-hf
- facebook/opt-30b
- tiiuae/falcon-40b
- bigscience/bloom
- THUDM/chatglm2-6b
- Salesforce/codegen-2B-multi
- HuggingFaceH4/starchat-beta
- bigcode/starcoder
- Baichuan-inc/Baichuan-13B-Chat
- google/flan-t5-xl

.. note::

   The ``config.json`` file needs to be modified to work with the ``modeling_falcon.py`` script. In the following scripts, we need to pass an extra configuration file like "--config-file=model_config/tiiuae_falcon-40b_config.json". This is optional for BF16, but must for INT8.

Install Dependencies
--------------------

.. code:: shell

  conda install -y gperftools -c conda-forge
  conda install -y intel-openmp
  python -m pip install transformers==4.31.0 cpuid accelerate datasets sentencepiece protobuf==3.20.3

  # Used for accuracy test only
  git clone https://github.com/EleutherAI/lm-evaluation-harness
  cd lm-evaluation-harness
  pip install -e .

  # [Optional] install neural-compressor for GPT-J INT8 only
  python -m pip install neural-compressor==2.3.1

  # [Optional] The following is only for DeepSpeed case
  #Install oneccl-bind-pt(also named torch-ccl)
  git clone https://github.com/intel/torch-ccl.git
  cd torch-ccl && git checkout ccl_torch_dev_0905
  git submodule sync && git submodule update --init --recursive
  python setup.py install
  cd ../
  #Install DeepSpeed
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

Example python scripts are provided in Github repo `example directory <https://github.com/intel/intel-extension-for-pytorch/tree/llm_feature_branch/examples/cpu/inference/python/llm/>`_ to launch inference workloads with supported models.

Preparations
^^^^^^^^^^^^

A *prompt.json* file is required to run performance benchmarks. You can use the command below to download a sample file. For simple testing, an argument *\-\-prompt* is provided by the scripts to take a text for processing.

To get these Python scripts, you can either get the entire Github repository down with git command, or use the following wget commands to get individual scripts.

.. code:: shell

  # Get the example scripts with git command
  git clone https://github.com/intel/intel-extension-for-pytorch.git
  cd intel-extension-for-pytorch
  git checkout llm_feature_branch
  cd examples/cpu/inference/python/llm

  # Get the sample prompt.json
  # Make sure the downloaded prompt.json file is under the same directory as that of the example python scripts.
  wget https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/prompt.json

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

Single Instance
^^^^^^^^^^^^^^^

Single Instance Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
  ## Llama 2 quantization
  python run_llama_int8.py --ipex-smooth-quant --lambada --output-dir "saved_results" --jit --int8-bf16-mixed -m <LLAMA MODEL_ID>
  ## GPT-NEOX quantization
  python run_gpt-neox_int8.py --ipex-weight-only-quantization --lambada --output-dir "saved_results" --jit --int8 -m <GPT-NEOX MODEL_ID>
  ## Falcon quantization (example of config-file: model_config/tiiuae_falcon-40b_config.json)
  python run_falcon_int8.py --ipex-weight-only-quantization --output-dir "saved_results" --jit --int8-bf16-mixed -m <MODEL_ID> --config-file <CONFIG_FILE>
  ## Bloom quantization
  python run_bloom_int8.py --ipex-weight-only-quantization --output-dir "saved_results" --jit --int8-bf16-mixed -m <MODEL_ID>
  ## CodeGen quantization
  python run_codegen_int8.py --ipex-weight-only-quantization --output-dir "saved_results" --jit --int8-bf16-mixed -m <MODEL_ID>
  ## GPTBigCode quantization
  python run_gptbigcode_int8.py --ipex-weight-only-quantization --output-dir "saved_results" --jit --int8-bf16-mixed -m <MODEL_ID>
  ## T5 quantization
  python run_t5_int8.py --ipex-weight-only-quantization --output-dir "saved_results" --jit --int8-bf16-mixed -m <MODEL_ID> --input-tokens <INPUT_PROMPT_SIZE> --max-new-tokens <OUTPUT_MAX_NEW_TOKENS>

  ## (2) Run int8 performance test (note that GPT-NEOX uses --int8 instead of --int8-bf16-mixed)
  OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_<MODEL>_int8.py -m <MODEL_ID> --quantized-model-path "./saved_results/best_model.pt" --benchmark --jit --int8-bf16-mixed

  ## ChatGLM2 quantization and run int8 performance
  OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_chatglm_int8.py  -m <MODEL_ID> --ipex-weight-only-quantization  --benchmark --jit --int8-bf16-mixed

  ## Baichuan quantization and run int8 performance
  OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_baichuan_int8.py  -m <MODEL_ID> --ipex-weight-only-quantization  --benchmark --jit --int8-bf16-mixed

Single Instance Accuracy
~~~~~~~~~~~~~~~~~~~~~~~~

Replace the placeholder ``{TASK_NAME}`` in the commands below with choices from [lm-evaluation-harness task table](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md). By default, ``lambada_openai`` is used.

.. code:: shell

  # bfloat16
  OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run_accuracy.py --accuracy-only -m <MODEL_ID> --dtype bfloat16 --ipex --jit --tasks {TASK_NAME}

  # Quantization as a performance part
  # (1) Do quantization to get the quantized model as mentioned above
  # (2) Run int8 accuracy test (note that GPT-NEOX please remove --int8-bf16-mixed)
  OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_accuracy.py --model <MODEL ID> --quantized-model-path "./saved_results/best_model.pt" --dtype int8 --accuracy-only --jit --int8-bf16-mixed --tasks {TASK_NAME}


Distributed Inference
^^^^^^^^^^^^^^^^^^^^^

Shard model for Distributed Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before performing Distributed Performance with DeepSpeed (saving memory usage purpose), the model has to be well sharded with the command below. Afterwards, use ``-m <SHARD MODEL NEW PATH>`` in later test commands.

.. code:: shell

  python create_shard_model.py -m <MODEL ID>  --save-path <SHARD MODEL NEW PATH>

Distributed Performance with DeepSpeed (autoTP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: shell

  unset KMP_AFFINITY

  # Get prompt file to the path of scripts
  mv PATH/TO/prompt.json WORK_DIR

  # Run GPTJ/LLAMA/OPT/Falcon/Bloom/ChatGLM/CodeGen/GPTBigCode/Baichuan/T5 with bfloat16 DeepSpeed
  deepspeed --bind_cores_to_rank run_generation_with_deepspeed.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex --jit

  # Run GPT-NeoX with ipex weight only quantization
  deepspeed --bind_cores_to_rank run_generation_with_deepspeed.py --benchmark -m EleutherAI/gpt-neox-20b --dtype float32 --ipex --jit --ipex-weight-only-quantization

.. warning::

  Limitation:
    - Falcon: from_config is not enabled in transformers to use 'meta' device load to reduce peak memory.
    - ChatGLM2: Auto-tp has no policy for ChatGLM2.

Distributed Accuracy with DeepSpeed (autoTP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run distributed accuracy with 2 ranks of one node for bfloat16 with ipex and jit

.. code:: shell

  source ${ONECCL_DIR}/build/_install/env/setvars.sh
  
  export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so:${CONDA_PREFIX}/lib/libtcmalloc.so
  export LD_LIBRARY_PATH=${ONECCL_DIR}/lib:$LD_LIBRARY_PATH
  unset KMP_AFFINITY
  
  deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py --device cpu --model <MODEL_ID> --dtype bfloat16 --ipex --jit --tasks <TASK_NAME> --accuracy-only

DeepSpeed with weight only quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: shell

  deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank run_accuracy_with_deepspeed.py --device cpu --model <MODEL_ID> --int8-bf16-mixed --ipex --jit --tasks <TASK_NAME> --accuracy-only --ipex-weight-only-quantization
