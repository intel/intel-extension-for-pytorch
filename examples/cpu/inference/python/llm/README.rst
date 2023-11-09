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
     - CentOS/RHEL 8, Ubuntu 22.04 (Docker)
   * - Linux Kernel
     - Intel® 4th Gen Xeon® Platinum: 5.15.0; Intel® 4th Gen Xeon® Max: 5.19.0
   * - Python
     - 3.8-3.11, conda is required.
   * - Compiler
     - Preset in the compilation script below, if compile from source

Supported Models
================

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

Environment Setup
=================

1. Get Intel® Extension for PyTorch* source code

.. code:: shell

  # Get the example scripts with git command
  git clone https://github.com/intel/intel-extension-for-pytorch.git
  cd intel-extension-for-pytorch
  git checkout llm_feature_branch
  cd examples/cpu/inference/python/llm

1.a Use provided `Dockerfile` to build a docker image.

.. code:: shell

  docker build -t ipex-llm:dev .
  
  # Run the container with command below
  docker run --rm -it --privileged ipex-llm:dev bash
  
  # In docker environment, make sure you are in the llm examples directory.
  cd llm

1.b Alternatively, you can take advantage of an environment configuration script to setup the environment.

.. code:: shell

  # GCC 12.3 is required
  # Create environment
  conda create -n llm python=3.9 -y
  conda activate llm
  
  bash ./tools/env_setup.sh

2. Once an environment is configured with either method above, set necessary environment variables with an environment variables activation script and download the sample `prompt.json`.

The *prompt.json* file is required to run performance benchmarks. You can use the command below to download a sample file. For simple testing, an argument *\-\-prompt* is provided by the scripts to take a text for processing.

.. code:: shell

  # Activate environment variables
  source ./tools/env_activate.sh
  wget https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/prompt.json

Run Examples
============

Example python scripts are provided in Github repo `example directory <https://github.com/intel/intel-extension-for-pytorch/tree/llm_feature_branch/examples/cpu/inference/python/llm/>`_ to launch inference workloads with supported models.

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
