.. meta::
   :description: This website introduces Intel® Extension for PyTorch*
   :keywords: Intel optimization, PyTorch, Intel® Extension for PyTorch*, LLM

==========================================================
Intel® Extension for PyTorch* - Large Language Model (LLM)
==========================================================

Intel® Extension for PyTorch* extends optimizations to large language models (LLM). Optimizations are at development and experimental phase at this moment. You are welcomed to have a try with these optimizations on 4th Gen Intel® Xeon® Scalable processors.

Install Dependencies
--------------------

.. code:: shell

  WORK_DIR=$PWD
  # GCC 12.3 is required, please set it firstly
  # Create environment (conda recommended)
  conda create -n llm python=3.9 -y
  # install deps
  conda install cmake ninja mkl mkl-include -y
  conda install gperftools -c conda-forge -y

  python -m pip install torch==2.2.0.dev20231006+cpu torchvision==0.17.0.dev20231006+cpu torchaudio==2.2.0.dev20231006+cpu --index-url https://download.pytorch.org/whl/nightly/cpu
  python -m pip install https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_dev/cpu/intel_extension_for_pytorch-2.2.0.dev0%2Bcpu.llm-cp39-cp39-linux_x86_64.whl
  conda install -y libstdcxx-ng=12 -c conda-forge

  # Used for accuracy test only
  git clone https://github.com/EleutherAI/lm-evaluation-harness
  cd lm-evaluation-harness
  pip install -e .

  # Install transformers
  pip install transformers==4.31.0
  # Install others deps
  pip install cpuid accelerate datasets sentencepiece protobuf==3.20.3

  # Setup environment variables for performance on Xeon
  export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so.6
  export KMP_BLOCKTIME=INF
  export KMP_TPAUSE=0
  export KMP_SETTINGS=1
  export KMP_FORJOIN_BARRIER_PATTERN=dist,dist
  export KMP_PLAIN_BARRIER_PATTERN=dist,dist
  export KMP_REDUCTION_BARRIER_PATTERN=dist,dist
  export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so # Intel OpenMP
  # Tcmalloc is a recommended malloc implementation that emphasizes fragmentation avoidance and scalable concurrency support.
  export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

  # [Optional] install neural-compressor for GPT-J static quantization and running GPTQ (see below)
  pip install neural-compressor==2.3.1

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
  #Install OneCCL
  git clone https://github.com/oneapi-src/oneCCL.git
  cd oneCCL
  mkdir build
  cd build
  cmake ..
  make -j install
  source _install/env/setvars.sh
  cd ../..

  # Get the sample prompt.json
  # Make sure the downloaded prompt.json file is under the same directory as that of the python scripts mentioned above.
  wget https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/prompt.json

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

Single NUMA Node Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: shell

  # Get prompt file to the path of scripts
  mv PATH/TO/prompt.json WORK_DIR

  # bfloat16 benchmark
  OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python test_multiple_instance.py --benchmark -m "meta-llama/Llama-2-7b-hf" --dtype bfloat16 --ipex --jit --request-number xxx --instance-number xxx --prompt <PROMPT_FILE>  --token-latency 

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

  # Run LLAMA with bfloat16 DeepSpeed
  deepspeed --bind_cores_to_rank test_multiple_instance_with_deepspeed.py --benchmark -m <MODEL_ID> --dtype bfloat16 --ipex --jit --request-number xxx --instance-number xxx --token-latency

Run scripts with multiple process to enable concrrent request 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1) Add the above runing command into a script test.sh, which include the export xxx and the runing command. 
2) use mpiexec to launch multiple process of test.sh. 
   e.g. you have 8 nodes and 16 sockets,  “mpiexec   -np 16 -ppn 2 –host hostfile sh test.sh “, in this case, there will be 10 * 16 = 160 model instance in the memory, and 16 model instances are active. 

  
