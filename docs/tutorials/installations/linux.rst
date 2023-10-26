Installation Guide (Linux/WSL2)
===============================

System Requirements
-------------------

Hardware Requirement
~~~~~~~~~~~~~~~~~~~~

Verified Hardware Platforms:
 - Intel® Data Center GPU Flex Series 170
 - Intel® Data Center GPU Max Series
 - Intel® Arc™ A-Series GPUs (Experimental support)

Software Requirements
~~~~~~~~~~~~~~~~~~~~~

- OS & Intel GPU Drivers

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Hardware
     - OS
     - Driver
   * - Intel® Data Center GPU Flex Series
     - Ubuntu 22.04 (Validated), Red Hat 8.6
     - `Stable 647.21 <https://dgpu-docs.intel.com/releases/stable_647_21_20230714.html>`_
   * - Intel® Data Center GPU Max Series
     - Ubuntu 22.04, Red Hat 8.6, Sles 15sp3/sp4 (Validated)
     - `Stable 647.21 <https://dgpu-docs.intel.com/releases/stable_647_21_20230714.html>`_
   * - Intel® Arc™ A-Series Graphics
     - Ubuntu 22.04
     - `Stable 647.21 <https://dgpu-docs.intel.com/releases/stable_647_21_20230714.html>`_
   * - Intel® Arc™ A-Series Graphics
     - Windows 10, Windows 11 (21H2, 22H2) (via WSL2) (Validated)
     - `Intel® Arc™ & Iris® Xe Graphics - WHQL - Windows* <https://www.intel.com/content/www/us/en/download/726609/intel-arc-iris-xe-graphics-whql-windows.html>`_
   * - CPU (3\ :sup:`rd`\  and 4\ :sup:`th`\   Gen of Intel® Xeon® Scalable Processors)
     - Linux* distributions with glibc>=2.17. Validated on CentOS 8.
     - N/A

- `Intel® oneAPI Base Toolkit 2023.2.0 <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html>`_
- Python 3.8-3.11
- Verified with GNU GCC 11

Preparations
------------

Install Intel GPU Driver
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1

   * - OS
     - Instructions for installing Intel GPU Driver
   * - Linux*
     - Refer to the `Installation Guides <https://dgpu-docs.intel.com/installation-guides/index.html>`_ for the driver installation on individual Linux* distributions. When installing the verified driver mentioned in the table above, use the specific version of each component packages mentioned in the installation guide page, such as `sudo apt-get install intel-opencl-icd=<version>`
   * - Windows 10, Windows 11 (21H2, 22H2) (via WSL2) (Validated)
     - Install the driver for Windows mentioned in the table above first on Windows. Then, follow Steps 4 & 5 of the `Installation Guides <https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-jammy-arc.html#step-4-install-run-time-packages>`_ on WSL2 to install drivers for Linux.

Install oneAPI Base Toolkit
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following components of Intel® oneAPI Base Toolkit are required:
 - Intel® oneAPI DPC++ Compiler (Placeholder `DPCPPROOT` as its installation path)
 - Intel® oneAPI Math Kernel Library (oneMKL) (Placeholder `MKLROOT` as its installation path)

PyTorch-Intel® Extension for PyTorch* Version Mapping
------------------------------------------------------

Intel® Extension for PyTorch* has to work with a corresponding version of PyTorch. Here are the PyTorch versions that we support and the mapping relationship:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - PyTorch Version
     - Extension Version
   * - `v2.0.* <https://github.com/pytorch/pytorch/tree/v2.0.1>`_ (patches needed)
     - `v2.0.* <https://github.com/intel/intel-extension-for-pytorch/tree/v2.0.110+xpu>`_
   * - `v1.13.* <https://github.com/pytorch/pytorch/tree/v1.13.1>`_ (patches needed)
     - `v1.13.* <https://github.com/intel/intel-extension-for-pytorch/tree/v1.13.120+xpu>`_
   * - `v1.10.* <https://github.com/pytorch/pytorch/tree/v1.10.0>`_ (patches needed)
     - `v1.10.* <https://github.com/intel/intel-extension-for-pytorch/tree/v1.10.200+gpu>`_

Install via prebuilt wheel files
--------------------------------

via pip command
~~~~~~~~~~~~~~~

Generic Python
**************

Prebuilt wheel files availability matrix for Python versions:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Extension Version 
     - Python 3.6 
     - Python 3.7 
     - Python 3.8 
     - Python 3.9 
     - Python 3.10 
     - Python 3.11 
   * - 2.0.110+xpu 
     -  
     -  
     - ✔️ 
     - ✔️ 
     - ✔️ 
     - ✔️ 
   * - 1.13.120+xpu 
     -  
     - ✔️ 
     - ✔️ 
     - ✔️ 
     - ✔️ 
     -  
   * - 1.13.10+xpu 
     -  
     - ✔️ 
     - ✔️ 
     - ✔️ 
     - ✔️ 
     -  
   * - 1.10.200+gpu 
     - ✔️ 
     - ✔️ 
     - ✔️ 
     - ✔️ 
     -  
     -  

.. code:: shell

  python -m pip install torch==2.0.1a0 torchvision==0.15.2a0 intel_extension_for_pytorch==2.0.110+xpu -f https://developer.intel.com/ipex-whl-stable-xpu

.. note::

  Under generic Python environments, do NOT install numpy from Intel conda channel.

`Intel® Distribution for Python* <https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html>`_
********************************************************************************************************************************

Prebuit wheel files only support Python 3.9 for Intel® Distribution for Python* environment. Supported version starts from 1.13.10+xpu.

.. code:: shell

  python -m pip install torch==2.0.1a0 torchvision==0.15.2a0 intel_extension_for_pytorch==2.0.110+xpu -f https://developer.intel.com/ipex-whl-stable-xpu-idp

via conda command
~~~~~~~~~~~~~~~~~

Prebuilt conda packages availability matrix for Python versions:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Extension Version 
     - Python 3.6 
     - Python 3.7 
     - Python 3.8 
     - Python 3.9 
     - Python 3.10 
     - Python 3.11
   * - 2.0.110+xpu 
     -  
     -  
     - ✔️ 
     - ✔️ 
     - ✔️ 
     - ✔️
   * - 1.13.120+xpu 
     -  
     - ✔️ 
     - ✔️ 
     - ✔️ 
     - ✔️ 
     - 

Prebuilt conda packages are stored in Intel channel on Anaconda: `intel-extension-for-pytorch <https://anaconda.org/intel/intel-extension-for-pytorch/files>`_ and `pytorch <https://anaconda.org/intel/pytorch/files>`_.

.. code:: shell

  conda install intel-extension-for-pytorch=2.0.110 pytorch=2.0.1 -c intel -c main

Important Notes
~~~~~~~~~~~~~~~

.. note::

  - Installation of TorchVision is optional.
  - You may need to have gomp package in your system (`apt install libgomp1` or `yum/dnf install libgomp`).
  - Since DPC++ compiler doesn't support old `C++ ABI <https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html>`_ (`_GLIBCXX_USE_CXX11_ABI=0`), ecosystem packages, including PyTorch and TorchVision, need to be compiled with the new C++ ABI (`_GLIBCXX_USE_CXX11_ABI=1`).
  - If you need TorchAudio, please follow the `instructions <https://pytorch.org/audio/main/build.html>`_ to compile it from source. According to torchaudio-pytorch dependency table, torchaudio 2.0.2 is recommended.

Install via compiling from source
---------------------------------

Configure the AOT (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please refer to `AOT documentation <../technical_details/AOT.md>`_ for how to configure `USE_AOT_DEVLIST`. Without configuring AOT, the start-up time for processes using Intel® Extension for PyTorch* will be long, so this step is important.

.. toctree::
   :hidden:
   :maxdepth: 1

   ../technical_details/AOT

Compile the bundle (PyTorch*, torchvision, torchaudio, Intel® Extension for PyTorch*) with script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To ensure a smooth compilation of the bundle, including PyTorch*, torchvision, torchaudio, Intel® Extension for PyTorch*, a script is provided in the Github repo. If you would like to compile the binaries from source, it is highly recommended to utilize this script.

.. code:: shell

  $ wget https://raw.githubusercontent.com/intel/intel-extension-for-pytorch/xpu-main/scripts/compile_bundle.sh
  $ bash compile_bundle.sh <DPCPPROOT> <MKLROOT> [AOT]
    DPCPPROOT and MKLROOT are mandatory, should be absolute or relative path to the root directory of DPC++ compiler and oneMKL respectively.
    AOT is optional, should be the text string for environment variable USE_AOT_DEVLIST.

.. note::

  - Recommend to use the `compile_bundle.sh` script in a clean docker container with Intel GPU driver packages installed.
  - Use the `compile_bundle.sh` script under a `conda` environment.
  - Depends on what applications are available on your OS, you probably need to install some Linux commands, like `patch`, `git`, etc. Installation of these Linux commands are not included in this script.
  - The `compile_bundle.sh` script downloads source code of PyTorch*, torchvision, torchaudio, Intel® Extension for PyTorch* into individual folders in its directory. You can consider to create a specific folder to use this script. Wheel files will be generated under `dist` folder of each source code directory. Besides, compilation progress is dumped into a log file `build.log` in each source code directory. The log file is helpful to identify errors occurred during compilation. Should any failure happened, after addressing the issue, you can simply run the `compile_bundle.sh` script again with the same command.

.. code:: shell

  $ mkdir ipex_bundle
  $ cd ipex_bundle
  $ wget .../compile_bundle.sh
  $ bash compile_bundle.sh ...
  $ ls
  audio  compile_bundle.sh  intel_extension_for_pytorch  torch  vision
  $ tree -L 3 .
  .
  ├── audio
  │   ├── dist
  │   │   └── torchaudio-....whl
  │   ├ build.log
  │   └ ...
  ├── compile_bundle.sh
  ├── intel_extension_for_pytorch
  │   ├── dist
  │   │   └── intel_extension_for_pytorch-....whl
  │   ├ build.log
  │   └ ...
  ├── torch
  │   ├── dist
  │   │   └── torch-....whl
  │   ├ build.log
  │   └ ...
  └── vision
      ├── dist
      │   └── torchvision-....whl
      ├ build.log
      └ ...

Sanity Test
-----------

You can run a simple sanity test to double confirm if the correct version is installed, and if the software stack can get correct hardware information onboard your system. Replace the placeholders *{DPCPPROOT}* and *{MKLROOT}* with the installation path on your system in the commands below.

.. code:: shell

  source {DPCPPROOT}/env/vars.sh
  source {MKLROOT}/env/vars.sh
  python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__); [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];"

Install C++ SDK
---------------

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Version
     - cxx11 ABI
   * - 2.0.110+xpu
     - `libtorch-cxx11-abi-shared-with-deps-2.0.0a0.zip <https://intel-extension-for-pytorch.s3.amazonaws.com/libipex/xpu/libtorch-cxx11-abi-shared-with-deps-2.0.0a0.zip>`_, `libintel-ext-pt-cxx11-abi-2.0.110+xpu.run <https://intel-extension-for-pytorch.s3.amazonaws.com/libipex/xpu/libintel-ext-pt-cxx11-abi-2.0.110%2Bxpu.run>`_

**Usage:** Download one run file above according to your scenario, run the following command to install it and follow the `C++ example <./examples.md#c>`_.

.. code:: shell

  unzip libtorch-cxx11-abi-shared-with-deps-2.0.0a0.zip
  bash <libintel-ext-pt-name>.run install <libtorch_path>

You can get full usage help message by running the run file alone, as the following command.

.. code:: shell

  bash <libintel-ext-pt-name>.run
