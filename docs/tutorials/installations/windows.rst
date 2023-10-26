Installation Guide (Windows)
============================

System Requirements
-------------------

Hardware Requirement
~~~~~~~~~~~~~~~~~~~~

Verified Hardware Platforms:
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
   * - Intel® Arc™ A-Series Graphics
     - Windows 10, Windows 11 (21H2, 22H2) (Validated)
     - `Intel® Arc™ & Iris® Xe Graphics - WHQL - Windows* <https://www.intel.com/content/www/us/en/download/726609/intel-arc-iris-xe-graphics-whql-windows.html>`_

- `Build Tool for Visual Studio 2022 <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022>`_
- `Intel® oneAPI Base Toolkit 2023.2.0 <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html>`_
- Python 3.8-3.11

Preparations
------------

Install Intel GPU Driver
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1

   * - OS
     - Instructions for installing Intel GPU Driver
   * - Windows 10, Windows 11 (21H2, 22H2) (Validated)
     - Please download drivers for Intel® Arc™ A-Series from the web page mentioned in the table above.

Install Microsoft Visual Studio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install **Desktop development with C++** in the Workloads tab. 2 separate options **MSVC** and **Windows SDK** are required.

Install oneAPI Base Toolkit
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   * Do NOT install the following Intel® oneAPI Base Toolkit components:

     * Intel® oneAPI Deep Neural Network Library (oneDNN)

   * Make sure the installation includes Miscrosoft C++ Build Tools integration.

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
     - Python 3.8 
     - Python 3.9 
     - Python 3.10 
     - Python 3.11 
   * - 2.0.110+xpu 
     - ✔️ 
     - ✔️ 
     - ✔️ 
     - ✔️ 

.. code:: shell

  conda install pkg-config libuv
  python -m pip install torch==2.0.1a0 intel_extension_for_pytorch==2.0.110+xpu -f https://developer.intel.com/ipex-whl-stable-xpu

Important Notes
~~~~~~~~~~~~~~~

.. note::

  - Installation of TorchVision is optional.
  - If you need TorchAudio, please follow the `instructions <https://pytorch.org/audio/main/build.html>`_ to compile it from source. According to torchaudio-pytorch dependency table, torchaudio 2.0.2 is recommended.

Install via compiling from source
---------------------------------

Compile the bundle (PyTorch*, torchvision, Intel® Extension for PyTorch*) with script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To ensure a smooth compilation of the bundle, including PyTorch*, torchvision, Intel® Extension for PyTorch*, a script is provided in the Github repo. If you would like to compile the binaries from source, it is highly recommended to utilize this script.

.. note::

  Make sure to surround values of the placeholders *DPCPPROOT* and *MKLROOT* with double quotes (*"*) to work with directory paths containing spaces.

.. code:: shell

  Download https://github.com/intel/intel-extension-for-pytorch/blob/xpu-main/scripts/compile_bundle.bat
  $ compile_bundle.bat "<DPCPPROOT>" "<MKLROOT>"
    DPCPPROOT and MKLROOT are mandatory, should be absolute or relative path to the root directory of DPC++ compiler and oneMKL respectively.

.. note::

  - Recommend to use the `compile_bundle.bat` script under a `conda` environment.
  - The `compile_bundle.bat` script downloads source code of PyTorch*, torchvision, torchaudio, Intel® Extension for PyTorch* into individual folders in its directory. You can consider to create a specific folder to use this script. Wheel files will be generated under `dist` folder of each source code directory.

.. code:: shell

  $ mkdir ipex_bundle
  $ cd ipex_bundle
  Download .../compile_bundle.bat
  $ compile_bundle.bat ...
  $ dir
  compile_bundle.bat  intel_extension_for_pytorch  torch  vision

Sanity Test
-----------

You can run a simple sanity test to double confirm if the correct version is installed, and if the software stack can get correct hardware information onboard your system. Replace the placeholders *{DPCPPROOT}* and *{MKLROOT}* with the installation path on your system in the commands below.

.. code:: shell

  call {DPCPPROOT}\env\vars.bat
  call {MKLROOT}\env\vars.bat
  python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__); [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];"
