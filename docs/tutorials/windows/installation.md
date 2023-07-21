# Building Intel® Extension for PyTorch* on Native Windows

## 1. Install MSVC and Windows SDK fom Visual Studio Build Tools
Follow this [link](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) to download and install MSVC and Windows SDK. Select **Desktop development with C++** workload, and under optional components select **MSVC** and **Windows SDK**.

## 2. Install Intel oneAPI Base Toolkit
Follow this [link](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=window&distributions=offline) to download and install the latest Windows distribution of oneAPI toolkit.

**Note**: While installing oneAPI base toolkit, please use the custom install:

- **DO NOT** select **Intel® oneAPI Deep Neural Network Library**.
- Make sure to select **Intel® oneAPI DPC++ Compiler** and **Intel® oneAPI Math Kernel Library**. 

Integrate **Miscrosoft C++ Build Tools 2022** when the option pops up.

The following instruction assumes the oneAPI base toolkit is installed under folder `C:\oneAPI`.

## 3. Install Anaconda Environment
Follow this [link](https://www.anaconda.com/products/distribution) to download and install the Windows distribution of Anaconda.

**Optional**: Create new conda environment by `conda create -n [env_name] python=3.9`

Inside an Anaconda Prompt, activate oneAPI environment first:
`{YOUR_PATH_TO_ONEAPI}\setvars.bat` e.g. `C:\oneAPI\setvars.bat`

## 4. Install PyTorch
Get the PyTorch repo:
```
git clone https://github.com/pytorch/pytorch -b v2.0.1
```
Get the Intel® Extension for PyTorch* repo to apply PyTorch patch needed:
```
git clone https://github.com/intel/intel-extension-for-pytorch.git -b xpu-master
cd pytorch
git apply ../intel-extension-for-pytorch/torch_patches/*.patch
```
Install the required packages:
```
pip install -r requirements.txt
conda install libuv
```
Install PyTorch:
```
python setup.py install
```

## 5. Install Intel® Extension for PyTorch*
Get the Intel® Extension for PyTorch* repo:

Skip if already done in previous step.
```
git clone https://github.com/intel/intel-extension-for-pytorch.git -b xpu-master
cd intel-extension-for-pytorch
```
Install the required packages:
```
pip install -r requirements.txt
```
Install Intel® Extension for PyTorch*:
```
set BUILD_WITH_CPU=0
set USE_MULTI_CONTEXT=1
set DISTUTILS_USE_SDK=1
set CMAKE_CXX_COMPILER=icx
python setup.py install
```


