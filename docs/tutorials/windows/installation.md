# Building IPEX on Native Windows

## 1. Install Microsoft Visual Studio 2022
Follow this [link](https://visualstudio.microsoft.com/vs/) to download and install Microsoft Visual Studio 2022. In order to link oneAPI to VS environment in the next step, select **Desktop development with C++** workload.

## 2. Install Intel oneAPI Base Toolkit
Follow this [link](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=window&distributions=offline) to download and install the latest Windows distribution of oneAPI toolkit.

**Note**: While installing oneAPI base toolkit, please use the custom install:

- **DO NOT** select **Intel® oneAPI Deep Neural Network Library**.
- Make sure to select **Intel® oneAPI DPC++ Compiler** and **Intel® oneAPI Math Kernel Library**. 

**Integrate Microsoft Visual Studio 2022** when the option pops up.

The following instruction assumes the oneAPI base toolkit is installed under folder `C:\oneAPI`.

## 3. Install Anaconda Environment
Follow this [link](https://www.anaconda.com/products/distribution) to download and install the Windows distribution of Anaconda.

**Optional**: Create new conda environment by `conda create -n [env_name] python=3.9`

Inside an Anaconda Prompt, activate oneAPI environment first:
`{YOUR_PATH_TO_ONEAPI}\setvars.bat` e.g. `C:\oneAPI\setvars.bat`

## 4. Install PyTorch
Get the PyTorch repo:
```
git clone https://github.com/pytorch/pytorch -b v1.13.1
```
Get the IPEX repo to apply PyTorch patch needed:
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

## 5. Install IPEX
Get the IPEX repo:

Skip if already done in previous step.
```
git clone https://github.com/intel/intel-extension-for-pytorch.git -b xpu-master
cd intel-extension-for-pytorch
```
Install the required packages:
```
pip install -r requirements.txt
```
Install IPEX:

**Note**: Windows compiler check needs to be **comment out** at **third_party/oneDNN/cmake/dpcpp_driver_check.cmake#L36**
    
```
set BUILD_WITH_CPU=0
set USE_MULTI_CONTEXT=1
set DISTUTILS_USE_SDK=1
set CMAKE_CXX_COMPILER=icx
python setup.py install
```


