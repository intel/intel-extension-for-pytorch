# Intel GPU Extension for PyTorch

> This is Intel GPU Extension for PyTorch Repo

## Pre-requirements:

### **HW proxy:** Intel Gen9 Graphics

### **OS:** Ubuntu-18.04

### **Python:** 3.6.x

### **GitPython:**
```bash
python3 -m pip install GitPython==3.1.1
```

### **UMD Component Installation:**
```bash
sudo wget https://github.com/intel/compute-runtime/releases/download/20.21.16886/intel-gmmlib_20.1.1_amd64.deb
sudo wget https://github.com/intel/compute-runtime/releases/download/20.21.16886/intel-igc-core_1.0.4053_amd64.deb
sudo wget https://github.com/intel/compute-runtime/releases/download/20.21.16886/intel-igc-opencl_1.0.4053_amd64.deb
sudo wget https://github.com/intel/compute-runtime/releases/download/20.21.16886/intel-opencl_20.21.16886_amd64.deb
sudo wget https://github.com/intel/compute-runtime/releases/download/20.21.16886/intel-ocloc_20.21.16886_amd64.deb
sudo wget https://github.com/intel/compute-runtime/releases/download/20.21.16886/intel-level-zero-gpu_0.8.16886_amd64.deb
sudo dpkg -i *.deb
```

### **Intel OpenCL SDK Installation (Only for ComputeCpp):**
Download from http://registrationcenter-download.intel.com/akdlm/irc_nas/vcp/12526/intel_sdk_for_opencl_2017_7.0.0.2568_x64.gz
```bash
tar xzf intel_sdk_for_opencl_2017_7.0.0.2568_x64.gz
cd intel_sdk_for_opencl_2017_7.0.0.2568_x64/
sudo ./install.sh
```

### **Add ubuntu user to video group:**
```bash
sudo usermod -a -G video $USER
sudo shutdown -r 0
```
**Note:** please update $USER to your ubuntu username.

## **Compiler Version and Setting:**

### **DPC++ compiler:** **0420** nightly build (**Contact:** [Kurkov, Vasiliy A](vasiliy.a.kurkov@intel.com) or [Maslov, Oleg](oleg.maslov@intel.com))
- Environment Variables Setting for DPC++:
```bash
export DPCPP_ROOT=/${PATH_TO_Your_Compiler}/linux_prod/compiler/linux
export LD_LIBRARY_PATH=${DPCPP_ROOT}/lib:${DPCPP_ROOT}/compiler/lib/intel64_lin:${LD_LIBRARY_PATH}
export INTELOCLSDKROOT=${DPCPP_ROOT}
export PATH=${DPCPP_ROOT}/bin:$PATH
```

### **ComputeCpp compiler:** CE 1.1.3 Device Compiler - clang version 6.0.0  (based on LLVM 6.0.0svn)
Please download from https://developer.codeplay.com/products/computecpp/ce/download and install.
- Environment Variables Setting for ComputeCpp:
```bash
export INTELOCLSDKROOT=/${PATH_TO_Your_Intel_OpenCL_SDK}
export COMPUTECPP_DIR=/${PATH_TO_Your_ComputeCpp} 
export CXX=${ COMPUTECPP_DIR}/bin/compute++
export LD_LIBRARY_PATH=${COMPUTECPP_DIR}/lib/:${LD_LIBRARY_PATH}
export PATH=${COMPUTECPP_DIR}/bin:$PATH
```
**Note:** Please update ${PATH_TO_Your_Intel_OpenCL_SDK} to where you install Intel OpenCL SDK. It is /opt/intel/opencl by default.

**Note:** Please update ${PATH_TO_Your_ComputeCpp} to where you install ComputeCpp compiler. 

### **Validation:**
Finally, compile and execute the following program and check the result. It is optional.
- Source Code:
```c++
// source file: device_enum.cpp
#include <CL/sycl.hpp>
#include <stdlib.h>
#include <iostream>

void enumDevices(void) {
  auto platform_list = cl::sycl::platform::get_platforms();
  int pidx = 1;
  for(const auto& platform : platform_list){
    auto device_list = platform.get_devices();
    int didx = 1;
    for(const auto& device : device_list){
      printf("platform-%d device-%d ...\n", pidx, didx);
      if (device.is_gpu()) {
        std::cout << device.get_info<cl::sycl::info::device::name>() << std::endl;
        std::cout << device.get_info<cl::sycl::info::device::vendor>() << std::endl;
      } else {
        printf("Non-GPU device\n");
      }
      didx++;
    }
    pidx++;
  }
}

int main() {
  enumDevices();
  return 0;
}

```

- Compile Command:

| Compiler | Command |
| ------ | ------ |
| DPC++ | `$ clang++ -I $DPCPP_ROOT/include/sycl device_enum.cpp -L~/.local/lib -fsycl -o device_enum` |
| ComputeCpp | `$ compute++ -I $COMPUTECPP_DIR/include device_enum.cpp -L~/.local/lib -lComputeCpp -o device_enum` | 

- Desired Result:
```bash
./device_enum

  platform-1 device-1 ...
  Non-GPU device
  platform-2 device-1 ...
  Intel(R) Gen9 HD Graphics NEO
  Intel(R) Corporation 
```

## Repo preparation:
1.  Download source code of corresponding PyTorch
```bash
git clone https://gitlab.devtools.intel.com/intel-pytorch-extension/pytorch.git -b pytorch-1.5
cd pytorch
git submodule update --init --recursive
```
**Note:** Please upload SSH public keys of your building machine onto gitlab "settings", refer to [**this link**](https://gitlab.devtools.intel.com/help/ssh/README#locating-an-existing-ssh-key-pair) for more details.

2.  Download source code of Intel GPU Extension for PyTorch
```bash
git clone https://gitlab.devtools.intel.com/intel-pytorch-extension/intel-pytorch-extension -b <RELEASE_TAG>
cd intel-pytorch-extension
git submodule update --init --recursive
```

## Build and Install PyTorch:
```bash
export USE_CUDNN=0 USE_FBGEMM=0 USE_NNPACK=0 BUILD_CAFFE2_OPS=0
cd pytorch
python3 setup.py install --user
```
**Note:** You can choose your favorite compiler for building PyTorch, which could be the same or different from the one for building Intel PyTorch Extension.
We recommend using **GCC** compiler for building PyTorch (unset $CXX if already set for ComputeCpp before). 

## Build and Install Intel PyTorch Extension:

### Downgrade to gcc5 (only for ***ComputeCpp***, not needed for DPC++)
```bash
sudo apt install gcc-5 g++-5
sudo mv /usr/include/c++/7 /usr/include/c++/7_bak
sudo ln -s /usr/include/c++/5 /usr/include/c++/7
sudo mv /usr/lib/gcc/x86_64-linux-gnu/7 /usr/lib/gcc/x86_64-linux-gnu/7_bak
sudo ln -s /usr/lib/gcc/x86_64-linux-gnu/5 /usr/lib/gcc/x86_64-linux-gnu/7
```

### Build intel-pytorch-extension
```bash
cd intel-pytorch-extension
python3 setup.py install --user
```

## Caveat:
### 1. Set https proxy:
Please configure http(s).proxy for git, otherwise you will get an error similar to “fatal: unable to access 'https://gitlab.devtools.intel.com/intel-pytorch-extension/pytorch.git/': gnutls_handshake() failed: The TLS connection was non-properly terminated.”
```bash
git config --global http.proxy YourAddress:Port
git config --global https.proxy YourAddress:Port
```
### 2. Build order of PyTorch and extension:
Please build intel-pytorch-extension after pytorch is built and installed, otherwise you will get an error “ModuleNotFoundError: No module named 'torch'”.

### 3. Cannot enumerate GPU device through DPC++ runtime:
- If clinfo can enumerate GPU device :
DPC++ runtime has basic requirement for qualified hardware/software condition including Driver Name, Driver version, Device Name, etc.
If these conditions are not fulfilled, corresponding platform/device won’t be enumerated.
It might be workaround by setting SYCL_DEVICE_WHITE_LIST as following: `SYCL_DEVICE_WHITE_LIST="" ./device_enum`
Please compile device_enum in above section “Validation” section.
- If no GPU device is enumerated by clinfo:
No solid solution so far. Try to reboot your machine and see whether it disappeared.

### 4. Downgrade to gcc5 for ComputeCpp build:
In ComputeCpp build, gcc5 is required while gcc7.5 is the default Ubuntu18.04 compiler. Without downgrading to gcc5, you will get following compile error:
```bash
/usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0/tuple:1452:14: error: no matching conversion for functional-style cast from '__global double' to '__result_type'
      (aka 'tuple<__global double>')
      return __result_type(std::forward<_Elements>(__args)...);
```
While this is not TRUE for DPC++ build. Please use gcc7.5 for DPC++ environment.
