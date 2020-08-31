# Intel GPU Extension for PyTorch

*  The Intel GPU Extension for PyTorch is a directed optimized solution for PyTorch end-users to run PyTorch workloads on Intel Graphics cards.

## Pre-requirements:

| **HW proxy** | **OS** | **GPU User Mode Driver** | Python |
| ------ | ------ | ------ | ------ |
| **Gen9** | Ubuntu-18.04 | [20.30.17454](https://github.com/intel/compute-runtime/releases/tag/20.30.17454) | 3.6.x |
| **DG1** | Ubuntu-20.04 | [agama-dg1-29/engineering build](http://10.239.87.81/zhenjie/agama/agama-dg1-29/) and replace IGC with [**igc3436**](http://10.239.87.81/zhenjie/igc3436/) | 3.6.x |

### **Dependence:**
```bash
# Install python dependences
python3 -m pip install -r requirements.txt
# Add ubuntu user to video and render group
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER
sudo reboot
```

## **Compiler Version and Setting:**

### **DPC++ compiler:** **Xmain 0716** nightly build (**Contact:** [Kurkov, Vasiliy A](vasiliy.a.kurkov@intel.com) or [Maslov, Oleg](oleg.maslov@intel.com))
- Environment Variables Setting for DPC++:
```bash
export DPCPP_ROOT=/${PATH_TO_Your_Compiler}/linux_prod/compiler/linux
export LD_LIBRARY_PATH=${DPCPP_ROOT}/lib:${DPCPP_ROOT}/compiler/lib/intel64_lin:${LD_LIBRARY_PATH}
export INTELOCLSDKROOT=${DPCPP_ROOT}
export PATH=${DPCPP_ROOT}/bin:$PATH
```

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

```bash
$ clang++ -I $DPCPP_ROOT/include/sycl device_enum.cpp -L $DPCPP_ROOT/lib -fsycl -o device_enum
```

- Expected result:
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
git clone https://github.com/pytorch/pytorch.git -b v1.5.0
cd pytorch
git submodule update --init --recursive
```

2.  Download source code of Intel GPU Extension for PyTorch
```bash
git clone ssh://git@gitlab.devtools.intel.com:29418/intel-pytorch-extension/intel-pytorch-extension.git
cd intel-pytorch-extension
git submodule update --init --recursive
```
**Note:**
<br>Please upload SSH public keys of your building machine onto gitlab "settings", refer to [**this link**](https://gitlab.devtools.intel.com/help/ssh/README#locating-an-existing-ssh-key-pair) for more details.

## Build and Install PyTorch:
```bash
cd pytorch
git am <PATH_To_intel-pytorch-extension>/torch_patches/*
python3 setup.py install --user
```
**Note:**
<br>You can choose your favorite compiler for building PyTorch, which could be the same or different from the one for building Intel PyTorch Extension.
<br>We recommend using **GCC** compiler for building PyTorch. 

## Build and Install Intel GPU Extension for PyTorch:

### Build intel-pytorch-extension
```bash
cd intel-pytorch-extension
python3 setup.py install --user
```

## Programming Model:
*  ```import torch_ipex``` is a MUST before running any cases with Intel GPU Extension for PyTorch.
*  New devcie "dpcpp" is added into PyTorch proper. Must convert Tensors/Operators/Models onto DPCPP device before running with this Extension.

## Supported Models:
Please download pre-optimized models for this Extension through below command:
```bash
git clone ssh://git@gitlab.devtools.intel.com:29418/intel-pytorch-extension/gpu-optimized-models.git
```

## Known issues:
*  Tensor.new() is not supported on DPCPP device. The alternative solution is Tensor.to("cpu").new().to("dpcpp").
*  Model.storage() is not supported on DPCPP device. The alternative solution is Model.to("cpu").storage().

## Caveat:
### 1. Set https proxy:
Please configure http(s).proxy for git, otherwise you will get an error similar to “fatal: unable to access 'https://git@gitlab.devtools.intel.com:29418/intel-pytorch-extension/intel-pytorch-extension.git': gnutls_handshake() failed: The TLS connection was non-properly terminated.”
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

