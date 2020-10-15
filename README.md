# Intel GPU Extension for PyTorch

*  The Intel GPU Extension for PyTorch is a directed optimized solution for PyTorch end-users to run PyTorch workloads on Intel Graphics cards.

## Pre-requirements:

| **HW proxy** | **OS** | **GPU User Mode Driver** | Python |
| ------ | ------ | ------ | ------ |
| **Gen9** | Ubuntu-18.04 | [compute runtime 20.40.18075](https://github.com/intel/compute-runtime/releases/tag/20.40.18075) + [level zero](https://github.com/oneapi-src/level-zero/releases/tag/v1.0) | 3.6 + |
| **DG1** | Ubuntu-20.04 | [agama-dg1-29/engineering build](http://10.239.87.81/zhenjie/agama/agama-dg1-29/) and replace IGC with [**igc3436**](http://10.239.87.81/zhenjie/igc3436/) | 3.6 + |

### **Dependence:**
```bash
# Install python dependences
python3 -m pip install -r requirements.txt
# Add ubuntu user to video and render group
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER
## logout and relogin
logout
```

## **Compiler Version and Setting:**

### Intel® oneAPI Base Toolkit(Beta) for Linux (version beta09) [https://dynamicinstaller.intel.com/oneapi/toolkits/base-kit/linux/]

### install complier

```bash
chmod +x ./l_[Toolkit Name]Kit_[version].sh
bash ./l_[Toolkit Name]Kit_[version].sh -s -a --silent --eula accept
```
- Environment Variables Setting for DPC++:
```bash
export DPCPP_ROOT=${HOME}/intel/oneapi/compiler/latest/linux
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
#include <iostream>
#include <map>
#include <stdlib.h>
#include <string>
#include <vector>

using namespace std;

map<string, vector<string>> enumDevices() {
  map<string, vector<string>> enummap;
  auto platform_list = cl::sycl::platform::get_platforms();
  for (const auto &platform : platform_list) {
    if (platform.is_host() == false) {
      auto platform_name = platform.get_info<cl::sycl::info::platform::name>();
      auto devices = platform.get_devices();
      for (const auto &device : devices) {
        vector<string> temp;
        if (device.is_gpu()) {
          auto name = device.get_info<cl::sycl::info::device::name>();
          temp.push_back(name);
        }
        enummap.insert(pair<string, vector<string>>(platform_name, temp));
      }
    }
  }
  return enummap;
}

int main() {
  auto enummap = enumDevices();
  cout << "================================================================"
       << endl;
  cout << "                   All Available Backend                        "
       << endl;
  cout << "================================================================"
       << endl;
  for (map<string, vector<string>>::iterator each = enummap.begin();
       each != enummap.end(); ++each) {
    cout << "|Platform:" << endl << "|" << (*each).first << endl;
    cout << "|\t|__|Devices:" << endl;
    for (vector<string>::iterator itr = (*each).second.begin();
         itr != (*each).second.end(); ++itr) {
      cout << "|\t   |" << *itr << endl;
    };
    cout << "----------------------------------------------------------------"
         << endl;
  }

  return 0;
}


```

- Compile Command:

```bash
$ clang++ -I $DPCPP_ROOT/include/sycl device_enum.cpp -L $DPCPP_ROOT/lib -fsycl -o device_enum
```

- Expected result like:
```bash
./device_enum

================================================================
                   All Available Backend                        
================================================================
|Platform:
|Intel(R) CPU Runtime for OpenCL(TM) Applications
|       |__|Devices:
|          |Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz(NonGPU)
----------------------------------------------------------------
|Platform:
|Intel(R) Level-Zero
|       |__|Devices:
|          |Intel(R) Gen9(GPU)
----------------------------------------------------------------
|Platform:
|Intel(R) OpenCL HD Graphics
|       |__|Devices:
|          |Intel(R) Gen9 HD Graphics NEO(GPU)
----------------------------------------------------------------
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

