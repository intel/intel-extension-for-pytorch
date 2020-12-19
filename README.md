# Intel GPU Extension for PyTorch

*  The Intel GPU Extension for PyTorch is a directed optimized solution for PyTorch end-users to run PyTorch workloads on Intel Graphics cards.

## Pre-requirements:

| **HW proxy** | **OS** | **GPU User Mode Driver** | Python |
| ------ | ------ | ------ | ------ |
| **Gen9** | Ubuntu-20.04 | [compute runtime 20.43.18277](https://github.com/intel/compute-runtime/releases/tag/20.43.18277) | 3.6 + |
| **DG1** | Ubuntu-20.04 |  [Pytorch BKC](https://wiki.ith.intel.com/pages/viewpage.action?spaceKey=OSGCSH&title=PyTorch+Environment+BKC)| 3.6 + |
| **DG1** | OpenSUSE Leap 15sp2| [BKC](about:blank) | 3.6+ |
| **ATS** | Ubuntu-20.04 |  [PyTorch BKC](https://wiki.ith.intel.com/pages/viewpage.action?spaceKey=OSGCSH&title=PyTorch+Environment+BKC) | 3.6 + |
| **ATS** | OpenSUSE Leap 15sp2| [BKC](about:blank) | 3.6+ |

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

### Using oneAPI BaseKit 

#### [Intel® oneAPI Base Toolkit(Beta) for Linux](https://dynamicinstaller.intel.com/oneapi/toolkits/base-kit/linux/)

#### install complier

```bash
chmod +x ./l_[Toolkit Name]Kit_[version].sh
bash ./l_[Toolkit Name]Kit_[version].sh -s -a --silent --eula accept
```

- Evironment Variables Setting for DPC++:

```bash
export DPCPP_ROOT=${HOME}/intel/oneapi/compiler/latest/linux
export LD_LIBRARY_PATH=${DPCPP_ROOT}/lib:${DPCPP_ROOT}/compiler/lib/intel64_lin:${LD_LIBRARY_PATH}
export PATH=${DPCPP_ROOT}/bin:$PATH
```
### Using Intel LLVM for DPC++ Preview Vesion

#### install complier

```bash
tar zxvf xxxx.tar.gz
```

- Evironment Variables Setting for DPC++:

```bash
export DPCPP_ROOT=path/to/compiler/latest/linux
export LD_LIBRARY_PATH=${DPCPP_ROOT}/lib:${DPCPP_ROOT}/compiler/lib/intel64_lin:${LD_LIBRARY_PATH}
export PATH=${DPCPP_ROOT}/bin:$PATH
```

### **Validation:**
Finally, compile and execute the following program and check the result. It is optional.

- Source Code:

```c++
// source file: device_enum.cpp
#include <CL/sycl.hpp>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  std::cout
      << "================================================================\n";
  std::cout
      << "           Available DPC++ Platforms / Devices                  \n";
  std::cout
      << "================================================================\n";
  sycl::vector_class<sycl::platform> platforms =
      sycl::platform::get_platforms();
  for (size_t pid = 0; pid < platforms.size(); pid++) {
    sycl::string_class pname =
        platforms[pid].get_info<sycl::info::platform::name>();
    std::cout << "|Platform" << pid << " :\n"
              << "|" << pname << std::endl;
    sycl::vector_class<sycl::device> devices =
        platforms[pid].get_devices(sycl::info::device_type::all);
    for (size_t device_id = 0; device_id < devices.size(); device_id++) {
      sycl::string_class dname =
          devices[device_id].get_info<sycl::info::device::name>();
      sycl::string_class dtype;
      if (devices[device_id].is_gpu()) {
        dtype = "GPU";
      } else {
        dtype = "NonGPU";
      }
      std::cout << "|\t|__|Device" << device_id << " :\n"
                << "|\t|  |" << dname << " (" << dtype << ")" << std::endl;
    }
    std::cout
        << "----------------------------------------------------------------\n";
  }
}

```

- Compile Command:

```bash
$ clang++ device_enum.cpp -fsycl -o device_enum
```

- Expected result like:

```bash
./device_enum

==============================================================
                    All Available Backend                     
==============================================================
|Platform:
|Intel(R) CPU Runtime for OpenCL(TM) Applications
|       |__|Devices:
|          |Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz(NonGPU)
--------------------------------------------------------------
|Platform:
|Intel(R) Level-Zero
|       |__|Devices:
|          |Intel(R) Gen9(GPU)
--------------------------------------------------------------
|Platform:
|Intel(R) OpenCL HD Graphics
|       |__|Devices:
|          |Intel(R) Gen9 HD Graphics NEO(GPU)
--------------------------------------------------------------
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
*  New devcie "xpu" is added into PyTorch proper. Must convert Tensors/Operators/Models onto DPCPP device before running with this Extension.

## Supported Models:

Please download pre-optimized models for this Extension through below command:

```bash
git clone ssh://git@gitlab.devtools.intel.com:29418/intel-pytorch-extension/gpu-optimized-models.git
```

## Known issues:

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

