# Intel GPU Extension for PyTorch

*  The Intel GPU Extension for PyTorch is a directed optimized solution for PyTorch end-users to run PyTorch workloads on Intel Graphics cards.

## Pre-requirements

| **GPU HW** | **OS** | **GPU User Mode Driver** | Python |
| ------ | ------ | ------ | ------ |
| **ATS** | Ubuntu-20.04 |  [agama-ci-embargo-ats-353](https://ubit-gfx.intel.com/build/10100871) | 3.7 |
| **ATS** | OpenSUSE Leap 15sp2| [agama-ci-embargo-ats-353](https://ubit-gfx.intel.com/build/10100871) | 3.7 |
| **Gen9** | Ubuntu-20.04 | [compute runtime 21.11.19310](https://github.com/intel/compute-runtime/releases/tag/21.11.19310) | 3.7 |


### **Dependencies**

```bash
# Install python dependencies
python3 -m pip install -r requirements.txt
# Add ubuntu user to video and render group
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER
## logout and relogin
logout
```

## **Compiler Version and Setting**


- Intel DPC++ Compiler: **oneAPI 2021.2 RC version**
  
   Obtain from: \\\nncv03a-cifs.inn.intel.com\icl_xarch\archive\deploy_oneapi\linux\20210401\build

- Environment Variables Setting for DPC++:

```bash
<PATH_To_Your_Compiler>
      +-- env
      +-- linux
          +-- bin
          +-- compiler
          ¦   +-- include
          ¦   +-- lib
          +-- include
          ¦   +-- sycl
          +-- lib

export DPCPP_ROOT=${PATH_To_Your_Compiler}/linux
source ${PATH_To_Your_Compiler}/env/vars.sh
```
**Note:**
please update ${PATH_To_Your_Compiler} to where you install DPC++ compiler with absolute path.
<br>Example: /20210401/build/linux_prod/compiler

### **Validation of Compiler Installation**
Compile and execute the following program and check the result. It is optional.

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

- Expected Result:

```bash
## dual dg1 card host on a Xeon hw env with Level-Zero
./device_enum
================================================================
           Available DPC++ Platforms / Devices
================================================================
|Platform0 :
|Intel(R) OpenCL HD Graphics
|       |__|Device0 :
|       |  |Intel(R) Graphics [0x4905] (GPU)
|       |__|Device1 :
|       |  |Intel(R) Graphics [0x4905] (GPU)
----------------------------------------------------------------
|Platform1 :
|Intel(R) Level-Zero
|       |__|Device0 :
|       |  |Intel(R) Graphics [0x4905] (GPU)
|       |__|Device1 :
|       |  |Intel(R) Graphics [0x4905] (GPU)
----------------------------------------------------------------
|Platform2 :
|SYCL host platform
|       |__|Device0 :
|       |  |SYCL host device (NonGPU)
----------------------------------------------------------------

```

## **oneMKL Version and Setting**

- oneMKL Version: [**oneMKL 2021.2 NDA RC2**](https://phoenix.intel.com/next/#/packages/1100:142154/overview)
 
- Environment Variables Setting for oneMKL:

```bash
export MKL_DPCPP_ROOT=${PATH_To_Your_oneMKL}/2021.2.0-prerelease
export LD_LIBRARY_PATH=${MKL_DPCPP_ROOT}/lib:${MKL_DPCPP_ROOT}/lib64:${MKL_DPCPP_ROOT}/lib/intel64:${LD_LIBRARY_PATH}

```
**Note:**
please update ${PATH_To_Your_oneMKL} to where you install oneMKL library with absolute path.


## Repo preparation

1.  Download source code of corresponding PyTorch

```bash
git clone https://github.com/pytorch/pytorch.git -b v1.7.1
cd pytorch
git submodule update --init --recursive
```

2.  Download source code of Intel GPU Extension for PyTorch

```bash
git clone https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-gpu.git
cd frameworks.ai.pytorch.ipex-gpu
git submodule update --init --recursive
```


## Build and Install PyTorch

```bash
cd pytorch
git am <PATH_To_frameworks.ai.pytorch.ipex-gpu>/torch_patches/*
python3 setup.py install --user
```
**Note:** We recommend using **GCC** compiler for building PyTorch. 

## Build and Install Intel GPU Extension for PyTorch

```bash
cd frameworks.ai.pytorch.ipex-gpu
python3 setup.py install --user
```

## Programming Model

*  ```import torch_ipex``` is a MUST before running any cases with Intel GPU Extension for PyTorch.
*  Must convert tensors and models to xpu device before running with this Extension. Example:

```bash
import torch_ipex

input = input.to("xpu")
model = model.to("xpu")
```

## Verified Models

Please download pre-optimized models for this Extension through below command:

```bash
git clone ssh://git@gitlab.devtools.intel.com:29418/intel-pytorch-extension/gpu-optimized-models.git
```
| **Model** | **ATS Level Zero** | **Gen9 Level Zero** | 
| ------ | ------ | ------ |
| **ResNet50** | **Inference**:INT8/FP32/FP16/BF16 <br> **Training**: FP32/BF16 | **Inference**:INT8/FP32/FP16/BF16 <br> **Training**: FP32/BF16 |
| **DLRM** | **Inference**:FP32/FP16/BF16 <br> **Training**: FP32/BF16 |  **Inference**:FP32/FP16/BF16 <br> **Training**: FP32/BF16 | 
| **BERT** | **Inference**:FP32/FP16 <br> **Training**: FP32/BF16 | Note* | 
| **Transformer-LT** | **Inference**:FP32/BF16 | **Inference**:FP32/FP16 <br> **Training**: FP32 |
| **SSD-MobileNetV1** | **Inference**:FP32/FP16/BF16 | **Inference**:FP32/FP16/BF16 | 
| **SSD-ResNet50** | **Inference**:FP32/FP16 <br> **Training**:FP32/BF16 | **Inference**:FP32/FP16 <br> **Training**:FP32  | 
| **RCAN** | **Inference**:FP32/FP16/BF16 | **Inference**:FP32/FP16/BF16 |
| **ResNext101** | **Inference**:FP32/FP16 | **Inference**:FP32/FP16 |
| **3D-Unet** | **Inference**:FP32 | **Inference**:FP32/FP16 |

**Note:** BERT is not validated on Gen9 due to bad hardware performance in this release.


## Build Option List
The following build options are supported in Intel GPU Extension for PyTorch.

| **Build Option** | **Default<br> Value** | **Description** |
| ------ | ------ | ------ |
| USE_AOT_DEVLIST | "" | device list for AOT compilation. Now only ats is supported. | 
| USE_ONEDPL | ON | Use oneDPL library under <PATH_To_frameworks.ai.pytorch.ipex-gpu>/<br>third_party/ if set to ON. |
| USE_ONEMKL | ON | Use oneMKL BLAS library if set to ON. |
| USE_PERSIST_STREAM | ON | Use persistent oneDNN stream if set to ON.|
| USE_PRIMITIVE_CACHE | OFF | Use IPEX solution to cache oneDNN primtives if set to ON. <br> Otherwise use oneDNN cache solution.|
| USE_MULTI_CONTEXT | ON | Create DPC++ runtime context per device.
| USE_ITT | OFF | (Experimental) Use Intel(R) VTune Profiler ITT functionality if set to ON. |
| BUILD_BY_PER_KERNEL | OFF | Build by DPC++ per_kernel option if set to ON. |
| BUILD_NO_L0_ONEDNN | OFF | Build oneDNN without LevelZero support if set to ON. |
| BUILD_INTERNAL_DEBUG | OFF | Use internal debug code path if set to ON. |
| BUILD_DOUBLE_KERNEL | OFF | Build double data type kernels. This option is set to ON only if <br> BUILD_INTERNAL_DEBUG is set to ON. |

## Launch Option List
The following lauch options are supported in Intel GPU Extension for PyTorch.

| **Launch Option** | **Description** |
| ------ | ------ |
| IPEX_VERBOSE | Provide verbose output for IPEX customized kernel. |
| IPEX_FORCE_SYNC | Enable synchronized execution mode. This mode will perform blocking <br> wait for the completion of submitted kernel. |
| IPEX_DISABLE_PROFILING | Disable IPEX profiling solution. If set to 1, the queue profiling flag will be unset and kernel profiling information will be reset. |
| IPEX_LAZY_REORDER | Enable lazy reorder. When enabled, reorders on activation tensors are <br> eliminated when oneDNN block format can be propagated for more than one operator.  |
| IPEX_WEIGHT_CACHE | Cache packed weight. When enabled, the packed weight is cached <br> in original weight tensor. This option can be enabled only when <br> IPEX_LAZY_REORDER is set to 1. |
| IPEX_DISABLE_TILE_PARTITION | Device partition. When enabled, tile partition will be disabled and map frameworkd device to physical device. |


All these options are set to zero by default. User may configure one or more options like below examples.</br>

1. Set single option before running model
```bash
export IPEX_VERBOSE=1
python ResNet50.py
```
2. Set single option when running model
```bash
IPEX_VERBOSE=1 python ResNet50.py
```

3. Set multiple options when running model
```bash
IPEX_VERBOSE=1 IPEX_LAZY_REORDER=1 IPEX_WEIGHT_CACHE=1 python ResNet50.py
```

## Feature Introduction

### AOT compilation:
* AOT compilation is supported on ATS and enabled by default. For other GPUs, user need configure USE_AOT=OFF to disable AOT compilation.
* The AOT binary compiled by ATS can be used in other GPUs.

### Models with lazy reorder support:
The following models can run with IPEX_LAZY_REORDER=1 on ATS L0 backend.

| **Model** | **Inference** | **Training** |
| ------ | ------ | ------ |
| **ResNet50** | INT8/FP32/FP16/BF16 | FP32/BF16 |
| **DLRM** | FP32/FP16/BF16 | FP32/BF16 |
| **BERT** | FP32/FP16 |  FP32/BF16 |
| **Transformer-LT** | FP32/BF16 |  - |
| **SSD-MobileNetV1** | FP32/FP16/BF16 | - |
| **SSD-ResNet50** | FP32/FP16 | FP32/BF16 |
| **RCAN** | FP32/FP16/BF16 | - |
| **ResNext101** | FP32/FP16 | - |
| **3D-Unet** | FP32 | - |


### Fusion pattern support:
All fusions patterns are only available in PyTorch JIT mode.

| Supported Fusion | Supported Precision |
| ------ | ------ |
| Conv2D + ReLU |  FP32/FP16/BF16/INT8  |
| Conv2D + Sum | FP32/FP16/BF16/INT8 | 
| Conv2D + Sum + ReLU | FP32/FP16/BF16/INT8 |
| Conv3D + ReLU | FP32/FP16/BF16 | 
| Conv3D + Sum | FP32/FP16/BF16 |
| Conv3D + Sum + ReLU | FP32/FP16/BF16 | 
| Linear + ReLU | FP32/FP16/BF16  |
| Linear + Sigmoid | FP32/FP16/BF16 | 
| Linear + Div(scalar) | FP32/FP16/BF16 |
| Mul + Add | FP32/FP16/BF16 | 
| Add + ReLU | INT8 |

### Multi-tile training:
Multi-tile ResNet50 training is verified with Horovod on 2-tile ATS. For supporting this scenario, oneCCL provides corresponding patch to Horovod so that a private version of Horovod need be used. Besides that, following launch options need be set: <br>

```bash
source $ONEAPI_INSTALL_DIR/mpi/latest/env/vars.sh -i_mpi_library_kind=release_mt
source $ONEAPI_INSTALL_DIR/ccl/latest/env/vars.sh --ccl-configuration=cpu_gpu_dpcpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
export CCL_STAGING_BUFFER=regular
```

### Profile tool:
autograd.profiler supports profiling kernel time spent on "xpu" device. Update model as below:

```bash
with torch.autograd.profiler.profile(enabled=True, use_xpu=True) as prof:
    # put what you want to profile here. Such as:
    # output = YourModel(input)
print(prof.table())
```

Flag ```enabled``` and ```use_xpu``` should be set to True to enable this feature.

### ITT support (Experimental):
ITT is Intel® VTune™ Profiler's Instrumentation and Tracing Technology. To enable this feature, <br>
build Intel GPU Extension for PyTorch with USE_ITT=ON and update model as below:

```bash
with torch_ipex.profiler.emit_itt():
    output = YourModel(input)
```
Then start VTune for profiling kernels. Make sure ```INTELONEAPIROOT``` is set for VTune.

### Operator Coverage:
| **Operator Type** | **Implemented**| **Completion ratio** |
| ------ | ------ | ------ |
| PyTorch NN functions | 115 | 100.00%　|
| PyTorch Tensor functions | 322 | 94.15%　|
| PyTorch Methods | 192 | 80.00%　|
| Total | 629 | 90.24% | 

## Caveat
### 1. Build order of PyTorch and extension:
Please build IPEX after pytorch is built and installed, otherwise you will get an error “ModuleNotFoundError: No module named 'torch'”.
