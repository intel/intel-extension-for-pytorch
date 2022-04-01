# Intel® Extension for PyTorch* 

*  Intel® Extension for PyTorch* is a directed optimized solution for PyTorch end-users to run PyTorch workloads on Intel Graphics cards. The latest release version is 0.3.0gpu.

## Pre-requirements

| **GPU HW** | **OS** | **GPU User Mode Driver** | Python |
| ------ | ------ | ------ | ------ |
| **ATS-P** | Ubuntu-20.04.3 |  agama-ci-prerelease-335 | 3.x |
| **ATS-P** | OpenSUSE Leap 15sp3| agama-ci-prerelease-335 | 3.x |
| **PVC** | Ubuntu-20.04.3 |  agama-ci-prerelease-335 | 3.x |
| **PVC** | OpenSUSE Leap 15sp3| agama-ci-prerelease-335 | 3.x |

### **Dependencies**

```bash
# Install python dependencies
python3 -m pip install -r requirements.txt
```

## Code organization

```
Code organization
    ├── cmake                 // cmake files for build process and dependencies
    ├── csrc                  // IPEX native source code
    │   ├── aten              // XPU aten implementations
    │   │   ├── core          // [Export] aten integration layer
    │   │   │   └── detail    // Mutable implementations
    │   │   ├── operators     // aten operator implementations
    │   │   │   └── comm      // [Header only] Common code for operators
    │   │   └── quantized     // Quantization utilities
    |   ├── autograd          // IPEX autograd support
    │   ├── intrinsic         // IPEX intrinsic
    │   ├── itt               // ITT support
    │   ├── jit               // JIT passes and patterns
    │   ├── oneDNN            // [Header only] oneDNN integration layer
    │   ├── runtime           // DPCPP runtime intergation & utilities
    │   ├── tensor            // IPEX tensor details
    │   └── utils             // [Export] IPEX utilities
    ├── scripts               // Build scripts
    ├── tests                 // IPEX test suites
    │   └── gpu               // IPEX gpu test suites
    │       ├── examples      // IPEX gpu examples and unit tests
    │       └── pytorch       // Test suites ported from PyTorch proper
    ├── third_party           // third party modules
    │   ├── ittapi            // Intel® Instrumentation and Tracing Technology (ITT) API
    │   ├── oneDNN            // oneAPI Deep Neural Network Library
    │   └── oneDPL            // oneAPI DPC++ Library
    ├── ipex                  // IPEX Python layer
    │   ├── autograd          // IPEX autograd implementation for Python
    │   ├── csrc              // IPEX native implementation for Python
    │   │   ├── gpu           // IPEX gpu Python API implementation
    │   │   └── itt           // ITT support
    │   ├── optim             // Customized optimizer implementation for Python
    |   └── xpu               // XPU Python API implementation
    └── torch_patches         // Remaining patches for PyTorch proper
```

## **Compiler Version and Setting**


- Intel DPC++ Compiler Version: **Intel(R) oneAPI Base Toolkit 2022.1 RC3 NDA** 

- Environment Variables Setting for DPC++:

```bash
source ${PATH_To_Your_Compiler}/env/vars.sh
```
**Note:**
please update ${PATH_To_Your_Compiler} to where you install DPC++ compiler with absolute path.

## **oneMKL Version and Setting**

- oneMKL Version: **Intel(R) oneAPI Base Toolkit 2022.1 RC3 NDA**
 
- Environment Variables Setting for oneMKL:

```bash
export MKL_DPCPP_ROOT=${PATH_To_Your_oneAPI_basekit}/intel/oneapi/mkl/latest
export LD_LIBRARY_PATH=${MKL_DPCPP_ROOT}/lib:${MKL_DPCPP_ROOT}/lib64:${MKL_DPCPP_ROOT}/lib/intel64:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${MKL_DPCPP_ROOT}/lib:${MKL_DPCPP_ROOT}/lib64:${MKL_DPCPP_ROOT}/lib/intel64:${LIBRARY_PATH}
```
**Note:**
please update ${PATH_To_Your_oneAPI_basekit} to where you install oneAPI basekit with absolute path.
If you are using different version of oneMKL, the MKL path might be different.

## Repo preparation

1.  Download source code of corresponding PyTorch

```bash
git clone https://github.com/pytorch/pytorch.git -b v1.7.1
cd pytorch
git submodule update --init --recursive
```

2.  Download source code of Intel® Extension for PyTorch* GPU

```bash
git clone https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-gpu/
cd frameworks.ai.pytorch.ipex-gpu
git submodule update --init --recursive
```

### **Validation of Compiler Installation**
Follow instrcutions in <PATH_To_frameworks.ai.pytorch.ipex-gpu>/tests/gpu/device_enum.cpp to check the compiler and device. It is optional.

## Build and Install PyTorch

```bash
cd pytorch
git am <PATH_To_frameworks.ai.pytorch.ipex-gpu>/torch_patches/*
python3 setup.py install --user
```
**Note:** We recommend using **GCC** compiler for building PyTorch.

## Build and Install Intel® Extension for PyTorch* GPU

```bash
cd frameworks.ai.pytorch.ipex-gpu
python3 setup.py install --user
```

## Programming Model

*  ```import intel_extension_for_pytorch``` is a MUST before running any cases with Intel® Extension for PyTorch* GPU.
*  Must convert tensors and models to xpu device before running. Example:

```bash
import intel_extension_for_pytorch

input = input.to("xpu")
model = model.to("xpu")
```

## Verified Models

Please download pre-optimized models for Intel® Extension for PyTorch* GPU through below command:

```bash
git clone https://github.com/intel-innersource/frameworks.ai.pytorch.gpu-models/
```
| **Model** | **Inference** | **Training** |
| ------ | ------ | ------ |
| **ResNet50** | FP32/FP16/BF16/INT8 | FP32/BF16 |
| **DLRM** | FP32/FP16 | FP32/BF16 |
| **BERT** | FP32/FP16 | FP32/BF16 |
| **3D-Unet** | FP32/INT8 | N/A |
| **SSD-ResNet34** | INT8 | BF16 |
| **Transformer** | FP32 | FP32 |
| **SSD-ResNet50** | N/A | FP32/BF16 |
| **SE-ResNeXt50-32x4d** | FP32 | N/A |


## Build Option List
The following build options are supported in Intel® Extension for PyTorch* GPU.

| **Build Option** | **Default<br> Value** | **Description** |
| ------ | ------ | ------ |
| USE_ONEMKL | ON | Use oneMKL BLAS library if set to ON. |
| USE_CHANNELS_LAST_1D | ON | Support channels last 1D memory format if set to ON. |
| USE_PERSIST_STREAM | ON | Use persistent oneDNN stream if set to ON.|
| USE_PRIMITIVE_CACHE | OFF | Use Intel® Extension for PyTorch* GPU solution to cache oneDNN primtives if set to ON. <br> Otherwise use oneDNN cache solution.|
| USE_QUEUE_BARRIER | ON | Default is ON. Use queue submit barrier if set to ON. Otherwise use dummy kernel. |
| USE_SCRATCHPAD_MODE | ON | Default is ON. Use oneDNN scratchpad user mode.|
| USE_MULTI_CONTEXT | ON | Create DPC++ runtime context per device. |
| USE_ITT | ON | Use Intel(R) VTune Profiler ITT functionality if set to ON. |
| USE_AOT_DEVLIST | "" | device list for AOT compilation. Now only ATS-P and PVC are supported. |
| BUILD_STATS | OFF | Count statistics for each component during build process if set to ON. |
| BUILD_BY_PER_KERNEL | OFF | Build by DPC++ per_kernel option if set to ON. |
| BUILD_STRIPPED_BIN | OFF | Strip all symbols when building Intel® Extension for PyTorch* GPU libraries. |
| BUILD_SEPARATE_OPS | OFF | Build each operator in separate library if set to ON. |
| BUILD_SIMPLE_TRACE | OFF | Build simple trace for each registered operators
| BUILD_OPT_LEVEL | OFF | Add build option -Ox, accept values: 0/1
| BUILD_NO_CLANGFORMAT | OFF | Build without force clang-format if set to ON. |
| BUILD_INTERNAL_DEBUG | OFF | Use internal debug code path if set to ON. |
| BUILD_DOUBLE_KERNEL | OFF | Build double data type kernels. This option is set to ON only if <br> BUILD_INTERNAL_DEBUG is set to ON. |

## Launch Option List
The following lauch options are supported in Intel® Extension for PyTorch* GPU.

| **Launch Option** | **Description** |
| ------ | ------ |
| IPEX_SHOW_OPTION | Show all available launch option values. |
| IPEX_VERBOSE | Verbose level in integer. Provide verbose output for Intel® Extension for PyTorch* GPU customized kernel. |
| IPEX_XPU_SYNC_MODE | Enable synchronized execution mode. This mode will perform blocking <br> wait for the completion of submitted kernel. |
| IPEX_TILE_AS_DEVICE | Device partition. If set to 0, tile partition will be disabled and map device to physical device. |
| IPEX_LAYOUT_OPT | Enable oneDNN specific layouts. If set to 1, Intel® Extension for PyTorch* GPU tries to use blocked layouts querying from oneDNN.  |

All these options are set to zero by default. User may enable one or more options like below examples.</br>

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
IPEX_VERBOSE=1 IPEX_ONEDNN_LAYOUT=1 python ResNet50.py
```

## Feature Introduction

### AOT compilation:
AOT compilation is supported on ATS-P or PVC separately with below config:
| Supported HW | Setting |
| ------ | ------ |
| ATS-P B0 |  USE_AOT_DEVLIST='xe_hp_sdv'  |
| PVC XT A0 | USE_AOT_DEVLIST='12.4.0'  |
| PVC XT B3 | USE_AOT_DEVLIST='12.4.1' |
<br>
Multi-target AOT compilation to support both ATS-P and PVC is not allowed currently.

### oneDNN specific layouts:
All models listed in above "Verified Models" can run with IPEX_ONEDNN_LAYOUT=1 on ATS-P L0 backend. However, not all supported models can gain performance improvement through this feature.

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

### Multi-tile training with Horovod:
Multi-tile ResNet50 training is verified with Horovod on 2-tile ATS-P. Following launch options need be set during the validation: <br>

```bash
source $ONEAPI_INSTALL_DIR/mpi/latest/env/vars.sh -i_mpi_library_kind=release_mt
source $ONEAPI_INSTALL_DIR/ccl/latest/env/vars.sh --ccl-configuration=cpu_gpu_dpcpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
export CCL_STAGING_BUFFER=regular
```

### Multi-tile training with DistributedDataParallel:
Multi-tile ResNet50 training is verified with DistributedDataParallel (DDP) on 2-tile ATS-P. For supporting this scenario, oneCCL Bindings for Pytorch* based on oneCCL 2021.5 version shall be built and used. 

```bash
git clone -b chengjun/ccl_torch1.7_gpu https://github.com/intel-innersource/frameworks.ai.pytorch.torch-ccl.git
git submodule update --init --recursive
COMPUTE_BACKEND=dpcpp_level_zero python setup.py install
```
Example of running multi-tile ResNet50 training with DDP:
```bash
cd frameworks.ai.pytorch.gpu-models/ResNet50
source `python -c "import torch_ccl;print(torch_ccl.cwd)"`/env/setvars.sh
mpiexec -n 2 python main.py -a resnet50 -e -b 1024 --pretrained --jit --xpu 0 $dataset
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

### ITT support:
ITT is Intel® VTune™ Profiler's Instrumentation and Tracing Technology. To enable this feature, <br>
build Intel® Extension for PyTorch* GPU with USE_ITT=ON and update model as below:

```bash
with torch.xpu.emit_itt():
    torch.xpu.itt.mark('single shot marker')
    torch.xpu.itt.range_push('custom range')
    output = YourModel(input)
    torch.xpu.itt.range_pop()
```
Then start VTune for profiling kernels. Make sure ```INTELONEAPIROOT``` is set for VTune.

### User mode scratchpad:
oneDNN defines two scratchpad modes: library and user. User mode means framework will manage the scratchpad allocation by querying and providing the scratchpad memory to oneDNN primitives. User mode scratchpad is default enabled for all oneDNN primitives in this release. 

To switch to library mode scratchpad, please set USE_SCRATCHPAD_MODE=OFF and rebuild. Please expect negative impact to performance when changing the mode.

### Master weights support:
Master weights is partially supported in this release, for assuring accuracy by using FP32 weights in BF16 training gradient update. To use this feature, model shall be updated as following: 
```bash
...
# optimizer = torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=1e-4)
optimizer = torch.xpu.optim.SGDMasterWeight(model.parameters(),lr=0.1,momentum=0.9,weight_decay=1e-4)
model.bfloat16()

output = model(input)
loss = criterion(output)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```
1) optimizer shall be updated to use a new API such as ```SGDMasterWeight``` which contains master weights support
2) model shall be converted to bfloat16 after optimizer is initiated. Otherwise, optimizer can't get FP32 weights since it is already converted to BF16.


### Sparse tensor support:
This release supports sparse backend and several sparse related operators, including dense_dim, sparse_dim, to_sparse, sparse_mask, values, indices, coalesce and embeddingbag. It adds additional support for performing embeddingbag on sparse tensor in DLRM training.

### Coding Style Alignment:
This release uses clang-format and flake8 to enhance the code in Intel® Extension for PyTorch* GPU and make sure the coding style align with PyTorch proper.

### Operator Coverage:
| **Operator Type** | **Implemented**| **Completion ratio** |
| ------ | ------ | ------ |
| PyTorch NN functions | 115 | 100.00%　|
| PyTorch Tensor functions | 340 | 100.00%　|
| PyTorch Methods | 231 | 100.00%　|
| Total | 686 | 100.00% |

## Caveat
### 1. Build order of PyTorch and extension:
Please build Intel® Extension for PyTorch* GPU after pytorch is built and installed, otherwise you will get an error “ModuleNotFoundError: No module named 'torch'”.

### 2. MKL related issues:
1) undefined symbol: mkl_lapack_dspevd. Intel MKL FATAL ERROR: cannot load libmkl_vml_avx512.so.2 or libmkl_vml_def.so.2 <br>
This issue may raise when Intel® Extension for PyTorch* is built with oneMKL library while PyTorch is not build with any MKL library. oneMKL kernel may run into CPU occasionally which causes this issue. Please install MKL library from conda as following:
```bash
conda install mkl
conda install mkl-include
```
then clean build PyTorch to solve this issue.

2) OSError: libmkl_intel_lp64.so.1: cannot open shared object file: No such file or directory <br>
Wrong MKL library is used when multiple MKL libraries exist in system. To solve this issue, preload oneMKL by:

```bash
export LD_PRELOAD=${MKL_DPCPP_ROOT}/lib/intel64/libmkl_intel_lp64.so.1:${MKL_DPCPP_ROOT}/lib/intel64/libmkl_intel_ilp64.so.1:${MKL_DPCPP_ROOT}/lib/intel64/libmkl_sequential.so.1:${MKL_DPCPP_ROOT}/lib/intel64/libmkl_core.so.1:${MKL_DPCPP_ROOT}/lib/intel64/libmkl_sycl.so.1
```

If you still meet the similar issue which cannot open shared object file not listed above, please add corresponding files under ${MKL_DPCPP_ROOT}/lib/intel64/ to LD_PRELOAD.

3) Can't find oneMKL library when build Intel® Extension for PyTorch* without oneMKL <br>
Error info: <br>
/usr/bin/ld: cannot find -lmkl_sycl <br>
/usr/bin/ld: cannot find -lmkl_intel_ilp64 <br>
/usr/bin/ld: cannot find -lmkl_core <br>
/usr/bin/ld: cannot find -lmkl_tbb_thread <br>
dpcpp: error: linker command failed with exit code 1 (use -v to see invocation) <br>

When PyTorch is built with oneMKL library while Intel® Extension for PyTorch* is built without oneMKL library, we need to set following configuration to solve the link issue:
```bash
export USE_ONEMKL=OFF
export MKL_DPCPP_ROOT=${PATH_To_Your_oneAPI_basekit}/intel/oneapi/mkl/latest
export LD_LIBRARY_PATH=${MKL_DPCPP_ROOT}/lib:${MKL_DPCPP_ROOT}/lib64:${MKL_DPCPP_ROOT}/lib/intel64:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${MKL_DPCPP_ROOT}/lib:${MKL_DPCPP_ROOT}/lib64:${MKL_DPCPP_ROOT}/lib/intel64:${LIBRARY_PATH}
```

Please note that the solutions to above issues are workaround. When MKLD-11291 is solved, mkl library for CPU and oneMKL library for GPU can work together in same system, we will no longer meet such issues. For now, we recommend to build PyTorch with mkl from conda channel and build Intel® Extension for PyTorch* with oneMKL library from oneAPI Base toolkit, which is the typical usage scenario validated regularly.

### 3. symbol undefined caused by _GLIBCXX_USE_CXX11_ABI:
#### Error info: <br>
```bash
File "/root/.local/lib/python3.9/site-packages/ipex/__init__.py", line 4, in <module>
    from . import _C
ImportError: /root/.local/lib/python3.9/site-packages/ipex/lib/libipex_gpu_core.so: undefined symbol: _ZNK5torch8autograd4Node4nameB5cxx11Ev
```

This issue appears when Intel® Extension for PyTorch* is compiled with _GLIBCXX_USE_CXX11_ABI=1 and PyTorch is compiled with _GLIBCXX_USE_CXX11_ABI=0, which causes inconsistent.
<BR>

#### Background：
1. DPC++ has no plan to support _GLIBCXX_USE_CXX11_ABI=0 (CMPLRLLVM-34202), Intel® Extension for PyTorch* is always compiled with _GLIBCXX_USE_CXX11_ABI=1. <br>
2. PyTorch detects the setting of _GLIBCXX_USE_CXX11_ABI by checking user config and compiler capability. If compiler in use does not support _GLIBCXX_USE_CXX11_ABI=1, PyTorch is compiled with _GLIBCXX_USE_CXX11_ABI=0. PyTorch publishes official binary package with _GLIBCXX_USE_CXX11_ABI=0. <br>

#### Solution：
User shall update PyTorch CMAKE file to set _GLIBCXX_USE_CXX11_ABI=1 and compile PyTorch with particular compiler which supports _GLIBCXX_USE_CXX11_ABI=1. We recommend to use gcc version 9.3.0 (Ubuntu 9.3.0-17ubuntu1~20.04) on ubuntu 20.04 which is validated by us. <br>

This issue won't exist with future version of PyTorch as community agrees to provide _GLIBCXX_USE_CXX11_ABI=1 binary for versions after PyTorch 1.10. 
