# Intel® Extension for PyTorch\* GPU

Intel® Extension for PyTorch\* extends [PyTorch\*](https://github.com/pytorch/pytorch) with optimizations for extra performance boost on Intel hardware. It is a heterogeneous, high performance deep learning implementation for Intel® XPU (GPU, CPU, etc.) devices. This repo introduces optimized GPU solution for PyTorch end-users to get up-to-date features and optimizations on Intel Graphics cards. Eventually, it will be merged with CPU solution and released in the same public repo: https://github.com/intel/intel-extension-for-pytorch/.

Intel® Extension for PyTorch\* is loaded as a Python module for Python programs or linked as a C++ library for C++ programs. Users can enable it dynamically in script by importing intel_extension_for_pytorch. It covers optimizations for both imperative mode and graph mode. Optimized operators and kernels are registered to XPU backend through PyTorch dispatching mechanism. These operators and kernels are accelerated from native vectorization feature and matrix calculation feature of Intel hardware. In graph mode, further operator fusions are supported to reduce operator/kernel invocation overheads, and thus increase performance.

The latest release version for GPU solution of Intel® Extension for PyTorch\* is 1.10.100+gpu.

# Table of Contents
- [Intel® Extension for PyTorch* GPU](#intel-extension-for-pytorch-gpu)
- [Table of Contents](#table-of-contents)
- [Pre-requirements](#pre-requirements)
  - [Dependencies](#dependencies)
- [Code organization](#code-organization)
- [Compiler Version and Setting](#compiler-version-and-setting)
- [oneMKL Version and Setting](#onemkl-version-and-setting)
  - [MKL related issues](#mkl-related-issues)
- [Repo preparation](#repo-preparation)
  - [Validation of Compiler Installation](#validation-of-compiler-installation)
- [Build and Install PyTorch](#build-and-install-pytorch)
- [Build and Install Intel® Extension for PyTorch* GPU](#build-and-install-intel-extension-for-pytorch-gpu)
- [Programming Model](#programming-model)
  - [How to get accurate End to End model execution time](#how-to-get-accurate-end-to-end-model-execution-time)
- [Build Option List](#build-option-list)
- [Launch Option List](#launch-option-list)
- [Feature Introduction](#feature-introduction)
  - [AOT compilation](#aot-compilation)
  - [Auto Mixed Precision (AMP)](#auto-mixed-precision-amp)
  - [Coding Style Alignment](#coding-style-alignment)
  - [Distributed Training with DistributedDataParallel (DDP)](#distributed-training-with-distributeddataparallel-ddp)
  - [Distributed Training with Horovod](#distributed-training-with-horovod)
  - [Fusion pattern support](#fusion-pattern-support)
  - [Master weights support](#master-weights-support)
  - [Operator Coverage](#operator-coverage)
  - [Profile tool](#profile-tool)
  - [Sparse tensor support](#sparse-tensor-support)
  - [TF32 math mode](#tf32-math-mode)
  - [torch.inference_mode](#torchinference_mode)
  - [User mode scratchpad](#user-mode-scratchpad)
- [Caveat](#caveat)
  - [1. Build order of PyTorch and extension](#1-build-order-of-pytorch-and-extension)
  - [2. symbol undefined caused by _GLIBCXX_USE_CXX11_ABI](#2-symbol-undefined-caused-by-_glibcxx_use_cxx11_abi)
  - [3. Distributed Training Issues](#3-distributed-training-issues)
  - [4. UT failures](#4-ut-failures)

## Pre-requirements

| **GPU HW** | **OS** | **GPU User Mode Driver** | Python |
| ------ | ------ | ------ | ------ |
| **PVC-XT B4** | Ubuntu-20.04.3 |  agama-ci-prerelease-522 | 3.x |
| **PVC-XT B4** | OpenSUSE Leap 15sp3| agama-ci-prerelease-522 | 3.x |
| **ATS-P B0** | Ubuntu-20.04.3 |  agama-ci-prerelease-522 | 3.x |
| **ATS-P B0** | OpenSUSE Leap 15sp3| agama-ci-prerelease-522 | 3.x |
| **ATS-M M1** | Ubuntu-20.04.3 | agama-ci-devel-419.2 | 3.x |

### **Dependencies**

```bash
# Install python dependencies
python3 -m pip install -r requirements.txt
```

## Code organization

```
Code organization
    ├── cmake                       // cmake files for build process and dependencies
    ├── csrc                        // IPEX native source code
    │   ├── aten                    // XPU aten implementations
    |   |   ├── amp                 // Auto mixed precision implementations 
    │   │   ├── core                // [Export] aten integration layer
    │   │   │   └── detail          // Mutable implementations
    │   │   ├── operators           // aten operator implementations
    │   │   │   └── comm            // [Header only] Common code for operators
    │   │   └── quantized           // Quantization utilities
    │   ├── intrinsic               // IPEX intrinsic
    │   ├── jit                     // JIT passes and patterns
    │   ├── oneDNN                  // [Header only] oneDNN integration layer
    │   ├── runtime                 // DPCPP runtime intergation & utilities
    │   ├── tensor                  // IPEX tensor details
    │   └── utils                   // [Export] IPEX utilities
    ├── intel_extension_for_pytorch // IPEX Python layer
    │   ├── autograd                // IPEX autograd implementation for Python
    │   ├── csrc                    // IPEX native implementation for Python
    │   │   └── gpu                 // IPEX gpu Python API implementation
    │   ├── optim                   // Customized optimizer implementation for Python
    |   └── xpu                     // XPU Python API implementation  
    ├── scripts                     // Build scripts
    ├── tests                       // IPEX test suites
    │   └── gpu                     // IPEX gpu test suites
    │       ├── examples            // IPEX gpu examples and unit tests
    │       ├── experimental        // Test suites ported from PyTorch 1.10   
    │       ├── pytorch             // Test suites ported from PyTorch proper
    │       └── regression          // unit tests for regression issues
    ├── third_party                 // third party modules
    │   └── oneDNN                  // oneAPI Deep Neural Network Library
    └── torch_patches               // Remaining patches for PyTorch proper
```

## **Compiler Version and Setting**

| **GPU HW** | **oneMKL Version** |
| ------ | ------ |
| **PVC-XT B4** | 20220630_ms284 |
| **ATS-P B0** | 20220630_ms284 |
| **ATS-M M1** | XMAIN-REL_0730 |

- Environment Variables Setting for DPC++:

```bash
source ${PATH_To_Your_Compiler}/env/vars.sh
```

**Note:**
please update ${PATH_To_Your_Compiler} to where you install DPC++ compiler with absolute path.

## **oneMKL Version and Setting**

| **GPU HW** | **oneMKL Version** |
| ------ | ------ |
| **PVC-XT B4** | 20220708_ms284 |
| **ATS-P B0** | 20220708_ms284 |
| **ATS-M M1** | 2022u2_20220804 |
 
- Environment Variables Setting for oneMKL:

```bash
export MKL_DPCPP_ROOT=${PATH_To_Your_oneMKL}/__release_lnx/mkl
```

**Note:**
please update ${PATH_To_Your_oneMKL} to where you install oneAPI basekit with absolute path.
If you are using different version of oneMKL, the MKL path might be different.

### MKL related issues:

Story MKLD-13445 is not completed in verified oneMKL versions listed above, which makes different versions of MKL library in one system might conflict with each other. For now, we recommend to build PyTorch with mkl from conda channel and build Intel® Extension for PyTorch* with the verified oneMKL library, which is the typical usage scenario validated regularly. You may meet build error or runtime errors listed below in other usage scenarios:

#### Can't find oneMKL library when build Intel® Extension for PyTorch* without oneMKL <br>

Error info: <br>

```bash
/usr/bin/ld: cannot find -lmkl_sycl <br>
/usr/bin/ld: cannot find -lmkl_intel_ilp64 <br>
/usr/bin/ld: cannot find -lmkl_core <br>
/usr/bin/ld: cannot find -lmkl_tbb_thread <br>
dpcpp: error: linker command failed with exit code 1 (use -v to see invocation) <br>
```

When PyTorch is built with oneMKL library while Intel® Extension for PyTorch* is built without oneMKL library, we need to set following configuration to solve the link issue:

```bash
export USE_ONEMKL=OFF
export MKL_DPCPP_ROOT=${PATH_To_Your_oneMKL}/__release_lnx/mkl
```

#### undefined symbol: mkl_lapack_dspevd. Intel MKL FATAL ERROR: cannot load libmkl_vml_avx512.so.2 or libmkl_vml_def.so.2 <br>

This issue may raise when Intel® Extension for PyTorch* is built with oneMKL library while PyTorch is not build with any MKL library. oneMKL kernel may run into CPU occasionally which causes this issue. Please install MKL library from conda as following:

```bash
conda install mkl
conda install mkl-include
```

then clean build PyTorch to solve this issue.

#### OSError: libmkl_intel_lp64.so.1: cannot open shared object file: No such file or directory <br>

Wrong MKL library is used when multiple MKL libraries exist in system. To solve this issue, preload oneMKL by:

```bash
export LD_PRELOAD=${MKL_DPCPP_ROOT}/lib/intel64/libmkl_intel_lp64.so.1:${MKL_DPCPP_ROOT}/lib/intel64/libmkl_intel_ilp64.so.1:${MKL_DPCPP_ROOT}/lib/intel64/libmkl_sequential.so.1:${MKL_DPCPP_ROOT}/lib/intel64/libmkl_core.so.1:${MKL_DPCPP_ROOT}/lib/intel64/libmkl_sycl.so.1
```

If you still meet the similar issue which cannot open shared object file not listed above, please add corresponding files under ${MKL_DPCPP_ROOT}/lib/intel64/ to LD_PRELOAD. Please also note that the suffix of the libraries may change (e.g. from .1 to .2), if more than one oneMKL libraries are installed to the system.

## Repo preparation

1.  Download source code of corresponding PyTorch

```bash
git clone https://github.com/intel-innersource/frameworks.ai.pytorch.private-gpu -b pytorch-1.10
cd frameworks.ai.pytorch.private-gpu
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
cd frameworks.ai.pytorch.private-gpu
python3 setup.py install --user
```

**Note:** We recommend using **GCC** compiler for building PyTorch.

## Build and Install Intel® Extension for PyTorch* GPU

```bash
cd frameworks.ai.pytorch.ipex-gpu
python3 setup.py install --user
```

## Programming Model

*  Must ```import intel_extension_for_pytorch``` before running any cases with Intel® Extension for PyTorch* GPU.
*  Must convert tensors and models to xpu device before running. Example:

```bash
import intel_extension_for_pytorch

input = input.to("xpu")
model = model.to("xpu")
```

*  Should enable inference mode to get better performance for inference model. Example:

```bash
with torch.inference_mode():
  model(input)
```

### How to get accurate End to End model execution time

To get accurate End to End model execution time, users need call torch.xpu.synchronize() in model script right before calculating elapsed time. This API waits for all GPU kernels which are executing on device being completed, so that calculting the elasped time after the call can cover both CPU and GPU execution time.

#### Training Model Example

```bash
  # compute gradient and do SGD step
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

  # sync for time measurement
  torch.xpu.synchronize()

  # measure elapsed time
  end_Batch = time.time()
  batch_time.update(time.time() - start_Batch)
  iter_time.append(end_Batch - start_Batch)
```

#### Inference Model Example

```bash
with torch.inference_mode():
  # compute output
  output = model(input)

  # sync for time measurement
  torch.xpu.synchronize()

  # measure elapsed time
  end = time.time()
  batch_time.update(end - start)
  iter_time.append(end - start)
```

### Supported torchvision

Intel® Extension for PyTorch\* GPU supports torchvision 0.8.2 for now. The supported torchvision version will be uplifted to align with PyTorch version we supported eventually, for example, torchvision 0.11 will be supported in future release. Currently, for models rely on torchvision, users may follow below steps to install required torchvision package:

```bash
    Install Private PyTorch
    Install Intel® Extension for PyTorch* GPU
    python3 -m pip install pillow
    python3 -m pip install torchvision==0.8.2 --no-deps
```
Please skip the first two steps if Private Pytorch is already installed (follow steps in [Build and Install PyTorch](#build-and-install-pytorch) or through binary) and Intel® Extension for PyTorch\* GPU is already installed (follow steps in [Build and Install Intel® Extension for PyTorch* GPU](#build-and-install-intel-extension-for-pytorch-gpu) or through binary).

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
| USE_AOT_DEVLIST | "" | device list for AOT compilation. |
| BUILD_STATS | OFF | Count statistics for each component during build process if set to ON. |
| BUILD_BY_PER_KERNEL | OFF | Build by DPC++ per_kernel option if set to ON. |
| BUILD_STRIPPED_BIN | OFF | Strip all symbols when building Intel® Extension for PyTorch* GPU libraries. |
| BUILD_SEPARATE_OPS | OFF | Build each operator in separate library if set to ON. |
| BUILD_SIMPLE_TRACE | OFF | Build simple trace for each registered operators
| BUILD_OPT_LEVEL | OFF | Add build option -Ox, accept values: 0/1
| BUILD_NO_CLANGFORMAT | OFF | Build without force clang-format if set to ON. |
| BUILD_INTERNAL_DEBUG | OFF | Use internal debug code path if set to ON. |

## Launch Option List

The following lauch options are supported in Intel® Extension for PyTorch* GPU.

| **Launch Option** | **Default<br> Value** | **Description** |
| ------ | ------ | ------ |
| IPEX_VERBOSE | 0 | Verbose level in integer. Provide verbose output for Intel® Extension for PyTorch* GPU customized kernel. |
| IPEX_FP32_MATH_MODE | FP32 | FP32 math mode. Set to TF32 for using TF32 math mode,  BF32 for using BF32 math mode.|
| IPEX_TILE_AS_DEVICE | 1 | Device partition. If set to 0, tile partition will be disabled and map device to physical device. |
| IPEX_XPU_SYNC_MODE | 0 | Kernel Execution mode. If set to 1, use synchronized execution mode and perform blocking wait for the completion of submitted kernel. |

Examples to config the launch options:</br>

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
IPEX_VERBOSE=1 IPEX_XPU_SYNC_MODE=1 python ResNet50.py
```

## Feature Introduction

### AOT compilation:

AOT compilation is supported on ATS-P, ATS-M or PVC with below config using agama-ci-prerelease-522:
| Supported HW | Setting |
| ------ | ------ |
| ATS-P B0 |  USE_AOT_DEVLIST='xehp-sdv'  |
| ATS-M M1 |  USE_AOT_DEVLIST='dg2-g10-c0'
| PVC XT B3 | USE_AOT_DEVLIST='pvc-xt-b1' |
| PVC XT B4| USE_AOT_DEVLIST='pvc-xt-c0' |
| ATS-M M1 + PVC XT B4 | USE_AOT_DEVLIST='dg2-g10-c0,pvc-xt-c0' |
| ATS-M M1 + PVC XT B4 + ATS-P B0 | USE_AOT_DEVLIST='dg2-g10-c0,pvc-xt-c0,xehp-sdv' |

Multi-target AOT compilation is supported with application side workaround. We still need product solution of CMPLRLLVM-25864 to support large object file (>2GB).

On agama-ci-devel-419.2, set USE_AOT_DEVLIST='12.55.8' to enable single-target AOT compilation on ATS-M M1. Multi-target AOT is not verified on that driver.

### Auto Mixed Precision (AMP):

This release supports the fundamental functionality of AMP, and enables automatic mixed precision in ResNet50-v1.5 inference and training workloads. We will support more workloads and make this feature mature in future release.

#### Inference Model Example using AMP (float16)

```bash
with torch.inference_mode():
  with torch.xpu.amp.autocast(dtype=torch.half):
    output = model(input)
```

#### Inference Model Example using AMP (bfloat16)

```bash
with torch.inference_mode():
  with torch.xpu.amp.autocast(dtype=torch.bfloat16):
    output = model(input)
```

#### Training Model Example using AMP (bfloat16)

```bash
for images, label in train_loader():
  with torch.xpu.amp.autocast(dtype=torch.bfloat16):
    loss = criterion(model(images), label)
  loss.backward()
  optimizer.step()
```

### Coding Style Alignment:

This release uses clang-format and flake8 to enhance the code in Intel® Extension for PyTorch* GPU and make sure the coding style align with PyTorch proper.

### Distributed Training with DistributedDataParallel (DDP):

ResNet50, BERT, CosmicTagger training are verified with DDP on PVC B4. For supporting this scenario, oneCCL Bindings for Pytorch* based on oneCCL 2021.8-eng02 version shall be built and used. 

```bash
git clone -b torch-ccl-xpu-1.10-rc2 https://github.com/intel-innersource/frameworks.ai.pytorch.torch-ccl.git
git submodule update --init --recursive
COMPUTE_BACKEND=dpcpp python setup.py install
```
Example of running multi-tile ResNet50 training with DDP:

```bash
cd frameworks.ai.pytorch.gpu-models/ResNet50
source `python -c "from oneccl_bindings_for_pytorch import cwd;print(cwd)"`/env/setvars.sh
mpiexec -n 2 python main.py -a resnet50 -e -b 1024 --pretrained --jit --xpu 0 $dataset
```

### Distributed Training with Horovod:

ResNet50, CosmicTagger, PointNet-ATLAS are verified with Horovod on PVC B4. Following launch options need be set during the validation: <br>

```bash
source $ONEAPI_INSTALL_DIR/mpi/latest/env/vars.sh -i_mpi_library_kind=release_mt
source $ONEAPI_INSTALL_DIR/ccl/latest/env/vars.sh --ccl-configuration=cpu_gpu_dpcpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
export CCL_STAGING_BUFFER=regular
```

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
| Permute + Contiguous | FP32/FP16/BF16/INT8 |
| Conv2D + Leaky_relu | INT8 |
| Conv2D + Leaky_relu_ | INT8 |
| Conv2D + Sigmoid | INT8 |
| Conv2D + Dequantize | INT8 |
| Conv2D + Dequantize + Softplus + Tanh + Mul + Quantize + Add | INT8 |

### Master weights support:

Master weights is enabled for assuring accuracy by using FP32 weights in BF16 training gradient update. To use this feature, model shall be updated as following: 

```bash
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

### Operator Coverage:

| **Operator Type** | **Implemented**| **Completion ratio** |
| ------ | ------ | ------ |
| PyTorch NN functions | 176 | 100.00%　|
| PyTorch Tensor functions | 176 | 100.00%　|
| PyTorch Methods | 217 | 100.00%　|
| Others | 178 | 100.00%　|
| Total | 747 | 100.00% |

Note that we count 61 NN backwards op to make the total op number accurate and increased from 686 to 747.

### Profile tool:

torch.autograd.profiler_legacy supports profiling kernel time spent on "xpu" device. Pesudo example looks like:

```bash
with torch.autograd.profiler_legacy.profile(enabled=True, use_xpu=True) as prof:
   fwd
   bwd
   weight update
print(prof.key_averages().table(sort_by="self_xpu_time_total"))
print(prof.table(sort_by="id", row_limit=100000))
```

#### Training Model Example

```bash
  with torch.autograd.profiler_legacy.profile(use_xpu=True, record_shapes=False) as prof:

      if args.gpu is not None:
          input = input.xpu(args.gpu, non_blocking=True)
          target = target.xpu(args.gpu, non_blocking=True)
      elif args.xpu is not None:
          input = input.to("xpu")
          target = target.to("xpu")

      # compute output
      output = model(input)
      loss = criterion(output, target)

      # compute gradient and do SGD step
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()

  # sync for time measurement
  torch.xpu.synchronize()
  profiling_path = os.path.abspath('../') + '/report/'
  torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), profiling_path + 'rn50_training_profiling.pt')
  prof.export_chrome_trace(profiling_path + 'rn50_training_profiling.json')
  print(prof.key_averages().table(sort_by="self_xpu_time_total"))
  print(prof.key_averages(group_by_input_shape=True).table())
  print(prof.table(sort_by="id", row_limit=100000))
```

#### Inference Model Example

```bash
  with torch.autograd.profiler_legacy.profile(use_xpu=True, record_shapes=False) as prof:

      if args.xpu is not None:
          input = input.to("xpu")

      if args.channels_last:
          input = input.to(memory_format=torch.channels_last)

      # compute output
      output = model(input)

  # sync for time measurement
  torch.xpu.synchronize()
  profiling_path = os.path.abspath('../') + '/report/'
  torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), profiling_path + 'rn50_inference_profiling.pt')
  prof.export_chrome_trace(profiling_path + 'rn50_inference_profiling.json')
  print(prof.key_averages().table(sort_by="self_xpu_time_total"))
  print(prof.key_averages(group_by_input_shape=True).table())
  print(prof.table(sort_by="id", row_limit=100000))
```

#### Profiling Results

The output from BERT training model looks like (omitting some columns and rows):

```bash
#---------------------------- ----------  ----------    ----------    ----------  ------------  ----------     ----------     ---------   ------------   ------------
# Name                         Self CPU %  Self CPU       CPU total %  CPU total  CPU time avg    Self XPU       Self XPU %     XPU total  XPU time avg    # of Calls  
#---------------------------- ----------  ----------    ----------    ----------  ------------  ----------     ----------     ---------   ------------   ------------
# aten::mm                      15.06%     122.520ms        15.49%     126.016ms     431.563us     172.356ms        24.70%     172.356ms     590.262us           292  
# aten::bmm                     5.35%      43.484ms         5.58%      45.342ms     314.877us      70.894ms        10.16%      70.894ms     492.318us           144  
# aten::addmm                   5.00%      40.692ms         5.16%      41.965ms     285.478us      64.013ms         9.17%      64.013ms     435.461us           147  
# bernoulliDistr                0.28%       2.303ms         0.28%       2.303ms      31.541us      36.179ms         5.18%      36.179ms     495.599us            73  
# aten::_fused_dropout          0.85%       6.915ms         1.23%      10.004ms     137.045us      35.081ms         5.03%      71.260ms     976.164us            73  
# dnnl_reorder                  7.01%      57.012ms         7.06%      57.433ms     297.579us      30.084ms         4.31%      30.084ms     155.876us           193  
# transformer_adamWMasterWeight 2.01%      16.347ms         2.07%      16.874ms      42.827us      25.825ms         3.70%      25.825ms      65.546us           394  
# aten::norm                    5.31%      43.159ms         5.55%      45.163ms     114.337us      22.140ms         3.17%      22.140ms      56.051us           395  
#---------------------------- ----------  ----------    ----------    ----------  ------------  ----------     ----------     ---------   ------------   ------------
# Self CPU time total: 813.282ms
# XPU time total: 697.936ms
```

Note the difference between Self XPU time and XPU total time - operators can call other operators, Self XPU time excludes time spent in children operator calls, while XPU total time includes it. You can choose to sort by the self xpu time by passing sort_by="self_xpu_time_total" into the table call.

### Sparse tensor support:

This release supports sparse backend and several sparse related operators, including dense_dim, sparse_dim, to_sparse, sparse_mask, values, indices, coalesce and embeddingbag. It adds additional support for performing embeddingbag on sparse tensor in DLRM training.

### TF32 math mode:

This release supports TF32 math mode and provides launch option 'IPEX_FP32_MATH_MODE' to configure. The default math mode is FP32. Use 'IPEX_FP32_MATH_MODE=TF32' to change the math mode to TF32.

### torch.inference_mode():

The inference_mode is recommended by PyTorch official to get better performance by disabling view tracking and version counter bumps. For Intel® Extension for PyTorch* GPU, we also strongly recommend it, as inference_mode is able to significantly enhance performance by removing redundant layout conversions when using oneDNN specific layouts.

### User mode scratchpad:

oneDNN defines two scratchpad modes: library and user. User mode means framework will manage the scratchpad allocation by querying and providing the scratchpad memory to oneDNN primitives. User mode scratchpad is default enabled for all oneDNN primitives in this release. 

To switch to library mode scratchpad, please set USE_SCRATCHPAD_MODE=OFF and rebuild. Please expect negative impact to performance when changing the mode.

## Caveat

### 1. Build order of PyTorch and extension:

Please build Intel® Extension for PyTorch* GPU after pytorch is built and installed, otherwise you will get an error “ModuleNotFoundError: No module named 'torch'”.

### 2. symbol undefined caused by _GLIBCXX_USE_CXX11_ABI:

#### Error info: <br>

```bash
File "/root/.local/lib/python3.9/site-packages/ipex/__init__.py", line 4, in <module>
    from . import _C
ImportError: /root/.local/lib/python3.9/site-packages/ipex/lib/libipex_gpu_core.so: undefined symbol: _ZNK5torch8autograd4Node4nameB5cxx11Ev
```

This issue appears when Intel® Extension for PyTorch* is compiled with \_GLIBCXX_USE_CXX11_ABI=1 and PyTorch is compiled with \_GLIBCXX_USE_CXX11_ABI=0, which causes inconsistent.
<br>

#### Background：

1. DPC++ has no plan to support \_GLIBCXX_USE_CXX11_ABI=0 (CMPLRLLVM-34202), Intel® Extension for PyTorch* is always compiled with \_GLIBCXX_USE_CXX11_ABI=1. <br>
2. PyTorch detects the setting of \_GLIBCXX_USE_CXX11_ABI by checking user config and compiler capability. If compiler in use does not support \_GLIBCXX_USE_CXX11_ABI=1, PyTorch is compiled with \_GLIBCXX_USE_CXX11_ABI=0. PyTorch publishes official binary package with \_GLIBCXX_USE_CXX11_ABI=0. <br>

#### Solution：

User shall update PyTorch CMAKE file to set \_GLIBCXX_USE_CXX11_ABI=1 and compile PyTorch with particular compiler which supports \_GLIBCXX_USE_CXX11_ABI=1. We recommend to use gcc version 9.3.0 (Ubuntu 9.3.0-17ubuntu1~20.04) on ubuntu 20.04 which is validated by us. <br>

This issue won't exist with future version of PyTorch as community agrees to provide \_GLIBCXX_USE_CXX11_ABI=1 binary for versions after PyTorch 1.10. 

### 3. Distributed Training Issues:

#### ResNet50 training with Horovod

Explicit scaling of ResNet50 training with Horovod hang on 2-cards PVC B4 (XDEPS-4646). This was identified as driver issue. 

#### BERT training with DDP

Explicit scaling of BERT training with DDP hang on 2-cards PVC B4 (PYTORCHDGQ-1768). This issue disappeared when roll back driver from agama-ci-prerelease-522 to agama-ci-prerelease-438. The root cause is still under investigation.

### 4. UT failures:

#### test_groupnorm_channels_last.py AssertionError

test_groupnorm_channels_last.py AssertionError (MFDNN-8290) was identified as HW bug related to alignment for 2D messages and was fixed in oneDNN master. We will uplift oneDNN in IPEX master to get such fix.
