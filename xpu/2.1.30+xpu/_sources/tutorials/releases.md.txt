Releases
=============

## 2.1.30+xpu

Intel® Extension for PyTorch\* v2.1.30+xpu is an update release which supports Intel® GPU platforms (Intel® Data Center GPU Flex Series, Intel® Data Center GPU Max Series and Intel® Arc™ A-Series Graphics) based on PyTorch\* 2.1.0.

### Highlights

- Intel® oneDNN v3.4.1 integration
- Intel® oneAPI Base Toolkit 2024.1 compatibility
- Large Language Model (LLM) optimizations for FP16 inference on Intel® Data Center GPU Max Series (Beta): Intel® Extension for PyTorch* provides a lot of specific optimizations for LLM workloads in this release on Intel® Data Center GPU Max Series.  In operator level, we provide highly efficient GEMM kernel to speed up Linear layer and customized fused operators to reduce HBM access/kernel launch overhead. To reduce memory footprint, we define a segment KV Cache policy to save device memory and improve the throughput. Such optimizations are added in this release to enhance existing optimized LLM FP16 models and more Chinese LLM models such as Baichuan2-13B, ChatGLM3-6B and Qwen-7B.

- LLM optimizations for INT4 inference on Intel® Data Center GPU Max Series and Intel® Arc™ A-Series Graphics (Prototype): Intel® Extension for PyTorch* shows remarkable performance when executing LLM models on Intel® GPU. However, deploying such models on GPUs with limited resources is challenging due to their high computational and memory requirements. To achieve a better trade-off, a low-precision solution, e.g., weight-only-quantization for INT4 is enabled to allow Llama 2-7B, GPT-J-6B and Qwen-7B to be executed efficiently on Intel® Arc™ A-Series Graphics. The same optimization makes INT4 models achieve 1.5x speeded up in total latency performance compared with FP16 models with the same configuration and parameters on Intel® Data Center GPU Max Series.

- Opt-in collective performance optimization with oneCCL Bindings for Pytorch*: This opt-in feature can be enabled by setting `TORCH_LLM_ALLREDUCE=1` to provide better scale-up performance by enabling optimized collectives such as `allreduce`, `allgather`, `reducescatter` algorithms in Intel® oneCCL. This feature requires XeLink enabled for cross-cards communication.

### Known Issues

Please refer to [Known Issues webpage](./known_issues.md).

## 2.1.20+xpu

Intel® Extension for PyTorch\* v2.1.20+xpu is a minor release which supports Intel® GPU platforms (Intel® Data Center GPU Flex Series, Intel® Data Center GPU Max Series and Intel® Arc™ A-Series Graphics) based on PyTorch\* 2.1.0.

### Highlights

- Intel® oneAPI Base Toolkit 2024.1 compatibility
- Intel® oneDNN v3.4 integration
- LLM inference scaling optimization based on Intel® oneCCL 2021.12 (Prototype)
- Bug fixing and other optimization
  - Uplift XeTLA to v0.3.4.1 [#3696](https://github.com/intel/intel-extension-for-pytorch/commit/dc0f6d39739404d38226ccf444c421706f14f2de)
  - [SDP] Fallback unsupported bias size to native impl [#3706](https://github.com/intel/intel-extension-for-pytorch/commit/d897ebd585da05a90295165584efc448e265a38d)
  - Error handling enhancement [#3788](https://github.com/intel/intel-extension-for-pytorch/commit/bd034e7a37822f84706f0068ec85d989fb766529), [#3841](https://github.com/intel/intel-extension-for-pytorch/commit/7d4f297ecb4c076586a22908ecadf4689cb2d5ef)
  - Fix beam search accuracy issue in workgroup reduce [#3796](https://github.com/intel/intel-extension-for-pytorch/commit/f2f20a523ee85ed1f44c7fa6465b8e5e1e2edfea)
  - Support int32 index tensor in index operator [#3808](https://github.com/intel/intel-extension-for-pytorch/commit/f7bb4873c0416a9f56d1f7ecfbcdbe7ad58b47cd)
  - Add deepspeed in LLM dockerfile [#3829](https://github.com/intel/intel-extension-for-pytorch/commit/6266f89833f8010d6c683f9b45cfb2031575ad92)
  - Fix batch norm accuracy issue [#3882](https://github.com/intel/intel-extension-for-pytorch/commit/a1e2271717ff61dc3ea7d8d471c2356b3e469b93)
  - Prebuilt wheel dockerfile update [#3887](https://github.com/intel/intel-extension-for-pytorch/commit/8d5d71522910c1f622dac6a52cb0025e469774b2#diff-022fb5910f470cc5c44ab38cb20586d014f37c06ac8f3378e146ed35ee202a46), [#3970](https://github.com/intel/intel-extension-for-pytorch/commit/54b8171940cd694ba91c928c99acc440c9993881)
  - Fix windows build failure with Intel® oneMKL 2024.1 in torch_patches [#18](https://github.com/intel/intel-extension-for-pytorch/blob/release/xpu/2.1.20/torch_patches/0018-use-ONEMKL_LIBRARIES-for-mkl-libs-in-torch-to-not-ov.patch)
  - Fix FFT core dump issue with Intel® oneMKL 2024.1 in torch_patches [#20](https://github.com/intel/intel-extension-for-pytorch/blob/release/xpu/2.1.20/torch_patches/0020-Hide-MKL-symbols-211-212.patch), [#21](https://github.com/intel/intel-extension-for-pytorch/blob/release/xpu/2.1.20/torch_patches/0021-Fix-Windows-Build-214-215.patch)

### Known Issues

Please refer to [Known Issues webpage](./known_issues.md).

## 2.1.10+xpu

Intel® Extension for PyTorch\* v2.1.10+xpu is the new Intel® Extension for PyTorch\* release supports both CPU platforms and GPU platforms (Intel® Data Center GPU Flex Series, Intel® Data Center GPU Max Series and Intel® Arc™ A-Series Graphics) based on PyTorch\* 2.1.0. It extends PyTorch\* 2.1.0 with up-to-date features and optimizations on `xpu` for an extra performance boost on Intel hardware. Optimizations take advantage of AVX-512 Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX) on Intel CPUs as well as Intel Xe Matrix Extensions (XMX) AI engines on Intel discrete GPUs. Moreover, through PyTorch* `xpu` device, Intel® Extension for PyTorch* provides easy GPU acceleration for Intel discrete GPUs with PyTorch*.

### Highlights

This release provides the following features:

- Large Language Model (LLM) optimizations for FP16 inference on Intel® Data Center GPU Max Series (Prototype): Intel® Extension for PyTorch* provides a lot of specific optimizations for LLM workloads on Intel® Data Center GPU Max Series in this release. In operator level, we provide highly efficient GEMM kernel to speedup Linear layer and customized fused operators to reduce HBM access and kernel launch overhead. To reduce memory footprint, we define a segment KV Cache policy to save device memory and improve the throughput. To better trade-off the performance and accuracy, low-precision solution e.g., weight-only-quantization for INT4 is enabled. Besides, tensor parallel can also be adopted to get lower latency for LLMs.

  - A new API function, `ipex.optimize_transformers`, is designed to optimize transformer-based models within frontend Python modules, with a particular focus on LLMs. It provides optimizations for both model-wise and content-generation-wise. You just need to invoke the `ipex.optimize_transformers` API instead of the `ipex.optimize` API to apply all optimizations transparently. More detailed information can be found at [Large Language Model optimizations overview](https://intel.github.io/intel-extension-for-pytorch/xpu/2.1.10+xpu/tutorials/llm.html).
  - A typical usage of this new feature is quite simple as below:
    ```            
    import torch
    import intel_extension_for_pytorch as ipex
    ...
    model = ipex.optimize_transformers(model, dtype=dtype)
    ```

- `Torch.compile` functionality on Intel® Data Center GPU Max Series (Beta): Extends Intel® Extension for PyTorch* capabilities to support [torch.compile](https://pytorch.org/docs/stable/generated/torch.compile.html#torch-compile) APIs on Intel® Data Center GPU Max Series. And provides Intel GPU support on top of [Triton*](https://github.com/openai/triton) compiler to reach competitive performance speed-up over eager mode by default "inductor" backend of Intel® Extension for PyTorch*.
  
- Intel® Arc™ A-Series Graphics on WSL2, native Windows and native Linux are officially supported in this release. Intel® Arc™ A770 Graphic card has been used as primary verification vehicle for product level test.
  
- Other features are listed as following, more detailed information can be found in [public documentation](https://intel.github.io/intel-extension-for-pytorch/xpu/2.1.10+xpu/):
  - FP8 datatype support (Prototype): Add basic data type and FP8 Linear operator support based on emulation kernel.
  - Kineto Profiling (Prototype): An extension of PyTorch* profiler for profiling operators on Intel® GPU devices.
  - Fully Sharded Data Parallel (FSDP):  Support new PyTorch* [FSDP](https://pytorch.org/docs/stable/fsdp.html) API which provides an industry-grade solution for large-scale model training.
  - Asymmetric INT8 quantization: Support asymmetric quantization to align with stock PyTorch* and provide better accuracy in INT8.

- CPU support has been merged in this release. CPU features and optimizations are equivalent to what has been released in [Intel® Extension for PyTorch* v2.1.0+cpu release](https://github.com/intel/intel-extension-for-pytorch/releases/tag/v2.1.0+cpu) that was made publicly available in Oct 2023. For customers who would like to evaluate workloads on both GPU and CPU, they can use this package. For customers who are focusing on CPU only, we still recommend them to use Intel® Extension for PyTorch* v2.1.0+cpu release for smaller footprint, less dependencies and broader OS support.

### Known Issues

Please refer to [Known Issues webpage](./known_issues.md).

## 2.0.110+xpu

Intel® Extension for PyTorch\* v2.0.110+xpu is the new Intel® Extension for PyTorch\* release supports both CPU platforms and GPU platforms (Intel® Data Center GPU Flex Series and Intel® Data Center GPU Max Series) based on PyTorch\* 2.0.1. It extends PyTorch\* 2.0.1 with up-to-date features and optimizations on `xpu` for an extra performance boost on Intel hardware. Optimizations take advantage of AVX-512 Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX) on Intel CPUs as well as Intel Xe Matrix Extensions (XMX) AI engines on Intel discrete GPUs. Moreover, through PyTorch* `xpu` device, Intel® Extension for PyTorch* provides easy GPU acceleration for Intel discrete GPUs with PyTorch*.

### Highlights

This release introduces specific XPU solution optimizations on Intel discrete GPUs which include Intel® Data Center GPU Flex Series and Intel® Data Center GPU Max Series. Optimized operators and kernels are implemented and registered through PyTorch\* dispatching mechanism for the `xpu` device. These operators and kernels are accelerated on Intel GPU hardware from the corresponding native vectorization and matrix calculation features. In graph mode, additional operator fusions are supported to reduce operator/kernel invocation overheads, and thus increase performance.

This release provides the following features:
- oneDNN 3.3 API integration and adoption
- Libtorch support
- ARC support on Windows, WSL2 and Ubuntu (Prototype)
- OOB models improvement
  - More fusion patterns enabled for optimizing OOB models
- CPU support is merged in this release:
  - CPU features and optimizations are equivalent to what has been released in Intel® Extension for PyTorch* v2.0.100+cpu release that was made publicly available in May 2023. For customers who would like to evaluate workloads on both GPU and CPU, they can use this package. For customers who are focusing on CPU only, we still recommend them to use Intel® Extension for PyTorch* v2.0.100+cpu release for smaller footprint, less dependencies and broader OS support.

This release adds the following fusion patterns in PyTorch\* JIT mode for Intel GPU:
- `add` + `softmax`
- `add` + `view` + `softmax`

### Known Issues

Please refer to [Known Issues webpage](./known_issues.md).

## 1.13.120+xpu

Intel® Extension for PyTorch\* v1.13.120+xpu is the updated Intel® Extension for PyTorch\* release supports both CPU platforms and GPU platforms (Intel® Data Center GPU Flex Series and Intel® Data Center GPU Max Series) based on PyTorch\* 1.13.1. It extends PyTorch\* 1.13.1 with up-to-date features and optimizations on `xpu` for an extra performance boost on Intel hardware. Optimizations take advantage of AVX-512 Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX) on Intel CPUs as well as Intel Xe Matrix Extensions (XMX) AI engines on Intel discrete GPUs. Moreover, through PyTorch* `xpu` device, Intel® Extension for PyTorch* provides easy GPU acceleration for Intel discrete GPUs with PyTorch*.

### Highlights

This release introduces specific XPU solution optimizations on Intel discrete GPUs which include Intel® Data Center GPU Flex Series and Intel® Data Center GPU Max Series. Optimized operators and kernels are implemented and registered through PyTorch\* dispatching mechanism for the `xpu` device. These operators and kernels are accelerated on Intel GPU hardware from the corresponding native vectorization and matrix calculation features. In graph mode, additional operator fusions are supported to reduce operator/kernel invocation overheads, and thus increase performance.

This release provides the following features:
- oneDNN 3.1 API integration and adoption
- OOB models improvement
  - More fusion patterns enabled for optimizing OOB models
- CPU support is merged in this release:
  - CPU features and optimizations are equivalent to what has been released in Intel® Extension for PyTorch* v1.13.100+cpu release that was made publicly available in Feb 2023. For customers who would like to evaluate workloads on both GPU and CPU, they can use this package. For customers who are focusing on CPU only, we still recommend them to use Intel® Extension for PyTorch* v1.13.100+cpu release for smaller footprint, less dependencies and broader OS support.

This release adds the following fusion patterns in PyTorch\* JIT mode for Intel GPU:
- `Matmul` + UnaryOp(`abs`, `sqrt`, `square`, `exp`, `log`, `round`, `Log_Sigmoid`, `Hardswish`, `HardSigmoid`, `Pow`, `ELU`, `SiLU`, `hardtanh`, `Leaky_relu`)
- `Conv2d` + BinaryOp(`add`, `sub`, `mul`, `div`, `max`, `min`, `eq`, `ne`, `ge`, `gt`, `le`, `lt`)
- `Linear` + BinaryOp(`add`, `sub`, `mul`, `div`, `max`, `min`)
- `Conv2d` + `mul` + `add`
- `Conv2d` + `mul` + `add` + `relu`
- `Conv2d` + `sigmoid` + `mul` + `add`
- `Conv2d` + `sigmoid` + `mul` + `add` + `relu`

### Known Issues

Please refer to [Known Issues webpage](./known_issues.md).

## 1.13.10+xpu

Intel® Extension for PyTorch\* v1.13.10+xpu is the first Intel® Extension for PyTorch\* release supports both CPU platforms and GPU platforms (Intel® Data Center GPU Flex Series and Intel® Data Center GPU Max Series) based on PyTorch\* 1.13. It extends PyTorch\* 1.13 with up-to-date features and optimizations on `xpu` for an extra performance boost on Intel hardware. Optimizations take advantage of AVX-512 Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX) on Intel CPUs as well as Intel Xe Matrix Extensions (XMX) AI engines on Intel discrete GPUs. Moreover, through PyTorch* `xpu` device, Intel® Extension for PyTorch* provides easy GPU acceleration for Intel discrete GPUs with PyTorch*.

### Highlights

This release introduces specific XPU solution optimizations on Intel discrete GPUs which include Intel® Data Center GPU Flex Series and Intel® Data Center GPU Max Series. Optimized operators and kernels are implemented and registered through PyTorch\* dispatching mechanism for the `xpu` device. These operators and kernels are accelerated on Intel GPU hardware from the corresponding native vectorization and matrix calculation features. In graph mode, additional operator fusions are supported to reduce operator/kernel invocation overheads, and thus increase performance.

This release provides the following features:
- Distributed Training on GPU:
  - support of distributed training with DistributedDataParallel (DDP) on Intel GPU hardware
  - support of distributed training with Horovod (prototype feature) on Intel GPU hardware
- Automatic channels last format conversion on GPU:
  - Automatic channels last format conversion is enabled. Models using `torch.xpu.optimize` API running on Intel® Data Center GPU Max Series will be converted to channels last memory format, while models running on Intel® Data Center GPU Flex Series will choose oneDNN block format.
- CPU support is merged in this release:
  - CPU features and optimizations are equivalent to what has been released in Intel® Extension for PyTorch* v1.13.0+cpu release that was made publicly available in Nov 2022. For customers who would like to evaluate workloads on both GPU and CPU, they can use this package. For customers who are focusing on CPU only, we still recommend them to use Intel® Extension for PyTorch* v1.13.0+cpu release for smaller footprint, less dependencies and broader OS support.

This release adds the following fusion patterns in PyTorch\* JIT mode for Intel GPU:
- `Conv2D` + UnaryOp(`abs`, `sqrt`, `square`, `exp`, `log`, `round`, `GeLU`, `Log_Sigmoid`, `Hardswish`, `Mish`, `HardSigmoid`, `Tanh`, `Pow`, `ELU`, `hardtanh`)
- `Linear` + UnaryOp(`abs`, `sqrt`, `square`, `exp`, `log`, `round`, `Log_Sigmoid`, `Hardswish`, `HardSigmoid`, `Pow`, `ELU`, `SiLU`, `hardtanh`, `Leaky_relu`)

### Known Issues

Please refer to [Known Issues webpage](./known_issues.md).

## 1.10.200+gpu

Intel® Extension for PyTorch\* v1.10.200+gpu extends PyTorch\* 1.10 with up-to-date features and optimizations on XPU for an extra performance boost on Intel Graphics cards. XPU is a user visible device that is a counterpart of the well-known CPU and CUDA in the PyTorch\* community. XPU represents an Intel-specific kernel and graph optimizations for various “concrete” devices. The XPU runtime will choose the actual device when executing AI workloads on the XPU device. The default selected device is Intel GPU. XPU kernels from Intel® Extension for PyTorch\* are written in [DPC++](https://github.com/intel/llvm#oneapi-dpc-compiler) that supports [SYCL language](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html) and also a number of [DPC++ extensions](https://github.com/intel/llvm/tree/sycl/sycl/doc/extensions).

### Highlights

This release introduces specific XPU solution optimizations on Intel® Data Center GPU Flex Series 170. Optimized operators and kernels are implemented and registered through PyTorch\* dispatching mechanism for the XPU device. These operators and kernels are accelerated on Intel GPU hardware from the corresponding native vectorization and matrix calculation features. In graph mode, additional operator fusions are supported to reduce operator/kernel invocation overheads, and thus increase performance.

This release provides the following features:
- Auto Mixed Precision (AMP)
  - support of AMP with BFloat16 and Float16 optimization of GPU operators
- Channels Last
  - support of channels\_last (NHWC) memory format for most key GPU operators
- DPC++ Extension
  - mechanism to create PyTorch\* operators with custom DPC++ kernels running on the XPU device
- Optimized Fusion
  - support of SGD/AdamW fusion for both FP32 and BF16 precision

This release supports the following fusion patterns in PyTorch\* JIT mode:

- Conv2D + ReLU
- Conv2D + Sum
- Conv2D + Sum + ReLU
- Pad + Conv2d
- Conv2D + SiLu
- Permute + Contiguous
- Conv3D + ReLU
- Conv3D + Sum
- Conv3D + Sum + ReLU
- Linear + ReLU
- Linear + Sigmoid
- Linear + Div(scalar)
- Linear + GeLu
- Linear + GeLu\_
- T + Addmm
- T + Addmm + ReLu
- T + Addmm + Sigmoid
- T + Addmm + Dropout
- T + Matmul
- T + Matmul + Add
- T + Matmul + Add + GeLu
- T + Matmul + Add + Dropout
- Transpose + Matmul
- Transpose + Matmul + Div
- Transpose + Matmul + Div + Add
- MatMul + Add
- MatMul + Div
- Dequantize + PixelShuffle
- Dequantize + PixelShuffle + Quantize
- Mul + Add
- Add + ReLU
- Conv2D + Leaky\_relu
- Conv2D + Leaky\_relu\_
- Conv2D + Sigmoid
- Conv2D + Dequantize
- Softplus + Tanh
- Softplus + Tanh + Mul
- Conv2D + Dequantize + Softplus + Tanh + Mul
- Conv2D + Dequantize + Softplus + Tanh + Mul + Quantize
- Conv2D + Dequantize + Softplus + Tanh + Mul + Quantize + Add

### Known Issues

- [CRITICAL ERROR] Kernel 'XXX' removed due to usage of FP64 instructions unsupported by the targeted hardware

    FP64 is not natively supported by the [Intel® Data Center GPU Flex Series](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/data-center-gpu/flex-series/overview.html) platform. If you run any AI workload on that platform and receive this error message, it means a kernel requiring FP64 instructions is removed and not executed, hence the accuracy of the whole workload is wrong.

- symbol undefined caused by \_GLIBCXX\_USE\_CXX11\_ABI

    ```bash
    ImportError: undefined symbol: _ZNK5torch8autograd4Node4nameB5cxx11Ev
    ```
    
    DPC++ does not support \_GLIBCXX\_USE\_CXX11\_ABI=0, Intel® Extension for PyTorch\* is always compiled with \_GLIBCXX\_USE\_CXX11\_ABI=1. This symbol undefined issue appears when PyTorch\* is compiled with \_GLIBCXX\_USE\_CXX11\_ABI=0. Update PyTorch\* CMAKE file to set \_GLIBCXX\_USE\_CXX11\_ABI=1 and compile PyTorch\* with particular compiler which supports \_GLIBCXX\_USE\_CXX11\_ABI=1. We recommend to use gcc version 9.4.0 on ubuntu 20.04.

- Can't find oneMKL library when build Intel® Extension for PyTorch\* without oneMKL

    ```bash
    /usr/bin/ld: cannot find -lmkl_sycl
    /usr/bin/ld: cannot find -lmkl_intel_ilp64
    /usr/bin/ld: cannot find -lmkl_core
    /usr/bin/ld: cannot find -lmkl_tbb_thread
    dpcpp: error: linker command failed with exit code 1 (use -v to see invocation)
    ```
    
    When PyTorch\* is built with oneMKL library and Intel® Extension for PyTorch\* is built without oneMKL library, this linker issue may occur. Resolve it by setting:
    
    ```bash
    export USE_ONEMKL=OFF
    export MKL_DPCPP_ROOT=${PATH_To_Your_oneMKL}/__release_lnx/mkl
    ```
    
    Then clean build Intel® Extension for PyTorch\*.

- undefined symbol: mkl\_lapack\_dspevd. Intel MKL FATAL ERROR: cannot load libmkl\_vml\_avx512.so.2 or libmkl\_vml\_def.so.2

    This issue may occur when Intel® Extension for PyTorch\* is built with oneMKL library and PyTorch\* is not build with any MKL library. The oneMKL kernel may run into CPU backend incorrectly and trigger this issue. Resolve it by installing MKL library from conda:
    
    ```bash
    conda install mkl
    conda install mkl-include
    ```
    
    then clean build PyTorch\*.

- OSError: libmkl\_intel\_lp64.so.1: cannot open shared object file: No such file or directory

    Wrong MKL library is used when multiple MKL libraries exist in system. Preload oneMKL by:
    
    ```bash
    export LD_PRELOAD=${MKL_DPCPP_ROOT}/lib/intel64/libmkl_intel_lp64.so.1:${MKL_DPCPP_ROOT}/lib/intel64/libmkl_intel_ilp64.so.1:${MKL_DPCPP_ROOT}/lib/intel64/libmkl_sequential.so.1:${MKL_DPCPP_ROOT}/lib/intel64/libmkl_core.so.1:${MKL_DPCPP_ROOT}/lib/intel64/libmkl_sycl.so.1
    ```
    
    If you continue seeing similar issues for other shared object files, add the corresponding files under ${MKL\_DPCPP\_ROOT}/lib/intel64/ by `LD_PRELOAD`. Note that the suffix of the libraries may change (e.g. from .1 to .2), if more than one oneMKL library is installed on the system.


