Examples
========

These examples will help you get started using Intel® Extension for PyTorch\*
with Intel GPUs.

For examples on Intel CPUs, check the [CPU examples](../../../cpu/latest/tutorials/examples.html).

**Prerequisites**:
Before running these examples, install the `torchvision` and `transformers` Python packages.

- [Python](#python) examples demonstrate usage of Python APIs:

  - [Training](#training)
  - [Inference](#inference)

- [C++](#c) examples demonstrate usage of C++ APIs
- [Intel® AI Reference Models](#intel-ai-reference-models) provide out-of-the-box use cases, demonstrating the performance benefits achievable with Intel Extension for PyTorch\*


## Python

### Training

#### Single-Instance Training

To use Intel® Extension for PyTorch\* on training, you need to make the following changes in your code:

1. Import `intel_extension_for_pytorch` as `ipex`.
2. Use the `ipex.optimize` function, which applies optimizations against the model object, as well as an optimizer object.
3. Use Auto Mixed Precision (AMP) with BFloat16 data type.
4. Convert input tensors, loss criterion and model to XPU, as shown below:

```
...
import torch
import intel_extension_for_pytorch as ipex
...
model = Model()
criterion = ...
optimizer = ...
model.train()
# For Float32
model, optimizer = ipex.optimize(model, optimizer=optimizer)
# For BFloat16
model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)
...
# For Float32
output = model(data)
...
# For BFloat16
with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
  output = model(input)
...
```

Below you can find complete code examples demonstrating how to use the extension on training for different data types:

##### Float32

[//]: # (marker_train_single_fp32_complete)
[//]: # (marker_train_single_fp32_complete)

##### BFloat16

[//]: # (marker_train_single_bf16_complete)
[//]: # (marker_train_single_bf16_complete)

### Inference

Get additional performance boosts for your computer vision and NLP workloads by
applying the Intel® Extension for PyTorch\* `optimize` function against your
model object.

#### Float32

##### Imperative Mode

###### Resnet50

[//]: # (marker_inf_rn50_imp_fp32)
[//]: # (marker_inf_rn50_imp_fp32)

###### BERT

[//]: # (marker_inf_bert_imp_fp32)
[//]: # (marker_inf_bert_imp_fp32)

##### TorchScript Mode

We recommend using Intel® Extension for PyTorch\* with [TorchScript](https://pytorch.org/docs/stable/jit.html) for further optimizations.

###### Resnet50

[//]: # (marker_inf_rn50_ts_fp32)
[//]: # (marker_inf_rn50_ts_fp32)

###### BERT

[//]: # (marker_inf_bert_ts_fp32)
[//]: # (marker_inf_bert_ts_fp32)

#### BFloat16

The `optimize` function works for both Float32 and BFloat16 data type. For BFloat16 data type, set the `dtype` parameter to `torch.bfloat16`.
We recommend using Auto Mixed Precision (AMP) with BFloat16 data type.


##### Imperative Mode

###### Resnet50

[//]: # (marker_inf_rn50_imp_bf16)
[//]: # (marker_inf_rn50_imp_bf16)

###### BERT

[//]: # (marker_inf_bert_imp_bf16)
[//]: # (marker_inf_bert_imp_bf16)

##### TorchScript Mode

We recommend using Intel® Extension for PyTorch\* with [TorchScript](https://pytorch.org/docs/stable/jit.html) for further optimizations.

###### Resnet50

[//]: # (marker_inf_rn50_ts_bf16)
[//]: # (marker_inf_rn50_ts_bf16)

###### BERT

[//]: # (marker_inf_bert_ts_bf16)
[//]: # (marker_inf_bert_ts_bf16)

#### Float16

The `optimize` function works for both Float32 and Float16 data type. For Float16 data type, set the `dtype` parameter to `torch.float16`.
We recommend using Auto Mixed Precision (AMP) with Float16 data type.

##### Imperative Mode

###### Resnet50

[//]: # (marker_inf_rn50_imp_fp16)
[//]: # (marker_inf_rn50_imp_fp16)

###### BERT

[//]: # (marker_inf_bert_imp_fp16)
[//]: # (marker_inf_bert_imp_fp16)

##### TorchScript Mode

We recommend using Intel® Extension for PyTorch\* with [TorchScript](https://pytorch.org/docs/stable/jit.html) for further optimizations.

###### Resnet50

[//]: # (marker_inf_rn50_ts_fp16)
[//]: # (marker_inf_rn50_ts_fp16)

###### BERT

[//]: # (marker_inf_bert_ts_fp16)
[//]: # (marker_inf_bert_ts_fp16)

#### INT8

We recommend using TorchScript for INT8 model because it has wider support for models. TorchScript mode also auto-enables our optimizations. For TorchScript INT8 model, inserting observer and model quantization is achieved through `prepare_jit` and `convert_jit` separately. Calibration process is required for collecting statistics from real data. After conversion, optimizations such as operator fusion would be auto-enabled.

[//]: # (marker_int8_static)
[//]: # (marker_int8_static)

#### torch.xpu.optimize

The `torch.xpu.optimize` function is an alternative to `ipex.optimize` in Intel® Extension for PyTorch\*, and provides identical usage for XPU devices only. The motivation for adding this alias is to unify the coding style in user scripts base on `torch.xpu` modular. Refer to the example below for usage.

[//]: # (marker_inf_rn50_imp_fp32_alt)
[//]: # (marker_inf_rn50_imp_fp32_alt)

## C++

To work with libtorch, the PyTorch C++ library, Intel® Extension for PyTorch\* provides its own C++ dynamic library. The C++ library only handles inference workloads, such as service deployment. For regular development, use the Python interface. Unlike using libtorch, no specific code changes are required. Compilation follows the recommended methodology with CMake. Detailed instructions can be found in the [PyTorch tutorial](https://pytorch.org/tutorials/advanced/cpp_export.html#depending-on-libtorch-and-building-the-application).

During compilation, Intel optimizations will be activated automatically after the C++ dynamic library of Intel® Extension for PyTorch\* is linked.

The example code below works for all data types.

### Basic Usage

**example-app.cpp**

[//]: # (marker_cppsdk_sample_app)
[//]: # (marker_cppsdk_sample_app)

**CMakeLists.txt**

[//]: # (marker_cppsdk_cmake_app)
[//]: # (marker_cppsdk_cmake_app)

**Command for compilation**

```bash
$ cd examples/gpu/inference/cpp/example-app
$ mkdir build
$ cd build
$ CC=icx CXX=icpx cmake -DCMAKE_PREFIX_PATH=<LIBPYTORCH_PATH> ..
$ make
```

If *Found IPEX* is shown as dynamic library paths, the extension was linked into the binary. This can be verified with the Linux command *ldd*.

```bash
$ CC=icx CXX=icpx cmake -DCMAKE_PREFIX_PATH=/workspace/libtorch ..
-- The C compiler identification is IntelLLVM 2023.2.0
-- The CXX compiler identification is IntelLLVM 2023.2.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /workspace/intel/oneapi/compiler/2023.2.0/linux/bin/icx - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /workspace/intel/oneapi/compiler/2023.2.0/linux/bin/icpx - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Looking for pthread.h
-- Looking for pthread.h - found
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE
-- Found Torch: /workspace/libtorch/lib/libtorch.so
-- Found IPEX: /workspace/libtorch/lib/libintel-ext-pt-cpu.so;/workspace/libtorch/lib/libintel-ext-pt-gpu.so
-- Configuring done
-- Generating done
-- Build files have been written to: examples/gpu/inference/cpp/example-app/build

$ ldd example-app
        ...
        libtorch.so => /workspace/libtorch/lib/libtorch.so (0x00007fd5bb927000)
        libc10.so => /workspace/libtorch/lib/libc10.so (0x00007fd5bb895000)
        libtorch_cpu.so => /workspace/libtorch/lib/libtorch_cpu.so (0x00007fd5a44d8000)
        libintel-ext-pt-cpu.so => /workspace/libtorch/lib/libintel-ext-pt-cpu.so (0x00007fd5a1a1b000)
        libintel-ext-pt-gpu.so => /workspace/libtorch/lib/libintel-ext-pt-gpu.so (0x00007fd5862b0000)
        ...
        libmkl_intel_lp64.so.2 => /workspace/intel/oneapi/mkl/2023.2.0/lib/intel64/libmkl_intel_lp64.so.2 (0x00007fd584ab0000)
        libmkl_core.so.2 => /workspace/intel/oneapi/mkl/2023.2.0/lib/intel64/libmkl_core.so.2 (0x00007fd5806cc000)
        libmkl_gnu_thread.so.2 => /workspace/intel/oneapi/mkl/2023.2.0/lib/intel64/libmkl_gnu_thread.so.2 (0x00007fd57eb1d000)
        libmkl_sycl.so.3 => /workspace/intel/oneapi/mkl/2023.2.0/lib/intel64/libmkl_sycl.so.3 (0x00007fd55512c000)
        libOpenCL.so.1 => /workspace/intel/oneapi/compiler/2023.2.0/linux/lib/libOpenCL.so.1 (0x00007fd55511d000)
        libsvml.so => /workspace/intel/oneapi/compiler/2023.2.0/linux/compiler/lib/intel64_lin/libsvml.so (0x00007fd553b11000)
        libirng.so => /workspace/intel/oneapi/compiler/2023.2.0/linux/compiler/lib/intel64_lin/libirng.so (0x00007fd553600000)
        libimf.so => /workspace/intel/oneapi/compiler/2023.2.0/linux/compiler/lib/intel64_lin/libimf.so (0x00007fd55321b000)
        libintlc.so.5 => /workspace/intel/oneapi/compiler/2023.2.0/linux/compiler/lib/intel64_lin/libintlc.so.5 (0x00007fd553a9c000)
        libsycl.so.6 => /workspace/intel/oneapi/compiler/2023.2.0/linux/lib/libsycl.so.6 (0x00007fd552f36000)
        ...
```

### Use SYCL code

Using SYCL code in an C++ application is also possible. The example below shows how to invoke SYCL codes. You need to explicitly pass `-fsycl` into `CMAKE_CXX_FLAGS`.

**example-usm.cpp**

[//]: # (marker_cppsdk_sample_usm)
[//]: # (marker_cppsdk_sample_usm)

**CMakeLists.txt**

[//]: # (marker_cppsdk_cmake_usm)
[//]: # (marker_cppsdk_cmake_usm)

### Customize DPC++ kernels

Intel® Extension for PyTorch\* provides its C++ dynamic library to allow users to implement custom DPC++ kernels to run on the XPU device. Refer to the [DPC++ extension](./features/DPC++_Extension.md) for details.


## Intel® AI Reference Models

Use cases that have already been optimized by Intel engineers are available at [Intel® AI Reference Models](https://github.com/IntelAI/models/tree/v2.12.0) (former Model Zoo). A number of PyTorch use cases for benchmarking are also available in the [Use Cases](https://github.com/IntelAI/models/tree/v2.12.0#use-cases) section. Models verified on Intel GPUs are marked in the `Model Documentation` column. You can get performance benefits out-of-the-box by simply running scripts in the Intel® AI Reference Models.
