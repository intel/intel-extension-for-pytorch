Examples
========

These examples will guide you through using the Intel® Extension for PyTorch\* on Intel CPUs.

You can also refer to the [Features](./features.rst) section to get the examples and usage instructions related to particular features.

The source code for these examples, as well as the feature examples, can be found in the GitHub source tree under the [examples](https://github.com/intel/intel-extension-for-pytorch/tree/main/examples/cpu) directory.

- [Python](#python) examples demonstrate usage of Python APIs:

  - [Training](#training)
  - [Inference](#inference)

- [C++](#c) examples demonstrate usage of C++ APIs
- [Intel® AI Reference Models](#intel-ai-reference-models) provide out-of-the-box use cases, demonstrating the performance benefits achievable with Intel Extension for PyTorch\*

**Prerequisites**:
Before running these examples, please note the following:

- Examples using the BFloat16 data type require machines with the  Intel® Advanced Vector Extensions 512 (Intel® AVX-512) BF16 and Intel® Advanced Matrix Extensions (Intel® AMX) BF16 instruction sets.


## Python

### Training

#### Single-instance Training

To use Intel® Extension for PyTorch\* on training, you need to make the following changes in your code:

1. Import `intel_extension_for_pytorch` as `ipex`.
2. Invoke the `ipex.optimize` function to apply optimizations against the model and optimizer objects, as shown below:


```python
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
# Invoke the code below to enable beta feature torch.compile
model = torch.compile(model, backend="ipex")
...
optimizer.zero_grad()
output = model(data)
...
```

Below you can find complete code examples demonstrating how to use the extension on training for different data types:

##### Float32

**Note:** You need to install `torchvision` Python package to run the following example.

[//]: # (marker_train_single_fp32_complete)
[//]: # (marker_train_single_fp32_complete)

##### BFloat16

**Note:** You need to install `torchvision` Python package to run the following example.

[//]: # (marker_train_single_bf16_complete)
[//]: # (marker_train_single_bf16_complete)

#### Distributed Training

Distributed training with PyTorch DDP is accelerated by oneAPI Collective Communications Library Bindings for Pytorch\* (oneCCL Bindings for Pytorch\*). The extension supports FP32 and BF16 data types. More detailed information and examples are available at the [Github repo](https://github.com/intel/torch-ccl).

**Note:** You need to install `torchvision` Python package to run the following example.

[//]: # (marker_train_ddp_complete)
[//]: # (marker_train_ddp_complete)

### Inference

The `optimize` function of Intel® Extension for PyTorch\* applies optimizations to the model, bringing additional performance boosts. For both computer vision workloads and NLP workloads, we recommend applying the `optimize` function against the model object.

#### Float32

##### Eager Mode

###### Resnet50

**Note:** You need to install `torchvision` Python package to run the following example.

[//]: # (marker_inf_rn50_imp_fp32)
[//]: # (marker_inf_rn50_imp_fp32)

###### BERT

**Note:** You need to install `transformers` Python package to run the following example.

[//]: # (marker_inf_bert_imp_fp32)
[//]: # (marker_inf_bert_imp_fp32)

##### TorchScript Mode

We recommend using Intel® Extension for PyTorch\* with [TorchScript](https://pytorch.org/docs/stable/jit.html) for further optimizations.

###### Resnet50

**Note:** You need to install `torchvision` Python package to run the following example.

[//]: # (marker_inf_rn50_ts_fp32)
[//]: # (marker_inf_rn50_ts_fp32)

###### BERT

**Note:** You need to install `transformers` Python package to run the following example.

[//]: # (marker_inf_bert_ts_fp32)
[//]: # (marker_inf_bert_ts_fp32)

##### TorchDynamo Mode (Beta, _NEW feature from 2.0.0_)

###### Resnet50

**Note:** You need to install `torchvision` Python package to run the following example.

[//]: # (marker_inf_rn50_dynamo_fp32)
[//]: # (marker_inf_rn50_dynamo_fp32)

###### BERT

**Note:** You need to install `transformers` Python package to run the following example.

[//]: # (marker_inf_bert_dynamo_fp32)
[//]: # (marker_inf_bert_dynamo_fp32)

**Note:** In TorchDynamo mode, since the native PyTorch operators like `aten::convolution` and `aten::linear` are well supported and optimized in `ipex` backend, we need to disable weights prepacking by setting `weights_prepack=False` in `ipex.optimize()`.

#### BFloat16

The `optimize` function works for both Float32 and BFloat16 data type. For BFloat16 data type, set the `dtype` parameter to `torch.bfloat16`.
We recommend using Auto Mixed Precision (AMP) with BFloat16 data type.

##### Eager Mode

###### Resnet50

**Note:** You need to install `torchvision` Python package to run the following example.

[//]: # (marker_inf_rn50_imp_bf16)
[//]: # (marker_inf_rn50_imp_bf16)

###### BERT

**Note:** You need to install `transformers` Python package to run the following example.

[//]: # (marker_inf_bert_imp_bf16)
[//]: # (marker_inf_bert_imp_bf16)

##### TorchScript Mode

We recommend using Intel® Extension for PyTorch\* with [TorchScript](https://pytorch.org/docs/stable/jit.html) for further optimizations.

###### Resnet50

**Note:** You need to install `torchvision` Python package to run the following example.

[//]: # (marker_inf_rn50_ts_bf16)
[//]: # (marker_inf_rn50_ts_bf16)

###### BERT

**Note:** You need to install `transformers` Python package to run the following example.

[//]: # (marker_inf_bert_ts_bf16)
[//]: # (marker_inf_bert_ts_bf16)

##### TorchDynamo Mode (Beta, _NEW feature from 2.0.0_)

###### Resnet50

**Note:** You need to install `torchvision` Python package to run the following example.

[//]: # (marker_inf_rn50_dynamo_bf16)
[//]: # (marker_inf_rn50_dynamo_bf16)

###### BERT

**Note:** You need to install `transformers` Python package to run the following example.

[//]: # (marker_inf_bert_dynamo_bf16)
[//]: # (marker_inf_bert_dynamo_bf16)

#### Fast Bert (*Prototype*)

**Note:** You need to install `transformers` Python package to run the following example.

[//]: # (marker_feature_fastbert_bf16)
[//]: # (marker_feature_fastbert_bf16)

#### INT8

Starting from Intel® Extension for PyTorch\* 1.12.0, quantization feature supports both static and dynamic modes.

##### Static Quantization

###### Calibration

Please follow the steps below to perform calibration for static quantization:

1. Import `intel_extension_for_pytorch` as `ipex`.
2. Import `prepare` and `convert` from `intel_extension_for_pytorch.quantization`.
3. Instantiate a config object from `torch.ao.quantization.QConfig` to save configuration data during calibration.
4. Prepare model for calibration.
5. Perform calibration against dataset.
6. Invoke `ipex.quantization.convert` function to apply the calibration configure object to the fp32 model object to get an INT8 model.
7. Save the INT8 model into a `pt` file.

**Note:** You need to install `torchvision` Python package to run the following example.

[//]: # (marker_int8_static)
[//]: # (marker_int8_static)

###### Deployment

For deployment, the INT8 model is loaded from the local file and can be used directly for sample inference.

Follow the steps below:

1. Import `intel_extension_for_pytorch` as `ipex`.
2. Load the INT8 model from the saved file.
3. Run inference.

[//]: # (marker_int8_deploy)
[//]: # (marker_int8_deploy)

##### Dynamic Quantization

Please follow the steps below to perform dynamic quantization:

1. Import `intel_extension_for_pytorch` as `ipex`.
2. Import `prepare` and `convert` from `intel_extension_for_pytorch.quantization`.
3. Instantiate a config object from `torch.ao.quantization.QConfig` to save configuration data during calibration.
4. Prepare model for quantization.
5. Convert the model.
6. Run inference to perform dynamic quantization.
7. Save the INT8 model into a `pt` file.

**Note:** You need to install `transformers` Python package to run the following example.

[//]: # (marker_int8_dynamic)
[//]: # (marker_int8_dynamic)

### Large Language Model (LLM)

Intel® Extension for PyTorch\* provides dedicated optimization for running Large Language Models (LLM) faster.
A set of data types are supported for various scenarios, including FP32, BF16, Smooth Quantization INT8, Weight Only Quantization INT8/INT4 (prototype).

**Note:** You need to install `transformers==<VER_TRANSFORMERS>` Python package to run the following example.
In addition, you may need to log in your HuggingFace account to access the pretrained model files. 
Please refer to [HuggingFace login](https://huggingface.co/docs/huggingface_hub/quick-start#login).

#### FP32/BF16

[//]: # (marker_llm_optimize)
[//]: # (marker_llm_optimize)

#### Smooth Quantization INT8

The typical steps shown in the example are:

1. Calibration process: Run the example script specifying `--calibration`, along with other related arguments.
When the calibration process is completed, the quantization summary files would be generated.

2. Model inference process: Run the example script without specifying `--calibration`. In this process the quantized model 
will be generated via the original model and the quantization config and summary files, and will
generate results for the input prompt.

[//]: # (marker_llm_optimize_sq)
[//]: # (marker_llm_optimize_sq)

#### Weight Only Quantization INT8/INT4

[//]: # (marker_llm_optimize_woq)
[//]: # (marker_llm_optimize_woq)

**Note:** Please check [LLM Best Known Practice Page](https://github.com/intel/intel-extension-for-pytorch/tree/main/examples/cpu/inference/python/llm)
for detailed environment setup and LLM workload running instructions.

## C++

To work with libtorch, C++ library of PyTorch, Intel® Extension for PyTorch\* provides its C++ dynamic library as well. The C++ library is supposed to handle inference workload only, such as service deployment. For regular development, use the Python interface. Unlike using libtorch, no specific code changes are required. Compilation follows the recommended methodology with CMake. Detailed instructions can be found in [PyTorch tutorial](https://pytorch.org/tutorials/advanced/cpp_export.html#depending-on-libtorch-and-building-the-application).

During compilation, Intel optimizations will be activated automatically once C++ dynamic library of Intel® Extension for PyTorch\* is linked.

The example code below works for all data types.

**example-app.cpp**

[//]: # (marker_cppsdk_sample)
[//]: # (marker_cppsdk_sample)

**CMakeLists.txt**

[//]: # (marker_cppsdk_cmake)
[//]: # (marker_cppsdk_cmake)

**Command for compilation**

```bash
$ cd examples/cpu/inference/cpp
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=<LIBPYTORCH_PATH> ..
$ make
```

If *Found IPEX* is shown as with a dynamic library path, the extension had been linked into the binary. This can be verified with Linux command *ldd*.

```bash
$ cmake -DCMAKE_PREFIX_PATH=/workspace/libtorch ..
-- The C compiler identification is GNU XX.X.X
-- The CXX compiler identification is GNU XX.X.X
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
CMake Warning at /workspace/libtorch/share/cmake/Torch/TorchConfig.cmake:22 (message):
  static library kineto_LIBRARY-NOTFOUND not found.
Call Stack (most recent call first):
  /workspace/libtorch/share/cmake/Torch/TorchConfig.cmake:127 (append_torchlib_if_found)
  /workspace/libtorch/share/cmake/IPEX/IPEXConfig.cmake:84 (FIND_PACKAGE)
  CMakeLists.txt:4 (find_package)


-- Found Torch: /workspace/libtorch/lib/libtorch.so
-- Found IPEX: /workspace/libtorch/lib/libintel-ext-pt-cpu.so
-- Configuring done
-- Generating done
-- Build files have been written to: examples/cpu/inference/cpp/build

$ ldd example-app
        ...
        libtorch.so => /workspace/libtorch/lib/libtorch.so (0x00007f3cf98e0000)
        libc10.so => /workspace/libtorch/lib/libc10.so (0x00007f3cf985a000)
        libintel-ext-pt-cpu.so => /workspace/libtorch/lib/libintel-ext-pt-cpu.so (0x00007f3cf70fc000)
        libtorch_cpu.so => /workspace/libtorch/lib/libtorch_cpu.so (0x00007f3ce16ac000)
        ...
        libdnnl_graph.so.0 => /workspace/libtorch/lib/libdnnl_graph.so.0 (0x00007f3cde954000)
        ...
```

## Intel® AI Reference Models

Use cases that have already been optimized by Intel engineers are available at [Intel® AI Reference Models](https://github.com/IntelAI/models/tree/pytorch-r2.2.0-models) (former Model Zoo). A number of PyTorch use cases for benchmarking are also available in the [benchmarks](https://github.com/IntelAI/models/tree/pytorch-r2.2.0-models/benchmarks#pytorch-use-cases). You can get performance benefits out-of-the-box by simply running scripts in the Intel® AI Reference Models.

