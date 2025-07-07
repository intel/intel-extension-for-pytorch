Examples
========

These examples will help you get started using Intel® Extension for PyTorch\*
with Intel GPUs.

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
2. Use the `ipex.optimize` function for additional performance boost, which applies optimizations against the model object, as well as an optimizer object.
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
# Move model and loss criterion to xpu before calling ipex.optimize()
model = model.to("xpu")
criterion = criterion.to("xpu")

# For Float32
model, optimizer = ipex.optimize(model, optimizer=optimizer)
# For BFloat16
model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)
...
dataloader = ...
for (input, target) in dataloader:
    input = input.to("xpu")
    target = target.to("xpu")
    optimizer.zero_grad()
    # For Float32
    output = model(input)

    # For BFloat16
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
        output = model(input)

    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
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


##### Imperative Mode

###### Resnet50

[//]: # (marker_inf_rn50_imp_bf16)
[//]: # (marker_inf_rn50_imp_bf16)

###### BERT

[//]: # (marker_inf_bert_imp_bf16)
[//]: # (marker_inf_bert_imp_bf16)


##### Imperative Mode

###### Resnet50

[//]: # (marker_inf_rn50_imp_fp16)
[//]: # (marker_inf_rn50_imp_fp16)

###### BERT

[//]: # (marker_inf_bert_imp_fp16)
[//]: # (marker_inf_bert_imp_fp16)


#### torch.xpu.optimize

The `torch.xpu.optimize` function is an alternative to `ipex.optimize` in Intel® Extension for PyTorch\*, and provides identical usage for XPU devices only. The motivation for adding this alias is to unify the coding style in user scripts base on `torch.xpu` modular. Refer to the example below for usage.

[//]: # (marker_inf_rn50_imp_fp32_alt)
[//]: # (marker_inf_rn50_imp_fp32_alt)


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
