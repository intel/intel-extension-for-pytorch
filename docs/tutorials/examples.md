Examples
========

**Note:** For examples on CPU, please check [here](../../../cpu/latest/tutorials/examples.html).

## Training

### Single-instance Training

#### Code Changes Highlight

There are only a few lines of code change required to use Intel® Extension for PyTorch\* on training, as shown:
1. `ipex.optimize` function applies optimizations against the model object, as well as an optimizer object.
2.  Use Auto Mixed Precision (AMP) with BFloat16 data type.
3.  Convert input tensors, loss criterion and model to XPU.

The complete examples for Float32 and BFloat16 training on single-instance are illustrated in the sections.

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

#### Complete - Float32 Example

[//]: # (marker_train_single_fp32_complete)
[//]: # (marker_train_single_fp32_complete)

#### Complete - BFloat16 Example

[//]: # (marker_train_single_bf16_complete)
[//]: # (marker_train_single_bf16_complete)

## Inference

The `optimize` function of Intel® Extension for PyTorch\* applies optimizations to the model, bringing additional performance boosts. For both computer vision workloads and NLP workloads, we recommend applying the `optimize` function against the model object.

### Float32

#### Imperative Mode

##### Resnet50

[//]: # (marker_inf_rn50_imp_fp32)
[//]: # (marker_inf_rn50_imp_fp32)

##### BERT

[//]: # (marker_inf_bert_imp_fp32)
[//]: # (marker_inf_bert_imp_fp32)

#### TorchScript Mode

We recommend you take advantage of Intel® Extension for PyTorch\* with [TorchScript](https://pytorch.org/docs/stable/jit.html) for further optimizations.

##### Resnet50

[//]: # (marker_inf_rn50_ts_fp32)
[//]: # (marker_inf_rn50_ts_fp32)

##### BERT

[//]: # (marker_inf_bert_ts_fp32)
[//]: # (marker_inf_bert_ts_fp32)

### BFloat16

Similar to running with Float32, the `optimize` function also works for BFloat16 data type. The only difference is setting `dtype` parameter to `torch.bfloat16`.
We recommend using Auto Mixed Precision (AMP) with BFloat16 data type.


#### Imperative Mode

##### Resnet50

[//]: # (marker_inf_rn50_imp_bf16)
[//]: # (marker_inf_rn50_imp_bf16)

##### BERT

[//]: # (marker_inf_bert_imp_bf16)
[//]: # (marker_inf_bert_imp_bf16)

#### TorchScript Mode

We recommend you take advantage of Intel® Extension for PyTorch\* with [TorchScript](https://pytorch.org/docs/stable/jit.html) for further optimizations.

##### Resnet50

[//]: # (marker_inf_rn50_ts_bf16)
[//]: # (marker_inf_rn50_ts_bf16)

##### BERT

[//]: # (marker_inf_bert_ts_bf16)
[//]: # (marker_inf_bert_ts_bf16)

### Float16

Similar to running with Float32, the `optimize` function also works for Float16 data type. The only difference is setting `dtype` parameter to `torch.float16`.
We recommend using Auto Mixed Precision (AMP) with Float16 data type.

#### Imperative Mode

##### Resnet50

[//]: # (marker_inf_rn50_imp_fp16)
[//]: # (marker_inf_rn50_imp_fp16)

##### BERT

[//]: # (marker_inf_bert_imp_fp16)
[//]: # (marker_inf_bert_imp_fp16)

#### TorchScript Mode

We recommend you take advantage of Intel® Extension for PyTorch\* with [TorchScript](https://pytorch.org/docs/stable/jit.html) for further optimizations.

##### Resnet50

[//]: # (marker_inf_rn50_ts_fp16)
[//]: # (marker_inf_rn50_ts_fp16)

##### BERT

[//]: # (marker_inf_bert_ts_fp16)
[//]: # (marker_inf_bert_ts_fp16)

### INT8

We recommend to use TorchScript for INT8 model due to it has wider support for models. Moreover, TorchScript mode would auto enable our optimizations. For TorchScript INT8 model, inserting observer and model quantization is achieved through `prepare_jit` and `convert_jit` separately. Calibration process is required for collecting statistics from real data. After conversion, optimizations like operator fusion would be auto enabled.

[//]: # (marker_int8_static)
[//]: # (marker_int8_static)

### torch.xpu.optimize

`torch.xpu.optimize` is an alternative of `ipex.optimize` in Intel® Extension for PyTorch\*, to provide identical usage for XPU device only. The motivation of adding this alias is to unify the coding style in user scripts base on torch.xpu modular. Refer to below example for usage.

#### ResNet50 FP32 imperative inference

[//]: # (marker_inf_rn50_imp_fp32_alt)
[//]: # (marker_inf_rn50_imp_fp32_alt)

## C++

Intel® Extension for PyTorch\* provides its C++ dynamic library to allow users to implement custom DPC++ kernels to run on the XPU device. Refer to the [DPC++ extension](./features/DPC++_Extension.md) for the details.

## Model Zoo

Use cases that had already been optimized by Intel engineers are available at [Model Zoo for Intel® Architecture](https://github.com/IntelAI/models/tree/v2.9.0). A bunch of PyTorch use cases for benchmarking are also available on the [GitHub page](https://github.com/IntelAI/models/tree/v2.9.0#use-cases). Models verified on Intel dGPUs are marked in `Model Documentation` Column. You can get performance benefits out-of-box by simply running scipts in the Model Zoo.
