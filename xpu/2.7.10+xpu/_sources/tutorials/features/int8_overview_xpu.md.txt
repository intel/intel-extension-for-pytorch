Intel® Extension for PyTorch\* Optimizations for Quantization [GPU]
===================================================================

Intel® Extension for PyTorch\* currently supports imperative mode and TorchScript mode for post-training static quantization on GPU. This section illustrates the quantization workflow on Intel GPUs.

The overall view is that our usage follows the API defined in official PyTorch. Therefore, only small modification like moving model and data to GPU with `to('xpu')` is required. We highly recommend using the TorchScript for quantizing models. With graph model created via TorchScript, optimization like operator fusion (e.g. `conv_relu`) is enabled automatically. This delivers the best performance for int8 workloads.

## Imperative Mode
```python
import torch
import intel_extension_for_pytorch

# Define model
model = Model().to("xpu")
model.eval()
modelImpe = torch.quantization.QuantWrapper(model)

# Define QConfig
qconfig = torch.quantization.QConfig(activation=torch.quantization.observer.MinMaxObserver .with_args(qscheme=torch.per_tensor_symmetric),
    weight=torch.quantization.default_weight_observer)  # weight could also be perchannel

modelImpe.qconfig = qconfig

# Prepare model for inserting observer
torch.quantization.prepare(modelImpe, inplace=True)

# Calibration to obtain statistics for Observer
for data in calib_dataset:
    modelImpe(data)

# Convert model to create a quantized module
torch.quantization.convert(modelImpe, inplace=True)

# Inference
modelImpe(inference_data)
```

Imperative mode usage follows official Pytorch and more details can be found at [PyTorch doc](https://pytorch.org/docs/1.9.1/quantization.html).  

Defining the quantized config (QConfig) for model is the first stage of quantization. Per-tensor quantization is supported for activation quantization, while both per-tensor and per-channel are supported for weight quantization. Weight can be quantized to `int8` data type only. As for activation quantization, both symmetric and asymmetric are supported. Also, both `uint8` and `int8` data types are supported.

If the best performance is desired, we recommend using the `symmetric+int8` combination. Other configuration may have lower performance due to the existence of `zero_point`.

After defining a QConfig, the `prepare` function is used to insert observer in models. The observer is responsible for collecting statistics for quantization. A calibration stage is needed for observer to collect info. 

After calibration, function `convert` would quantize weight in module and swap FP32 module to quantized ones. Then, an int8 module is created. Be free to use it for inference.

## TorchScript Mode
```python
import torch
import intel_extension_for_pytorch
from torch.quantization.quantize_jit import (
    convert_jit,
    prepare_jit,
)

# Define model
model = Model().to("xpu")
model.eval()

# Generate a ScriptModule
modelJit = torch.jit.trace(model, example_input) # or torch.jit.script(model)

# Defin QConfig
qconfig = torch.quantization.QConfig(
    activation=torch.quantization.observer.MinMaxObserver.with_args(
        qscheme=qscheme,
        reduce_range=False,
        dtype=dtype
    ),
    weight=torch.quantization.default_weight_observer
)

# Prepare model for inserting observer
modelJit = prepare_jit(modelJit, {'': qconfig}, inplace=True)

# Calibration 
for data in calib_dataset:
    modelJit(data)

# Convert model to quantized one
modelJit = convert_jit(modelJit)

# Warmup to fully trigger fusion patterns
for i in range(5):
    modelJit(warmup_data) 
# Inference
modelJit(inference_data)

# Debug
print(modelJit.graph_for(inference_dta))
```

We need to define `QConfig`` for TorchScript module, use `prepare_jit` for inserting observer and use `convert_jit` for replacing FP32 modules.

Before `prepare_jit`, create a ScriptModule using `torch.jit.script` or `torch.jit.trace`. `jit.trace` is recommended for capable of catching the whole graph in most scenarios.

Fusion operations like `conv_unary`, `conv_binary`, `linear_unary` (e.g. `conv_relu`, `conv_sum_relu`) are automatically enabled after model conversion (`convert_jit`). A warmup stage is required for bringing the fusion into effect. With the benefit from fusion, ScriptModule can deliver better performance than eager mode. Hence, we recommend using ScriptModule as for performance consideration.

`modelJit.graph_for(input)` is useful to dump the inference graph and other graph related information for performance analysis.

