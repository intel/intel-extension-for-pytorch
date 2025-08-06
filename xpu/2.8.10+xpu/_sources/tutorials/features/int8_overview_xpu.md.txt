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
