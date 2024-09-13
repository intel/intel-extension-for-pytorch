Transformers Optimization Frontend API
======================================

The new API function, `ipex.llm.optimize`, is designed to optimize transformer-based models within frontend Python modules, with a particular focus on Large Language Models (LLMs). It provides optimizations for both model-wise and content-generation-wise. You just need to invoke the `ipex.llm.optimize` function instead of the `ipex.optimize` function to apply all optimizations transparently.

This API currently works for inference workloads. Support for training is undergoing. Currently, this API supports certain models. Supported model list can be found at [Overview](../llm.html#optimized-models).

API documentation is available at [API Docs page](../api_doc.html#ipex.llm.optimize).

## Pseudocode of Common Usage Scenarios

The following sections show pseudocode snippets to invoke Intel® Extension for PyTorch\* APIs to work with LLMs. Complete examples can be found at [the Example directory](https://github.com/intel/intel-extension-for-pytorch/tree/v2.1.30%2Bxpu/examples/gpu/inference/python/llm).

### FP16

``` python
import torch
import intel_extension_for_pytorch as ipex
import transformers


device = "xpu"
model= transformers.AutoModelForCausalLM(model_name_or_path).eval().to(device)

amp_dtype = torch.float16 
model = ipex.llm.optimize(model.eval(), dtype=amp_dtype, device=device, inplace=True)

# inference with model.generate()
...
```

### SmoothQuant

Supports INT8.

#### Imperative mode

``` python
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

#### TorchScript Mode

``` python

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

### Distributed Inference with DeepSpeed

Distributed inference can be performed with `DeepSpeed`. Based on original Intel® Extension for PyTorch\* scripts, the following code changes are required.

Check Distributed Examples in [LLM example](https://github.com/intel/intel-extension-for-pytorch/tree/v2.1.30%2Bxpu/examples/gpu/inference/python/llm) for complete codes.





