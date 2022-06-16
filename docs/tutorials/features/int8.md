Intel® Extension for PyTorch\* optimizations for quantization
=============================================================

The quantization functionality in Intel® Extension for PyTorch\* currently only supports post-training quantization. This tutorial introduces how the quantization works in the Intel® Extension for PyTorch\* side.

We fully utilize Pytorch quantization components as much as possible, such as PyTorch [Observer method](https://pytorch.org/docs/1.11/quantization-support.html#torch-quantization-observer). To make a PyTorch user be able to easily use the quantization API, API for quantization in Intel® Extension for PyTorch\* is very similar to those in PyTorch. Intel® Extension for PyTorch\* quantization supports a default recipe to decide which operators should be quanized or not. This brings a satisfying performance and accuracy tradeoff.

## Static Quantization

```python
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert
```

### Define QConfig

Define a **QConfig** which set the activation and weight's observer methond:

```python
from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
qconfig = QConfig(activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
```

Note: we fully use of PyTorch [observer methonds](https://pytorch.org/docs/stable/quantization-support.html#torch-quantization-observer), so you can use a different PyTorch obsever methond to define the [QConfig](https://pytorch.org/docs/1.11/generated/torch.quantization.qconfig.QConfig.html). For weight observer, we only support **torch.qint8** dtype now.

**Suggestion**: For activation observer, if your set **qscheme** with **torch.per_tensor_affine**, the dtype prefer to **torch.quint8**, if you set the **qscheme** with **torch.per_tensor_symmetric**, the dtype prefer to **torch.qint8**. For weight observer, setting **qscheme** to **torch.per_channel_symmetric** can get a better accuracy,

### Prepare Model and Do Calibration

```python
# prepare model, do conv+bn folding, and init model quant_state.
user_model = ...
user_model.eval()
example_inputs = ..
prepared_model = prepare(user_model, qconfig, example_inputs=example_inputs, inplace=False)

for data in calibration_data_set:
    prepared_model(x)

# Optional, if you want to tuning(performance or accuracy), you can save the qparams as json file which
# including the quantization state, such as scales, zero points and inference dtype.
# And then you can change the json file's settings, loading the changed json file
# to model which will override the model's original quantization's settings.  
#  
# prepared_model.save_qconf_summary(qconf_summary = "configure.json")
# prepared_model.load_qconf_summary(qconf_summary = "configure.json")
```

### Convert to Static Quantized Model and Deploy

```python
# make sure the example_inputs's size is same as the real input's size 
convert_model = convert(prepared_model)
with torch.no_grad():
    traced_model = torch.jit.trace(convert_model, example_input)
    traced_model = torch.jit.freeze(traced_model)
# for inference 
y = traced_model(x)

# or save the model to deploy

# traced_model.save("quantized_model.pt")
# quantized_model = torch.jit.load("quantized_model.pt")
# quantized_model = torch.jit.freeze(quantized_model.eval())
# ...
```

## Dynamic Quantization

```python
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert
```

### Define QConfig

```python
from torch.ao.quantization import MinMaxObserver, PlaceholderObserver, QConfig
dynamic_qconfig = QConfig(
        activation = PlaceholderObserver.with_args(dtype=torch.float, compute_dtype=torch.quint8),
        weight = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
```

Note: For weight observer, it only support dtype **torch.qint8**, and the qscheme only can be **torch.per_tensor_symmetric** or **torch.per_channel_symmetric**.

### Prepare Model

```python
prepared_model = prepare(user_model, dynamic_qconfig, example_inputs=example_inputs)
```

## Convert to Dynamic Quantized Model and Deploy

```python
# make sure the example_inputs's size is same as the real input's size
convert_model = convert(prepared_model)
# Optional: convert the model to traced model
#with torch.no_grad():
#    traced_model = torch.jit.trace(convert_model, example_input)
#    traced_model = torch.jit.freeze(traced_model)

# or save the model to deploy
# traced_model.save("quantized_model.pt")
# quantized_model = torch.jit.load("quantized_model.pt")
# quantized_model = torch.jit.freeze(quantized_model.eval())
# ...
# for inference 
y = convert_model(x)

```

Note: we only support the following ops to do dynamic quantization:
- torch.nn.Linear
- torch.nn.LSTM
- torch.nn.GRU
- torch.nn.LSTMCell
- torch.nn.RNNCell
- torch.nn.GRUCell
