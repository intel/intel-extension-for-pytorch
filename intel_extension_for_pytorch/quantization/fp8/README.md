# FP8 User API overview

We refer to implementation of [TransformerEngine](https://github.com/NVIDIA/TransformerEngine), including features like `fp8_autocast` and `Delayed scaling`, etc. At the current stage, only `FP8Linear` is enabled for both inference and training.

## FP8 data type support

FP8 is a natural evolution beyonds 16-bit formats common in modern processors. The primary motivation of the format is to accelerate deep learning training and inference, by enabling smaller and more power efficient math pipelines, as well as reducing memory bandwidth pressure with ensured accuracy.

Intel® Extension for PyTorch\* FP8 support only focuses on two formats of FP8: E4M3 (4-bit exponent and 3-bit mantissa) and E5M2 (5-bit exponent and 2-bit mantissa) formats, proposed by the Nvidia-Arm-Intel [joint paper](https://arxiv.org/pdf/2209.05433.pdf).

## FP8 Recipe

Default FP8 recipe is `DelayedScaling`, choosing a scaling factor based on the maximums of absolute values seen in some numbers of previous iterations. This enables full performance of the FP8 computation, but requires storing history of maximums as additional parameters of the FP8 operators.

In the `DelayedScaling` recipe, `fp8_format` contains three formats. `HYBRID` is the default format:

E4M3: All FP8 tensors are in E4M3 format.

E5M2: All FP8 tensors are in E5M2 format.

HYBRID: FP8 tensors in the forward pass are in E4M3 format, FP8 tensors in the backward pass are in E5M2 format.

## FP8 Usage Example

```python
from intel_extension_for_pytorch.quantization.fp8 import (
    fp8_autocast,
    DelayedScaling,
    Format,
    prepare_fp8,
)
```

### Inference

#### Convert Model

Use `prepare_fp8` to convert modules to FP8 modules (e.g, convert `nn.Linear` to `FP8Linear`) in the model:

```python
fp8_model = prepare_fp8(model)
```

#### Perform Calibration

FP8 calibration is performed in FP32 mode, by using `fp8_autocast` with `enabled=False` and `calibrating=True`. It allows collecting statistics such as a max and scale data of FP8 tensors without FP8 enabled.

```python
with fp8_autocast(enabled=False, calibrating=True, fp8_recipe=DelayedScaling(fp8_format=Format.E4M3), device="cpu"):
    output = fp8_model(input)
```

Then use `fp8_model.state_dict()` to save FP8 model. FP8 related information is saved by `get_extra_state`, which will be called in `state_dict()`.

```python
torch.save(fp8_model.state_dict(), "fp8_model.pt")
```

#### Load and Run FP8 Model

Use `model.load_state_dict()` to load FP8 model for inference. FP8 related information is loaded by `set_extra_state`, which will be called in `load_state_dict()`.

```python
fp8_model_with_calibration = prepare_fp8(model)
fp8_model_with_calibration.load_state_dict(torch.load("fp8_model.pt"))
fp8_model_with_calibration.eval()
```

Use context manager `fp8_autocast` to run FP8 model:

```python
with fp8_autocast(enabled=True, calibrating=False, fp8_recipe=DelayedScaling(fp8_format=Format.E4M3), device="cpu"):
    output = fp8_model_with_calibration(input)
```

### Training

#### Convert Model

Use `prepare_fp8` to convert modules to FP8 modules (e.g, convert `nn.Linear` to `FP8Linear`) in the model:

```python
fp8_model, ipex_optimizer = prepare_fp8(model, optimizer)
```

#### Run FP8 Model

Use context manager `fp8_autocast` to run FP8 model:

```python
fp8_model.train()
with fp8_autocast(enabled=True, fp8_recipe=DelayedScaling(fp8_format=Format.E4M3), device="cpu"):
    out = fp8_model(input)
    ipex_optimizer.zero_grad()
    out.mean().backward()
    ipex_optimizer.step()
```
