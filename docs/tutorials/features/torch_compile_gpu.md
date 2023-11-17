torch.compile for GPU
=====================

## Introduction

Intel® Extension for PyTorch\* now empowers users to seamlessly harness graph compilation capabilities for optimal PyTorch model performance on Intel GPU via the flagship [torch.compile](https://pytorch.org/docs/stable/generated/torch.compile.html#torch-compile) API through the default "inductor" backend ([TorchInductor](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747/1)). The Triton compiler has been the core of the Inductor codegen supporting various accelerator devices. Intel has extended TorchInductor by adding Intel GPU support to Triton. Additionally, post-op fusions for convolution and matrix multiplication, facilitated by oneDNN fusion kernels, contribute to enhanced efficiency for computational intensive operations. Leveraging these features is as simple as using the default "inductor" backend, making it easier than ever to unlock the full potential of your PyTorch models on Intel GPU platforms.

`torch.compile` for GPU is an experimental feature and available from 2.1.10. So far, the feature is functional on Intel® GPU Max Series.

### Inferenece with torch.compile

```python
import torch
import intel_extension_for_pytorch

# create model
model = SimpleNet().to("xpu")

# compile model
compiled_model = torch.compile(model, options={"freezing": True})

# inference main
input = torch.rand(64, 3, 224, 224, device=torch.device("xpu"))
with torch.no_grad():
    with torch.xpu.amp.autocast(dtype=torch.float16):
        output = compiled_model(input)
```

### Training with torch.compile

```python
import torch
import intel_extension_for_pytorch

# create model and optimizer
model = SimpleNet().to("xpu")
optimizer = torch.optim.SGD(model.parameters(), lr=..., momentum=..., weight_decay=...)

# compile model
compiled_model = torch.compile(model)

# training main
input = torch.rand(64, 3, 224, 224, device=torch.device("xpu"))
with torch.xpu.amp.autocast(dtype=torch.bfloat16):
    output = compiled_model(input)
    loss = loss_function(output)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```
