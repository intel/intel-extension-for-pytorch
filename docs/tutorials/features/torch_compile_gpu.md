torch.compile for GPU (Beta)
============================

# Introduction

Intel® Extension for PyTorch\* now empowers users to seamlessly harness graph compilation capabilities for optimal PyTorch model performance on Intel GPU via the flagship [torch.compile](https://pytorch.org/docs/stable/generated/torch.compile.html#torch-compile) API through the default "inductor" backend ([TorchInductor](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747/1)). The Triton compiler has been the core of the Inductor codegen supporting various accelerator devices. Intel has extended TorchInductor by adding Intel GPU support to Triton. Additionally, post-op fusions for convolution and matrix multiplication, facilitated by oneDNN fusion kernels, contribute to enhanced efficiency for computational intensive operations. Leveraging these features is as simple as using the default "inductor" backend, making it easier than ever to unlock the full potential of your PyTorch models on Intel GPU platforms.


# Required Dependencies

**Verified version**:
- `torch` : v2.3
- `intel_extension_for_pytorch` : v2.3
- `triton` : >= v3.0.0

Install [Intel® oneAPI Base Toolkit 2024.2.1](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html).

Follow [Intel® Extension for PyTorch\* Installation](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/) to install `torch` and `intel_extension_for_pytorch` firstly.

Triton could be directly installed using the following command:

```Bash
pip install --pre pytorch-triton-xpu==3.0.0+1b2f15840e --index-url https://download.pytorch.org/whl/nightly/xpu
```

Remember to activate the oneAPI basekit by following commands.

```bash
# {dpcpproot} is the location for dpcpp ROOT path and it is where you installed oneAPI DPCPP, usually it is /opt/intel/oneapi/compiler/latest or ~/intel/oneapi/compiler/latest
source {dpcpproot}/env/vars.sh
```


# Example Usage

## Inferenece with torch.compile

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

## Training with torch.compile

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
