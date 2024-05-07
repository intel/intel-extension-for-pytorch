torch.compile for GPU (Beta)
============================

## Introduction

Intel® Extension for PyTorch\* now empowers users to seamlessly harness graph compilation capabilities for optimal PyTorch model performance on Intel GPU via the flagship [torch.compile](https://pytorch.org/docs/stable/generated/torch.compile.html#torch-compile) API through the default "inductor" backend ([TorchInductor](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747/1)). The Triton compiler has been the core of the Inductor codegen supporting various accelerator devices. Intel has extended TorchInductor by adding Intel GPU support to Triton. Additionally, post-op fusions for convolution and matrix multiplication, facilitated by oneDNN fusion kernels, contribute to enhanced efficiency for computational intensive operations. Leveraging these features is as simple as using the default "inductor" backend, making it easier than ever to unlock the full potential of your PyTorch models on Intel GPU platforms.

**Note**: `torch.compile` for GPU is a beta feature and available from 2.1.10. So far, the feature is functional on Intel® Data Center GPU Max Series.

## Required Dependencies

**Verified version**:
- `torch` : v2.1.0
- `intel_extension_for_pytorch` : > v2.1.10
- `triton` : [v2.1.0](https://github.com/intel/intel-xpu-backend-for-triton/releases/tag/v2.1.0) with Intel® XPU Backend for Triton* backend enabled.

Follow [Intel® Extension for PyTorch\* Installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.1.30%2bxpu) to install `torch` and `intel_extension_for_pytorch` firstly.

Then install [Intel® XPU Backend for Triton\* backend](https://github.com/intel/intel-xpu-backend-for-triton) for `triton` package. You may install it via prebuilt wheel package or build it from the source. We recommend installing via prebuilt package:

- Download the wheel package from [release page](https://github.com/intel/intel-xpu-backend-for-triton/releases). Note that you don't need to install the LLVM release manually. 
- Install the wheel package by `pip install`. Note that this wheel package is a `triton` package with Intel GPU support, so you don't need to `pip install triton` again.
  
```Bash
python -m pip install --force-reinstall  triton-2.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

Please follow the [Intel® XPU Backend for Triton\* Installation](https://github.com/intel/intel-xpu-backend-for-triton?tab=readme-ov-file#setup-guide) for more detailed installation steps. 

Note that if you install `triton` using `make triton` command inside PyTorch\* repo, the installed `triton` does not compile with Intel GPU support by default, you will need to manually set `TRITON_CODEGEN_INTEL_XPU_BACKEND=1` for enabling Intel GPU support. In addition, for building from the source via the `triton` [repo](https://github.com/openai/triton.git), the commit needs to be pinned at a tested [triton commit](https://github.com/intel/intel-xpu-backend-for-triton/blob/main/triton_hash.txt). Please follow the [Intel® XPU Backend for Triton\* Installation #build from the source](https://github.com/intel/intel-xpu-backend-for-triton?tab=readme-ov-file#option-2-build-from-the-source) section for more information about build `triton` package from the source. 


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

