# Intel® Extension for PyTorch\*

Intel® Extension for PyTorch* extends [PyTorch\*](https://github.com/pytorch/pytorch) with up-to-date features and optimizations for an extra performance boost on Intel hardware. It is a heterogeneous, high-performance deep-learning implementation for both CPU and XPU. XPU is a user visible device that is a counterpart of the well-known CPU and CUDA in the PyTorch* community. XPU represents an Intel-specific kernel and graph optimizations for various “concrete” devices. The XPU runtime will choose the actual device when executing AI workloads on the XPU device. The default selected device is Intel GPU. This release introduces specific XPU solution optimizations and gives PyTorch end-users up-to-date features and optimizations on Intel Graphics cards.

Intel® Extension for PyTorch* on XPU provides aggressive optimizations for both eager mode and graph mode. Graph mode in PyTorch* normally yields better performance from optimization techniques such as operation fusion, and Intel® Extension for PyTorch* amplifies them with more comprehensive graph optimizations. This extension can be loaded as a Python module for Python programs or linked as a C++ library for C++ programs. In Python scripts users can enable it dynamically by ``import intel_extension_for_pytorch``. To execute AI workloads on XPU, the input tensors and models must be converted to XPU beforehand by ``input = input.to("xpu")`` and ``model = model.to("xpu")``.

More detailed XPU tutorials are available at [Intel® Extension for PyTorch* online document website](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/).

## Installation

Begin by installing optimized PyTorch\*:

```bash
python -m pip install torch==1.10.0a0 -f https://developer.intel.com/ipex-whl-stable-xpu
```

then install Intel® Extension for PyTorch\*:

```bash
python -m pip install intel_extension_for_pytorch==1.10.200+gpu -f https://developer.intel.com/ipex-whl-stable-xpu
```

More installation methods can be found at [Installation Guide](./docs/tutorials/installation.md).

## Getting Started

Only a few code changes are required to use Intel® Extension for PyTorch\* on XPU. Both PyTorch imperative mode and TorchScript mode are supported. Import the Intel® Extension for PyTorch\* package and apply its optimize function against the model object. If it is a training workload, the optimize function also needs to be applied against the optimizer object.

The following code snippet shows an inference code with FP32 data type. More examples, including training and C++ examples, are available at [Example page](./docs/tutorials/examples.md).

```python
import torch
import torchvision.models as models
########## code changes ##########
import intel_extension_for_pytorch
########## code changes ##########

model = models.resnet50(pretrained=True)
model.eval()
data = torch.rand(1, 3, 224, 224)

#################### code changes ####################
model = model.to("xpu")
data = data.to("xpu")
model = torch.xpu.optimize(model, dtype=torch.float32)
#################### code changes ####################

with torch.no_grad():
  model(data)
```

### End to End model performance measurement

To get accurate End to End model performance, call `torch.xpu.synchronize()` in model script right before calculating elapsed time. This API waits for the completion of all executing GPU kernels on the device, so calculating the elapsed time after the call covers both CPU and GPU execution time.

#### Training Model Example

```bash
  # compute gradient and do SGD step
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

  # sync for time measurement
  torch.xpu.synchronize()

  # measure elapsed time
  end_Batch = time.time()
  batch_time.update(time.time() - start_Batch)
  iter_time.append(end_Batch - start_Batch)
```

#### Inference Model Example

```bash
with torch.inference_mode():
  # compute output
  output = model(input)

  # sync for time measurement
  torch.xpu.synchronize()

  # measure elapsed time
  end = time.time()
  batch_time.update(end - start)
  iter_time.append(end - start)
```

## License

_Apache License_, Version _2.0_. As found in [LICENSE](https://github.com/intel/intel-extension-for-pytorch/blob/xpu-master/LICENSE) file.

## Security

See Intel's [Security Center](https://www.intel.com/content/www/us/en/security-center/default.html)
for information on how to report a potential security issue or vulnerability.

See also: [Security Policy](SECURITY.md)
