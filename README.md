# Intel® Extension for PyTorch*

Intel Extension for PyTorch* extends PyTorch with optimizations for extra performance boost on Intel hardware. Most of the optimizations will be included in stock PyTorch releases eventually, and the intention of the extension is to deliver up to date features and optimizations for PyTorch on Intel hardware, examples include AVX-512 Vector Neural Network Instructions (AVX512 VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX).

**NOTE:** Currently the extension supports AVX-512 instruction sets and AVX-2 support is WIP. The latest extension is compatible with PyTorch 1.10 and changes the device underhood from XPU to CPU which means that the model and tensor does not need to be coverted to the XPU device. The original XPU code is hosted at [xpu-cpu](https://github.com/intel/intel-extension-for-pytorch/tree/xpu-cpu) branch

 - [Installation](#installation)
     - [Install PyTorch (Optional)](#install-pytorch-optional)
     - [Install via wheel file](#install-via-wheel-file)
     - [Install Intel Extension for PyTorch from Source](#install-extension-by-compiling-from-source)
 - [Features](#features)
     - [Ease-of-use Python API](#ease-of-use-python-api) 
     - [Channels Last](#channels-last)
     - [Auto Mixed Precision (AMP)](#auto-mixed-precision-amp)
     - [Graph Optimization](#graph-optimization)
     - [Operator Optimization](#operator-optimization)
 - [Getting Started](#getting-started)
     - [Training](#training)
         - [Float32](#float32)
         - [BFloat16](#bfloat16)
     - [Inference - Imperative Mode](#inference---imperative-mode)
         - [Float32](#float32-1)
         - [BFloat16](#bfloat16-1)
     - [Inference - TorchScript Mode](#inference---torchscript-mode)
         - [Float32](#float32-2)
         - [BFloat16](#bfloat16-2)
     - [Inference - C++](#inference---c)
 - [Operator Optimizations](operator-optimizations)
     - [Supported Customized Operators](#supported-customized-operators)
     - [Supported Fusion Patterns](#supported-fusion-patterns)
 - [Tutorials](#tutorials)
 - [Joint blogs](#joint-blogs)
 - [Contribution](#contribution)
 - [License](#license)

## Installation

### Install PyTorch (Optional)
 |Extension Version|PyTorch Version|
 |--|--|
 |[v1.9.0](https://github.com/intel/intel-extension-for-pytorch/tree/v1.9.0)|[v1.9.0](https://github.com/pytorch/pytorch/tree/v1.9.0 "v1.9.0")|
 |[v1.8.0](https://github.com/intel/intel-extension-for-pytorch/tree/v1.8.0)|[v1.8.0](https://github.com/pytorch/pytorch/tree/v1.8.0 "v1.8.0")|
 |[v1.2.0](https://github.com/intel/intel-extension-for-pytorch/tree/v1.2.0)|[v1.7.0](https://github.com/pytorch/pytorch/tree/v1.7.0 "v1.7.0")|
 |[v1.1.0](https://github.com/intel/intel-extension-for-pytorch/tree/v1.1.0)|[v1.5.0-rc3](https://github.com/pytorch/pytorch/tree/v1.5.0-rc3 "v1.5.0-rc3")|
 |[v1.0.2](https://github.com/intel/intel-extension-for-pytorch/tree/v1.0.2)|[v1.5.0-rc3](https://github.com/pytorch/pytorch/tree/v1.5.0-rc3 "v1.5.0-rc3")|
 |[v1.0.1](https://github.com/intel/intel-extension-for-pytorch/tree/v1.0.1)|[v1.5.0-rc3](https://github.com/pytorch/pytorch/tree/v1.5.0-rc3 "v1.5.0-rc3")|
 |[v1.0.0](https://github.com/intel/intel-extension-for-pytorch/tree/v1.0.0)|[v1.5.0-rc3](https://github.com/pytorch/pytorch/tree/v1.5.0-rc3 "v1.5.0-rc3")|

For the extension version earlier than 1.8.0, a patch has to be manually applied to PyTorch source code. Please check previous installation guide.

From 1.8.0, compiling PyTorch from source is not required. If you still want to compile PyTorch, please follow instructions [here](https://github.com/pytorch/pytorch#installation). Please make sure to checkout the correct PyTorch version according to the table above.

**Note:** Compiling with gcc 7 on some environments, like CentOS 7, may fail. Please use GCC >= 8 to compile.

**Note:** Installing the extension will automatically invoke installation of the corresponding version of PyTorch.

### Install via wheel file

```python
python -m pip install torch_ipex==1.9.0 -f https://software.intel.com/ipex-whl-stable
```

:information_source: Wheel files availability for Python versions

| Extension Version | Python 3.6 | Python 3.7 | Python 3.8 | Python 3.9 |
| :--: | :--: | :--: | :--: | :--: |
| 1.9.0 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| 1.8.0 |  | :heavy_check_mark: |  |  |

### Install Extension by compiling from source

**NOTE** The master branch adapts to PyTorch 1.10. Please clone the PyTorch 1.10 branch, then build and install from source first.

```bash
git clone --recursive https://github.com/intel/intel-extension-for-pytorch
cd intel-extension-for-pytorch

# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive

# run setup.py to compile and install the binaries
python setup.py install
```

## Features

### Ease-of-use Python API

Intel® Extension for PyTorch* provides simple frontend Python APIs and utilities for users to get performance optimizations  such as graph optimization and operator optimization with minor code changes. Typically, only 2 to 3 clauses are required to be added to the original code.

### Channels Last

Comparing to the default NCHW memory format, channels_last (NHWC) memory format could further accelerate convolutional neural networks.In Intel® Extension for PyTorch*, NHWC memory format  has been enabled for most key CPU operators, though not all of them have been merged to PyTorch master branch yet. They are expected to be fully landed in PyTorch upstream soon.

### Auto Mixed Precision (AMP)

Low precision data type BFloat16 has been natively supported on the 3rd Generation Xeon scalable Servers (aka Cooper Lake) with AVX512 instruction set and will be  supported on the next generation of Intel® Xeon® Scalable Processors with Intel® Advanced Matrix Extensions (Intel® AMX) instruction set with further boosted performance. The support of Auto Mixed Precision (AMP) with BFloat16 for CPU and BFloat16 optimization of operators have been  massively enabled in Intel® Extension for PyTorch*, and partially upstreamed to PyTorch master branch. Most of these optimizations will be landed in PyTorch master through PRs that are being submitted and reviewed.

### Graph Optimization

To optimize performance further with torchscript, Intel® Extension for PyTorch* supports fusion of frequently used operator patterns, like Conv2D+ReLU, Linear+ReLU, etc.  The benefit of the fusions are delivered to users in a transparant fashion.

### Operator Optimization

Intel® Extension for PyTorch* also optimizes operators and implements several customized operators for performance. A few ATen operators are replaced by their optimized counterparts in Intel® Extension for PyTorch* via ATen registration mechanism. Moreover, some customized operators are implemented for several popular topologies . For instance, ROIAlign and NMS are defined in Mask R-CNN. To improve performance of these topologies, Intel® Extension for PyTorch* also optimized these customized operators.

## Getting Started

Minor code changes are required for users to get start with Intel® Extension for PyTorch*. Both PyTorch imperative mode and TorchScript mode are supported. This section introduces usage of Intel® Extension for PyTorch* API functions for both imperative mode and TorchScript mode, covering data type Float32 and BFloat16. C++ usage will also be introduced at the end.

You just need to import Intel® Extension for PyTorch* package and apply its optimize function against the model object. If it is a training workload, the optimize function also needs to be applied against the optimizer object.

For training and inference with BFloat16 data type, torch.cpu.amp has been enabled in PyTorch upstream to support mixed precision with convenience, and BFloat16 datatype has been enabled excessively for CPU operators in PyTorch upstream and Intel® Extension for PyTorch*. Running torch.cpu.amp will match each operator to its appropriate datatype and returns the best possible performance.

The code changes that are required for Intel® Extension for PyTorch* are highlighted with comments in a line above.

### Training

#### Float32

```python
import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(2, 3, 2)

    def forward(self, x):
        return self.conv(x)

model = Model()
model.set_state_dict(torch.load(PATH))
optimizer.set_state_dict(torch.load(PATH))

model.train()
# Setting memory_format to torch.channels_last could improve performance with 4D input data. This is optional.
model = model.to(memory_format=torch.channels_last)
# Invoke optimize function against the model object and optimizer object
model, optimizer = ipex.optimize(model, optimizer=optimizer)

for images, label in train_loader():
    # Optional.
    images = images.to(memory_format=torch.channels_last)

    loss = criterion(model(images), label)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), PATH)
torch.save(optimizer.state_dict(), PATH)
```

#### BFloat16

```python
import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(2, 3, 2)

    def forward(self, x):
        return self.conv(x)

model = Model()
model.set_state_dict(torch.load(PATH))
optimizer.set_state_dict(torch.load(PATH))

model.train()
# Setting memory_format to torch.channels_last could improve performance with 4D input data. This is optional.
model = model.to(memory_format=torch.channels_last)
# Invoke optimize function against the model object and optimizer object with data type set to torch.bfloat16
model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)

for images, label in train_loader():
    with torch.cpu.amp.autocast():
        # Optional.
        images = images.to(memory_format=torch.channels_last)
        loss = criterion(model(images), label)
    loss.backward()
    optimizer.step()
torch.save(model.state_dict(), PATH)
torch.save(optimizer.state_dict(), PATH)
```

### Inference - Imperative Mode

#### Float32

```python
import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(2, 3, 2)

    def forward(self, x):
        return self.conv(x)

model = Model()
model.eval()
# Setting memory_format to torch.channels_last could improve performance with 4D input data. This is optional.
model = model.to(memory_format=torch.channels_last)
# Invoke optimize function against the model object
model = ipex.optimize(model)
with torch.no_grad():
    # Optional.
    images = images.to(memory_format=torch.channels_last)

    res = model(images)
```

#### BFloat16

```python
import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(2, 3, 2)

    def forward(self, x):
        return self.conv(x)

model = Model()
model.eval()
# Setting memory_format to torch.channels_last could improve performance with 4D input data. This is optional.
model = model.to(memory_format=torch.channels_last)
# Invoke optimize function against the model object with data type set to torch.bfloat16
model = ipex.optimize(model, dtype=torch.bfloat16)
with torch.no_grad(), torch.cpu.amp.autocast():
    # Optional.
    images = images.to(memory_format=torch.channels_last)

    res = model(images)
```

### Inference - TorchScript Mode

TorchScript mode makes graph optimization possible , hence improves performance for some topologies. Intel® Extension for PyTorch* enables most commonly used operator pattern fusion, and users can get the performance benefit without additional code changes

#### Float32

```python
import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex

# oneDNN graph fusion is enabled by default, uncomment the line below to disable it explicitly 
# ipex.enable_onednn_fusion(False)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(2, 3, 2)

    def forward(self, x):
        return self.conv(x)

model = Model()
model.eval()
# Setting memory_format to torch.channels_last could improve performance with 4D input data. This is optional.
model = model.to(memory_format=torch.channels_last)
# Invoke optimize function against the model object
model = ipex.optimize(model)
with torch.no_grad():
    # Optional.
    images = images.to(memory_format=torch.channels_last)

    model = torch.jit.trace(model, images)
    model = torch.jit.freeze(model)
    res = model(images)
```

#### BFloat16

```python
import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex

# oneDNN graph fusion is enabled by default, uncomment the line below to disable it explicitly 
# ipex.enable_onednn_fusion(False)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(2, 3, 2)

    def forward(self, x):
        return self.conv(x)

model = Model()
model.eval()
# Setting memory_format to torch.channels_last could improve performance with 4D input data. This is optional.
model = model.to(memory_format=torch.channels_last)
# Invoke optimize function against the model with data type set to torch.bfloat16
model = ipex.optimize(model, dtype=torch.bfloat16)
with torch.no_grad(), torch.cpu.amp.autocast():
    # Optional.
    images = images.to(memory_format=torch.channels_last)

    model = torch.jit.trace(model, torch.rand(args.batch_size, 3, 224, 224))
    model = torch.jit.freeze(model)
    res = model(images)
```

### Inference - C++

To work with libtorch, C++ library of PyTorch, Intel® Extension for PyTorch* provides its C++ dynamic library as well. The C++ library is supposed to handle inference workload only, such as service deployment. For regular development, please use Python interface. Comparing to usage of libtorch, no specific code changes are required, except for converting input data into channels last data format. During compilation, Intel optimizations will be activated automatically once C++ dynamic library of Intel® Extension for PyTorch* is linked.

```C++
#include <torch/script.h>
#include <iostream> 
#include <memory> 
 
int main(int argc, const char* argv[]) { 
  torch::jit::script::Module module; 
  try { 
    module = torch::jit::load(argv[1]); 
  } 
  catch (const c10::Error& e) { 
    std::cerr << "error loading the model\n"; 
    return -1; 
  } 
  std::vector<torch::jit::IValue> inputs; 
  // make sure input data are converted to channels last format
  inputs.push_back(torch::ones({1, 3, 224, 224}).to(c10::MemoryFormat::ChannelsLast)); 
 
  at::Tensor output = module.forward(inputs).toTensor(); 
 
  return 0; 
} 
```

## Operator Optimizations

### Supported Customized Operators

* ROIAlign
* NMS
* BatchScoreNMS
* MLP
* Interaction
* FrozenBatchNorm2d

### Supported Fusion Patterns
* Conv2D + ReLU
* Conv2D + SUM
* Conv2D + SUM + ReLU
* Conv2D + Sigmoid
* Conv2D + Sigmoid + MUL
* Conv2D + HardTanh
* Conv2D + ELU
* Conv3D + ReLU
* Conv3D + SUM
* Conv3D + SUM + ReLU
* Linear + ReLU
* Linear + GELU
* View + Transpose + Contiguous + View

## Tutorials

*  [Performance Tuning](tutorials/Performance_Tuning.md)

## Joint-blogs

* [Intel and Facebook Accelerate PyTorch Performance with 3rd Gen Intel® Xeon® Processors and Intel® Deep Learning Boost’s new BFloat16 capability](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/intel-facebook-boost-bfloat16.html)
* [Accelerate PyTorch with the extension and oneDNN using Intel BF16 Technology](https://medium.com/pytorch/accelerate-pytorch-with-ipex-and-onednn-using-intel-bf16-technology-dca5b8e6b58f)
* [Scaling up BERT-like model Inference on modern CPU - Part 1 by the launcher of the extension](https://huggingface.co/blog/bert-cpu-scaling-part-1)

## Contribution

Please submit PR or issue to communicate with us or contribute code.

## License

_Apache License_, Version _2.0_. As found in [LICENSE](https://github.com/intel/intel-extension-for-pytorch/blob/master/LICENSE.txt) file.
