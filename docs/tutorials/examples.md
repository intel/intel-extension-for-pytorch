Examples
========

## Training

### Single-instance Training

#### Code Changes Highlight

```
...
import torch
import intel_extension_for_pytorch as ipex
...
model = Model()
criterion = ...
optimizer = ...
model.train()
# For Float32
model, optimizer = ipex.optimize(model, optimizer=optimizer)
# For BFloat16
model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)
...
# Setting memory_format to torch.channels_last could improve performance with 4D input data. This is optional.
data = data.to(memory_format=torch.channels_last)
optimizer.zero_grad()
output = model(data)
...
```

#### Complete - Float32


```
import torch
import torchvision
import intel_extension_for_pytorch as ipex

LR = 0.001
DOWNLOAD = True
DATA = 'datasets/cifar10/'

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = torchvision.datasets.CIFAR10(
        root=DATA,
        train=True,
        transform=transform,
        download=DOWNLOAD,
)
train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=128
)

model = torchvision.models.resnet50()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = LR, momentum=0.9)
model.train()
model, optimizer = ipex.optimize(model, optimizer=optimizer)

for batch_idx, (data, target) in enumerate(train_loader):
    # Setting memory_format to torch.channels_last could improve performance with 4D input data. This is optional.
    data = data.to(memory_format=torch.channels_last)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print(batch_idx)
torch.save({
     'model_state_dict': model.state_dict(),
     'optimizer_state_dict': optimizer.state_dict(),
     }, 'checkpoint.pth')
```

#### Complete - BFloat16

```
import torch
import torchvision
import intel_extension_for_pytorch as ipex

LR = 0.001
DOWNLOAD = True
DATA = 'datasets/cifar10/'

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = torchvision.datasets.CIFAR10(
        root=DATA,
        train=True,
        transform=transform,
        download=DOWNLOAD,
)
train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=128
)

model = torchvision.models.resnet50()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = LR, momentum=0.9)
model.train()
model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)

for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    with torch.cpu.amp.autocast():
        # Setting memory_format to torch.channels_last could improve performance with 4D input data. This is optional.
        data = data.to(memory_format=torch.channels_last)
        output = model(data)
        loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print(batch_idx)
torch.save({
     'model_state_dict': model.state_dict(),
     'optimizer_state_dict': optimizer.state_dict(),
     }, 'checkpoint.pth')
```

### Distributed Training

Distributed training with PyTorch DDP is accelerated by oneAPI Collective Communications Library Bindings for Pytorch\* (oneCCL Bindings for Pytorch\*). More detailed information and examples are available at its [Github repo](https://github.com/intel/torch-ccl).

## Inference

### Float32

#### Imperative Mode

##### Resnet50

```
import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)
model.eval()
data = torch.rand(1, 3, 224, 224)

import intel_extension_for_pytorch as ipex
model = model.to(memory_format=torch.channels_last)
model = ipex.optimize(model, dtype=torch.float32, level='O1')
data = data.to(memory_format=torch.channels_last)

with torch.no_grad():
  model(data)
```

##### BERT

```
import torch
from transformers import BertModel

model = BertModel.from_pretrained(args.model_name)
model.eval()

vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])

import intel_extension_for_pytorch as ipex
model = ipex.optimize(model, dtype=torch.float32, level="O1")

with torch.no_grad():
  model(data)
```

#### TorchScript Mode

##### Resnet50

```
import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)
model.eval()
data = torch.rand(1, 3, 224, 224)

import intel_extension_for_pytorch as ipex
model = model.to(memory_format=torch.channels_last)
model = ipex.optimize(model, dtype=torch.float32, level='O1')
data = data.to(memory_format=torch.channels_last)

with torch.no_grad():
  d = torch.rand(1, 3, 224, 224)
  model = torch.jit.trace(model, d)
  model = torch.jit.freeze(model)

  model(data)
```

##### BERT

```
import torch
from transformers import BertModel

model = BertModel.from_pretrained(args.model_name)
model.eval()

vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])

import intel_extension_for_pytorch as ipex
model = ipex.optimize(model, dtype=torch.float32, level="O1")

with torch.no_grad():
  d = torch.randint(vocab_size, size=[batch_size, seq_length])
  model = torch.jit.trace(model, (d,), check_trace=False, strict=False)
  model = torch.jit.freeze(model)

  model(data)
```

### BFloat16

#### Imperative Mode

##### Resnet50

```
import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)
model.eval()
data = torch.rand(1, 3, 224, 224)

import intel_extension_for_pytorch as ipex
model = model.to(memory_format=torch.channels_last)
model = ipex.optimize(model, dtype=torch.bfloat16, level='O1')
data = data.to(memory_format=torch.channels_last)

with torch.no_grad():
  with torch.cpu.amp.autocast():
    model(data)
```

##### BERT

```
import torch
from transformers import BertModel

model = BertModel.from_pretrained(args.model_name)
model.eval()

vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])

import intel_extension_for_pytorch as ipex
model = ipex.optimize(model, dtype=torch.bfloat16, level="O1")

with torch.no_grad():
  with torch.cpu.amp.autocast():
    model(data)
```

#### TorchScript Mode

##### Resnet50

```
import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)
model.eval()
data = torch.rand(1, 3, 224, 224)

import intel_extension_for_pytorch as ipex
model = model.to(memory_format=torch.channels_last)
model = ipex.optimize(model, dtype=torch.bfloat16, level='O1')
data = data.to(memory_format=torch.channels_last)

with torch.no_grad():
  with torch.cpu.amp.autocast():
    model = torch.jit.trace(model, torch.rand(1, 3, 224, 224))
    model = torch.jit.freeze(model)

    model(data)
```

##### BERT

```
import torch
from transformers import BertModel

model = BertModel.from_pretrained(args.model_name)
model.eval()

vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])

import intel_extension_for_pytorch as ipex
model = ipex.optimize(model, dtype=torch.bfloat16, level="O1")

with torch.no_grad():
  with torch.cpu.amp.autocast():
    d = torch.randint(vocab_size, size=[batch_size, seq_length])
    model = torch.jit.trace(model, (d,), check_trace=False, strict=False)
    model = torch.jit.freeze(model)

    model(data)
```

### INT8

#### Calibration

```
import os
import torch

model = Model()
model.eval()
data = torch.rand(<shape>)

import intel_extension_for_pytorch as ipex
# For first-time calibration, pass dtype into the QuantConf function to generate a conf
if os.path.isfile('int8_conf.json'):
  conf = ipex.QuantConf(dtype=torch.int8) 
# For re-calibration, pass the generated json file into the QuantConf function to generate a conf
else:
  conf = ipex.QuantConf('int8_conf.json')
model, conf = ipex.quantization.prepare(model, conf)
for d in calibration_data_loader(): 
  # conf will be updated with observed statistics during calibrating with the dataset 
  with ipex.quantization.calibrate(conf):
    model(d) 
conf.save('int8_conf.json', default_recipe=True)
model = ipex.quantization.convert(model, conf, torch.rand(<shape>)) 

with torch.no_grad():
  model(data)
```

#### Deployment

```
import torch

model = models.Model()
model.eval()
data = torch.rand(<shape>)

import intel_extension_for_pytorch as ipex
conf = ipex.QuantConf('int8_conf.json')
model = ipex.quantization.convert(model, conf, torch.rand(<shape>)) 

with torch.no_grad():
  model(data)
```

## C++

To work with libtorch, C++ library of PyTorch, Intel® Extension for PyTorch\* provides its C++ dynamic library as well. The C++ library is supposed to handle inference workload only, such as service deployment. For regular development, please use Python interface. Comparing to usage of libtorch, no specific code changes are required, except for converting input data into channels last data format. Compilation follows the recommended methodology with CMake. Detailed instructions can be found in [PyTorch tutorial](https://pytorch.org/tutorials/advanced/cpp_export.html#depending-on-libtorch-and-building-the-application).

During compilation, Intel optimizations will be activated automatically once C++ dynamic library of Intel® Extension for PyTorch\* is linked.

**example-app.cpp**

```
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

**CMakeList.txt**

```
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -Wl,--no-as-needed")

add_executable(example-app example-app.cpp)
# Link the binary against the C++ dynamic library file of Intel® Extension for PyTorch*
target_link_libraries(example-app "${TORCH_LIBRARIES}" "${INTEL_EXTENSION_FOR_PYTORCH_PATH}/lib/libintel-ext-pt-cpu.so")

set_property(TARGET example-app PROPERTY CXX_STANDARD 14)
```

**Note:** Since Intel® Extension for PyTorch\* is still under development, name of the c++ dynamic library in the master branch may defer to *libintel-ext-pt-cpu.so* shown above. Please check the name out in the installation folder. The so file name starts with *libintel-*.

**Command for compilation**

```
$ cmake -DCMAKE_PREFIX_PATH=<LIBPYTORCH_PATH> -DINTEL_EXTENSION_FOR_PYTORCH_PATH=<INTEL_EXTENSION_FOR_PYTORCH_INSTALLATION_PATH> ..
$ make
```
