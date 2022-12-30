Examples
========

**Note:** For examples on CPU, please check [here](../../../cpu/latest/tutorials/examples.html).

## Training

### Single-instance Training

#### Code Changes Highlight

There are only a few lines of code change required to use Intel® Extension for PyTorch\* on training, as shown:
1. `ipex.optimize` function applies optimizations against the model object, as well as an optimizer object.
2.  Use Auto Mixed Precision (AMP) with BFloat16 data type.
3.  Convert input tensors, loss criterion and model to XPU.

The complete examples for Float32 and BFloat16 training on single-instance are illustrated in the sections.

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
# For Float32
output = model(data)
...
# For BFloat16
with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
  output = model(input)
...
```

#### Complete - Float32 Example

```
import torch
import torchvision
############# code changes ###############
import intel_extension_for_pytorch as ipex
############# code changes ###############

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
######################## code changes #######################
model = model.to("xpu")
criterion = criterion.to("xpu")
model, optimizer = ipex.optimize(model, optimizer=optimizer)
######################## code changes #######################

for batch_idx, (data, target) in enumerate(train_loader):
    ########## code changes ##########
    data = data.to("xpu")
    target = target.to("xpu")
    ########## code changes ##########
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

#### Complete - BFloat16 Example

```
import torch
import torchvision
############# code changes ###############
import intel_extension_for_pytorch as ipex
############# code changes ###############

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
##################################### code changes ################################
model = model.to("xpu")
criterion = criterion.to("xpu")
model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)
##################################### code changes ################################

for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    ######################### code changes ######################### 
    data = data.to("xpu")
    target = target.to("xpu")
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
    ######################### code changes ######################### 
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
#### Complete - Model checkpoint load Example

```
model = torchvision.models.resnet50()
optimizer = torch.optim.SGD(model.parameters(), lr = LR, momentum=0.9)
model.train()
##### code changes #####
model = model.to("xpu")
##### code changes #####

checkpoint = torch.load('checkpoint.pth', map_location="xpu")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

##################################### code changes ####################################
model, optimizer = torch.xpu.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)
##################################### code changes ####################################
```

## Inference

The `optimize` function of Intel® Extension for PyTorch\* applies optimizations to the model, bringing additional performance boosts. For both computer vision workloads and NLP workloads, we recommend applying the `optimize` function against the model object.

### Float32

#### Imperative Mode

##### Resnet50

```
import torch
import torchvision.models as models
############# code changes ###############
import intel_extension_for_pytorch as ipex
############# code changes ###############

model = models.resnet50(pretrained=True)
model.eval()
data = torch.rand(1, 3, 224, 224)

######## code changes #######
model = model.to("xpu")
data = data.to("xpu")
model = ipex.optimize(model)
######## code changes #######

with torch.no_grad():
    model(data)
```

##### BERT

```
import torch
from transformers import BertModel
############# code changes ###############
import intel_extension_for_pytorch as ipex
############# code changes ###############

model = BertModel.from_pretrained(args.model_name)
model.eval()

vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])

######## code changes #######
model = model.to("xpu")
data = data.to("xpu")
model = ipex.optimize(model)
######## code changes #######

with torch.no_grad():
    model(data)
```

#### TorchScript Mode

We recommend you take advantage of Intel® Extension for PyTorch\* with [TorchScript](https://pytorch.org/docs/stable/jit.html) for further optimizations.

##### Resnet50

```
import torch
import torchvision.models as models
############# code changes ###############
import intel_extension_for_pytorch as ipex
############# code changes ###############

model = models.resnet50(pretrained=True)
model.eval()
data = torch.rand(1, 3, 224, 224)

######## code changes #######
model = model.to("xpu")
data = data.to("xpu")
model = ipex.optimize(model)
######## code changes #######

with torch.no_grad():
    d = torch.rand(1, 3, 224, 224)
    ##### code changes #####  
    d = d.to("xpu")
    ##### code changes #####  
    model = torch.jit.trace(model, d)
    model = torch.jit.freeze(model)

    model(data)
```

##### BERT

```
import torch
from transformers import BertModel
############# code changes ###############
import intel_extension_for_pytorch as ipex
############# code changes ###############

model = BertModel.from_pretrained(args.model_name)
model.eval()

vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])

######## code changes #######
model = model.to("xpu")
data = data.to("xpu")
model = ipex.optimize(model)
######## code changes #######

with torch.no_grad():
    d = torch.randint(vocab_size, size=[batch_size, seq_length])
    ##### code changes #####
    d = d.to("xpu")
    ##### code changes #####
    model = torch.jit.trace(model, (d,), strict=False)
    model = torch.jit.freeze(model)

    model(data)
```

### BFloat16

Similar to running with Float32, the `optimize` function also works for BFloat16 data type. The only difference is setting `dtype` parameter to `torch.bfloat16`.
We recommend using Auto Mixed Precision (AMP) with BFloat16 data type.


#### Imperative Mode

##### Resnet50

```
import torch
import torchvision.models as models
############# code changes ###############
import intel_extension_for_pytorch as ipex
############# code changes ###############

model = models.resnet50(pretrained=True)
model.eval()
data = torch.rand(1, 3, 224, 224)

#################### code changes #################
model = model.to("xpu")
data = data.to("xpu")
model = ipex.optimize(model, dtype=torch.bfloat16)
#################### code changes #################

with torch.no_grad():
    ############################# code changes #####################
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):    
    ############################ code changes ######################
        model(data)
```

##### BERT

```
import torch
from transformers import BertModel
############# code changes ###############
import intel_extension_for_pytorch as ipex
############# code changes ###############

model = BertModel.from_pretrained(args.model_name)
model.eval()

vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])

#################### code changes #################
model = model.to("xpu")
data = data.to("xpu")
model = ipex.optimize(model, dtype=torch.bfloat16)
#################### code changes #################

with torch.no_grad():
    ########################### code changes ########################
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):    
    ########################### code changes ########################
        model(data)
```

#### TorchScript Mode

We recommend you take advantage of Intel® Extension for PyTorch\* with [TorchScript](https://pytorch.org/docs/stable/jit.html) for further optimizations.

##### Resnet50

```
import torch
import torchvision.models as models
############# code changes ###############
import intel_extension_for_pytorch as ipex
############# code changes ###############

model = models.resnet50(pretrained=True)
model.eval()
data = torch.rand(1, 3, 224, 224)

#################### code changes #################
model = model.to("xpu")
data = data.to("xpu")
model = ipex.optimize(model, dtype=torch.bfloat16)
#################### code changes #################

with torch.no_grad():
    d = torch.rand(1, 3, 224, 224)
    ############################# code changes #####################
    d = d.to("xpu")
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):     
    ############################# code changes #####################
        model = torch.jit.trace(model, d)
        model = torch.jit.freeze(model)
        model(data)
```

##### BERT

```
import torch
from transformers import BertModel
############# code changes ###############
import intel_extension_for_pytorch as ipex
############# code changes ###############

model = BertModel.from_pretrained(args.model_name)
model.eval()

vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])

#################### code changes #################
model = model.to("xpu")
data = data.to("xpu")
model = ipex.optimize(model, dtype=torch.bfloat16)
#################### code changes #################

with torch.no_grad():
    d = torch.randint(vocab_size, size=[batch_size, seq_length])
    ############################# code changes #####################
    d = d.to("xpu")
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):     
    ############################# code changes #####################
        model = torch.jit.trace(model, (d,), strict=False)
        model = torch.jit.freeze(model)
    
        model(data)
```
### Float16

Similar to running with Float32, the `optimize` function also works for Float16 data type. The only difference is setting `dtype` parameter to `torch.float16`.
We recommend using Auto Mixed Precision (AMP) with Float16 data type.

#### Imperative Mode

##### Resnet50

```
import torch
import torchvision.models as models
############# code changes ###############
import intel_extension_for_pytorch as ipex
############# code changes ###############

model = models.resnet50(pretrained=True)
model.eval()
data = torch.rand(1, 3, 224, 224)

#################### code changes ################
model = model.to("xpu")
data = data.to("xpu")
model = ipex.optimize(model, dtype=torch.float16)
#################### code changes ################

with torch.no_grad():
    ############################# code changes #####################
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16):    
    ############################# code changes #####################
        model(data)
```

##### BERT

```
import torch
from transformers import BertModel
############# code changes ###############
import intel_extension_for_pytorch as ipex
############# code changes ###############

model = BertModel.from_pretrained(args.model_name)
model.eval()

vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])

#################### code changes ################
model = model.to("xpu")
data = data.to("xpu")
model = ipex.optimize(model, dtype=torch.float16)
#################### code changes ################

with torch.no_grad():
    ############################# code changes #####################
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16):    
    ############################# code changes #####################
        model(data)
```

#### TorchScript Mode

We recommend you take advantage of Intel® Extension for PyTorch\* with [TorchScript](https://pytorch.org/docs/stable/jit.html) for further optimizations.

##### Resnet50

```
import torch
import torchvision.models as models
############# code changes ###############
import intel_extension_for_pytorch as ipex
############# code changes ###############

model = models.resnet50(pretrained=True)
model.eval()
data = torch.rand(1, 3, 224, 224)

#################### code changes ################
model = model.to("xpu")
data = data.to("xpu")
model = ipex.optimize(model, dtype=torch.float16)
#################### code changes ################

with torch.no_grad():
    d = torch.rand(1, 3, 224, 224)
    ############################# code changes #####################
    d = d.to("xpu")
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16):
    ############################# code changes #####################
        model = torch.jit.trace(model, d)
        model = torch.jit.freeze(model)
        model(data)
```

##### BERT

```
import torch
from transformers import BertModel
############# code changes ###############
import intel_extension_for_pytorch as ipex
############# code changes ###############

model = BertModel.from_pretrained(args.model_name)
model.eval()

vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])

#################### code changes ################
model = model.to("xpu")
data = data.to("xpu")
model = ipex.optimize(model, dtype=torch.float16)
#################### code changes ################

with torch.no_grad():
    d = torch.randint(vocab_size, size=[batch_size, seq_length])
    ############################# code changes #####################
    d = d.to("xpu")
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16):
    ############################# code changes #####################
        model = torch.jit.trace(model, (d,), strict=False)
        model = torch.jit.freeze(model)

        model(data)
```

### INT8

#### Imperative Mode

Compared to FP32 module, INT8 model requires less memory usage and usually has higher performance. To convert a FP32 model to INT8 model, statistic information of model activation and weight need be obtained firstly. `QConfig` specifies which method to calculate statistics. `torch.quantization.prepare` inserts observers in model, which is responsible for computing statistics. Then, a calibration dataset should be fed to model for collecting information from real data. At last, FP32 model is converted to INT8 model by using `torch.quantization.convert`. Then, the model can inference with testing dataset.

```
import torch

########## code changes ##########
import intel_extension_for_pytorch
model = model.to('xpu')
########## code changes ##########

modelImpe = torch.quantization.QuantWrapper(model)
modelImpe.eval()
qconfig = torch.quantization.QConfig(
        activation=torch.quantization.observer.MinMaxObserver.with_args(
            qscheme=torch.per_tensor_symmetric),
            weight=torch.quantization.default_weight_observer
)

modelImpe.qconfig = qconfig
torch.quantization.prepare(modelImpe, inplace=True)

with torch.no_grad():
    for input in enumerate(calib_dataset):
        ########## code changes ##########
        calib = input.to("xpu")
        ########## code changes ##########
        modelImpe(calib)
        if i == self.calib_iters - 1:
            break
torch.quantization.convert(modelImpe, inplace=True)

modelImpe(test_data)
```

#### TorchScript Mode

We recommend to use TorchScript for INT8 model due to it has wider support for models. Moreover, TorchScript mode would auto enable our optimizations. For TorchScript INT8 model, inserting observer and model quantization is achieved through `prepare_jit` and `convert_jit` separately. Calibration process is required for collecting statistics from real data. After conversion, optimizations like operator fusion would be auto enabled.

```
import torch
import torchvision.models as models
from torch.jit._recursive import wrap_cpp_module
from torch.quantization.quantize_jit import (
      convert_jit,
      prepare_jit,
)
########## code changes ##########
import intel_extension_for_pytorch
########## code changes ##########

model = models.resnet50(pretrained=True)
model.eval()
data = torch.rand(1, 3, 224, 224)

##### code changes #####
model = model.to("xpu")
data = data.to("xpu")
##### code changes #####

with torch.no_grad():
  d = torch.rand(1, 3, 224, 224)
  ##### code changes #####
  d = d.to("xpu")
  ##### code changes #####
  model = torch.jit.trace(model, d)

#################### code changes ##################################
qconfig = torch.quantization.QConfig(
    activation=torch.quantization.observer.MinMaxObserver.with_args(
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
        dtype=torch.quint8
    ),
    weight=torch.quantization.default_weight_observer
)
modelJit = prepare_jit(modelJit, {'': qconfig}, True)
for i, input in enumerate(calib_dataset):
    calib = input.to("xpu")
    modelJit(calib)

    if i == calib_iters - 1:
        break
modelJit = convert_jit(modelJit, True)
#################### code changes ##################################

modelJit(data)
```

### `torch.xpu.optimize`

`torch.xpu.optimize` is an alternative of `ipex.optimize` in Intel® Extension for PyTorch*, to provide identical usage for XPU device only. The motivation of adding this alias is to unify the coding style in user scripts base on torch.xpu modular. Refer to below example for usage.

#### ResNet50 FP32 imperative inference

```
import torch
import torchvision.models as models
############# code changes #########
import intel_extension_for_pytorch
############# code changes #########

model = models.resnet50(pretrained=True)
model.eval()
data = torch.rand(1, 3, 224, 224)

model = model.to(memory_format=torch.channels_last)
data = data.to(memory_format=torch.channels_last)

########## code changes #########
model = model.to("xpu")
data = data.to("xpu")
model = torch.xpu.optimize(model)
########## code changes #########

with torch.no_grad():
  model(data)
```

## C++

Intel® Extension for PyTorch\* provides its C++ dynamic library to allow users to implement custom DPC++ kernels to run on the XPU device. Refer to the [DPC++ extension](./features/DPC++_Extension.md) for the details.

## Model Zoo

Use cases that had already been optimized by Intel engineers are available at [Model Zoo for Intel® Architecture](https://github.com/IntelAI/models/tree/v2.9.0). A bunch of PyTorch use cases for benchmarking are also available on the [GitHub page](https://github.com/IntelAI/models/tree/v2.9.0#use-cases). Models verified on Intel dGPUs are marked in `Model Documentation` Column. You can get performance benefits out-of-box by simply running scipts in the Model Zoo.
