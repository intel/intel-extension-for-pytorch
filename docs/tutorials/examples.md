Examples
========

## Training

### Single-instance Training

#### Code Changes Highlight

There are only a few lines of code change required to use Intel® Extension for PyTorch\* on training, as shown:
1. `torch.xpu.optimize` function applies optimizations against the model object, as well as an optimizer object.
2.  Use Auto Mixed Precision (AMP) with BFloat16 data type.
3.  Convert both input tensors, loss criterion and model to XPU.

The complete examples for Float32 and BFloat16 training on single-instance are illustrated in the sections.

```
...
import torch
import intel_extension_for_pytorch
...
model = Model()
criterion = ...
optimizer = ...
model.train()
# For Float32
model, optimizer = torch.xpu.optimize(model, optimizer=optimizer, dtype=torch.float32)
# For BFloat16
model, optimizer = torch.xpu.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)
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
########## code changes ##########
import intel_extension_for_pytorch
########## code changes ##########

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
#################################### code changes ####################################
model = model.to("xpu")
criterion = criterion.to("xpu")
model, optimizer = torch.xpu.optimize(model, optimizer=optimizer, dtype=torch.float32)
#################################### code changes ####################################

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
########## code changes ##########
import intel_extension_for_pytorch
########## code changes ##########

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
##################################### code changes ####################################
model = model.to("xpu")
criterion = criterion.to("xpu")
model, optimizer = torch.xpu.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)
##################################### code changes ####################################

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
##################################### code changes ####################################
model = model.to("xpu")
##################################### code changes ####################################

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
######################################################

with torch.no_grad():
    model(data)
```

##### BERT

```
import torch
from transformers import BertModel
########## code changes ##########
import intel_extension_for_pytorch
########## code changes ##########

model = BertModel.from_pretrained(args.model_name)
model.eval()

vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])

#################### code changes ####################
model = model.to("xpu")
data = data.to("xpu")
model = torch.xpu.optimize(model, dtype=torch.float32)
######################################################

with torch.no_grad():
    model(data)
```

#### TorchScript Mode

We recommend you take advantage of Intel® Extension for PyTorch\* with [TorchScript](https://pytorch.org/docs/stable/jit.html) for further optimizations.

##### Resnet50

```
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
######################################################

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
########## code changes ##########
import intel_extension_for_pytorch
########## code changes ##########

model = BertModel.from_pretrained(args.model_name)
model.eval()

vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])

#################### code changes ####################
model = model.to("xpu")
data = data.to("xpu")
model = torch.xpu.optimize(model, dtype=torch.float32)
######################################################

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
########## code changes ##########
import intel_extension_for_pytorch
########## code changes ##########

model = models.resnet50(pretrained=True)
model.eval()
data = torch.rand(1, 3, 224, 224)

#################### code changes ####################
model = model.to("xpu")
data = data.to("xpu")
model = torch.xpu.optimize(model, dtype=torch.bfloat16)
######################################################

with torch.no_grad():
    ################################# code changes ######################################
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):    
    ################################# code changes ######################################
        model(data)
```

##### BERT

```
import torch
from transformers import BertModel
########## code changes ##########
import intel_extension_for_pytorch
########## code changes ##########

model = BertModel.from_pretrained(args.model_name)
model.eval()

vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])

#################### code changes ####################
model = model.to("xpu")
data = data.to("xpu")
model = torch.xpu.optimize(model, dtype=torch.bfloat16)
######################################################

with torch.no_grad():
    ################################# code changes ######################################
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):    
    ################################# code changes ######################################
        model(data)
```

#### TorchScript Mode

We recommend you take advantage of Intel® Extension for PyTorch\* with [TorchScript](https://pytorch.org/docs/stable/jit.html) for further optimizations.

##### Resnet50

```
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
model = torch.xpu.optimize(model, dtype=torch.bfloat16)
######################################################

with torch.no_grad():
    d = torch.rand(1, 3, 224, 224)
    ################################# code changes ######################################
    d = d.to("xpu")
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
    ################################# code changes ######################################
        model = torch.jit.trace(model, d)
        model = torch.jit.freeze(model)

    model(data)
```

##### BERT

```
import torch
from transformers import BertModel
########## code changes ##########
import intel_extension_for_pytorch
########## code changes ##########

model = BertModel.from_pretrained(args.model_name)
model.eval()

vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])

#################### code changes ####################
model = model.to("xpu")
data = data.to("xpu")
model = torch.xpu.optimize(model, dtype=torch.bfloat16)
######################################################

with torch.no_grad():
    d = torch.randint(vocab_size, size=[batch_size, seq_length])
    ################################# code changes ######################################
    d = d.to("xpu")
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):     
    ################################# code changes ######################################
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
########## code changes ##########
import intel_extension_for_pytorch
########## code changes ##########

model = models.resnet50(pretrained=True)
model.eval()
data = torch.rand(1, 3, 224, 224)

#################### code changes ####################
model = model.to("xpu")
data = data.to("xpu")
model = torch.xpu.optimize(model, dtype=torch.float16)
######################################################

with torch.no_grad():
    ################################# code changes ######################################
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16):    
    ################################# code changes ######################################
        model(data)
```

##### BERT

```
import torch
from transformers import BertModel
########## code changes ##########
import intel_extension_for_pytorch
########## code changes ##########

model = BertModel.from_pretrained(args.model_name)
model.eval()

vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])

#################### code changes ####################
model = model.to("xpu")
data = data.to("xpu")
model = torch.xpu.optimize(model, dtype=torch.float16)
######################################################

with torch.no_grad():
    ################################# code changes ######################################
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16):    
    ################################# code changes ######################################
        model(data)
```

#### TorchScript Mode

We recommend you take advantage of Intel® Extension for PyTorch\* with [TorchScript](https://pytorch.org/docs/stable/jit.html) for further optimizations.

##### Resnet50

```
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
model = torch.xpu.optimize(model, dtype=torch.float16)
######################################################

with torch.no_grad():
    d = torch.rand(1, 3, 224, 224)
    ################################# code changes ######################################
    d = d.to("xpu")
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16):
    ################################# code changes ######################################
        model = torch.jit.trace(model, d)
        model = torch.jit.freeze(model)

    model(data)
```

##### BERT

```
import torch
from transformers import BertModel
########## code changes ##########
import intel_extension_for_pytorch
########## code changes ##########

model = BertModel.from_pretrained(args.model_name)
model.eval()

vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])

#################### code changes ####################
model = model.to("xpu")
data = data.to("xpu")
model = torch.xpu.optimize(model, dtype=torch.float16)
######################################################

with torch.no_grad():
    d = torch.randint(vocab_size, size=[batch_size, seq_length])
    ################################# code changes ######################################
    d = d.to("xpu")
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16):
    ################################# code changes ######################################
        model = torch.jit.trace(model, (d,), strict=False)
        model = torch.jit.freeze(model)

    model(data)
```

## C++
Intel® Extension for PyTorch\* provides its C++ dynamic library to allow users to implement custom DPC++ kernels to run on the XPU backend. Refer to the [DPC++ extension](./features/DPC++_Extension.md) for the details.

## Model Zoo

Use cases that are already optimized by Intel engineers are available at [Model Zoo for Intel® Architecture](https://github.com/IntelAI/models). You can get performance benefits out-of-box by simply running scripts in the Model Zoo.
