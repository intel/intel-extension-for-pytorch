Auto Mixed Precision (AMP)
==========================

## Introduction

`torch.cpu.amp` provides convenience for auto data type conversion at runtime. Deep learning workloads could benefit from lower precision floating point data types like `torch.float16` or `torch.bfloat16`, because of its lighter calculation workload and less memory usage. However, because of the nature character of lower precision floating point data types, accuracy is sacrificed. Using lower precision floating point data types is a trade-off between accuracy and performance. Thus, some operations need to keep in `torch.float32`, while others can be converted to lower precision floating point data types. The Auto Mixed Precision (AMP) feature automates the tuning of data type conversions over all operators.

Currently, `torch.cpu.amp` only supports `torch.bfloat16`. It is the default lower precision floating point data type when `torch.cpu.amp` is enabled. `torch.cpu.amp` primarily benefits on Intel CPU with BFloat16 instruction set support.

## Use Case

The following simple network should show a speedup with mixed precision.

```
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        return self.conv(x)
```

### Default Precision

Without `torch.cpu.amp`, the network executes all operators with default precision (`torch.float32`).
```
model = SimpleNet()
x = torch.rand(64, 64, 224, 224)
y = model(x)
```

### Inference with Imperative Path

`torch.cpu.amp.autocast` is designed to be context managers that allow scopes of your script to run in mixed precision. In these scopes, operations run in a data type chosen by the `autocast` class to improve performance while maintaining accuracy. See the operations category section for details on what precision the `autocast` class chooses for each operator, and under what circumstances.

```
model = SimpleNet().eval()
x = torch.rand(64, 64, 224, 224)
with torch.cpu.amp.autocast():
    y = model(x)
```

### Inference with TorchScript Path

`torch.cpu.amp.autocast` can be used with `torch.jit.trace` to apply graph optimization.

```
model = SimpleNet().eval()
x = torch.rand(64, 64, 224, 224)
with torch.cpu.amp.autocast():
    model = torch.jit.trace(model, x)
    model = torch.jit.freeze(model)
    y = model(x)
```

### Training Support

`torch.cpu.amp.autocast` can be used in training to improve performance.

```
model = SimpleNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
for images, label in train_loader():
    with torch.cpu.amp.autocast():
        loss = criterion(model(images), label)
    loss.backward()
    optimizer.step()
```

## Design Details

### Frontend API Design

`torch.cpu.amp` is designed to be context managers that allow scopes of your script to run in mixed precision. It takes input parameter `dtype`, which is `torch.bfloat16` by default.

### Dedicated Dispatch Key

`torch.cpu.amp` extends the design of the original pytorch `Auto Mixed Precision` using the dedicated dispatch key of `AutocastCPU`. Each tensor during creation will have an `Autocast` Dispatchkey corresponding to the device (`CUDA` or `CPU`). Thus, for every tensor on CPU, `AutocastCPU` exists along with the tensor. During the dispatch phase, operators with input tensors of `AutocastCPU` are dispatched to the `Autocast` layers. The `Autocast` layer decides what precision to chooses for each operator. `AutocastCPU` has higher dispatch priority comparing to `Autograd` which makes sure the `Autocast` layer runs before `Autograd`.

### Operations category

The operations are generally divided into 3 major categories and registered under Dispatch Key `AutocastCPU`:
* `lower_precision_fp` category: Computation bound operators that could get performance boost with BFloat16 data type through acceleration by Intel CPU BFloat16 instruction set. Inputs of them are casted into `torch.bfloat16` before execution. `convolutions` and `linear` are examples of this category.
* `fallthrough` category: Operators that support running with both Float32 and BFloat16 data types, but could not get performance boost with BFloat16 data type. `relu` and `max_pool2d` are examples of this category.
* `fp32` category: Operators that are not enabled with BFloat16 support yet. Inputs of them are casted into `float32` before execution. `max_pool3d` and `group_norm` are examples of this category.
