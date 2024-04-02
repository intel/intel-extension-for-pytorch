Runtime Extension (Prototype)
==========================

Intel® Extension for PyTorch\* Runtime Extension provides a runtime CPU pool API to bind threads to cores. It also features async tasks. Please **note**: Intel® Extension for PyTorch\* Runtime extension is still in the **POC** stage. The API is subject to change. More detailed descriptions are available at [API Documentation page](../api_doc.html).

## Requirements

Intel® Extension for PyTorch\* Runtime Extension relies on `iomp` to bind threads to cores. If you want to use it in your application, please run models with extra flag: `LD_PRELOAD=$LD_PRELOAD:$PATH/libiomp5.so  python model_script.py`.

## Use Cases

### Example of multi Stream Module

Runtime extension support weight-sharing multi-stream inference on CPU. You just need to convert the original model into multi stream object and run the new multi stream object as normal.

The `MultiStreamModule` creates streams with numbers based on input parameter `num_streams`. If the number of cores inside `cpu_pool` is divisible by `num_streams`, the CPU cores in `cpu_pool` will be allocated equally to each stream. If the number of cores inside `cpu_pool` is not divisible by `num_streams` with remainder N, one extra core will be allocated to the first N streams. Similarly, the input tensor with batch size B will be allocated equally to each stream if divisible by `num_streams`, otherwise the first N streams will be allocated with extra input.

There are 2 motivations to use the `MultiStreamModule`: 1. Better Cache locality: With `MultiStreamModule`, the activations will be limited in the CPU cores allocated to this stream instead of the whole cpu_pool. 2. Reduce the OMP sync overhead: if one CPU core allocated to one stream, the whole execution needs to do OMP sync once after all streams finish execution instead of sync per layer. Thus, `MultiStreamModule` may benefit performance for inference of throughput mode.

```
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        x1 = self.conv(x)
        y = torch.flatten(x1, start_dim=1)
        return y

model = SimpleNet()
model.eval()
x = torch.rand(16, 64, 3, 3)

# Convert the model into multi_Stream_model
cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
multi_Stream_model = ipex.cpu.runtime.MultiStreamModule(model, num_streams=2, cpu_pool=cpu_pool)
y = multi_Stream_model(x)
```

### Example of Python API without Task

Runtime Extension provides API of `intel_extension_for_pytorch.cpu.runtime.pin` to a CPU Pool for binding physical cores. We can use it without the async task feature. There are 2 different ways to use `intel_extension_for_pytorch.cpu.runtime.pin`: use `decorator` or use `with` context.

#### Use the decorator

```
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        x1 = self.conv(x)
        y = torch.flatten(x1, start_dim=1)
        return y

model = SimpleNet()
model.eval()
x = torch.rand(64, 64, 3, 3)

cpu_pool = intel_extension_for_pytorch.cpu.runtime.CPUPool(node_id=0)
@intel_extension_for_pytorch.cpu.runtime.pin(cpu_pool)
def test(model, x):
    return model(x)

y_runtime = test(model, x)

y = model(x)
self.assertEqual(y, y_runtime)
```

#### Use the `with` context

```
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        x1 = self.conv(x)
        y = torch.flatten(x1, start_dim=1)
        return y
model = SimpleNet()
model.eval()
x = torch.rand(64, 64, 3, 3)

cpu_pool = intel_extension_for_pytorch.cpu.runtime.CPUPool(node_id=0)
with intel_extension_for_pytorch.cpu.runtime.pin(cpu_pool):
    y_runtime = model(x)

y = model(x)
self.assertEqual(y, y_runtime)
```

### Example of Python API with Task

Here is an example about how to use the Python API, suppose you have 2 modules to run.
* In native implementation, you will run these 2 modules one by one in sequence.
* With the support of runtime API, you can run these 2 modules simultaneously. Each modules runs on the corresponding cpu pool.

#### Native Implementation without runtime API

```
import torch

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        x1 = self.conv(x)
        y = torch.flatten(x1, start_dim=1)
        return y

x1 = torch.rand(64, 64, 3, 3)
x2 = torch.rand(64, 64, 3, 3)

model = SimpleNet()
model.eval()
y1 = model(x1)
y2 = model(x2)

```

#### Implementation with runtime API

```
import torch
import intel_extension_for_pytorch as ipex

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        x1 = self.conv(x)
        y = torch.flatten(x1, start_dim=1)
        return y

model = SimpleNet()
model.eval()

# Create the cpu pool and numa aware memory allocator
cpu_pool1 = ipex.runtime.CPUPool([0, 1, 2, 3])
cpu_pool2 = ipex.runtime.CPUPool([4, 5, 6, 7])

task1 = ipex.runtime.Task(model, cpu_pool1)
task2 = ipex.runtime.Task(model, cpu_pool2)

y1_future = task1(x1)
y2_future = task2(x2)

y1 = y1_future.get()
y2 = y2_future.get()
```

You will need to run the script with command `LD_PRELOAD=$LD_PRELOAD:$PATH/libiomp5.so python test.py`.

**Note**: you need to preload `Intel OMP library` if you build Intel® Extension for PyTorch\* with Runtime API support. `Intel OMP library` generally will be installed with anaconda. So, you can preload `libiomp5.so` in your conda environment.

### Example of C++ API without Task

The runtime extension provides purely C++ API without async Task.
```
#include <torch/torch.h>
#include "torch_ipex/csrc/cpu/runtime/CPUPool.h"

at::Tensor input_tensor = at::rand({100, 8276});
std::vector<int32_t> cpu_core_list({0});
torch_ipex::runtime::CPUPool cpu_pool(cpu_core_list);
{
    torch_ipex::runtime::WithCPUPool with_cpu_pool(std::move(cpu_pool));
    auto res = at::softmax(input_tensor, -1);
}
```

### Example of C++ API with Task

The runtime extension also provides purely C++ API with async Task.
```
#include <torch/torch.h>
#include "TaskExecutor.h"
#include "Task.h"

// Encapulate your application into a task function
at::Tensor taskfunction(at::Tensor input) {
    at::Tensor output;
    output = at::softmax(input, -1);
    return input;
}

// Create TaskExecutor
std::vector<int32_t> cpu_core_list({0, 1, 2, 3});
std::shared_ptr<TaskExecutor> task_executor = std::make_shared<TaskExecutor>(cpu_core_list);
// Create Task
Task<at::Tensor (*)(at::Tensor), at::Tensor> task(taskfunction, task_executor);

// Create input
at::Tensor input_tensor = at::rand({100, 8276});

// Submit task into TaskExecutor
auto res_future = task(std::move(input_tensor));

// Block until finish executation and get the result
auto res = res_future.get();

```

## Detail Design

### How the core binding is implemented

The Runtime Extension relies on the `kmp_*` API inside `iomp` share library to fulfill the core binding. The idea is that during the initialization of async threads, `kmp_*` API functions are invoked internally to start up an openmp group with specified number of worker threads. Each worker thread is then bound to the designated physical core(s) inside this openmp group. After initialization, any time you submit a task, the openmp group will serve the requested task.

### Design of Task

Task is an abstraction of computation based on PyTorch module and is scheduled asynchronously. When a task with specific `nn.Module`, `jit module` or `C++ function` is created, a sub-thread which is bound to this task initialized. During the initialization, an openmp worker group is created and bound to this sub-thread. After initialization, the sub-thread spins to wait input. When the main thread submits an input to this task, the sub-thread will wake up and execute the input. The main thread returns a `FutureTensor` and not block until an explicit `FutureTensor.get()` invoking to get the results executed in sub-thread.

### IOMP preload or load during the runtime

Since Runtime Extension rely on the APIs from IOMP, we need to preload IOMP before executing the application. And we want Intel® Extension for PyTorch\* default build with Runtime API enabled, which means it should work fine w/o loading IOMP if user didn't use the runtime API.

Here we choose to `dlopen` IOMP library during runtime. And we ensure the IOMP symbols initialized once globally.
