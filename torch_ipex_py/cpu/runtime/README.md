Runtime Extension (Experimental)
==========================
## Runtime Extension
Intel® Extension for PyTorch\* Runtime extension: Runtime CPU Pool with thread to physical core binding. Built on top of the CPU Pool, there is also an async task feature. Please **note**: Intel® Extension for PyTorch\* Runtime extension is still in the **POC** stage. The API is subject to change.

## How to Use
Intel® Extension for PyTorch\* Runtime extension rely on `iomp` to bind the thread to cores. If you want to use it in your application, please run models with extra flag: `LD_PRELOAD=$LD_PRELOAD:$PATH/libiomp5.so  python model_script.py`.

## Use Cases
### Example of multi Stream Module
Runtime extension support weight-sharing multi-stream inference on CPU. You just need to convert the original model into multi stream object and run the new multi stream object as normal.
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
# batchsize must be divisible by instance number.
# instance number equals to cores_per_socket//cores_per_instance.
x = torch.rand(batchsize, 64, 3, 3)

# Convert the model into multi_Stream_model
cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
multi_Stream_model = ipex.cpu.runtime.MultiStreamModule(model, num_streams=2, cpu_pool=cpu_pool)
y = multi_Stream_model(x)
```

### Example of Python API without Task
Runtime extension provides API of `ipex.cpu.runtime.pin` to a CPU Pool for binding physical cores. We can use it without the async task feature. There are 2 different ways to use `ipex.cpu.runtime.pin`: use `decorator` or use `with` context.
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

cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
@ipex.cpu.runtime.pin(cpu_pool)
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

cpu_pool = ipex.cpu.runtime.CPUPool(node_id=0)
with ipex.cpu.runtime.pin(cpu_pool):
    y_runtime = model(x)

y = model(x)
self.assertEqual(y, y_runtime)
```

### Example of Python API with Task
Here is an example about how to use the Python API, suppose you have 2 modules to run.
* In the native implementation, you will run these 2 modules one by one in serial.
* With the support of runtime API, you can run these 2 modules simultaneously on corresponding each thread pool binding on corresponding cores.

#### Native Implementation w/o runtime API
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

#### Implementation w runtime API
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
Then you will run the test with command `LD_PRELOAD=$LD_PRELOAD:$PATH/libiomp5.so python test.py`
Please **note**: you need to preload `Intel OMP` if you build Intel® Extension for PyTorch\* with Runtime API support. `Intel OMP` generally will be installed with anaconda. So, you can preload `libiomp5.so` in your conda env.

### Example of C++ API with Task
The runtime extension also provides purely C++ API.
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
### How to implement the core binding
The Runtime Extension rely on the `kmp_*` API inside `iomp` share library to fulfill the core binding. The idea is during the initialization of async thread, we invoke the `kmp_*` API to start up a openmp group with specific number of worker threads. And we bind each of the worker thread to specific physical core. After initialization, any time you submit a task, the openmp group inside the async thread will serve as the active openmp group.

### Design of Async TaskExecutor
Async TaskExecutor is a thread which spins and waits for the input of task queue. When we create a task with specific `nn.Module`, `jit module` or `C++ function`, we will init TaskExecutor which is binding to this task. During the init of TaskExecutor, we will create a thread which initilizes openmp worker group, binding to specific cores, and spin to wait task input. When the main thread submit input to this task, the TaskExecutor will wake up to execute the task.

### IOMP preload or load during the runtime
Since Runtime Extension rely on the APIs from IOMP, we need to preload IOMP before executing the application. And we want Intel® Extension for PyTorch\* default build with Runtime API enabled, which means it should work fine w/o loading IOMP if user didn't use the runtime API.
Here we choose to `dlopen` IOMP library during runtime. And we ensure the IOMP symbols initialized once globally.
