Runtime Extension (Experimental)
================================

Intel® Extension for PyTorch\* Runtime Extension provides a couple of PyTorch frontend APIs for users to get finer-grained control of the thread runtime. It provides

1. Multi-stream inference via the Python frontend module `intel_extension_for_pytorch.cpu.runtime.MultiStreamModule`.
2. Spawn asynchronous tasks via the Python frontend module `intel_extension_for_pytorch.cpu.runtime.Task`.
3. Configure core bindings for OpenMP threads via the Python frontend `intel_extension_for_pytorch.cpu.runtime.pin`.

Please **note**: Intel® Extension for PyTorch\* Runtime extension is still in the **Experimental** stage. The API is subject to change. More detailed descriptions are available at [API Documentation page](../api_doc.html).

## Requirements

Intel® Extension for PyTorch\* Runtime Extension relies on `intel omp` to bind threads to cores. If you want to use it in your application, please start model script with extra flag: `LD_PRELOAD=$LD_PRELOAD:$PATH/libiomp5.so python model_script.py`.

## Use Cases

### Example of Multi Stream Module

Runtime extension supports weight-sharing multi-stream inference for throughput mode on CPU. You just need to convert the original model into multi stream model and run the new multi stream model as normal. The detailed description of parameters to create `MultiStreamModule` is available at [API Documentation page](../api_doc.html).

`MultiStreamModule` targets to improve performance of inference in throughput mode. We recommend creating a `MultiStreamModule` object with the `num_streams` parameter set to "AUTO" to heuristically decide the number of streams. Usually, it provides reasonable performance. However, it may still not be optimal for some cases (refer to the section [Performance recipes](#performance-recipes) for details) where manual tuning for the number of streams is needed.

The `MultiStreamModule` creates number of streams based on input parameter `num_streams` and bind cores to stream based on input parameter `cpu_pool`. If the number of cores inside `cpu_pool` is divisible by `num_streams`, the cores will be allocated equally to each stream. If the number of cores inside `cpu_pool` is not divisible by `num_streams` with remainder N, one extra core will be allocated to the first N streams. We suggest to set the `num_streams` as divisor of core number inside `cpu_pool`.

If the inputs' batchsize is larger than and divisible by ``num_streams``, the batchsize will be allocated equally to each stream. If batchsize is not divisible by ``num_streams`` with remainder N, one extra piece will be allocated to the first N streams. If the inputs' batchsize is less than ``num_streams``, only the first batchsize's streams are used with mini batch as one. We suggest to set inputs' batchsize larger than and divisible by ``num_streams``. When creating `MultiStreamModule`, if you leave num of streams as "AUTO", we suggest to set inputs' batchsize larger than and divisible by number of cores.

Firstly, creating some ExampleNets which will be used by below examples:
```
class ExampleNet1(torch.nn.Module):
    def __init__(self):
        super(ExampleNet1, self).__init__()
        self.conv = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        x1 = self.conv(x)
        y = torch.flatten(x1, start_dim=1)
        return y

class ExampleNet2(torch.nn.Module):
    def __init__(self):
        super(ExampleNet2, self).__init__()
        self.conv = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv2 = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x1, x2):
        y1 = self.conv(x1)
        y2 = self.conv2(x2)
        y = torch.flatten(y1, start_dim=1)
        return y1, y

model1 = ExampleNet1()
model1.eval()
x = torch.rand(16, 64, 3, 3)

with torch.no_grad():
    traced_model1 = torch.jit.trace(model1, x)
    traced_model1 = torch.jit.freeze(traced_model1)

model2 = ExampleNet2()
model2.eval()
x2 = torch.rand(16, 64, 3, 3)

with torch.no_grad():
    traced_model2 = torch.jit.trace(model2, (x, x2))
    traced_model2 = torch.jit.freeze(traced_model2)
```

#### Examples1: Basic Usage
Here is the example of a model with single tensor input/output. We create a CPUPool with all the cores available on numa node 0. And creating a `MultiStreamModule` with stream number of 2 to do inference.
```
# Convert the model into multi_Stream_model
cpu_pool = intel_extension_for_pytorch.cpu.runtime.CPUPool(node_id=0)
multi_Stream_model = intel_extension_for_pytorch.cpu.runtime.MultiStreamModule(traced_model1, num_streams=2, cpu_pool=cpu_pool)

with torch.no_grad():
    y = multi_Stream_model(x)
```

#### Examples2: Usage with "AUTO" setting
When creating a `MultiStreamModule`, we have default settings for `num_streams` ("AUTO") and `cpu_pool` (with all the cores available on numa node 0). For the `num_streams` of "AUTO", there are limitations to use with int8 datatype as we mentioned in below performance receipts section.
```
# Convert the model into multi_Stream_model
multi_Stream_model = intel_extension_for_pytorch.cpu.runtime.MultiStreamModule(traced_model1)

with torch.no_grad():
    y = multi_Stream_model(x)
```

#### Examples3: Usage for models with structure inputs/outputs
For module such as ExampleNet2 with structure input/output tensors, user needs to create `MultiStreamModuleHint` as input hint and output hint. `MultiStreamModuleHint` tells `MultiStreamModule` how to auto split the input into streams and concat the output from each steam.
```
# Convert the model into multi_Stream_model
cpu_pool = intel_extension_for_pytorch.cpu.runtime.CPUPool(node_id=0)
# Create the input hint object
input_hint = intel_extension_for_pytorch.cpu.runtime.MultiStreamModuleHint(0, 0)
# Create the output hint object
# When python module has multi output tensors, it will be auto pack into a tuple, So we pass a tuple(0, 0) to create the output_hint
output_hint = intel_extension_for_pytorch.cpu.runtime.MultiStreamModuleHint((0, 0))
multi_Stream_model = intel_extension_for_pytorch.cpu.runtime.MultiStreamModule(traced_model2, 
                                                                            num_streams=2,
                                                                            cpu_pool=cpu_pool,
                                                                            input_split_hint=input_hint,
                                                                            output_concat_hint=output_hint)

with torch.no_grad():
    y = multi_Stream_model(x, x2)
```

#### Performance recipes
There are 2 motivations to use the `MultiStreamModule`:
1. Better cache locality: With `MultiStreamModule`, the activations will be limited in the CPU cores allocated to this stream instead of the whole cpu_pool.
2. Reduce the OMP sync overhead: if one CPU core allocated to one stream, the whole execution needs to do OMP sync once after all streams finish execution instead of sync per layer.

Thus, `MultiStreamModule` may benefit performance for inference in throughput mode. However, the end-to-end performance is still subject to:
1. The kernels' efficiency which are different under different OMP threads' number.
2. The overhead of inputs' auto split and outputs' auto concat for each stream.
3. The overhead of pthread (stream async execution) wakes up and threads' synchronization after stream execution.

Below are some performance receipts we suggest to use for better multi stream performance.

* When creating `MultiStreamModule` with `torch.nn.Module` as imperative path module, each stream inside `MultiStreamModule` suffers the GIL issue when do inference together which hurts end-to-end performance. As the results, we suggest to create `MultiStreamModule` with the `torch.jit.ScriptModule`.

* For convolution network, `intel_extension_for_pytorch` has the quick path getting convolution primitive to mitigate overhead when `OMP_NUM_THREADS` is same between the phase of `torch.jit.trace` and model execution. To use this quick path for better performance, we suggest to set the `OMP_NUM_THREADS` environment before launch the model script. The suggested value of `OMP_NUM_THREADS` should equal to the threads number used by each stream. For example, creating `MultiStreamModule` as stream number of `s1`, CPUPool with core number `c1`, each stream will allocate threads number as `c1/s1`. Then we should set `OMP_NUM_THREADS` as this value.

* `Numactl` and the threads management in `MultiStreamModule` works in different levels. `MultiStreamModule` has the thread affinity setting for each stream which works in the thread level. However, for the python modules outside the stream, such as the dataloader, are out of radar for `MultiStreamModule`. As the result, we suggest to use `numactl -C core_ids -m node_id` for the process level core and memory resource management. For the core resource setting by `numactl`, suggest to set the same or superset of the core resource to create `CPUPool`. Otherwise, the behavior is undefined in current implementation.

#### Known issues
* Int8 data type does not support dynamic shape well. To avoid the performance issue, we suggest setting the batchsize to do `jit.trace` with same mini batchsize used by each stream. For example, creating `MultiStreamModule` as stream number of `s1`, input global batchsize as `gb`, each stream will inference with mini-batchsize as `gb/s1`. Then we should use this mini-batchsize value to do `jit.trace`. To be aware of the `num_streams` value, we suggest creating `MultiStreamModule` with `num_streams` setting explicitly instead of "AUTO". Due to the same limitation, the behavior that each stream inference with different mini batchsize of int8 data type is undefined and not supported.

### Example of asynchronous task

Here is an example about how to use the asynchronous task. With the support of runtime API, you can run 2 modules simultaneously. Each module runs on the corresponding cpu pool.

```
# Create the cpu pool and numa aware memory allocator
cpu_pool1 = ipex.runtime.CPUPool([0, 1, 2, 3])
cpu_pool2 = ipex.runtime.CPUPool([4, 5, 6, 7])

task1 = ipex.runtime.Task(traced_model1, cpu_pool1)
task2 = ipex.runtime.Task(traced_model1, cpu_pool2)

y1_future = task1(x)
y2_future = task2(x)

y1 = y1_future.get()
y2 = y2_future.get()
```

### Example of configuring core binding

Runtime Extension provides API of `intel_extension_for_pytorch.cpu.runtime.pin` to a CPU Pool for binding physical cores. We can use it without the async task feature. Here is the example to use `intel_extension_for_pytorch.cpu.runtime.pin` in the `with` context.

```
cpu_pool = intel_extension_for_pytorch.cpu.runtime.CPUPool(node_id=0)
with intel_extension_for_pytorch.cpu.runtime.pin(cpu_pool):
    y_runtime = traced_model1(x)
```

## Detail Design

### How the core binding is implemented

The Runtime Extension relies on the `kmp_*` API inside `iomp` share library to fulfill the core binding. The idea is that during the initialization of async threads, `kmp_*` API functions are invoked internally to start up an openmp group with specified number of worker threads. Each worker thread is then bound to the designated physical core(s) inside this openmp group. After initialization, any time you submit a task, the openmp group will serve the requested task.

### Design of Task

Task is an abstraction of computation based on PyTorch module and is scheduled asynchronously. When a task created with specific `nn.Module` or `jit module`, a sub-thread which is bound to this task initialized. During the initialization, an openmp worker group is created and bound to this sub-thread. After initialization, the sub-thread spins to wait input. When the main thread submits an input to this task, the sub-thread will wake up and execute the input. The main thread returns a `FutureTensor` and not block until an explicit `FutureTensor.get()` invoking to get the results executed in sub-thread.

### IOMP preload or load during the runtime

Since Runtime Extension rely on the APIs from IOMP, we need to preload IOMP before executing the application. And we want Intel® Extension for PyTorch\* default build with Runtime API enabled, which means it should work fine w/o loading IOMP if user didn't use the runtime API. Here we choose to `dlopen` IOMP library during runtime. And we ensure the IOMP symbols initialized once globally.
