# DPC++ Extension

## Introduction

C++ extension is a mechanism developed by PyTorch that lets you to create customized and highly efficient PyTorch operators defined out-of-source, i.e. separate from the PyTorch backend. (For more details, see https://pytorch.org/tutorials/advanced/cpp_extension.html). Based on the PyTorch C++ extension mechanism, Intel® Extension for PyTorch\* lets you to create PyTorch operators with custom DPC++ kernels to run on the XPU backend.

## Motivation and Example

This tutorial walks through a practical example of writing and using a DPC++ extension on the XPU backend with Intel® Extension for PyTorch\*.

## Writing a DPC++ Extension

You can build a DPC++ extension with `setuptools` or `CMake`. We’ll begin with the first approach and discuss the CMake extension later.

### Building with setuptools

For building with `setuptools`, we build our DPC++ extension by writing a `setup.py` script that uses `setuptools` to compile our C++ code. For the Long-Long-Term-Memory unit (LLTM), it looks like this:
```
from setuptools import setup
from intel_extension_for_pytorch.xpu.cpp_extension import DPCPPExtension, DpcppBuildExtension

setup(
    name='lltm',
    ext_modules=[
        DPCPPExtension('lltm_xpu', [
            'lltm_xpu.cpp',
            'lltm_xpu_kernel.cpp',
        ])
    ],
    cmdclass={
        'build_ext': DpcppBuildExtension
    })
```
In this code, `DPCPPExtension` is a convenience wrapper around `setuptools.Extension` that passes the correct include paths and sets the language of the extension to C++. The equivalent vanilla `setuptools` code would simply be:
```
Extension(
   name='lltm_xpu',
   sources=['lltm_xpu.cpp', 'lltm_xpu_kernel.cpp',],
   include_dirs=cpp_extension.include_paths(),
   language='c++')
```
`DpcppBuildExtension` performs a number of required configuration steps and checks and also manages compilation in the case of DPC++ extensions. And that’s all we really need to know about building DPC++ extensions for now.

 Let’s take a look at the implementation of our DPC++ extension, which goes into `lltm_xpu.cpp` and `lltm_xpu_kernel.cpp`.
After building the Python module with DPC++ extension, the `lltm_xpu` is available for importing as an extension plug-in.
```
import lltm_xpu
```
>**Note:**
>
> We still need to setup complier manually, example: ```CC=icx CXX=dpcpp python setup.py install```

### Building with CMake

For building with `CMake`, we build our DPC++ extension by writing a `CMakeLists.txt` file that uses CMake to build our C++ code. For the same example we showed using `setuptools`, the `CMakeLists.txt` looks like this:
CMakeLists.txt
```
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(lltm_xpu)

find_package(Python COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
find_package(IntelExtensionPytorch REQUIRED)

#The SYCL kernel should be compiled with "-fsycl"
set_source_files_properties(lltm_xpu_kernel.cpp PROPERTIES COMPILE_FLAGS "-fsycl")

add_library(lltm_xpu SHARED lltm_xpu.cpp lltm_xpu_kernel.cpp)
target_link_libraries(lltm_xpu "${TORCH_LIBRARIES}")
target_link_libraries(lltm_xpu "${TORCH_IPEX_LIBRARIES}")
target_include_directories(lltm_xpu PUBLIC "${Python_INCLUDE_DIRS}")
target_include_directories(lltm_xpu PUBLIC "${TORCH_IPEX_INCLUDE_DIRS}")

set_property(TARGET lltm_xpu PROPERTY CXX_STANDARD 17)
#DPCPP need 17
```
Commands for compilation:
```
$ cmake -DCMAKE_PREFIX_PATH=<torch/ipex cmake path in your env> -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=dpcpp ..
$ make
```
After build the python module with CMake, the `lltm_xpu` is also avalible for importing as a extension plug-in like setuptools method.
```
$ python
>>> import torch
>>> import intel_extension_for_pytorch
>>> import lltm_xpu
```
### Writing the DPC++ Op

The general strategy for writing a DPC++ extension is to write a C++ file that defines the functions that are called from Python, and binds those functions to Python with pybind11. The C++ functions do some checks and ultimately forward the calls to submit SYCL kernels. The `ipex.cpp_extension` package then takes care of compiling the C++ sources with a DPC++ compiler. 

Let's consider the PyTorch CUDA examples https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-mixed-c-cuda-extension. Here is how we implement it in DPC++ style:
```
#include <torch/extension.h>

#include <vector>

// XPU forward declarations

std::vector<torch::Tensor> lltm_xpu_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell);

std::vector<torch::Tensor> lltm_xpu_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights);
    
// C++ interface

#define CHECK_XPU(x) TORCH_CHECK(x.device().is_xpu(), #x " must be a XPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_XPU(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> lltm_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);
  CHECK_INPUT(old_h);
  CHECK_INPUT(old_cell);

  return lltm_xpu_forward(input, weights, bias, old_h, old_cell);
}

std::vector<torch::Tensor> lltm_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights) {
  CHECK_INPUT(grad_h);
  CHECK_INPUT(grad_cell);
  CHECK_INPUT(input_gate);
  CHECK_INPUT(output_gate);
  CHECK_INPUT(candidate_cell);
  CHECK_INPUT(X);
  CHECK_INPUT(gate_weights);
  CHECK_INPUT(weights);

  return lltm_xpu_backward(
      grad_h,
      grad_cell,
      new_cell,
      input_gate,
      output_gate,
      candidate_cell,
      X,
      gate_weights,
      weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lltm_forward, "LLTM forward (XPU)");
  m.def("backward", &lltm_backward, "LLTM backward (XPU)");
}
```

The bridge code checks and forwards the calls to functions that we’ll define in the DPC++ code file `lltm_xpu_kernel.cpp`. DPC++ supports compiling C++ naturally, thus we still have ATen and the C++ standard library available to us.

Let’s go through the DPC++ code step by step:

```
#include <torch/extension.h>

#include <ipex.h>

#include <vector>

template <typename scalar_t>
scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}
```

At the beginning of the code, we include `<torch/extension.h>` that will introduce all the torch definitions into the code. After that, the `<ipex.h>` line includes the SYCL header in DPC++. With the `<torch/extension.h>` and `<ipex.h>`, all the essential declarations have been included for writing the DPC++ kernel to run on the XPU backend device. The helper function `sigmoid` does the math calculation with the more efficient C++ language. Next are some more helper functions for LLTM:

```
template <typename scalar_t>
scalar_t d_sigmoid(scalar_t z) {
  const auto s = sigmoid(z);
  return (1.0 - s) * s;
}

template <typename scalar_t>
scalar_t d_tanh(scalar_t z) {
  const auto t = tanh(z);
  return 1 - (t * t);
}

template <typename scalar_t>
scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
  return fmax(0.0, z) + fmin(0.0, alpha * (exp(z) - 1.0));
}

template <typename scalar_t>
scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
  const auto e = exp(z);
  const auto d_relu = z < 0.0 ? 0.0 : 1.0;
  return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
}
```

Now we can implement the actual code for our extension with two functions in DPC++: 
* a function that performs operations we don’t wish to explicitly write by hand and calls into the function to submit the SYCL kernel, 
* a function that actual submits the SYCL kernel to the XPU device for the parts we want to speed up. 

For the forward pass, the first function looks like this:

```

std::vector<torch::Tensor> lltm_xpu_forward(
        torch::Tensor input,
        torch::Tensor weights,
        torch::Tensor bias,
        torch::Tensor old_h,
        torch::Tensor old_cell) {
  auto X = torch::cat({old_h, input}, /*dim=*/1);
  auto gates = torch::addmm(bias, X, weights.transpose(0, 1));

  const auto batch_size = old_cell.size(0);
  const auto state_size = old_cell.size(1);

  auto new_h = torch::zeros_like(old_cell);
  auto new_cell = torch::zeros_like(old_cell);
  auto input_gate = torch::zeros_like(old_cell);
  auto output_gate = torch::zeros_like(old_cell);
  auto candidate_cell = torch::zeros_like(old_cell);

  AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_xpu", ([&] {
    lltm_xpu_forward_kernel<scalar_t>(
          gates.data<scalar_t>(),
                  old_cell.data<scalar_t>(),
                  new_h.data<scalar_t>(),
                  new_cell.data<scalar_t>(),
                  input_gate.data<scalar_t>(),
                  output_gate.data<scalar_t>(),
                  candidate_cell.data<scalar_t>(),
                  state_size,
                  batch_size);
  }));

  return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
}
```

The purpose of `AT_DISPATCH_FLOATING_TYPES` is to take care of this dispatch for us. It takes a type (`gates.type()` in our case), a name (for error messages) and a lambda function. Inside this lambda function, the type alias `scalar_t` is available and is defined as the type that the tensor actually is at runtime in that context. As such, if we have a template function (which will submit the actual SYCL kernel), we can instantiate it with this `scalar_t` alias, and the correct function will be called. In this case, we also want to retrieve the data pointers of the tensors as pointers of that `scalar_t` type. If you wanted to dispatch over all types and not just floating point types (`Float` and `Double`), you can use `AT_DISPATCH_ALL_TYPES`.

Here's how to submit the actual kernel to the XPU device:

```
template <typename scalar_t>
void lltm_xpu_forward_kernel(
        const scalar_t* gates,
        const scalar_t* old_cell,
        scalar_t* new_h,
        scalar_t* new_cell,
        scalar_t* input_gate,
        scalar_t* output_gate,
        scalar_t* candidate_cell,
        size_t state_size,
        size_t batch_size) {

  const int threads = 1024;
  const int work_groups = (state_size + threads - 1) / threads;

  // define the kernel
  auto cgf = [&](sycl::handler& cgh) {
    auto kfn = [=](sycl::nd_item<2> item) {

      const int column = item.get_group(0) * item.get_group_range(0) + item.get_local_id(0);
      const int index = item.get_group(1) * state_size + column;
      const int gates_row = item.get_group(1) * (state_size * 3);

      if (column < state_size) {
        input_gate[index] = sigmoid(gates[gates_row + column]);
        output_gate[index] = sigmoid(gates[gates_row + state_size + column]);
        candidate_cell[index] = elu(gates[gates_row + 2 * state_size + column]);
        new_cell[index] =
                old_cell[index] + candidate_cell[index] * input_gate[index];
        new_h[index] = tanh(new_cell[index]) * output_gate[index];
      }

    };

    cgh.parallel_for(
            sycl::nd_range<2>(
                    sycl::range<2>(work_groups * threads, batch_size),
                    sycl::range<2>(threads, 1)),
            kfn);
  };

  // submit kernel
  auto device_type = c10::DeviceType::XPU;
  c10::impl::VirtualGuardImpl impl(device_type);
  c10::Stream dpcpp_stream = impl.getStream(c10::Device(device_type));
  auto queue = xpu::get_queue_from_stream(dpcpp_stream);

  queue.submit(cgf);
}
```

We're specifying that each work group has 1024 threads and that the entire GPU grid is split into as many work groups of 1 x 1024 threads as are required to fill our matrices with one thread per component. For example, if our state size was 2048 and our batch size 4, we’d launch a total of 4 x 2 = 8 work groups with 1024 threads each. If you are not familiar with the SYCL “work groups”, an introductory read about SYCL may help.

Note that the `c10::impl::VirtualGuardImpl` must get the current stream of the current XPU device and use the XPU API to get the corresponding SYCL underlaying queue. It can then submit the kernel to the queue for execution.

#### Using accessors

You can see in the SYCL kernel that we work directly on pointers with the right type. Indeed, working directly with high level type agnostic tensors inside SYCL kernels would be very inefficient.

However, this comes at a cost of ease of use and readability, especially for highly dimensional data. We can use torch's C++ utils to abstract access to high dimension data in the SYCL kernel directly.

The backwards pass follows much the same pattern but with the `torch::PackedTensorAccessor32`. You can get more information about these utils in torch documents:

```
template <typename scalar_t>
void lltm_xpu_backward_kernel(
        torch::PackedTensorAccessor32<scalar_t,2> d_old_cell,
        torch::PackedTensorAccessor32<scalar_t,3> d_gates,
        const torch::PackedTensorAccessor32<scalar_t,2> grad_h,
        const torch::PackedTensorAccessor32<scalar_t,2> grad_cell,
        const torch::PackedTensorAccessor32<scalar_t,2> new_cell,
        const torch::PackedTensorAccessor32<scalar_t,2> input_gate,
        const torch::PackedTensorAccessor32<scalar_t,2> output_gate,
        const torch::PackedTensorAccessor32<scalar_t,2> candidate_cell,
        const torch::PackedTensorAccessor32<scalar_t,3> gate_weights,
        size_t state_size,
        size_t batch_size) {

  const int threads = 1024;
  const int work_groups = (state_size + threads - 1) / threads;

  // define the kernel
  auto cgf = [&](sycl::handler& cgh) {
    auto kfn = [=](sycl::nd_item<2> item) {
      //batch index
      const int n = item.get_group(1);
      // column index
      const int c = item.get_group(0) * item.get_group_range(0) + item.get_local_id(0);
      auto d_gates_ = d_gates;
      auto d_old_cell_ = d_old_cell;
      if (c < d_gates.size(2)){
        const auto d_output_gate = tanh(new_cell[n][c]) * grad_h[n][c];
        const auto d_tanh_new_cell = output_gate[n][c] * grad_h[n][c];
        const auto d_new_cell =
                d_tanh(new_cell[n][c]) * d_tanh_new_cell + grad_cell[n][c];


        d_old_cell_[n][c] = d_new_cell;
        const auto d_candidate_cell = input_gate[n][c] * d_new_cell;
        const auto d_input_gate = candidate_cell[n][c] * d_new_cell;

        d_gates_[n][0][c] =
                d_input_gate * d_sigmoid(gate_weights[n][0][c]);
        d_gates_[n][1][c] =
                d_output_gate * d_sigmoid(gate_weights[n][1][c]);
        d_gates_[n][2][c] =
                d_candidate_cell * d_elu(gate_weights[n][2][c]);
      }
    };

    cgh.parallel_for(
            sycl::nd_range<2>(
                    sycl::range<2>(work_groups * threads, batch_size),
                    sycl::range<2>(threads, 1)),
            kfn);
  };

  // submit kernel
  auto device_type = c10::DeviceType::XPU;
  c10::impl::VirtualGuardImpl impl(device_type);
  c10::Stream dpcpp_stream = impl.getStream(c10::Device(device_type));
  auto queue = xpu::get_queue_from_stream(dpcpp_stream);

  queue.submit(cgf);
}

std::vector<torch::Tensor> lltm_xpu_backward(
        torch::Tensor grad_h,
        torch::Tensor grad_cell,
        torch::Tensor new_cell,
        torch::Tensor input_gate,
        torch::Tensor output_gate,
        torch::Tensor candidate_cell,
        torch::Tensor X,
        torch::Tensor gates,
        torch::Tensor weights) {
  auto d_old_cell = torch::zeros_like(new_cell);
  auto d_gates = torch::zeros_like(gates);

  const auto batch_size = new_cell.size(0);
  const auto state_size = new_cell.size(1);

  AT_DISPATCH_FLOATING_TYPES(X.type(), "lltm_backward_xpu", ([&] {
    lltm_xpu_backward_kernel<scalar_t>(
          d_old_cell.packed_accessor32<scalar_t,2>(),
                  d_gates.packed_accessor32<scalar_t,3>(),
                  grad_h.packed_accessor32<scalar_t,2>(),
                  grad_cell.packed_accessor32<scalar_t,2>(),
                  new_cell.packed_accessor32<scalar_t,2>(),
                  input_gate.packed_accessor32<scalar_t,2>(),
                  output_gate.packed_accessor32<scalar_t,2>(),
                  candidate_cell.packed_accessor32<scalar_t,2>(),
                  gates.packed_accessor32<scalar_t,3>(),
                  state_size,
                  batch_size);
  }));

  auto d_gate_weights = d_gates.reshape({batch_size, 3*state_size});
  auto d_weights = d_gate_weights.t().mm(X);
  auto d_bias = d_gate_weights.sum(/*dim=*/0, /*keepdim=*/true);

  auto d_X = d_gate_weights.mm(weights);
  auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates};
}
```
