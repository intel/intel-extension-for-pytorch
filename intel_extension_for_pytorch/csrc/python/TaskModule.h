#pragma once

#include <cassert>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>
#include "cpu/runtime/TaskExecutor.h"

namespace torch_ipex {
namespace runtime {
struct FutureTensor {
  // script module
  std::future<c10::IValue> future_script_tensor;
  bool script_module_initialized_{false};
  // nn module
  std::future<py::object> future_tensor;
  bool module_initialized_{false};
  // get the result
  py::object get();
};

/*TaskModule is used to handle Python input of nn.module or script module*/
class TORCH_API TaskModule {
 public:
  explicit TaskModule(
      const torch::jit::Module& module,
      const std::vector<int32_t>& cpu_core_list,
      bool traced_module);
  explicit TaskModule(
      const py::object& module,
      const std::vector<int32_t>& cpu_core_list);
  explicit TaskModule(
      const torch::jit::Module& module,
      const torch_ipex::runtime::CPUPool& cpu_pool,
      bool traced_module);
  explicit TaskModule(
      const py::object& module,
      const torch_ipex::runtime::CPUPool& cpu_pool);
  ~TaskModule();
  py::object run_sync(py::args&& args, py::kwargs&& kwargs); /*sync execution*/
  std::unique_ptr<FutureTensor> run_async(
      py::args&& args,
      py::kwargs&& kwargs); /*async execution in threadpool*/
 private:
  // Script module input
  torch::jit::Module script_module_;
  bool script_module_initialized_{false};
  // Module input
  py::object module_;
  bool module_initialized_{false};

  // TaskExecutor
  std::shared_ptr<TaskExecutor> task_executor;
  py::args args;
  py::kwargs kwargs;
};

} // namespace runtime
} // namespace torch_ipex
