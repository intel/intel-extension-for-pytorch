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
// #include "csrc/cpu/runtime/TaskExecutor.h"

/**********************Dummy Functions**************************/
#include <stdio.h>
#define DUMMY_FE_TRACE printf("[dummy_cpu_fe] %s\n", __FUNCTION__);
/***************************************************************/

namespace torch_ipex {
namespace runtime {
struct TORCH_API FutureTensor {
  // script module
  std::future<c10::IValue> future_script_tensor;
  bool script_module_initialized_{false};
  // nn module
  std::future<py::object> future_tensor;
  bool module_initialized_{false};
  // get the result
  py::object get();
};

typedef void* kmp_affinity_mask_t;
typedef void (*kmp_create_affinity_mask_p)(kmp_affinity_mask_t*);
typedef int (*kmp_set_affinity_mask_proc_p)(int, kmp_affinity_mask_t*);
typedef int (*kmp_set_affinity_p)(kmp_affinity_mask_t*);
typedef void (*kmp_destroy_affinity_mask_p)(kmp_affinity_mask_t*);
typedef int (*kmp_get_affinity_p)(kmp_affinity_mask_t*);
typedef int (*kmp_get_affinity_max_proc_p)();

class CPUPool {
 public:
  explicit CPUPool(const std::vector<int32_t>& cpu_core_list);
  explicit CPUPool(std::vector<kmp_affinity_mask_t>&& cpu_core_mask);
  CPUPool(CPUPool&& source_cpu_pool);

  const std::vector<int32_t>& get_cpu_core_list() const;
  const std::vector<kmp_affinity_mask_t>& get_cpu_affinity_mask() const;
  bool is_cpu_core_list_initialized() const;
  bool is_cpu_affinity_mask_initialized() const;
  ~CPUPool();

 private:
  // CPUPool has 2 kinds of expression: 1. cpu_core_list 2. cpu_affinity_mask
  // Notice: only one of these 2 expressions allow to use for specific CPUPool
  // object.
  std::vector<int32_t> cpu_core_list;
  bool cpu_core_list_initialized_{false};
  std::vector<kmp_affinity_mask_t> cpu_affinity_mask;
  bool cpu_affinity_mask_initialized_{false};

  // Put deleted function into private.
  CPUPool() = delete;
  CPUPool(const CPUPool& source_cpu_pool) =
      delete; // avoid potential risk of double destory masks.
  CPUPool& operator=(const CPUPool& source_cpu_pool) =
      delete; // avoid potential risk of double destory masks.
  CPUPool& operator=(CPUPool&& source_cpu_pool) = delete;
};

/*TaskModule is used to handle Python input of nn.module or script module*/
class TORCH_API TaskModule {
 public:
  explicit TaskModule(
      const torch::jit::Module& module,
      const torch_ipex::runtime::CPUPool& cpu_pool,
      bool traced_module);
  explicit TaskModule(
      const py::object& module,
      const torch_ipex::runtime::CPUPool& cpu_pool);
  TaskModule(const TaskModule& task_module) = delete;
  TaskModule(TaskModule&& task_module) = delete;
  TaskModule& operator=(const TaskModule& task_module) = delete;
  TaskModule& operator=(TaskModule&& task_module) = delete;
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
  /**********************Dummy Functions**************************/
  // std::shared_ptr<TaskExecutor> task_executor;
  /***************************************************************/

  py::args args;
  py::kwargs kwargs;
};

} // namespace runtime
} // namespace torch_ipex
