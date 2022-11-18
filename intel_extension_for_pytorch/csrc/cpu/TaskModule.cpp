#include "TaskModule.h"

namespace torch_ipex {
namespace runtime {

/**********************Dummy Functions**************************/
CPUPool::CPUPool(const std::vector<int32_t>& cpu_core_list) {
#if 0
  this->cpu_core_list = filter_cores_by_thread_affinity(cpu_core_list);
  this->cpu_core_list_initialized_ = true;
#endif
}

CPUPool::CPUPool(std::vector<kmp_affinity_mask_t>&& cpu_core_mask) {
#if 0
  // Notice: We shouldn't load iomp symbol in sub_thread, otherwise race
  // condition happens.
  if (!is_runtime_ext_enabled()) {
    throw std::runtime_error(
        "Fail to init CPUPool. Didn't preload IOMP before using the runtime API.");
  }
  this->cpu_affinity_mask = cpu_core_mask;
  this->cpu_affinity_mask_initialized_ = true;
#endif
}

CPUPool::CPUPool(CPUPool&& source_cpu_pool) {
#if 0
  if (!source_cpu_pool.is_cpu_core_list_initialized() &&
      !source_cpu_pool.is_cpu_affinity_mask_initialized()) {
    throw std::runtime_error(
        "Fail to CPUPool move construct. Neither cpu_core_list_initialized_ and cpu_affinity_mask_initialized_ init.");
  }
  if (source_cpu_pool.is_cpu_core_list_initialized()) {
    this->cpu_core_list = std::move(
        const_cast<std::vector<int32_t>&>(source_cpu_pool.get_cpu_core_list()));
    this->cpu_core_list_initialized_ = true;
  } else {
    this->cpu_affinity_mask =
        std::move(const_cast<std::vector<kmp_affinity_mask_t>&>(
            source_cpu_pool.get_cpu_affinity_mask()));
    this->cpu_affinity_mask_initialized_ = true;
  }
#endif
}

const std::vector<int32_t>& CPUPool::get_cpu_core_list() const {
#if 0
  if (!this->cpu_core_list_initialized_) {
    throw std::runtime_error(
        "Fail to get_cpu_core_list. Current CPUPool object didn't express as cpu_core_list format.");
  }
  return this->cpu_core_list;
#else
  return this->cpu_core_list;
#endif
}

const std::vector<kmp_affinity_mask_t>& CPUPool::get_cpu_affinity_mask() const {
#if 0
  if (!this->cpu_affinity_mask_initialized_) {
    throw std::runtime_error(
        "Fail to get_cpu_affinity_mask. Current CPUPool object didn't express as cpu_affinity_mask format.");
  }
  return this->cpu_affinity_mask;
#else
  return this->cpu_affinity_mask;
#endif
}

bool CPUPool::is_cpu_core_list_initialized() const {
  // return this->cpu_core_list_initialized_;
  return true;
}

bool CPUPool::is_cpu_affinity_mask_initialized() const {
  // return this->cpu_affinity_mask_initialized_;
  return true;
}

CPUPool::~CPUPool() {
#if 0
  if (this->cpu_affinity_mask_initialized_) {
    // If we are using the cpu_affinity_mask expression for CPUPool
    // Ensure we destory the mask in cpu_affinity_mask.
    for (int i = 0; i < this->cpu_affinity_mask.size(); i++) {
      kmp_affinity_mask_t mask = this->cpu_affinity_mask[i];
      kmp_destroy_affinity_mask_ext(&mask);
    }
  }
#endif
}

py::object FutureTensor::get() {
#if 0
  CHECK(this->script_module_initialized_ ^ this->module_initialized_);
  if (this->script_module_initialized_) {
    c10::IValue res;
    {
      pybind11::gil_scoped_release no_gil_guard;
      res = this->future_script_tensor.get();
    }
    return torch::jit::toPyObject(std::move(res));
  } else {
    CHECK(this->module_initialized_);
    {
      pybind11::gil_scoped_release no_gil_guard;
      return this->future_tensor.get();
    }
  }
#else
  py::object dummy;
  return dummy;
#endif
}

TaskModule::TaskModule(
    const torch::jit::Module& script_module,
    const torch_ipex::runtime::CPUPool& cpu_pool,
    bool traced_module)
    : script_module_(script_module) {
#if 0
  this->task_executor = std::make_shared<TaskExecutor>(cpu_pool);
  this->script_module_initialized_ = true;
#endif
}

TaskModule::TaskModule(
    const py::object& module,
    const torch_ipex::runtime::CPUPool& cpu_pool)
    : module_(module) {
#if 0
  this->task_executor = std::make_shared<TaskExecutor>(cpu_pool);
  this->module_initialized_ = true;
#endif
}

TaskModule::~TaskModule() {
#if 0
  pybind11::gil_scoped_release no_gil_guard;
  this->task_executor->stop_executor();
#endif
}

std::unique_ptr<FutureTensor> TaskModule::run_async(
    py::args&& args,
    py::kwargs&& kwargs) {
#if 0
  CHECK(this->script_module_initialized_ ^ this->module_initialized_);
  // FutureTensor is going to return
  std::unique_ptr<FutureTensor> future_tensor_result =
      std::make_unique<FutureTensor>();

  // Get the thread_local status such as grad_mode and set it into the Async
  // thread
  auto grad_mode = at::GradMode::is_enabled();
  if (this->script_module_initialized_) {
    {
      pybind11::gil_scoped_release no_gil_guard;
      auto& function = script_module_.get_method("forward").function();
      std::vector<at::IValue> stack = torch::jit::createStackForSchema(
          function.getSchema(),
          std::move(args),
          // NOLINTNEXTLINE(performance-move-const-arg)
          std::move(kwargs),
          script_module_._ivalue());

      typedef std::function<c10::IValue(std::vector<at::IValue>)>
          SubmitFunctionType;
      typedef decltype(SubmitFunctionType()(stack)) return_type;
      auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(
          std::forward<SubmitFunctionType>(
              [&](std::vector<at::IValue> stack) -> c10::IValue {
                return function(std::move(stack));
              }),
          std::forward<std::vector<at::IValue>>(stack)));

      future_tensor_result->script_module_initialized_ = true;
      future_tensor_result->future_script_tensor = task->get_future();

      {
        std::unique_lock<std::mutex> lock(this->task_executor->get_mutex());
        // submit task to a stopping the pool is not allowed
        if (this->task_executor->is_stop())
          throw std::runtime_error(
              "submit TaskModule(py::object) on stopped ThreadPool");
        this->task_executor->get_tasks().emplace([task, grad_mode]() {
          // set the thread local status, such as the grad mode before
          // execuating the status
          at::GradMode::set_enabled(grad_mode);
          // execuate the task
          (*task)();
        });
      }
      this->task_executor->get_condition().notify_one();
    }
  } else {
    CHECK(this->module_initialized_);
    this->args = args;
    this->kwargs = kwargs;

    typedef std::function<py::object()> SubmitFunctionType;
    typedef decltype(SubmitFunctionType()()) return_type;
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        [&, this]() -> py::object {
          {
            pybind11::gil_scoped_acquire gil_guard;
            return this->module_(*(this->args), **(this->kwargs));
          }
        });

    future_tensor_result->module_initialized_ = true;
    future_tensor_result->future_tensor = task->get_future();

    {
      std::unique_lock<std::mutex> lock(this->task_executor->get_mutex());
      // submit task to a stopping the pool is not allowed
      if (this->task_executor->is_stop())
        throw std::runtime_error(
            "submit TaskModule(py::object) on stopped ThreadPool");
      this->task_executor->get_tasks().emplace([task, grad_mode]() {
        // set the thread local status, such as the grad mode before execuating
        // the status
        at::GradMode::set_enabled(grad_mode);
        // execuate the task
        (*task)();
      });
    }
    this->task_executor->get_condition().notify_one();
  }
  return future_tensor_result;
#else
  std::unique_ptr<FutureTensor> dummy;
  return dummy;
#endif
}

py::object TaskModule::run_sync(py::args&& args, py::kwargs&& kwargs) {
#if 0
  // sync API to run application inside task
  std::unique_ptr<FutureTensor> future_tensor_result =
      this->run_async(std::move(args), std::move(kwargs));
  return future_tensor_result->get();
#else
  py::object dummy;
  return dummy;
#endif
}

/***************************************************************/
} // namespace runtime
} // namespace torch_ipex
