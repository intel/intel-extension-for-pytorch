#include "TaskModule.h"

namespace torch_ipex {
namespace runtime {

py::object FutureTensor::get() {
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
}

TaskModule::TaskModule(
    const torch::jit::Module& script_module,
    const std::vector<int32_t>& cpu_core_list,
    bool traced_module)
    : script_module_(script_module) {
  this->task_executor = std::make_shared<TaskExecutor>(cpu_core_list);
  this->script_module_initialized_ = true;
}

TaskModule::TaskModule(
    const py::object& module,
    const std::vector<int32_t>& cpu_core_list)
    : module_(module) {
  this->task_executor = std::make_shared<TaskExecutor>(cpu_core_list);
  this->module_initialized_ = true;
}

TaskModule::TaskModule(
    const torch::jit::Module& script_module,
    const torch_ipex::runtime::CPUPool& cpu_pool,
    bool traced_module)
    : script_module_(script_module) {
  this->task_executor =
      std::make_shared<TaskExecutor>(cpu_pool.get_cpu_core_list());
  this->script_module_initialized_ = true;
}

TaskModule::TaskModule(
    const py::object& module,
    const torch_ipex::runtime::CPUPool& cpu_pool)
    : module_(module) {
  this->task_executor =
      std::make_shared<TaskExecutor>(cpu_pool.get_cpu_core_list());
  this->module_initialized_ = true;
}

TaskModule::~TaskModule() {
  pybind11::gil_scoped_release no_gil_guard;
  this->task_executor->stop_executor();
}

std::unique_ptr<FutureTensor> TaskModule::run_async(
    py::args&& args,
    py::kwargs&& kwargs) {
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
      using return_type = typename std::result_of<SubmitFunctionType(
          std::vector<at::IValue>)>::type;
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
    using return_type = typename std::result_of<SubmitFunctionType()>::type;
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
}

py::object TaskModule::run_sync(py::args&& args, py::kwargs&& kwargs) {
  // sync API to run application inside task
  std::unique_ptr<FutureTensor> future_tensor_result =
      this->run_async(std::move(args), std::move(kwargs));
  return future_tensor_result->get();
}

} // namespace runtime
} // namespace torch_ipex
