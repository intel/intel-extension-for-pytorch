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
#include "TaskExecutor.h"

namespace torch_ipex {
namespace runtime {

// refer to http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2709.html
/*Task is used to handle input of general C++ functions*/
template <class F, class... Args>
class Task {
 public:
  explicit Task(F&& f, std::shared_ptr<TaskExecutor> task_executor);
  explicit Task(
      F&& f,
      Args&&... args,
      std::shared_ptr<TaskExecutor> task_executor);
  Task(const Task& task);
  ~Task();
  auto operator()(Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type>;

 private:
  F f;
  std::shared_ptr<TaskExecutor> task_executor;
};

template <class F, class... Args>
Task<F, Args...>::Task(F&& f, std::shared_ptr<TaskExecutor> task_executor) {
  this->f = f;
  this->task_executor = task_executor;
}

template <class F, class... Args>
Task<F, Args...>::Task(
    F&& f,
    Args&&... args,
    std::shared_ptr<TaskExecutor> task_executor) {
  this->f = f;
  this->task_executor = task_executor;
}

template <class F, class... Args>
Task<F, Args...>::Task(const Task& task) {
  this->f = task.f;
  this->task_executor = task.task_executor;
}

template <class F, class... Args>
Task<F, Args...>::~Task() {}

template <class F, class... Args>
auto Task<F, Args...>::operator()(Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
  using return_type = typename std::result_of<F(Args...)>::type;
  auto task = std::make_shared<std::packaged_task<return_type()>>(
      std::bind(std::forward<F>(this->f), std::forward<Args>(args)...));
  std::future<return_type> res = task->get_future();
  auto grad_mode = at::GradMode::is_enabled();
  {
    std::unique_lock<std::mutex> lock(this->task_executor->get_mutex());
    // submit task to a stopping the pool is not allowed
    if (this->task_executor->is_stop())
      throw std::runtime_error("Task submit on stopped ThreadPool");
    this->task_executor->get_tasks().emplace([task, grad_mode]() {
      // set the thread local status, such as the grad mode before execuating
      // the status
      at::GradMode::set_enabled(grad_mode);
      // execuate the task
      (*task)();
    });
  }

  this->task_executor->get_condition().notify_one();
  return res;
}

} // namespace runtime
} // namespace torch_ipex
