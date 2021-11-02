#include "TaskExecutor.h"

namespace torch_ipex {
namespace runtime {

TaskExecutor::TaskExecutor(const std::vector<int32_t>& cpu_core_list) {
  // Notice: We shouldn't load iomp symbol in sub_thread, otherwise race
  // condition happens.
  if (!is_runtime_ext_enabled()) {
    throw std::runtime_error(
        "Fail to init TaskExecutor. Didn't preload IOMP before using the runtime API.");
  }
  this->cpu_core_list = cpu_core_list;
  this->stop = false;

  this->worker = std::make_shared<std::thread>([&, this] {
    _pin_cpu_cores(this->cpu_core_list);
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(this->worker_mutex);
        this->worker_condition.wait(
            lock, [this] { return this->stop || !this->tasks.empty(); });

        if (this->stop && this->tasks.empty())
          return;

        task = std::move(this->tasks.front());
        this->tasks.pop();
      }
      task();
    }
  });
}

std::mutex& TaskExecutor::get_mutex() {
  return this->worker_mutex;
}

std::condition_variable& TaskExecutor::get_condition() {
  return this->worker_condition;
}

bool TaskExecutor::is_stop() {
  return this->stop;
}

std::queue<std::function<void()>>& TaskExecutor::get_tasks() {
  return this->tasks;
}

void TaskExecutor::stop_executor() {
  bool should_wait_worker_join = false;
  {
    std::unique_lock<std::mutex> lock(this->worker_mutex);
    if (this->stop == false) {
      should_wait_worker_join = true;
      this->stop = true;
    }
  }
  if (should_wait_worker_join) {
    this->worker_condition.notify_all();
    this->worker->join();
  }
  return;
}

TaskExecutor::~TaskExecutor() {
  this->stop_executor();
}

} // namespace runtime
} // namespace torch_ipex
