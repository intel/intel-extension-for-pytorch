#pragma once

#include <dlfcn.h>
#include <omp.h>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

#include <ATen/core/ivalue.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/jit/api/module.h>
#include "CPUPool.h"

namespace torch_ipex {
namespace runtime {

class TaskExecutor {
 public:
  explicit TaskExecutor(const std::vector<int32_t>& cpu_core_list);
  std::mutex& get_mutex();
  std::condition_variable& get_condition();
  bool is_stop();
  std::queue<std::function<void()>>& get_tasks();
  ~TaskExecutor();

 private:
  std::queue<std::function<void()>> tasks;
  std::shared_ptr<std::thread> worker;

  // Synchronization
  bool stop;
  std::mutex worker_mutex;
  std::condition_variable worker_condition;

  // Executor' thread_pool
  std::vector<int32_t> cpu_core_list;
};

} // namespace runtime
} // namespace torch_ipex
