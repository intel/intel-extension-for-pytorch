#pragma once
#include <dlfcn.h>
#include <omp.h>
#include <mutex>
#include <vector>

namespace torch_ipex {
namespace runtime {

void _pin_cpu_cores(const std::vector<int32_t>& cpu_core_list);
bool is_runtime_ext_enabled();
void init_runtime_ext();

class CPUPool {
 public:
  explicit CPUPool(const std::vector<int32_t>& cpu_core_list);
  const std::vector<int32_t>& get_cpu_core_list() const;
  ~CPUPool();

 private:
  // thread_number inside CPUPool
  std::vector<int32_t> cpu_core_list;
};

} // namespace runtime
} // namespace torch_ipex
