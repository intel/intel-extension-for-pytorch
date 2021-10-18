#include "CPUPool.h"
namespace torch_ipex {
namespace runtime {

namespace {
typedef void* kmp_affinity_mask_t;
typedef void (*kmp_create_affinity_mask_p)(kmp_affinity_mask_t*);
typedef int (*kmp_set_affinity_mask_proc_p)(int, kmp_affinity_mask_t*);
typedef int (*kmp_set_affinity_p)(kmp_affinity_mask_t*);
typedef void (*kmp_destroy_affinity_mask_p)(kmp_affinity_mask_t*);

kmp_create_affinity_mask_p kmp_create_affinity_mask_ext;
kmp_set_affinity_mask_proc_p kmp_set_affinity_mask_proc_ext;
kmp_set_affinity_p kmp_set_affinity_ext;
kmp_destroy_affinity_mask_p kmp_destroy_affinity_mask_ext;

std::once_flag
    iomp_symbol_loading_call_once_flag; // call_once_flag to ensure the iomp
                                        // symbol loaded once globally
bool iomp_symbol_loaded{
    false}; // Notice: iomp_symbol_loaded is not thread safe.
} // namespace

void loading_iomp_symbol() {
  void* handle = dlopen(NULL, RTLD_GLOBAL);

  if (dlsym(handle, "kmp_create_affinity_mask") == NULL ||
      dlsym(handle, "kmp_set_affinity_mask_proc") == NULL ||
      dlsym(handle, "kmp_set_affinity") == NULL ||
      dlsym(handle, "kmp_destroy_affinity_mask") == NULL) {
    iomp_symbol_loaded = false;
    return;
  }

  kmp_create_affinity_mask_ext =
      (kmp_create_affinity_mask_p)dlsym(handle, "kmp_create_affinity_mask");
  kmp_set_affinity_mask_proc_ext =
      (kmp_set_affinity_mask_proc_p)dlsym(handle, "kmp_set_affinity_mask_proc");
  kmp_set_affinity_ext = (kmp_set_affinity_p)dlsym(handle, "kmp_set_affinity");
  kmp_destroy_affinity_mask_ext =
      (kmp_destroy_affinity_mask_p)dlsym(handle, "kmp_destroy_affinity_mask");

  iomp_symbol_loaded = true;
  return;
}

bool is_runtime_ext_enabled() {
  std::call_once(iomp_symbol_loading_call_once_flag, loading_iomp_symbol);
  return iomp_symbol_loaded;
}

void init_runtime_ext() {
  std::call_once(iomp_symbol_loading_call_once_flag, loading_iomp_symbol);
  if (!iomp_symbol_loaded) {
    throw std::runtime_error(
        "Didn't preload IOMP before using the runtime API");
  }
  return;
}

void _pin_cpu_cores(const std::vector<int32_t>& cpu_core_list) {
  if (!iomp_symbol_loaded) {
    throw std::runtime_error(
        "Didn't preload IOMP before using the runtime API");
  }

  // Create the OMP thread pool and bind to cores of cpu_pools one by one
  omp_set_num_threads(cpu_core_list.size());
#pragma omp parallel num_threads(cpu_core_list.size())
  {
    // set the OMP thread affinity
    int thread_id = omp_get_thread_num();
    int phy_core_id = cpu_core_list[thread_id];
    kmp_affinity_mask_t mask;
    kmp_create_affinity_mask_ext(&mask);
    kmp_set_affinity_mask_proc_ext(phy_core_id, &mask);
    kmp_set_affinity_ext(&mask);
    kmp_destroy_affinity_mask_ext(&mask);
  }
  return;
}

CPUPool::CPUPool(const std::vector<int32_t>& cpu_core_list) {
  // Notice: We shouldn't load iomp symbol in sub_thread, otherwise race
  // condition happens.
  if (!is_runtime_ext_enabled()) {
    throw std::runtime_error(
        "Fail to init CPUPool. Didn't preload IOMP before using the runtime API.");
  }
  this->cpu_core_list = cpu_core_list;
}

const std::vector<int32_t>& CPUPool::get_cpu_core_list() const {
  return this->cpu_core_list;
}

CPUPool::~CPUPool() {}

} // namespace runtime
} // namespace torch_ipex
