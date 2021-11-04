#include "CPUPool.h"
namespace torch_ipex {
namespace runtime {

namespace {
kmp_create_affinity_mask_p kmp_create_affinity_mask_ext;
kmp_set_affinity_mask_proc_p kmp_set_affinity_mask_proc_ext;
kmp_set_affinity_p kmp_set_affinity_ext;
kmp_destroy_affinity_mask_p kmp_destroy_affinity_mask_ext;
kmp_get_affinity_p kmp_get_affinity_ext;

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
      dlsym(handle, "kmp_get_affinity") == NULL ||
      dlsym(handle, "kmp_destroy_affinity_mask") == NULL) {
    iomp_symbol_loaded = false;
    return;
  }

  kmp_create_affinity_mask_ext =
      (kmp_create_affinity_mask_p)dlsym(handle, "kmp_create_affinity_mask");
  kmp_set_affinity_mask_proc_ext =
      (kmp_set_affinity_mask_proc_p)dlsym(handle, "kmp_set_affinity_mask_proc");
  kmp_set_affinity_ext = (kmp_set_affinity_p)dlsym(handle, "kmp_set_affinity");
  kmp_get_affinity_ext = (kmp_get_affinity_p)dlsym(handle, "kmp_get_affinity");
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
  if (!is_runtime_ext_enabled()) {
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

CPUPool get_cpu_pool_from_mask_affinity() {
  if (!is_runtime_ext_enabled()) {
    throw std::runtime_error(
        "Didn't preload IOMP before using the runtime API");
  }
  int max_number_threads = omp_get_max_threads();
  // init the vector<mask>
  std::vector<kmp_affinity_mask_t> threads_mask(max_number_threads);
#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    kmp_affinity_mask_t mask;
    kmp_create_affinity_mask_ext(&mask);
    kmp_get_affinity_ext(&mask);
    threads_mask[thread_id] = mask;
  }
  return CPUPool(std::move(threads_mask));
}

void set_mask_affinity_from_cpu_pool(const CPUPool& cpu_pool) {
  if (!is_runtime_ext_enabled()) {
    throw std::runtime_error(
        "Didn't preload IOMP before using the runtime API");
  }
  std::vector<kmp_affinity_mask_t> threads_mask =
      cpu_pool.get_cpu_affinity_mask();
  omp_set_num_threads(threads_mask.size());
#pragma omp parallel num_threads(threads_mask.size())
  {
    // we will destory the mask inside the CPUPool deconstructor
    int thread_id = omp_get_thread_num();
    kmp_affinity_mask_t mask = threads_mask[thread_id];
    kmp_set_affinity_ext(&mask);
  }
}

CPUPool::CPUPool(const std::vector<int32_t>& cpu_core_list) {
  // Notice: We shouldn't load iomp symbol in sub_thread, otherwise race
  // condition happens.
  if (!is_runtime_ext_enabled()) {
    throw std::runtime_error(
        "Fail to init CPUPool. Didn't preload IOMP before using the runtime API.");
  }
  this->cpu_core_list = cpu_core_list;
  this->cpu_core_list_initialized_ = true;
}

CPUPool::CPUPool(std::vector<kmp_affinity_mask_t>&& cpu_core_mask) {
  // Notice: We shouldn't load iomp symbol in sub_thread, otherwise race
  // condition happens.
  if (!is_runtime_ext_enabled()) {
    throw std::runtime_error(
        "Fail to init CPUPool. Didn't preload IOMP before using the runtime API.");
  }
  this->cpu_affinity_mask = cpu_core_mask;
  this->cpu_affinity_mask_initialized_ = true;
}

CPUPool::CPUPool(CPUPool&& source_cpu_pool) {
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
}

const std::vector<int32_t>& CPUPool::get_cpu_core_list() const {
  if (!this->cpu_core_list_initialized_) {
    throw std::runtime_error(
        "Fail to get_cpu_core_list. Current CPUPool object didn't express as cpu_core_list format.");
  }
  return this->cpu_core_list;
}

const std::vector<kmp_affinity_mask_t>& CPUPool::get_cpu_affinity_mask() const {
  if (!this->cpu_affinity_mask_initialized_) {
    throw std::runtime_error(
        "Fail to get_cpu_affinity_mask. Current CPUPool object didn't express as cpu_affinity_mask format.");
  }
  return this->cpu_affinity_mask;
}

bool CPUPool::is_cpu_core_list_initialized() const {
  return this->cpu_core_list_initialized_;
}

bool CPUPool::is_cpu_affinity_mask_initialized() const {
  return this->cpu_affinity_mask_initialized_;
}

CPUPool::~CPUPool() {
  if (this->cpu_affinity_mask_initialized_) {
    // If we are using the cpu_affinity_mask expression for CPUPool
    // Ensure we destory the mask in cpu_affinity_mask.
    for (int i = 0; i < this->cpu_affinity_mask.size(); i++) {
      kmp_affinity_mask_t mask = this->cpu_affinity_mask[i];
      kmp_destroy_affinity_mask_ext(&mask);
    }
  }
}

} // namespace runtime
} // namespace torch_ipex
