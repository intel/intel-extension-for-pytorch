#include "CPUPool.h"

namespace torch_ipex {
namespace runtime {

std::vector<int32_t> available_cpu_cores = init_process_available_cores();

namespace {
// IOMP symbol
kmp_create_affinity_mask_p kmp_create_affinity_mask_ext;
kmp_set_affinity_mask_proc_p kmp_set_affinity_mask_proc_ext;
kmp_set_affinity_p kmp_set_affinity_ext;
kmp_destroy_affinity_mask_p kmp_destroy_affinity_mask_ext;
kmp_get_affinity_p kmp_get_affinity_ext;
kmp_get_affinity_max_proc_p kmp_get_affinity_max_proc_ext;

// IOMP symbol loading control flag
std::once_flag
    iomp_symbol_loading_call_once_flag; // call_once_flag to ensure the iomp
                                        // symbol loaded once globally
std::atomic<bool> iomp_symbol_loaded{false};

// current_cpu_core_list is only used to cache the cpu_core_list setting
// of _pin_cpu_cores. It's thread_local, so different task thread can have
// different settings to support task API.
thread_local std::vector<int32_t> current_cpu_core_list{-1};
} // namespace

void loading_iomp_symbol() {
  void* handle = dlopen(NULL, RTLD_NOW | RTLD_GLOBAL);
  if (handle == NULL || dlsym(handle, "kmp_create_affinity_mask") == NULL ||
      dlsym(handle, "kmp_set_affinity_mask_proc") == NULL ||
      dlsym(handle, "kmp_set_affinity") == NULL ||
      dlsym(handle, "kmp_get_affinity") == NULL ||
      dlsym(handle, "kmp_destroy_affinity_mask") == NULL ||
      dlsym(handle, "kmp_get_affinity_max_proc") == NULL) {
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
  kmp_get_affinity_max_proc_ext =
      (kmp_get_affinity_max_proc_p)dlsym(handle, "kmp_get_affinity_max_proc");

  iomp_symbol_loaded = true;
  return;
}

std::vector<int32_t> get_process_available_cores() {
  return torch_ipex::runtime::available_cpu_cores;
}

// Init the available cores when process starts up
std::vector<int32_t> init_process_available_cores() {
  std::vector<int32_t> available_cpu_cores_internal;

  if (is_runtime_ext_enabled()) {
    // When IOMP preloaded.
    // Step1: Get the main thread affinity information:
    // 2 knowning external command may change it during process starts up:
    //   * External Numactl.
    //   * Preload IOMP with KMP_AFFINITY settings.
    // We need to save this information firstly and restore it later.
    // Since main thread affinity may be changed in step2, when to query
    // available cores.
    kmp_affinity_mask_t main_thread_pre_mask;
    kmp_create_affinity_mask_ext(&main_thread_pre_mask);
    kmp_get_affinity_ext(&main_thread_pre_mask);

    // Step2: Test which cores the thread has privilege to use.
    // Step2 should work with IOMP preloaded. If not, We also need to support
    // it. Because we need to create a default cpupool in MultiStreamModule when
    // IPEX start up. We shouldn't break this behavior when IOMP is not
    // preloaded. But this information makes no sense when IOMP not preloaded.
    int nproc_online = kmp_get_affinity_max_proc_ext();
    for (int i = 0; i < nproc_online; i++) {
      kmp_affinity_mask_t mask;
      kmp_create_affinity_mask_ext(&mask);
      auto resutl1 = kmp_set_affinity_mask_proc_ext(i, &mask);
      auto resutl2 = kmp_set_affinity_ext(&mask);
      kmp_destroy_affinity_mask_ext(&mask);
      if ((resutl1 == 0) && (resutl2 == 0)) {
        // success to change main thread affinity to this core.
        // It means main thread has privilege to use this core.
        available_cpu_cores_internal.emplace_back(i);
      }
    }

    // Step3: restore the main thread affinity since it will be changed in
    // step2.
    kmp_set_affinity_ext(&main_thread_pre_mask);
    kmp_destroy_affinity_mask_ext(&main_thread_pre_mask);
  } else {
    // When IOMP didn't preload, We support for IPEX init without preload IOMP.
    // But this information makes no sense and shouldn't be used without preload
    // IOMP.
    // Step1: Get the main thread affinity
    cpu_set_t main_thread_pre_set;
    CPU_ZERO(&main_thread_pre_set);
    if (sched_getaffinity(0, sizeof(cpu_set_t), &main_thread_pre_set) != 0) {
      throw std::runtime_error("Fail to get the thread affinity information");
    }

    // Step2:
    // https://man7.org/linux/man-pages/man3/sysconf.3.html
    // Please note these value may not be standard.
    // _SC_NPROCESSORS_ONLN: processors available, may be less than
    // _SC_NPROCESSORS_CONF because processors may be offline.
    // _SC_NPROCESSORS_CONF: processors configured.
    int nproc_online = sysconf(_SC_NPROCESSORS_CONF);
    for (int i = 0; i < nproc_online; i++) {
      if (CPU_ISSET(i, &main_thread_pre_set)) {
        available_cpu_cores_internal.emplace_back(i);
      }
    }

    // Step3: restore main thread affinity
    if (sched_setaffinity(0, sizeof(cpu_set_t), &main_thread_pre_set) != 0) {
      throw std::runtime_error(
          "Fail to restore the main thread affinity in step3.");
    }
  }

  return available_cpu_cores_internal;
}

std::vector<int32_t> filter_cores_by_thread_affinity(
    const std::vector<int32_t>& cpu_core_list) {
  std::vector<int32_t> filter_cpu_core_list;
  for (int i = 0; i < cpu_core_list.size(); i++) {
    if (std::find(
            torch_ipex::runtime::available_cpu_cores.begin(),
            torch_ipex::runtime::available_cpu_cores.end(),
            cpu_core_list[i]) !=
        torch_ipex::runtime::available_cpu_cores.end()) {
      filter_cpu_core_list.emplace_back(cpu_core_list[i]);
    }
  }
  if (filter_cpu_core_list.size() == 0) {
    // When user tries to create a empty CPUPool.
    throw std::runtime_error(
        "Can't find available core id in current process with the core ids of CPUPool construction.");
  }
  return filter_cpu_core_list;
}

inline bool do_load_iomp_symbol() {
  // If invoking std::call_once concurrently, only one thread will invoke the
  // function as active execution. The other threads as passive execution will
  // not return until the finish of active execution.
  std::call_once(iomp_symbol_loading_call_once_flag, loading_iomp_symbol);
  return iomp_symbol_loaded;
}

bool is_runtime_ext_enabled() {
  return do_load_iomp_symbol();
}

void init_runtime_ext() {
  if (!do_load_iomp_symbol()) {
    throw std::runtime_error(
        "Didn't preload IOMP before using the runtime API");
  }
  return;
}

void _pin_cpu_cores(const torch_ipex::runtime::CPUPool& cpu_pool) {
  const std::vector<int32_t>& cpu_core_list = cpu_pool.get_cpu_core_list();
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
  // Cache the cpu_core_list for query.
  current_cpu_core_list = cpu_core_list;
  return;
}

bool is_same_core_affinity_setting(const std::vector<int32_t>& cpu_core_list) {
  return current_cpu_core_list == cpu_core_list;
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
  this->cpu_core_list = filter_cores_by_thread_affinity(cpu_core_list);
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