#pragma once

#include <CL/sycl.hpp>
#include <utils/Macros.h>
#include <utils/Profiler.h>
#include <utils/Settings.h>
#include <utils/Timer.h>

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

// alias for dpcpp namespace
namespace DPCPP = cl::sycl;

// Kernel inside print utils
#if defined(__SYCL_DEVICE_ONLY__)
#define DPCPP_CONSTANT __attribute__((opencl_constant))
#else
#define DPCPP_CONSTANT
#endif

#define DPCPP_KER_STRING(var, str) static const DPCPP_CONSTANT char var[] = str;

#if (__SYCL_COMPILER_VERSION >= 20200930)
#define DPCPP_KER_PRINTF DPCPP::ONEAPI::experimental::printf
#else
#define DPCPP_KER_PRINTF DPCPP::intel::experimental::printf
#endif

#define DPCPP_K_PRINT(fmt_str, ...)           \
  {                                           \
    DPCPP_KER_STRING(fmt_var, fmt_str);       \
    DPCPP_KER_PRINTF(fmt_var, ##__VA_ARGS__); \
  }

// macro-s for dpcpp command queue and kernel function
// Kernel name format
#define DPCPP_K_NAME(x) __##x##_dpcpp_kernel

// Kernel instance in parallel_for invocation
#define DPCPP_K(k, ...) DPCPP_K_NAME(k)<char, ##__VA_ARGS__>

// Global unique kernel declaration with variable arguments
// Full list of type arguments is NOT needed
#define DPCPP_DEF_K1(k)                                   \
  template <typename __dummy_typename_dpcpp, typename...> \
  class DPCPP_K_NAME(k) {}

// Global unique kernel declaration with variable arguments and constant
// specialization Full list type arguments is MUST due to constant
// specialization in class template
#define DPCPP_DEF_K2(k, ...)                                \
  template <typename __dummy_typename_dpcpp, ##__VA_ARGS__> \
  class DPCPP_K_NAME(k) {}

// Kernel function implementation
#define DPCPP_Q_KFN(...) [=](__VA_ARGS__)

// Command group function implementation
#define DPCPP_Q_CGF(h) [&](DPCPP::handler & h)

#define DPCPP_E_FORCE_SYNC(e)                                    \
  {                                                              \
    static auto force_sync = Settings::I().is_force_sync_exec(); \
    if (force_sync) {                                            \
      (e).wait_and_throw();                                      \
    }                                                            \
  }

#define DPCPP_EXT_SUBMIT(q, str, ker_submit)                                 \
  {                                                                          \
    static auto verbose = Settings::I().get_verbose_level();                 \
    if (verbose) {                                                           \
      IPEX_TIMER(t, verbose, __func__);                                      \
      auto start_evt = (q).submit_barrier();                                 \
      t.now("start barrier");                                                \
      auto e = (ker_submit);                                                 \
      t.now("submit");                                                       \
      auto end_evt = (q).submit_barrier();                                   \
      t.now("end barrier");                                                  \
      e.wait_and_throw();                                                    \
      t.now("event wait");                                                   \
      dpcpp_log((str), start_evt, end_evt);                                  \
      static auto event_prof_enabled =                                       \
          Settings::I().is_event_profiling_enabled();                        \
      if (event_prof_enabled) {                                              \
        start_evt.wait_and_throw();                                          \
        end_evt.wait_and_throw();                                            \
        auto se_end =                                                        \
            start_evt                                                        \
                .template get_profiling_info<dpcpp_event_profiling_end>();   \
        auto ee_start =                                                      \
            end_evt                                                          \
                .template get_profiling_info<dpcpp_event_profiling_start>(); \
        t.event_duration((ee_start - se_end) / 1000.0);                      \
      }                                                                      \
    } else if (is_profiler_enabled()) {                                      \
      auto start_evt = (q).submit_barrier();                                 \
      auto e = (ker_submit);                                                 \
      auto end_evt = (q).submit_barrier();                                   \
      dpcpp_mark((str), start_evt, end_evt);                                 \
      DPCPP_E_FORCE_SYNC(e);                                                 \
    } else {                                                                 \
      auto e = (ker_submit);                                                 \
      DPCPP_E_FORCE_SYNC(e);                                                 \
    }                                                                        \
    (q).throw_asynchronous();                                                \
  }

#define DPCPP_Q_SYNC_SUBMIT_VERBOSE(q, cgf, ...)                               \
  {                                                                            \
    IPEX_TIMER(t, verbose, __func__);                                          \
    auto e = (q).submit((cgf), ##__VA_ARGS__);                                 \
    t.now("submit");                                                           \
    e.wait_and_throw();                                                        \
    t.now("event wait");                                                       \
    dpcpp_log("dpcpp_kernel", e);                                              \
    static auto event_prof_enabled =                                           \
        Settings::I().is_event_profiling_enabled();                            \
    if (event_prof_enabled) {                                                  \
      auto e_start =                                                           \
          e.template get_profiling_info<dpcpp_event_profiling_start>();        \
      auto e_end = e.template get_profiling_info<dpcpp_event_profiling_end>(); \
      t.event_duration((e_end - e_start) / 1000.0);                            \
    }                                                                          \
  }

#define DPCPP_Q_SYNC_SUBMIT(q, cgf, ...)                      \
  {                                                           \
    static auto verbose = Settings::I().get_verbose_level();  \
    if (verbose) {                                            \
      DPCPP_Q_SYNC_SUBMIT_VERBOSE((q), (cgf), ##__VA_ARGS__); \
    } else {                                                  \
      auto e = (q).submit((cgf), ##__VA_ARGS__);              \
      dpcpp_log("dpcpp_kernel", e);                           \
      e.wait_and_throw();                                     \
    }                                                         \
  }

#define DPCPP_Q_ASYNC_SUBMIT(q, cgf, ...)                     \
  {                                                           \
    static auto verbose = Settings::I().get_verbose_level();  \
    if (verbose) {                                            \
      DPCPP_Q_SYNC_SUBMIT_VERBOSE((q), (cgf), ##__VA_ARGS__); \
    } else {                                                  \
      auto e = (q).submit((cgf), ##__VA_ARGS__);              \
      (q).throw_asynchronous();                               \
      dpcpp_log("dpcpp_kernel", e);                           \
      DPCPP_E_FORCE_SYNC(e);                                  \
    }                                                         \
  }

// the descriptor as entity attribute
#define DPCPP_HOST // for host only
#define DPCPP_DEVICE // for device only
#define DPCPP_BOTH // for both host and device

// dpcpp device configuration
// TODO: set subgourp size with api get_max_sub_group_size
#define DPCPP_SUB_GROUP_SIZE (1L)

#define NUM_THREADS (C10_WARP_SIZE * 2)
#define THREAD_WORK_SIZE 4
#define BLOCK_WORK_SIZE (THREAD_WORK_SIZE * NUM_THREADS)

// info value type
template <typename T, T v>
using dpcpp_info_t = typename DPCPP::info::param_traits<T, v>::return_type;

// dpcpp platform info
static constexpr auto dpcpp_platform_name = DPCPP::info::platform::name;

// dpcpp device info
// Returns the device name of this SYCL device
static constexpr auto dpcpp_dev_name = DPCPP::info::device::name;
// Returns the device type associated with the device.
static constexpr auto dpcpp_dev_type = DPCPP::info::device::device_type;
// Returns the SYCL platform associated with this SYCL device.
static constexpr auto dpcpp_dev_platform = DPCPP::info::device::platform;
// Returns the vendor of this SYCL device.
static constexpr auto dpcpp_dev_vendor = DPCPP::info::device::vendor;
// Returns a backend-defined driver version as a std::string.
static constexpr auto dpcpp_dev_driver_version =
    DPCPP::info::device::driver_version;
// Returns the SYCL version as a std::string in the form:
// <major_version>.<minor_version>
static constexpr auto dpcpp_dev_version = DPCPP::info::device::version;
// Returns a string describing the version of the SYCL backend associated with
// the device. static constexpr auto dpcpp_dev_backend_version =
// DPCPP::info::device::backend_version; Returns true if the SYCL device is
// available and returns false if the device is not available.
static constexpr auto dpcpp_dev_is_available =
    DPCPP::info::device::is_available;
// Returns the maximum size in bytes of the arguments that can be passed to a
// kernel.
static constexpr auto dpcpp_dev_max_param_size =
    DPCPP::info::device::max_parameter_size;
// Returns the number of parallel compute units available to the device.
static constexpr auto dpcpp_dev_max_compute_units =
    DPCPP::info::device::max_compute_units;
// Returns the maximum dimensions that specify the global and local work-item
// IDs used by the data parallel execution model.
static constexpr auto dpcpp_dev_max_work_item_dims =
    DPCPP::info::device::max_work_item_dimensions;
// Returns the maximum number of workitems that are permitted in a work-group
// executing a kernel on a single compute unit.
static constexpr auto dpcpp_dev_max_work_group_size =
    DPCPP::info::device::max_work_group_size;
// Returns the maximum number of subgroups in a work-group for any kernel
// executed on the device
static constexpr auto dpcpp_dev_max_num_subgroup =
    DPCPP::info::device::max_num_sub_groups;
// Returns a std::vector of size_t containing the set of sub-group sizes
// supported by the device
static constexpr auto dpcpp_dev_subgroup_sizes =
    DPCPP::info::device::sub_group_sizes;
// Returns the maximum configured clock frequency of this SYCL device in MHz.
static constexpr auto dpcpp_dev_max_clock_freq =
    DPCPP::info::device::max_clock_frequency;
// Returns the default compute device address space size specified as an
// unsigned integer value in bits. Must return either 32 or 64.
static constexpr auto dpcpp_dev_address_bits =
    DPCPP::info::device::address_bits;
// Returns the maximum size of memory object allocation in bytes
static constexpr auto dpcpp_dev_max_alloc_size =
    DPCPP::info::device::max_mem_alloc_size;
// Returns the minimum value in bits of the largest supported SYCL built-in data
// type if this SYCL device is not of device type info::device_type::custom.
static constexpr auto dpcpp_dev_mem_base_addr_align =
    DPCPP::info::device::mem_base_addr_align;
// Returns a std::vector of info::fp_config describing the half precision
// floating-point capability of this SYCL device.
static constexpr auto dpcpp_dev_half_fp_config =
    DPCPP::info::device::half_fp_config;
// Returns a std::vector of info::fp_config describing the single precision
// floating-point capability of this SYCL device.
static constexpr auto dpcpp_dev_single_fp_config =
    DPCPP::info::device::single_fp_config;
// Returns a std::vector of info::fp_config describing the double precision
// floatingpoint capability of this SYCL device.
static constexpr auto dpcpp_dev_double_fp_config =
    DPCPP::info::device::double_fp_config;
// Returns the size of global device memory in bytes
static constexpr auto dpcpp_dev_global_mem_size =
    DPCPP::info::device::global_mem_size;
// Returns the type of global memory cache supported.
static constexpr auto dpcpp_dev_global_mem_cache_type =
    DPCPP::info::device::global_mem_cache_type;
// Returns the size of global memory cache in bytes.
static constexpr auto dpcpp_dev_global_mem_cache_size =
    DPCPP::info::device::global_mem_cache_size;
// Returns the size of global memory cache line in bytes.
static constexpr auto dpcpp_dev_global_mem_cache_line_size =
    DPCPP::info::device::global_mem_cache_line_size;
// Returns the type of local memory supported.
static constexpr auto dpcpp_dev_local_mem_type =
    DPCPP::info::device::local_mem_type;
// Returns the size of local memory arena in bytes.
static constexpr auto dpcpp_dev_local_mem_size =
    DPCPP::info::device::local_mem_size;
// Returns the maximum number of sub-devices that can be created when this SYCL
// device is partitioned.
static constexpr auto dpcpp_dev_max_sub_devices =
    DPCPP::info::device::partition_max_sub_devices;
// Returns the resolution of device timer in nanoseconds.
static constexpr auto dpcpp_dev_profiling_resolution =
    DPCPP::info::device::profiling_timer_resolution;

// dpcpp event info
static constexpr auto dpcpp_event_exec_stat =
    DPCPP::info::event::command_execution_status;
// dpcpp event command status
static constexpr auto dpcpp_event_cmd_stat_submitted =
    DPCPP::info::event_command_status::submitted;
static constexpr auto dpcpp_event_cmd_stat_running =
    DPCPP::info::event_command_status::running;
static constexpr auto dpcpp_event_cmd_stat_complete =
    DPCPP::info::event_command_status::complete;

// dpcpp event profiling info
static constexpr auto dpcpp_event_profiling_submit =
    DPCPP::info::event_profiling::command_submit;
static constexpr auto dpcpp_event_profiling_start =
    DPCPP::info::event_profiling::command_start;
static constexpr auto dpcpp_event_profiling_end =
    DPCPP::info::event_profiling::command_end;

// dpcpp access mode
static constexpr auto dpcpp_r_mode = DPCPP::access::mode::read;
static constexpr auto dpcpp_w_mode = DPCPP::access::mode::write;
static constexpr auto dpcpp_rw_mode = DPCPP::access::mode::read_write;
static constexpr auto dpcpp_atomic_rw_mode = DPCPP::access::mode::atomic;
static constexpr auto dpcpp_discard_w_mode = DPCPP::access::mode::discard_write;
static constexpr auto dpcpp_discard_rw_mode =
    DPCPP::access::mode::discard_read_write;

// dpcpp access address space
static constexpr auto dpcpp_priv_space =
    DPCPP::access::address_space::private_space;
static constexpr auto dpcpp_const_space =
    DPCPP::access::address_space::constant_space;
static constexpr auto dpcpp_local_space =
    DPCPP::access::address_space::local_space;
static constexpr auto dpcpp_global_space =
    DPCPP::access::address_space::global_space;

// dpcpp access fence space
static constexpr auto dpcpp_local_fence =
    DPCPP::access::fence_space::local_space;
static constexpr auto dpcpp_global_fence =
    DPCPP::access::fence_space::global_space;
static constexpr auto dpcpp_global_and_local_fence =
    DPCPP::access::fence_space::global_and_local;

// dpcpp access target
static constexpr auto dpcpp_host_buf = DPCPP::access::target::host_buffer;
static constexpr auto dpcpp_const_buf = DPCPP::access::target::constant_buffer;
static constexpr auto dpcpp_local_buf = DPCPP::access::target::local;
static constexpr auto dpcpp_global_buf = DPCPP::access::target::global_buffer;

// dpcpp ptr type
template <typename T>
DPCPP_DEVICE using dpcpp_local_ptr = typename DPCPP::local_ptr<T>;

template <typename T>
DPCPP_DEVICE using dpcpp_priv_ptr = typename DPCPP::private_ptr<T>;

template <typename T>
DPCPP_DEVICE using dpcpp_global_ptr = typename DPCPP::global_ptr<T>;

template <typename T>
DPCPP_DEVICE using dpcpp_const_ptr = typename DPCPP::constant_ptr<T>;

template <typename T, DPCPP::access::address_space Space = dpcpp_global_space>
DPCPP_DEVICE using dpcpp_multi_ptr = typename DPCPP::multi_ptr<T, Space>;

// dpcpp pointer type
template <typename T>
DPCPP_DEVICE using dpcpp_local_ptr_pt = typename dpcpp_local_ptr<T>::pointer_t;

template <typename T>
DPCPP_DEVICE using dpcpp_priv_ptr_pt = typename dpcpp_priv_ptr<T>::pointer_t;

template <typename T>
DPCPP_DEVICE using dpcpp_global_ptr_pt =
    typename dpcpp_global_ptr<T>::pointer_t;

template <typename T>
DPCPP_DEVICE using dpcpp_const_ptr_pt = typename dpcpp_const_ptr<T>::pointer_t;

template <typename T, DPCPP::access::address_space Space = dpcpp_global_space>
DPCPP_DEVICE using dpcpp_multi_ptr_pt =
    typename dpcpp_multi_ptr<T, Space>::pointer_t;

template <typename T>
DPCPP_DEVICE using dpcpp_local_ptr_cpt =
    typename dpcpp_local_ptr<T>::const_pointer_t;

template <typename T>
DPCPP_DEVICE using dpcpp_priv_ptr_cpt =
    typename dpcpp_priv_ptr<T>::const_pointer_t;

template <typename T>
DPCPP_DEVICE using dpcpp_global_ptr_cpt =
    typename dpcpp_global_ptr<T>::const_pointer_t;

template <typename T>
DPCPP_DEVICE using dpcpp_const_ptr_cpt =
    typename dpcpp_const_ptr<T>::const_pointer_t;

template <typename T, DPCPP::access::address_space Space = dpcpp_global_space>
DPCPP_DEVICE using dpcpp_multi_ptr_cpt =
    typename dpcpp_multi_ptr<T, Space>::const_pointer_t;

// dpcpp reference type
template <typename T>
DPCPP_DEVICE using dpcpp_local_ptr_rt =
    typename dpcpp_local_ptr<T>::reference_t;

template <typename T>
DPCPP_DEVICE using dpcpp_priv_ptr_rt = typename dpcpp_priv_ptr<T>::reference_t;

template <typename T>
DPCPP_DEVICE using dpcpp_global_ptr_rt =
    typename dpcpp_global_ptr<T>::reference_t;

template <typename T>
DPCPP_DEVICE using dpcpp_const_ptr_rt =
    typename dpcpp_const_ptr<T>::reference_t;

template <typename T, DPCPP::access::address_space Space = dpcpp_global_space>
DPCPP_DEVICE using dpcpp_multi_ptr_rt =
    typename dpcpp_multi_ptr<T, Space>::reference_t;

template <typename T>
DPCPP_DEVICE using dpcpp_local_ptr_crt =
    typename dpcpp_local_ptr<T>::const_reference_t;

template <typename T>
DPCPP_DEVICE using dpcpp_priv_ptr_crt =
    typename dpcpp_priv_ptr<T>::const_reference_t;

template <typename T>
DPCPP_DEVICE using dpcpp_global_ptr_crt =
    typename dpcpp_global_ptr<T>::const_reference_t;

template <typename T>
DPCPP_DEVICE using dpcpp_const_ptr_crt =
    typename dpcpp_const_ptr<T>::const_reference_t;

template <typename T, DPCPP::access::address_space Space = dpcpp_global_space>
DPCPP_DEVICE using dpcpp_multi_ptr_crt =
    typename dpcpp_multi_ptr<T, Space>::const_reference_t;

// dpcpp accessor type
template <
    typename ScalarType,
    DPCPP::access::mode Mode = dpcpp_rw_mode,
    int Dims = 1>
DPCPP_DEVICE using dpcpp_local_acc_t =
    DPCPP::accessor<ScalarType, Dims, Mode, dpcpp_local_buf>;

template <
    typename ScalarType,
    DPCPP::access::mode Mode = dpcpp_rw_mode,
    int Dims = 1>
DPCPP_DEVICE using dpcpp_global_acc_t =
    DPCPP::accessor<ScalarType, Dims, Mode, dpcpp_global_buf>;

template <typename ScalarType, int Dims = 1>
DPCPP_DEVICE using dpcpp_const_acc_t =
    DPCPP::accessor<ScalarType, Dims, dpcpp_r_mode, dpcpp_const_buf>;

template <
    typename ScalarType,
    DPCPP::access::mode Mode = dpcpp_rw_mode,
    int Dims = 1>
DPCPP_HOST using dpcpp_host_acc_t =
    DPCPP::accessor<ScalarType, Dims, Mode, dpcpp_host_buf>;
