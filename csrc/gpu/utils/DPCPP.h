#pragma once

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

#include <ext/oneapi/atomic_enums.hpp>
#include <utils/Helpers.h>
#include <utils/Macros.h>
#include <utils/Profiler.h>
#include <utils/Settings.h>
#include <utils/Timer.h>

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

enum DPCPP_STATUS {
  DPCPP_SUCCESS = 0,
  DPCPP_FAILURE = 1,
};

// Kernel inside print utils
#if defined(__SYCL_DEVICE_ONLY__)
#define DPCPP_CONSTANT __attribute__((opencl_constant))
#else
#define DPCPP_CONSTANT
#endif

#define DPCPP_KER_STRING(var, str) static const DPCPP_CONSTANT char var[] = str;
#define DPCPP_KER_PRINTF sycl::ext::oneapi::experimental::printf

#define DPCPP_K_PRINT(fmt_str, ...)           \
  {                                           \
    DPCPP_KER_STRING(fmt_var, fmt_str);       \
    DPCPP_KER_PRINTF(fmt_var, ##__VA_ARGS__); \
  }

#define DPCPP_RESTRICT __restrict

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define DPCPP_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define DPCPP_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define DPCPP_LIKELY(expr) (expr)
#define DPCPP_UNLIKELY(expr) (expr)
#endif

// Kernel function implementation
#define DPCPP_Q_KFN(...) [=](__VA_ARGS__)

// Command group function implementation
#define DPCPP_Q_CGF(h) [&](sycl::handler & h)

#define DPCPP_E_SYNC_FOR_DEBUG(e)                                              \
  {                                                                            \
    static auto force_sync = xpu::dpcpp::Settings::I().is_sync_mode_enabled(); \
    if (force_sync) {                                                          \
      (e).wait_and_throw();                                                    \
    }                                                                          \
  }

inline constexpr std::string_view USES_FP64_MATH("uses-fp64-math");
inline constexpr std::string_view ASPECT_FP64_IS_NOT_SUPPORTED(
    "aspect fp64 is not supported");
inline constexpr std::string_view FP64_ERROR_FROM_MKL(
    "double type is not supported");
inline constexpr std::string_view OUT_OF_RESOURCES("PI_ERROR_OUT_OF_RESOURCES");

#define DPCPP_EXCEP_TRY try {
#define DPCPP_EXCEP_CATCH                                                  \
  }                                                                        \
  catch (const sycl::exception& ep) {                                      \
    const std::string_view err_msg(ep.what());                             \
    if (err_msg.find(USES_FP64_MATH) != std::string::npos ||               \
        err_msg.find(ASPECT_FP64_IS_NOT_SUPPORTED) != std::string::npos || \
        err_msg.find(FP64_ERROR_FROM_MKL) != std::string::npos) {          \
      throw std::runtime_error(                                            \
          "FP64 data type is unsupported on current platform.");           \
    } else if (err_msg.find(OUT_OF_RESOURCES) != std::string::npos) {      \
      throw std::runtime_error(                                            \
          "Allocation is out of device memory on current platform.");      \
    } else {                                                               \
      throw ep;                                                            \
    }                                                                      \
  }

#define DPCPP_EXT_SUBMIT(q, str, ker_submit)                                  \
  {                                                                           \
    DPCPP_EXCEP_TRY                                                           \
    static auto verbose = xpu::dpcpp::Settings::I().get_verbose_level();      \
    if (verbose) {                                                            \
      IPEX_TIMER(t, verbose, __func__);                                       \
      auto start_evt = xpu::dpcpp::queue_barrier(q);                          \
      t.now("start barrier");                                                 \
      auto e = (ker_submit);                                                  \
      t.now("submit");                                                        \
      auto end_evt = xpu::dpcpp::queue_barrier(q);                            \
      t.now("end barrier");                                                   \
      e.wait_and_throw();                                                     \
      t.now("event wait");                                                    \
      xpu::dpcpp::dpcpp_log((str), start_evt, end_evt);                       \
      start_evt.wait_and_throw();                                             \
      end_evt.wait_and_throw();                                               \
      auto se_end =                                                           \
          start_evt.template get_profiling_info<dpcpp_event_profiling_end>(); \
      auto ee_start =                                                         \
          end_evt.template get_profiling_info<dpcpp_event_profiling_start>(); \
      t.event_duration((ee_start - se_end) / 1000.0);                         \
    } else if (is_profiler_enabled()) {                                       \
      auto start_evt = xpu::dpcpp::queue_barrier(q);                          \
      auto e = (ker_submit);                                                  \
      auto end_evt = xpu::dpcpp::queue_barrier(q);                            \
      dpcpp_mark((str), start_evt, end_evt);                                  \
      DPCPP_E_SYNC_FOR_DEBUG(e);                                              \
    } else {                                                                  \
      auto e = (ker_submit);                                                  \
      DPCPP_E_SYNC_FOR_DEBUG(e);                                              \
    }                                                                         \
    (q).throw_asynchronous();                                                 \
    DPCPP_EXCEP_CATCH                                                         \
  }

#define DPCPP_Q_SUBMIT(q, cgf, ...)                                            \
  {                                                                            \
    DPCPP_EXCEP_TRY                                                            \
    static auto verbose = xpu::dpcpp::Settings::I().get_verbose_level();       \
    if (verbose) {                                                             \
      IPEX_TIMER(t, verbose, __func__);                                        \
      auto e = (q).submit((cgf), ##__VA_ARGS__);                               \
      t.now("submit");                                                         \
      e.wait_and_throw();                                                      \
      t.now("event wait");                                                     \
      xpu::dpcpp::dpcpp_log("dpcpp_kernel", e);                                \
      auto e_start =                                                           \
          e.template get_profiling_info<dpcpp_event_profiling_start>();        \
      auto e_end = e.template get_profiling_info<dpcpp_event_profiling_end>(); \
      t.event_duration((e_end - e_start) / 1000.0);                            \
    } else {                                                                   \
      auto e = (q).submit((cgf), ##__VA_ARGS__);                               \
      (q).throw_asynchronous();                                                \
      xpu::dpcpp::dpcpp_log("dpcpp_kernel", e);                                \
      DPCPP_E_SYNC_FOR_DEBUG(e);                                               \
    }                                                                          \
    DPCPP_EXCEP_CATCH                                                          \
  }

template <typename T>
using dpcpp_info_t = typename T::return_type;

// dpcpp platform info
using dpcpp_platform_name = sycl::info::platform::name;

// dpcpp device info
// Returns the device name of this SYCL device
using dpcpp_dev_name = sycl::info::device::name;
// Returns the device type associated with the device.
using dpcpp_dev_type = sycl::info::device::device_type;
// Returns the SYCL platform associated with this SYCL device.
using dpcpp_dev_platform = sycl::info::device::platform;
// Returns the vendor of this SYCL device.
using dpcpp_dev_vendor = sycl::info::device::vendor;
// Returns a backend-defined driver version as a std::string.
using dpcpp_dev_driver_version = sycl::info::device::driver_version;
// Returns the SYCL version as a std::string in the form:
// <major_version>.<minor_version>
using dpcpp_dev_version = sycl::info::device::version;
// Returns a string describing the version of the SYCL backend associated with
// the device. static constexpr auto dpcpp_dev_backend_version =
// sycl::info::device::backend_version; Returns true if the SYCL device is
// available and returns false if the device is not available.
using dpcpp_dev_is_available = sycl::info::device::is_available;
// Returns the maximum size in bytes of the arguments that can be passed to a
// kernel.
using dpcpp_dev_max_param_size = sycl::info::device::max_parameter_size;
// Returns the number of parallel compute units available to the device.
using dpcpp_dev_max_compute_units = sycl::info::device::max_compute_units;
// Returns the maximum dimensions that specify the global and local
// work-item IDs used by the data parallel execution model.
using dpcpp_dev_max_work_item_dims =
    sycl::info::device::max_work_item_dimensions;
// Returns the maximum number of workitems that are permitted in a work-group
// executing a kernel on a single compute unit.
using dpcpp_dev_max_work_group_size = sycl::info::device::max_work_group_size;
// Returns the maximum number of subgroups in a work-group for any kernel
// executed on the device
using dpcpp_dev_max_num_subgroup = sycl::info::device::max_num_sub_groups;
// Returns a std::vector of size_t containing the set of sub-group sizes
// supported by the device
using dpcpp_dev_subgroup_sizes = sycl::info::device::sub_group_sizes;
// Returns the maximum configured clock frequency of this SYCL device in MHz.
using dpcpp_dev_max_clock_freq = sycl::info::device::max_clock_frequency;
// Returns the default compute device address space size specified as an
// unsigned integer value in bits. Must return either 32 or 64.
using dpcpp_dev_address_bits = sycl::info::device::address_bits;
// Returns the maximum size of memory object allocation in bytes
using dpcpp_dev_max_alloc_size = sycl::info::device::max_mem_alloc_size;
// Returns the minimum value in bits of the largest supported SYCL built-in data
// type if this SYCL device is not of device type info::device_type::custom.
using dpcpp_dev_mem_base_addr_align = sycl::info::device::mem_base_addr_align;
// Returns a std::vector of info::fp_config describing the half precision
// floating-point capability of this SYCL device.
using dpcpp_dev_half_fp_config = sycl::info::device::half_fp_config;
// Returns a std::vector of info::fp_config describing the single precision
// floating-point capability of this SYCL device.
using dpcpp_dev_single_fp_config = sycl::info::device::single_fp_config;
// Returns a std::vector of info::fp_config describing the double precision
// floatingpoint capability of this SYCL device.
using dpcpp_dev_double_fp_config = sycl::info::device::double_fp_config;
// Returns the size of global device memory in bytes
using dpcpp_dev_global_mem_size = sycl::info::device::global_mem_size;
// Returns the type of global memory cache supported.
using dpcpp_dev_global_mem_cache_type =
    sycl::info::device::global_mem_cache_type;
// Returns the size of global memory cache in bytes.
using dpcpp_dev_global_mem_cache_size =
    sycl::info::device::global_mem_cache_size;
// Returns the size of global memory cache line in bytes.
using dpcpp_dev_global_mem_cache_line_size =
    sycl::info::device::global_mem_cache_line_size;
// Returns the type of local memory supported.
using dpcpp_dev_local_mem_type = sycl::info::device::local_mem_type;
// Returns the size of local memory arena in bytes.
using dpcpp_dev_local_mem_size = sycl::info::device::local_mem_size;
// Returns the maximum number of sub-devices that can be created when this
// device is partitioned.
using dpcpp_dev_max_sub_devices = sycl::info::device::partition_max_sub_devices;
// Returns the resolution of device timer in nanoseconds.
using dpcpp_dev_profiling_resolution =
    sycl::info::device::profiling_timer_resolution;
// Returns the preferred native vector width size for built-in
// scalar types that can be put into vectors.
using dpcpp_dev_pref_vec_width_char =
    sycl::info::device::preferred_vector_width_char;
using dpcpp_dev_pref_vec_width_short =
    sycl::info::device::preferred_vector_width_short;
using dpcpp_dev_pref_vec_width_int =
    sycl::info::device::preferred_vector_width_int;
using dpcpp_dev_pref_vec_width_long =
    sycl::info::device::preferred_vector_width_long;
using dpcpp_dev_pref_vec_width_float =
    sycl::info::device::preferred_vector_width_float;
using dpcpp_dev_pref_vec_width_double =
    sycl::info::device::preferred_vector_width_double;
using dpcpp_dev_pref_vec_width_half =
    sycl::info::device::preferred_vector_width_half;
// Returns the native ISA vector width. The vector width is defined as
// the number of scalar elements that can be stored in the vector.
using dpcpp_dev_native_vec_width_char =
    sycl::info::device::native_vector_width_char;
using dpcpp_dev_native_vec_width_short =
    sycl::info::device::native_vector_width_short;
using dpcpp_dev_native_vec_width_int =
    sycl::info::device::native_vector_width_int;
using dpcpp_dev_native_vec_width_long =
    sycl::info::device::native_vector_width_long;
using dpcpp_dev_native_vec_width_float =
    sycl::info::device::native_vector_width_float;
using dpcpp_dev_native_vec_width_double =
    sycl::info::device::native_vector_width_double;
using dpcpp_dev_native_vec_width_half =
    sycl::info::device::native_vector_width_half;

// intel extensions
using dpcpp_dev_ext_intel_gpu_eu_simd_width =
    sycl::ext::intel::info::device::gpu_eu_simd_width;
using dpcpp_dev_ext_intel_gpu_hw_threads_per_eu =
    sycl::ext::intel::info::device::gpu_hw_threads_per_eu;
using dpcpp_dev_ext_intel_gpu_eu_count =
    sycl::ext::intel::info::device::gpu_eu_count;

// aspects for extensions
static constexpr auto dpcpp_dev_aspect_gpu_eu_simd_width =
    sycl::aspect::ext_intel_gpu_eu_simd_width;
static constexpr auto dpcpp_dev_aspect_hw_threads_per_eu =
    sycl::aspect::ext_intel_gpu_hw_threads_per_eu;
static constexpr auto dpcpp_dev_aspect_gpu_eu_count =
    sycl::aspect::ext_intel_gpu_eu_count;
static constexpr auto dpcpp_dev_aspect_fp64 = sycl::aspect::fp64;
static constexpr auto dpcpp_dev_aspect_atomic64 = sycl::aspect::atomic64;

// dpcpp event info
using dpcpp_event_exec_stat = sycl::info::event::command_execution_status;
// dpcpp event command status
static constexpr auto dpcpp_event_cmd_stat_submitted =
    sycl::info::event_command_status::submitted;
static constexpr auto dpcpp_event_cmd_stat_running =
    sycl::info::event_command_status::running;
static constexpr auto dpcpp_event_cmd_stat_complete =
    sycl::info::event_command_status::complete;

// dpcpp event profiling info
using dpcpp_event_profiling_submit =
    sycl::info::event_profiling::command_submit;
using dpcpp_event_profiling_start = sycl::info::event_profiling::command_start;
using dpcpp_event_profiling_end = sycl::info::event_profiling::command_end;

// dpcpp access address space
static constexpr auto dpcpp_priv_space =
    sycl::access::address_space::private_space;
static constexpr auto dpcpp_local_space =
    sycl::access::address_space::local_space;
static constexpr auto dpcpp_global_space =
    sycl::access::address_space::global_space;

// dpcpp access fence space
static constexpr auto dpcpp_local_fence =
    sycl::access::fence_space::local_space;
static constexpr auto dpcpp_global_fence =
    sycl::access::fence_space::global_space;
static constexpr auto dpcpp_global_and_local_fence =
    sycl::access::fence_space::global_and_local;

// dpcpp memory ordering
static constexpr auto dpcpp_mem_odr_rlx = sycl::memory_order::relaxed;
static constexpr auto dpcpp_mem_odr_acq = sycl::memory_order::acquire;
static constexpr auto dpcpp_mem_odr_rel = sycl::memory_order::release;
static constexpr auto dpcpp_mem_odr_acq_rel = sycl::memory_order::acq_rel;
static constexpr auto dpcpp_mem_odr_seq_cst = sycl::memory_order::seq_cst;

// dpcpp memory scope
static constexpr auto dpcpp_mem_scp_wi = sycl::memory_scope::work_item;
static constexpr auto dpcpp_mem_scp_sg = sycl::memory_scope::sub_group;
static constexpr auto dpcpp_mem_scp_wg = sycl::memory_scope::work_group;
static constexpr auto dpcpp_mem_scp_dev = sycl::memory_scope::device;
static constexpr auto dpcpp_mem_scp_sys = sycl::memory_scope::system;

// dpcpp ptr type
template <typename T>
using dpcpp_local_ptr = typename sycl::local_ptr<T>;

template <typename T>
using dpcpp_global_ptr = typename sycl::global_ptr<T>;

// dpcpp pointer type
template <typename T>
using dpcpp_local_ptr_pt = typename dpcpp_local_ptr<T>::pointer_t;

template <typename T>
using dpcpp_global_ptr_pt = typename dpcpp_global_ptr<T>::pointer_t;

// dpcpp accessor type
template <typename ScalarType, int Dims = 1>
using dpcpp_local_acc_t = sycl::local_accessor<ScalarType, Dims>;

// dpcpp atomic
template <typename T>
using dpcpp_atomic_ref_rlx_dev_global_t = sycl::
    atomic_ref<T, dpcpp_mem_odr_rlx, dpcpp_mem_scp_dev, dpcpp_global_space>;

template <typename T>
using dpcpp_atomic_ref_rlx_wg_local_t =
    sycl::atomic_ref<T, dpcpp_mem_odr_rlx, dpcpp_mem_scp_wg, dpcpp_local_space>;

template <typename T, int Dims = 1>
inline T* IPEXGetLocalAccPointer(
    const sycl::local_accessor<T, Dims>& accessor) {
  if constexpr (std::is_same_v<
                    decltype(accessor.get_pointer()),
                    sycl::local_ptr<T>>) {
    return accessor.get_pointer().get();
  } else {
    return accessor.get_pointer();
  }
}
