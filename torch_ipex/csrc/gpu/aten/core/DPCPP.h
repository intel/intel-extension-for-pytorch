#pragma once

#include <CL/sycl.hpp>
#include <utils/Profiler.h>

#include <SYCL/event.h>
#include <chrono>

// alias for dpcpp namespace
namespace DPCPP = cl::sycl;

// macros for dpcpp command queue and kernel function
#define DPCPP_K_NAME(x) __##x##_dpcpp_kernel
#define DPCPP_K(k, ...) DPCPP_K_NAME(k)<char, ##__VA_ARGS__>
#define DPCPP_DEF_K1(k)                                   \
  template <typename __dummy_typename_dpcpp, typename...> \
  class DPCPP_K_NAME(k) {}
#define DPCPP_DEF_K2(k, ...)                                \
  template <typename __dummy_typename_dpcpp, ##__VA_ARGS__> \
  class DPCPP_K_NAME(k) {}

#define DPCPP_Q_KFN(...) [=](__VA_ARGS__)
#define DPCPP_Q_CGF(h) [&](DPCPP::handler & h)
#define DPCPP_Q_SUBMIT(q, cgf, ...) q.submit(cgf, ##__VA_ARGS__)

#ifndef DPCPP_PROFILING
#define DPCPP_Q_SYNC_SUBMIT(q, cgf, ...)   \
  {                                        \
    auto e = q.submit(cgf, ##__VA_ARGS__); \
    e.wait();                              \
  }
#define DPCPP_Q_ASYNC_SUBMIT(q, cgf, ...) \
  { auto e = q.submit(cgf, ##__VA_ARGS__); }
#else
#define DPCPP_Q_SYNC_SUBMIT(q, cgf, ...)   \
  {                                        \
    auto e = q.submit(cgf, ##__VA_ARGS__); \
    e.wait();                              \
    dpcpp_log("sycl_kernel", e);           \
  }
#define DPCPP_Q_ASYNC_SUBMIT(q, cgf, ...)  \
  {                                        \
    auto e = q.submit(cgf, ##__VA_ARGS__); \
    dpcpp_log("sycl_kernel", e);           \
  }
#endif

// the descriptor as entity attribute
#define DPCPP_HOST // for host only
#define DPCPP_DEVICE // for device only
#define DPCPP_BOTH // for both host and device

// dpcpp device configuration
// TODO: set subgourp size with api get_max_sub_group_size
#define DPCPP_SUB_GROUP_SIZE (1L)

// dpcpp get ptr from accessor
#if defined(USE_DPCPP)
#define GET_ACC_PTR(acc, T) acc.get_pointer().get()
#elif defined(USE_COMPUTECPP)
#define GET_ACC_PTR(acc, T) acc.template get_pointer<T>().get()
#endif

// dpcpp device info
static constexpr auto dpcpp_dev_type = DPCPP::info::device::device_type;
static constexpr auto dpcpp_dev_max_units =
    DPCPP::info::device::max_compute_units;
static constexpr auto dpcpp_dev_max_item_dims =
    DPCPP::info::device::max_work_item_dimensions;
static constexpr auto dpcpp_dev_max_item_sizes =
    DPCPP::info::device::max_work_item_sizes;
static constexpr auto dpcpp_dev_max_wgroup_size =
    DPCPP::info::device::max_work_group_size;
static constexpr auto dpcpp_dev_max_malloc_size =
    DPCPP::info::device::max_mem_alloc_size;
static constexpr auto dpcpp_dev_local_mem_type =
    DPCPP::info::device::local_mem_type;
static constexpr auto dpcpp_dev_local_mem_size =
    DPCPP::info::device::local_mem_size;
static constexpr auto dpcpp_dev_global_mem_size =
    DPCPP::info::device::global_mem_size;

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
