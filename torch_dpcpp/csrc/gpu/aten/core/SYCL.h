#pragma once

#include <CL/sycl.hpp>

// alias for dpcpp namespace
namespace DP = cl::sycl;

// macros for dpcpp command queue and kernel function
#define DP_K_NAME(x)        __##x##_dpcpp_kernel
#define DP_K(k, ...)        DP_K_NAME(k)<char, ##__VA_ARGS__>
#define DP_DEF_K1(k)        template <typename __dummy_typename_dpcpp, typename ...> class DP_K_NAME(k) {}
#define DP_DEF_K2(k, ...)   template <typename __dummy_typename_dpcpp, ##__VA_ARGS__> class DP_K_NAME(k) {}

#define DP_Q_KFN(...)                   [=](__VA_ARGS__)
#define DP_Q_CGF(h)                     [&](DP::handler &h)
#define DP_Q_SUBMIT(q, cgf, ...)        q.submit(cgf, ##__VA_ARGS__)
#define DP_Q_SYNC_SUBMIT(q, cgf, ...)   { auto e = DP_Q_SUBMIT(q, cgf, ##__VA_ARGS__); e.wait(); }
#define DP_Q_ASYNC_SUBMIT(q, cgf, ...)  { DP_Q_SUBMIT(q, cgf, ##__VA_ARGS__); }

// the descriptor as entity attribute
#define DP_HOST     // for host only
#define DP_DEVICE   // for device only
#define DP_BOTH     // for both host and device

// dpcpp device configuration
// TODO: set subgourp size with api get_max_sub_group_size
#define DP_SUB_GROUP_SIZE (1L)

// dpcpp get ptr from accessor
#if defined(USE_DPCPP)
  #define GET_ACC_PTR(acc, T) acc.get_pointer().get()
#elif  defined(USE_COMPUTECPP)
  #define GET_ACC_PTR(acc, T) acc.template get_pointer<T>().get()
#endif

// dpcpp device info
static constexpr auto dp_dev_type             = DP::info::device::device_type;
static constexpr auto dp_dev_max_units        = DP::info::device::max_compute_units;
static constexpr auto dp_dev_max_item_dims    = DP::info::device::max_work_item_dimensions;
static constexpr auto dp_dev_max_item_sizes   = DP::info::device::max_work_item_sizes;
static constexpr auto dp_dev_max_wgroup_size  = DP::info::device::max_work_group_size;
static constexpr auto dp_dev_max_malloc_size  = DP::info::device::max_mem_alloc_size;
static constexpr auto dp_dev_local_mem_type   = DP::info::device::local_mem_type;
static constexpr auto dp_dev_local_mem_size   = DP::info::device::local_mem_size;
static constexpr auto dp_dev_global_mem_size  = DP::info::device::global_mem_size;

// dpcpp access mode
static constexpr auto dp_r_mode           = DP::access::mode::read;
static constexpr auto dp_w_mode           = DP::access::mode::write;
static constexpr auto dp_rw_mode          = DP::access::mode::read_write;
static constexpr auto dp_atomic_rw_mode   = DP::access::mode::atomic;
static constexpr auto dp_discard_w_mode   = DP::access::mode::discard_write;
static constexpr auto dp_discard_rw_mode  = DP::access::mode::discard_read_write;

// dpcpp access address space
static constexpr auto dp_priv_space   = DP::access::address_space::private_space;
static constexpr auto dp_const_space  = DP::access::address_space::constant_space;
static constexpr auto dp_local_space  = DP::access::address_space::local_space;
static constexpr auto dp_global_space = DP::access::address_space::global_space;

// dpcpp access fence space
static constexpr auto dp_local_fence            = DP::access::fence_space::local_space;
static constexpr auto dp_global_fence           = DP::access::fence_space::global_space;
static constexpr auto dp_global_and_local_fence = DP::access::fence_space::global_and_local;

// dpcpp access target
static constexpr auto dp_host_buf   = DP::access::target::host_buffer;
static constexpr auto dp_const_buf  = DP::access::target::constant_buffer;
static constexpr auto dp_local_buf  = DP::access::target::local;
static constexpr auto dp_global_buf = DP::access::target::global_buffer;

// dpcpp ptr type
template <typename T>
DP_DEVICE using dp_local_ptr = typename DP::local_ptr<T>;

template <typename T>
DP_DEVICE using dp_priv_ptr = typename DP::private_ptr<T>;

template <typename T>
DP_DEVICE using dp_global_ptr = typename DP::global_ptr<T>;

template <typename T>
DP_DEVICE using dp_const_ptr = typename DP::constant_ptr<T>;

template <typename T, DP::access::address_space Space = dp_global_space>
DP_DEVICE using dp_multi_ptr = typename DP::multi_ptr<T, Space>;

// dpcpp pointer type
template <typename T>
DP_DEVICE using dp_local_ptr_pt = typename dp_local_ptr<T>::pointer_t;

template <typename T>
DP_DEVICE using dp_priv_ptr_pt = typename dp_priv_ptr<T>::pointer_t;

template <typename T>
DP_DEVICE using dp_global_ptr_pt = typename dp_global_ptr<T>::pointer_t;

template <typename T>
DP_DEVICE using dp_const_ptr_pt = typename dp_const_ptr<T>::pointer_t;

template <typename T, DP::access::address_space Space = dp_global_space>
DP_DEVICE using dp_multi_ptr_pt = typename dp_multi_ptr<T, Space>::pointer_t;

template <typename T>
DP_DEVICE using dp_local_ptr_cpt = typename dp_local_ptr<T>::const_pointer_t;

template <typename T>
DP_DEVICE using dp_priv_ptr_cpt = typename dp_priv_ptr<T>::const_pointer_t;

template <typename T>
DP_DEVICE using dp_global_ptr_cpt = typename dp_global_ptr<T>::const_pointer_t;

template <typename T>
DP_DEVICE using dp_const_ptr_cpt = typename dp_const_ptr<T>::const_pointer_t;

template <typename T, DP::access::address_space Space = dp_global_space>
DP_DEVICE using dp_multi_ptr_cpt = typename dp_multi_ptr<T, Space>::const_pointer_t;

// dpcpp reference type
template <typename T>
DP_DEVICE using dp_local_ptr_rt = typename dp_local_ptr<T>::reference_t;

template <typename T>
DP_DEVICE using dp_priv_ptr_rt = typename dp_priv_ptr<T>::reference_t;

template <typename T>
DP_DEVICE using dp_global_ptr_rt = typename dp_global_ptr<T>::reference_t;

template <typename T>
DP_DEVICE using dp_const_ptr_rt = typename dp_const_ptr<T>::reference_t;

template <typename T, DP::access::address_space Space = dp_global_space>
DP_DEVICE using dp_multi_ptr_rt = typename dp_multi_ptr<T, Space>::reference_t;

template <typename T>
DP_DEVICE using dp_local_ptr_crt = typename dp_local_ptr<T>::const_reference_t;

template <typename T>
DP_DEVICE using dp_priv_ptr_crt = typename dp_priv_ptr<T>::const_reference_t;

template <typename T>
DP_DEVICE using dp_global_ptr_crt = typename dp_global_ptr<T>::const_reference_t;

template <typename T>
DP_DEVICE using dp_const_ptr_crt = typename dp_const_ptr<T>::const_reference_t;

template <typename T, DP::access::address_space Space = dp_global_space>
DP_DEVICE using dp_multi_ptr_crt = typename dp_multi_ptr<T, Space>::const_reference_t;

// dpcpp accessor type
template <typename ScalarType, DP::access::mode Mode = dp_rw_mode, int Dims = 1>
DP_DEVICE using dp_local_acc_t = DP::accessor<ScalarType, Dims, Mode, dp_local_buf>;

template <typename ScalarType, DP::access::mode Mode = dp_rw_mode, int Dims = 1>
DP_DEVICE using dp_global_acc_t = DP::accessor<ScalarType, Dims, Mode, dp_global_buf>;

template <typename ScalarType, int Dims = 1>
DP_DEVICE using dp_const_acc_t = DP::accessor<ScalarType, Dims, dp_r_mode, dp_const_buf>;

template <typename ScalarType, DP::access::mode Mode = dp_rw_mode, int Dims = 1>
DP_HOST using dp_host_acc_t = DP::accessor<ScalarType, Dims, Mode, dp_host_buf>;

