#pragma once

#include <sycl/sycl.hpp>
#include "aten/operators/torch-xpu-ops/comm/Scalar.h"

// sycl access address space
static constexpr auto sycl_priv_space =
    sycl::access::address_space::private_space;
static constexpr auto sycl_local_space =
    sycl::access::address_space::local_space;
static constexpr auto sycl_global_space =
    sycl::access::address_space::global_space;

// sycl access fence space
static constexpr auto sycl_local_fence = sycl::access::fence_space::local_space;
static constexpr auto sycl_global_fence =
    sycl::access::fence_space::global_space;
static constexpr auto sycl_global_and_local_fence =
    sycl::access::fence_space::global_and_local;

// sycl memory ordering
static constexpr auto sycl_mem_odr_rlx = sycl::memory_order::relaxed;
static constexpr auto sycl_mem_odr_acq = sycl::memory_order::acquire;
static constexpr auto sycl_mem_odr_rel = sycl::memory_order::release;
static constexpr auto sycl_mem_odr_acq_rel = sycl::memory_order::acq_rel;
static constexpr auto sycl_mem_odr_seq_cst = sycl::memory_order::seq_cst;

// sycl memory scope
static constexpr auto sycl_mem_scp_wi = sycl::memory_scope::work_item;
static constexpr auto sycl_mem_scp_sg = sycl::memory_scope::sub_group;
static constexpr auto sycl_mem_scp_wg = sycl::memory_scope::work_group;
static constexpr auto sycl_mem_scp_dev = sycl::memory_scope::device;
static constexpr auto sycl_mem_scp_sys = sycl::memory_scope::system;

template <typename scalar_t, int dims = 1>
using sycl_local_acc_t = sycl::local_accessor<scalar_t, dims>;

template <typename T>
using sycl_local_ptr = typename sycl::local_ptr<T>;

template <typename T>
using sycl_global_ptr = typename sycl::global_ptr<T>;

template <typename T>
using sycl_atomic_ref_rlx_dev_global_t =
    sycl::atomic_ref<T, sycl_mem_odr_rlx, sycl_mem_scp_dev, sycl_global_space>;

template <typename ker_t, int dim>
static inline void sycl_kernel_submit(
    ::sycl::range<dim> range,
    ::sycl::queue q,
    ker_t ker) {
  auto cgf = [&](::sycl::handler& cgh) { cgh.parallel_for<ker_t>(range, ker); };
  q.submit(cgf);
}

// Additional convention of SYCL kernel configuration. Besides construct kernel
// functor, SYCL has some additional conventions to be called during setuping
// SYCL command group handler, e.g. declaring SYCL local accessor when the
// kernel requires shared local memory usage. Helpers below help simpilfiy
// submission of SYCL kernels requiring additional conventions.

// Defining additional convention. Can use `sycl_kernel_submit` simply to
// submit a kernel, if the kernel functor inherits from the struct below.
// Since cannot offload non-device-copyable (sycl::is_device_copyable) kernel
// functor, a structure has virtual function is non-device-copyable.
// Using an empty class, the kernel functor derived by it will be required to
// define member method `void convention(sycl::handler&)`, or fails in
// compilation.
struct __SYCL_KER_CONFIG_CONVENTION__ {};

template <typename ker_t, int dim>
static inline typename std::enable_if<
    std::is_base_of_v<__SYCL_KER_CONFIG_CONVENTION__, ker_t>,
    void>::type
sycl_kernel_submit(
    ::sycl::range<dim> global_range,
    ::sycl::range<dim> local_range,
    ::sycl::queue q,
    ker_t ker) {
  auto cgf = [&](::sycl::handler& cgh) {
    ker.sycl_ker_config_convention(cgh);
    cgh.parallel_for<ker_t>(
        ::sycl::nd_range<dim>(global_range, local_range), ker);
  };
  q.submit(cgf);
}

template <typename ker_t, int dim>
static inline typename std::enable_if<
    !std::is_base_of_v<__SYCL_KER_CONFIG_CONVENTION__, ker_t>,
    void>::type
sycl_kernel_submit(
    ::sycl::range<dim> global_range,
    ::sycl::range<dim> local_range,
    ::sycl::queue q,
    ker_t ker) {
  auto cgf = [&](::sycl::handler& cgh) {
    cgh.parallel_for<ker_t>(
        ::sycl::nd_range<dim>(global_range, local_range), ker);
  };
  q.submit(cgf);
}

template <typename ker_t>
static inline typename std::enable_if<
    std::is_base_of_v<__SYCL_KER_CONFIG_CONVENTION__, ker_t>,
    void>::type
sycl_kernel_submit(
    int64_t global_range,
    int64_t local_range,
    ::sycl::queue q,
    ker_t ker) {
  auto cgf = [&](::sycl::handler& cgh) {
    ker.sycl_ker_config_convention(cgh);
    cgh.parallel_for<ker_t>(
        ::sycl::nd_range<1>(
            ::sycl::range<1>(global_range), ::sycl::range<1>(local_range)),
        ker);
  };
  q.submit(cgf);
}

template <typename ker_t>
static inline typename std::enable_if<
    !std::is_base_of_v<__SYCL_KER_CONFIG_CONVENTION__, ker_t>,
    void>::type
sycl_kernel_submit(
    int64_t global_range,
    int64_t local_range,
    ::sycl::queue q,
    ker_t ker) {
  auto cgf = [&](::sycl::handler& cgh) {
    cgh.parallel_for<ker_t>(
        ::sycl::nd_range<1>(
            ::sycl::range<1>(global_range), ::sycl::range<1>(local_range)),
        ker);
  };
  q.submit(cgf);
}
