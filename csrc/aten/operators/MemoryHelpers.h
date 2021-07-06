#pragma once

#include <utils/DPCPP.h>
#include <runtime/Utils.h>

using namespace xpu::dpcpp;

DPCPP_DEF_K1(memory_scale);
DPCPP_DEF_K1(memory_scale1);
DPCPP_DEF_K1(memory_scale2);
DPCPP_DEF_K1(data_type_convert);

namespace at {
namespace AtenIpexTypeXPU {

template <typename dst_dt, typename src_dt>
static inline DPCPP_HOST void dpcppMemoryScale(
    dst_dt* dst,
    const src_dt* src,
    size_t n_elements,
    float alpha) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto total_threads = n_elements;

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for<DPCPP_K(memory_scale, dst_dt, src_dt)>(
        DPCPP::range<1>(total_threads), [=](DPCPP::item<1> item) {
          auto idx = item.get_id(0);
          dst[idx] = src[idx] * alpha;
        }
    );
  };

  // launch kernel
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

template <typename dst_dt, typename src_dt>
static inline DPCPP_HOST void dpcppMemoryScale1(
    dst_dt* dst,
    const src_dt* src,
    size_t n_elements,
    const double eps) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto total_threads = n_elements;

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for<DPCPP_K(memory_scale1, dst_dt, src_dt)>(
        DPCPP::range<1>(total_threads), [=](DPCPP::item<1> item) {
          auto idx = item.get_id(0);
          dst[idx] = src[idx] * eps + dst[idx] * (1 - eps);
        }
    );
  };

  // launch kernel
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

template <typename dst_dt, typename src_dt>
static inline DPCPP_HOST void dpcppMemoryScale2(
    dst_dt* dst,
    const src_dt* src,
    size_t n_elements,
    const float alpha,
    const double eps) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto total_threads = n_elements;

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for<DPCPP_K(memory_scale2, dst_dt, src_dt)>(
        DPCPP::range<1>(total_threads), [=](DPCPP::item<1> item) {
          auto idx = item.get_id(0);
          dst[idx] = src[idx] * alpha * eps + dst[idx] * (1 - eps);
        }
    );
  };

  // launch kernel
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

template <typename dst_dt, typename src_dt>
static inline DPCPP_HOST void
dtype_convert_by_scalar(dst_dt* dst, const src_dt* src, size_t n_elements) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto total_threads = dpcpp_queue.get_device().template get_info<dpcpp_dev_max_wgroup_size>();

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for<DPCPP_K(data_type_convert, dst_dt, src_dt)>(
      DPCPP::range<1>(total_threads), [=](DPCPP::item<1> itemId) {
        auto in_ptr = src;
        auto out_ptr = dst;
        auto id = itemId.get_id(0);
        for (auto i = id; i < n_elements; i += itemId.get_range()[0])
          out_ptr[i] = (dst_dt)in_ptr[i];
      });
  };

  // launch kernel
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

} // namespace AtenIpexTypeXPU
} // namespace at
