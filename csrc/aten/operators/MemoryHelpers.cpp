#include "MemoryHelpers.h"
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

template <typename dst_dt, typename src_dt>
DPCPP_HOST void dpcppMemoryScale(
    dst_dt* dst,
    const src_dt* src,
    size_t n_elements,
    float alpha) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto total_threads = n_elements;

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(DPCPP::range<1>(total_threads), [=](DPCPP::item<1> item) {
      auto idx = item.get_id(0);
      dst[idx] = src[idx] * alpha;
    });
  };

  // launch kernel
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

#define MEM_SCALE_EXPLICIT_INST(DST_T, SRC_T) \
  template DPCPP_HOST void dpcppMemoryScale(  \
      DST_T* dst, const SRC_T* src, size_t n_elements, float alpha);

#define MEM_SCALE_EXPLICIT_BI_INST(T1, T2) \
  MEM_SCALE_EXPLICIT_INST(T1, T2);         \
  MEM_SCALE_EXPLICIT_INST(T2, T1);

MEM_SCALE_EXPLICIT_INST(float, float);
MEM_SCALE_EXPLICIT_BI_INST(double, float);
MEM_SCALE_EXPLICIT_BI_INST(at::Half, float);
MEM_SCALE_EXPLICIT_BI_INST(at::BFloat16, float);

template <typename dst_dt, typename src_dt>
DPCPP_HOST void dpcppMemoryScale1(
    dst_dt* dst,
    const src_dt* src,
    size_t n_elements,
    const double eps) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto total_threads = n_elements;

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(DPCPP::range<1>(total_threads), [=](DPCPP::item<1> item) {
      auto idx = item.get_id(0);
      dst[idx] = src[idx] * eps + dst[idx] * (1 - eps);
    });
  };

  // launch kernel
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

#define MEM_SCALE1_EXPLICIT_INST(DST_T, SRC_T) \
  template DPCPP_HOST void dpcppMemoryScale1(  \
      DST_T* dst, const SRC_T* src, size_t n_elements, const double eps);

#define MEM_SCALE1_EXPLICIT_BI_INST(T1, T2) \
  MEM_SCALE1_EXPLICIT_INST(T1, T2);         \
  MEM_SCALE1_EXPLICIT_INST(T2, T1);

MEM_SCALE1_EXPLICIT_INST(float, float);
MEM_SCALE1_EXPLICIT_BI_INST(double, float);
MEM_SCALE1_EXPLICIT_BI_INST(at::Half, float);
MEM_SCALE1_EXPLICIT_BI_INST(at::BFloat16, float);

template <typename dst_dt, typename src_dt>
DPCPP_HOST void dpcppMemoryScale2(
    dst_dt* dst,
    const src_dt* src,
    size_t n_elements,
    const float alpha,
    const double eps) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto total_threads = n_elements;

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(DPCPP::range<1>(total_threads), [=](DPCPP::item<1> item) {
      auto idx = item.get_id(0);
      dst[idx] = src[idx] * alpha * eps + dst[idx] * (1 - eps);
    });
  };

  // launch kernel
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

#define MEM_SCALE2_EXPLICIT_INST(DST_T, SRC_T) \
  template DPCPP_HOST void dpcppMemoryScale2(  \
      DST_T* dst,                              \
      const SRC_T* src,                        \
      size_t n_elements,                       \
      const float alpha,                       \
      const double eps);

#define MEM_SCALE2_EXPLICIT_BI_INST(T1, T2) \
  MEM_SCALE2_EXPLICIT_INST(T1, T2);         \
  MEM_SCALE2_EXPLICIT_INST(T2, T1);

MEM_SCALE2_EXPLICIT_INST(float, float);
MEM_SCALE2_EXPLICIT_BI_INST(double, float);
MEM_SCALE2_EXPLICIT_BI_INST(at::Half, float);
MEM_SCALE2_EXPLICIT_BI_INST(at::BFloat16, float);

template <typename dst_dt, typename src_dt>
DPCPP_HOST void dtype_convert_by_scalar(
    dst_dt* dst,
    const src_dt* src,
    size_t n_elements) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto total_threads = dpcppMaxWorkGroupSize(dev_id);

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
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

#define DT_CONVERT_EXPLICIT_INST(DST_T, SRC_T)      \
  template DPCPP_HOST void dtype_convert_by_scalar( \
      DST_T* dst, const SRC_T* src, size_t n_elements);

#define DT_CONVERT_EXPLICIT_BI_INST(T1, T2) \
  DT_CONVERT_EXPLICIT_INST(T1, T2);         \
  DT_CONVERT_EXPLICIT_INST(T2, T1);

DT_CONVERT_EXPLICIT_INST(float, float);
DT_CONVERT_EXPLICIT_BI_INST(int, int64_t);
DT_CONVERT_EXPLICIT_BI_INST(at::Half, float);
DT_CONVERT_EXPLICIT_BI_INST(at::BFloat16, float);

} // namespace AtenIpexTypeXPU
} // namespace at
