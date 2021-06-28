#pragma once

#include <utils/DPCPP.h>
#include <runtime/Utils.h>

DPCPP_DEF_K1(data_type_convert);

namespace xpu {
namespace dpcpp {

union u32_to_f32 {
  uint32_t in;
  float out;
};

union f32_to_u32 {
  float in;
  uint32_t out;
};

union double_to_ull{
  double in;
  unsigned long long out;
};

union ull_to_double{
  unsigned long long in;
  double out;
};

union double_to_u32{
  double in;
  struct { uint32_t high, low;};
};

static inline DPCPP_DEVICE uint32_t __float_as_int(float val) {
  f32_to_u32 cn;
  cn.in = val;
  return cn.out;
}

static inline DPCPP_DEVICE float __int_as_float(uint32_t val) {
  u32_to_f32 cn;
  cn.in = val;
  return cn.out;
}

static inline DPCPP_DEVICE uint32_t __double_as_long_long(double val) {
  double_to_ull cn;
  cn.in = val;
  return cn.out;
}

static inline DPCPP_DEVICE float __long_long_as_double(unsigned long long val) {
  ull_to_double cn;
  cn.in = val;
  return cn.out;
}

static inline DPCPP_DEVICE uint32_t __double_as_int(double val) {
  double_to_u32 cn;
  cn.in = val;
  return cn.low;
}

static inline DPCPP_DEVICE float __int_as_double(uint32_t val) {
  return (double)val;
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

} // namespace dpcpp
} // namespace xpu
