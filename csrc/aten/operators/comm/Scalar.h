#pragma once

#include <utils/DPCPP.h>
#include <runtime/Utils.h>

namespace at {
namespace AtenIpexTypeXPU {

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

static inline DPCPP_DEVICE unsigned long long __double_as_long_long(double val) {
  double_to_ull cn;
  cn.in = val;
  return cn.out;
}

static inline DPCPP_DEVICE double __long_long_as_double(unsigned long long val) {
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

}} // at::AtenIpexTypeXPU
