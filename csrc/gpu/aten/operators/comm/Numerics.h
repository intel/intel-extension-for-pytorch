#pragma once

#include <ATen/ATen.h>

#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <utils/DPCPP.h>

#include "General.h"

template <typename T>
struct Numerics {};

template <typename T>
static inline T powi(T a, T b) {
  T result = 1;
  while (b) {
    if (b & 1) {
      result *= a;
    }
    b /= 2;
    a *= a;
  }
  return result;
}

template <typename T>
static inline T sgni(T z) {
  if (z == (T)(0)) {
    return (T)(0);
  } else {
    return z / abs(z);
  }
}

static inline c10::BFloat16 nextafteri(c10::BFloat16 from, c10::BFloat16 to) {
  // Reference:
  // https://git.musl-libc.org/cgit/musl/tree/src/math/nextafter.c
  using int_repr_t = uint16_t;
  using float_t = c10::BFloat16;
  constexpr uint8_t bits = 16;
  union {
    float_t f;
    int_repr_t i;
  } ufrom = {from}, uto = {to};

  // get a mask to get the sign bit i.e. MSB
  int_repr_t sign_mask = int_repr_t{1} << (bits - 1);

  // short-circuit: if either is NaN, return NaN
  if (from != from || to != to) {
    return from + to;
  }

  // short-circuit: if they are exactly the same.
  if (ufrom.i == uto.i) {
    return from;
  }

  // mask the sign-bit to zero i.e. positive
  // equivalent to abs(x)
  int_repr_t abs_from = ufrom.i & ~sign_mask;
  int_repr_t abs_to = uto.i & ~sign_mask;
  if (abs_from == 0) {
    // if both are zero but with different sign,
    // preserve the sign of `to`.
    if (abs_to == 0) {
      return to;
    }
    // smallest subnormal with sign of `to`.
    ufrom.i = (uto.i & sign_mask) | int_repr_t{1};
    return ufrom.f;
  }

  // if abs(from) > abs(to) or sign(from) != sign(to)
  if (abs_from > abs_to || ((ufrom.i ^ uto.i) & sign_mask)) {
    ufrom.i--;
  } else {
    ufrom.i++;
  }

  return ufrom.f;
}

template <typename T>
inline constexpr T pi_i() {
  return static_cast<T>(3.14159265358979323846L);
}

template <>
struct Numerics<uint8_t> {
  static inline uint8_t lower_bound() {
    return std::numeric_limits<uint8_t>::lowest();
  }
  static inline uint8_t upper_bound() {
    return std::numeric_limits<uint8_t>::max();
  }
  static inline constexpr uint8_t pi() {
    return pi_i<uint8_t>();
  }

  static inline bool lt(uint8_t a, uint8_t b) {
    return a < b;
  }
  static inline bool le(uint8_t a, uint8_t b) {
    return a <= b;
  }
  static inline bool gt(uint8_t a, uint8_t b) {
    return a > b;
  }
  static inline bool ge(uint8_t a, uint8_t b) {
    return a >= b;
  }
  static inline bool eq(uint8_t a, uint8_t b) {
    return a == b;
  }
  static inline bool ne(uint8_t a, uint8_t b) {
    return a != b;
  }

  static inline uint8_t ceil(uint8_t a) {
    return a;
  }
  static inline uint8_t floor(uint8_t a) {
    return a;
  }
  static inline uint8_t round(uint8_t a) {
    return a;
  }
  static inline uint8_t trunc(uint8_t a) {
    return a;
  }
  static inline uint8_t neg(uint8_t a) {
    return -a;
  }
  static inline uint8_t abs(uint8_t a) {
    return a;
  }
  static inline uint8_t add(uint8_t a, uint8_t b) {
    return a + b;
  }
  static inline uint8_t mul(uint8_t a, uint8_t b) {
    return a * b;
  }
  static inline uint8_t sub(uint8_t a, uint8_t b) {
    return a - b;
  }
  static inline uint8_t div(uint8_t a, uint8_t b) {
    return a / b;
  }
  static inline uint8_t pow(uint8_t a, uint8_t b) {
    return powi<uint8_t>(a, b);
  }
  static inline bool isnan(uint8_t a) {
    return false;
  }
  static inline bool isinf(uint8_t a) {
    return false;
  }
  static inline uint8_t min(uint8_t a, uint8_t b) {
    return sycl::min(a, b);
  }
  static inline uint8_t max(uint8_t a, uint8_t b) {
    return sycl::max(a, b);
  }
};

template <>
struct Numerics<bool> {
  static inline bool lower_bound() {
    return std::numeric_limits<bool>::lowest();
  }
  static inline bool upper_bound() {
    return std::numeric_limits<bool>::max();
  }
  static inline constexpr bool pi() {
    return pi_i<bool>();
  }

  static inline bool lt(bool a, bool b) {
    return a < b;
  }
  static inline bool le(bool a, bool b) {
    return a <= b;
  }
  static inline bool gt(bool a, bool b) {
    return a > b;
  }
  static inline bool ge(bool a, bool b) {
    return a >= b;
  }
  static inline bool eq(bool a, bool b) {
    return a == b;
  }
  static inline bool ne(bool a, bool b) {
    return a != b;
  }
  static inline bool add(bool a, bool b) {
    return a + b;
  }
  static inline bool mul(bool a, bool b) {
    return a && b;
  }
  static inline bool sub(bool a, bool b) {
    return a - b;
  }
  static inline bool div(bool a, bool b) {
    return a / b;
  }
  static inline bool abs(bool a) {
    return a;
  }
  static inline bool isnan(bool a) {
    return false;
  }
  static inline bool isinf(bool a) {
    return false;
  }
};

template <>
struct Numerics<int8_t> {
  static inline int8_t lower_bound() {
    return std::numeric_limits<int8_t>::lowest();
  }
  static inline int8_t upper_bound() {
    return std::numeric_limits<int8_t>::max();
  }
  static inline constexpr int8_t pi() {
    return pi_i<int8_t>();
  }

  static inline bool lt(int8_t a, int8_t b) {
    return a < b;
  }
  static inline bool le(int8_t a, int8_t b) {
    return a <= b;
  }
  static inline bool gt(int8_t a, int8_t b) {
    return a > b;
  }
  static inline bool ge(int8_t a, int8_t b) {
    return a >= b;
  }
  static inline bool eq(int8_t a, int8_t b) {
    return a == b;
  }
  static inline bool ne(int8_t a, int8_t b) {
    return a != b;
  }

  static inline int8_t ceil(int8_t a) {
    return a;
  }
  static inline int8_t floor(int8_t a) {
    return a;
  }
  static inline int8_t round(int8_t a) {
    return a;
  }
  static inline int8_t trunc(int8_t a) {
    return a;
  }
  static inline int8_t neg(int8_t a) {
    return -a;
  }
  static inline int8_t add(int8_t a, int8_t b) {
    return a + b;
  }
  static inline int8_t mul(int8_t a, int8_t b) {
    return a * b;
  }
  static inline int8_t sub(int8_t a, int8_t b) {
    return a - b;
  }
  static inline int8_t div(int8_t a, int8_t b) {
    return a / b;
  }
  static inline int8_t abs(int8_t a) {
    return std::abs((int)a);
  }
  static inline int8_t pow(int8_t a, int8_t b) {
    return powi<int8_t>(a, b);
  }
  static inline bool isnan(int8_t a) {
    return false;
  }
  static inline bool isinf(int8_t a) {
    return false;
  }
  static inline int8_t min(int8_t a, int8_t b) {
    return sycl::min(a, b);
  }
  static inline int8_t max(int8_t a, int8_t b) {
    return sycl::max(a, b);
  }
};

template <>
struct Numerics<int16_t> {
  static inline int16_t lower_bound() {
    return std::numeric_limits<int16_t>::lowest();
  }
  static inline int16_t upper_bound() {
    return std::numeric_limits<int16_t>::max();
  }
  static inline constexpr int16_t pi() {
    return pi_i<int16_t>();
  }

  static inline bool lt(int16_t a, int16_t b) {
    return a < b;
  }
  static inline bool le(int16_t a, int16_t b) {
    return a <= b;
  }
  static inline bool gt(int16_t a, int16_t b) {
    return a > b;
  }
  static inline bool ge(int16_t a, int16_t b) {
    return a >= b;
  }
  static inline bool eq(int16_t a, int16_t b) {
    return a == b;
  }
  static inline bool ne(int16_t a, int16_t b) {
    return a != b;
  }

  static inline int16_t ceil(int16_t a) {
    return a;
  }
  static inline int16_t floor(int16_t a) {
    return a;
  }
  static inline int16_t round(int16_t a) {
    return a;
  }
  static inline int16_t trunc(int16_t a) {
    return a;
  }
  static inline int16_t neg(int16_t a) {
    return -a;
  }
  static inline int16_t add(int16_t a, int16_t b) {
    return a + b;
  }
  static inline int16_t mul(int16_t a, int16_t b) {
    return a * b;
  }
  static inline int16_t sub(int16_t a, int16_t b) {
    return a - b;
  }
  static inline int16_t div(int16_t a, int16_t b) {
    return a / b;
  }
  static inline int16_t abs(int16_t a) {
    return std::abs(a);
  }
  static inline int16_t pow(int16_t a, int16_t b) {
    return powi<int16_t>(a, b);
  }
  static inline bool isnan(int16_t a) {
    return false;
  }
  static inline bool isinf(int8_t a) {
    return false;
  }
  static inline int16_t sgn(int16_t a) {
    return sgni<int16_t>(a);
  }
  static inline int16_t min(int16_t a, int16_t b) {
    return sycl::min(a, b);
  }
  static inline int16_t max(int16_t a, int16_t b) {
    return sycl::max(a, b);
  }
};

template <>
struct Numerics<int32_t> {
  static inline int32_t lower_bound() {
    return std::numeric_limits<int32_t>::lowest();
  }
  static inline int32_t upper_bound() {
    return std::numeric_limits<int32_t>::max();
  }
  static inline constexpr int32_t pi() {
    return pi_i<int32_t>();
  }

  static inline bool lt(int32_t a, int32_t b) {
    return a < b;
  }
  static inline bool le(int32_t a, int32_t b) {
    return a <= b;
  }
  static inline bool gt(int32_t a, int32_t b) {
    return a > b;
  }
  static inline bool ge(int32_t a, int32_t b) {
    return a >= b;
  }
  static inline bool eq(int32_t a, int32_t b) {
    return a == b;
  }
  static inline bool ne(int32_t a, int32_t b) {
    return a != b;
  }

  static inline int32_t ceil(int32_t a) {
    return a;
  }
  static inline int32_t floor(int32_t a) {
    return a;
  }
  static inline int32_t round(int32_t a) {
    return a;
  }
  static inline int32_t trunc(int32_t a) {
    return a;
  }
  static inline int32_t neg(int32_t a) {
    return -a;
  }
  static inline int32_t add(int32_t a, int32_t b) {
    return a + b;
  }
  static inline int32_t mul(int32_t a, int32_t b) {
    return a * b;
  }
  static inline int32_t sub(int32_t a, int32_t b) {
    return a - b;
  }
  static inline int32_t div(int32_t a, int32_t b) {
    return a / b;
  }
  static inline int32_t abs(int32_t a) {
    return std::abs(a);
  }
  static inline int32_t pow(int32_t a, int32_t b) {
    return powi<int32_t>(a, b);
  }
  static inline bool isnan(int32_t a) {
    return false;
  }
  static inline bool isinf(int8_t a) {
    return false;
  }
  static inline int32_t sgn(int32_t a) {
    return sgni<int32_t>(a);
  }
  static inline int32_t min(int32_t a, int32_t b) {
    return sycl::min(a, b);
  }
  static inline int32_t max(int32_t a, int32_t b) {
    return sycl::max(a, b);
  }
};

template <>
struct Numerics<int64_t> {
  static inline int64_t lower_bound() {
    return std::numeric_limits<int64_t>::lowest();
  }
  static inline int64_t upper_bound() {
    return std::numeric_limits<int64_t>::max();
  }
  static inline constexpr int64_t pi() {
    return pi_i<int64_t>();
  }

  static inline bool lt(int64_t a, int64_t b) {
    return a < b;
  }
  static inline bool le(int64_t a, int64_t b) {
    return a <= b;
  }
  static inline bool gt(int64_t a, int64_t b) {
    return a > b;
  }
  static inline bool ge(int64_t a, int64_t b) {
    return a >= b;
  }
  static inline bool eq(int64_t a, int64_t b) {
    return a == b;
  }
  static inline bool ne(int64_t a, int64_t b) {
    return a != b;
  }

  static inline int64_t ceil(int64_t a) {
    return a;
  }
  static inline int64_t floor(int64_t a) {
    return a;
  }
  static inline int64_t round(int64_t a) {
    return a;
  }
  static inline int64_t trunc(int64_t a) {
    return a;
  }
  static inline int64_t neg(int64_t a) {
    return -a;
  }
  static inline int64_t add(int64_t a, int64_t b) {
    return a + b;
  }
  static inline int64_t mul(int64_t a, int64_t b) {
    return a * b;
  }
  static inline int64_t sub(int64_t a, int64_t b) {
    return a - b;
  }
  static inline int64_t div(int64_t a, int64_t b) {
    return a / b;
  }
  static inline int64_t abs(int64_t a) {
    return std::abs(a);
  }
  static inline int64_t pow(int64_t a, int64_t b) {
    return powi<int64_t>(a, b);
  }
  static inline bool isnan(int64_t a) {
    return false;
  }
  static inline bool isinf(int8_t a) {
    return false;
  }
  static inline int64_t sgn(int64_t a) {
    return sgni<int64_t>(a);
  }
  static inline int64_t min(int64_t a, int64_t b) {
    return sycl::min(a, b);
  }
  static inline int64_t max(int64_t a, int64_t b) {
    return sycl::max(a, b);
  }
};

template <>
struct Numerics<at::Half> {
  static inline at::Half lower_bound() {
    return at::Half(0xFC00, at::Half::from_bits());
  }
  static inline at::Half upper_bound() {
    return at::Half(0x7C00, at::Half::from_bits());
  }
  static inline constexpr at::Half pi() {
    return at::Half(0x4248, at::Half::from_bits());
  }

  static inline bool lt(at::Half a, at::Half b) {
    return a < b;
  }
  static inline bool le(at::Half a, at::Half b) {
    return a <= b;
  }
  static inline bool gt(at::Half a, at::Half b) {
    return a > b;
  }
  static inline bool ge(at::Half a, at::Half b) {
    return a >= b;
  }
  static inline bool eq(at::Half a, at::Half b) {
    return a == b;
  }
  static inline bool ne(at::Half a, at::Half b) {
    return a != b;
  }

  static inline at::Half exp(at::Half a) {
    return std::exp(float(a));
  }
  static inline at::Half exp2(at::Half a) {
    return std::exp2(float(a));
  }
  static inline at::Half log(at::Half a) {
    return std::log(float(a));
  }
  static inline at::Half log10(at::Half a) {
    return std::log10(float(a));
  }
  static inline at::Half log1p(at::Half a) {
    return std::log1p(float(a));
  }
  static inline at::Half log2(at::Half a) {
    return std::log2(float(a));
  }
  static inline at::Half expm1(at::Half a) {
    return std::expm1(float(a));
  }

  static inline at::Half neg(at::Half a) {
    return -a;
  }
  static inline at::Half sin(at::Half a) {
    return std::sin(float(a));
  }
  static inline at::Half cos(at::Half a) {
    return std::cos(float(a));
  }
  static inline at::Half sqrt(at::Half a) {
    return std::sqrt(float(a));
  }
  static inline at::Half rsqrt(at::Half a) {
    return sycl::rsqrt(float(a));
  }
  static inline at::Half ceil(at::Half a) {
    return std::ceil(float(a));
  }
  static inline at::Half floor(at::Half a) {
    return std::floor(float(a));
  }
  static inline at::Half trunc(at::Half a) {
    return std::trunc(float(a));
  }
  static inline at::Half acos(at::Half a) {
    return std::acos(float(a));
  }
  static inline at::Half cosh(at::Half a) {
    return std::cosh(float(a));
  }
  static inline at::Half acosh(at::Half a) {
    return std::acosh(float(a));
  }
  static inline at::Half asin(at::Half a) {
    return std::asin(float(a));
  }
  static inline at::Half sinh(at::Half a) {
    return std::sinh(float(a));
  }
  static inline at::Half asinh(at::Half a) {
    return std::asinh(float(a));
  }
  static inline at::Half tan(at::Half a) {
    return std::tan(float(a));
  }
  static inline at::Half atan(at::Half a) {
    return std::atan(float(a));
  }
  static inline at::Half lgamma(at::Half a) {
    return std::lgamma(float(a));
  }
  static inline at::Half tanh(float a) {
    return std::tanh(float(a));
  }
  static inline at::Half atanh(float a) {
    return std::atanh(float(a));
  }
  static inline at::Half erf(float a) {
    return std::erf(float(a));
  }
  static inline at::Half erfc(float a) {
    return std::erfc(float(a));
  }
  static inline at::Half round(float a) {
    return std::round(float(a));
  }
  static inline at::Half frac(at::Half a) {
    return a - std::trunc(float(a));
  }
  static inline at::Half atan2(at::Half a, at::Half b) {
    return std::atan2(float(a), float(b));
  }
  static inline at::Half hypot(at::Half a, at::Half b) {
    return std::hypot(float(a), float(b));
  }
  static inline at::Half cinv(at::Half a) {
    return 1.0f / a;
  }
  static inline at::Half min(at::Half a, at::Half b) {
    if (a != a) {
      return a;
    } else if (b != b) {
      return b;
    } else {
      return std::fmin(float(a), float(b));
    }
  }
  static inline at::Half add(at::Half a, at::Half b) {
    return a + b;
  }
  static inline at::Half div(at::Half a, at::Half b) {
    return a / b;
  }
  static inline at::Half mul(at::Half a, at::Half b) {
    return a * b;
  }
  static inline at::Half sub(at::Half a, at::Half b) {
    return a - b;
  }

  static inline at::Half pow(at::Half a, at::Half b) {
    return std::pow(float(a), float(b));
  }
  static inline at::Half max(at::Half a, at::Half b) {
    if (a != a) {
      return a;
    } else if (b != b) {
      return b;
    } else {
      return std::fmax(float(a), float(b));
    }
  }
  static inline float fmin(at::Half a, at::Half b) {
    return std::fmin((float)a, (float)b);
  }
  static inline float fmax(at::Half a, at::Half b) {
    return std::fmax((float)a, (float)b);
  }
  static inline at::Half abs(at::Half a) {
    return std::fabs(float(a));
  }
  static inline at::Half fabs(at::Half a) {
    return std::fabs(float(a));
  }
  static inline bool isnan(at::Half a) {
    return std::isnan((float)a);
  }
  static inline bool isinf(at::Half a) {
    return std::isinf((float)a);
  }
  static inline at::Half copysign(at::Half a, at::Half b) {
    return std::copysign((float)a, (float)b);
  }
  static inline at::Half fmod(at::Half a, at::Half b) {
    return std::fmod((float)a, (float)b);
  }
};

template <>
struct Numerics<at::BFloat16> {
  static inline at::BFloat16 lower_bound() {
    return at::BFloat16(0xFF80, at::BFloat16::from_bits());
  }
  static inline at::BFloat16 upper_bound() {
    return at::BFloat16(0x7F80, at::BFloat16::from_bits());
  }
  static inline constexpr at::BFloat16 pi() {
    // According to
    // https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Special_values
    // pi is encoded as 4049
    return at::BFloat16(0x4049, at::BFloat16::from_bits());
  }

  static inline bool lt(at::BFloat16 a, at::BFloat16 b) {
    return a < b;
  }
  static inline bool le(at::BFloat16 a, at::BFloat16 b) {
    return a <= b;
  }
  static inline bool gt(at::BFloat16 a, at::BFloat16 b) {
    return a > b;
  }
  static inline bool ge(at::BFloat16 a, at::BFloat16 b) {
    return a >= b;
  }
  static inline bool eq(at::BFloat16 a, at::BFloat16 b) {
    return a == b;
  }
  static inline bool ne(at::BFloat16 a, at::BFloat16 b) {
    return a != b;
  }

  static inline at::BFloat16 exp(at::BFloat16 a) {
    return std::exp(float(a));
  }
  static inline at::BFloat16 exp2(at::BFloat16 a) {
    return std::exp2(float(a));
  }
  static inline at::BFloat16 log(at::BFloat16 a) {
    return std::log(float(a));
  }
  static inline at::BFloat16 log10(at::BFloat16 a) {
    return std::log10(float(a));
  }
  static inline at::BFloat16 log1p(at::BFloat16 a) {
    return std::log1p(float(a));
  }
  static inline at::BFloat16 log2(at::BFloat16 a) {
    return std::log2(float(a));
  }
  static inline at::BFloat16 expm1(at::BFloat16 a) {
    return std::expm1(float(a));
  }

  static inline at::BFloat16 neg(at::BFloat16 a) {
    return -a;
  }
  static inline at::BFloat16 sin(at::BFloat16 a) {
    return std::sin(float(a));
  }
  static inline at::BFloat16 cos(at::BFloat16 a) {
    return std::cos(float(a));
  }
  static inline at::BFloat16 sqrt(at::BFloat16 a) {
    return std::sqrt(float(a));
  }
  static inline at::BFloat16 rsqrt(at::BFloat16 a) {
    return sycl::rsqrt(float(a));
  }
  static inline at::BFloat16 ceil(at::BFloat16 a) {
    return std::ceil(float(a));
  }
  static inline at::BFloat16 floor(at::BFloat16 a) {
    return std::floor(float(a));
  }
  static inline at::BFloat16 trunc(at::BFloat16 a) {
    return std::trunc(float(a));
  }
  static inline at::BFloat16 acos(at::BFloat16 a) {
    return std::acos(float(a));
  }
  static inline at::BFloat16 cosh(at::BFloat16 a) {
    return std::cosh(float(a));
  }
  static inline at::BFloat16 acosh(at::BFloat16 a) {
    return std::acosh(float(a));
  }
  static inline at::BFloat16 asin(at::BFloat16 a) {
    return std::asin(float(a));
  }
  static inline at::BFloat16 sinh(at::BFloat16 a) {
    return std::sinh(float(a));
  }
  static inline at::BFloat16 asinh(at::BFloat16 a) {
    return std::asinh(float(a));
  }
  static inline at::BFloat16 tan(at::BFloat16 a) {
    return std::tan(float(a));
  }
  static inline at::BFloat16 atan(at::BFloat16 a) {
    return std::atan(float(a));
  }
  static inline at::BFloat16 tanh(float a) {
    return std::tanh(float(a));
  }
  static inline at::BFloat16 atanh(float a) {
    return std::atanh(float(a));
  }
  static inline at::BFloat16 erf(float a) {
    return std::erf(float(a));
  }
  static inline at::BFloat16 erfc(float a) {
    return std::erfc(float(a));
  }
  static inline at::BFloat16 lgamma(at::BFloat16 a) {
    return std::lgamma(float(a));
  }
  static inline at::BFloat16 round(float a) {
    return std::round(float(a));
  }
  static inline at::BFloat16 min(at::BFloat16 a, at::BFloat16 b) {
    if (a != a) {
      return a;
    } else if (b != b) {
      return b;
    } else {
      return std::fmin(float(a), float(b));
    }
  }
  static inline at::BFloat16 max(at::BFloat16 a, at::BFloat16 b) {
    if (a != a) {
      return a;
    } else if (b != b) {
      return b;
    } else {
      return std::fmax(float(a), float(b));
    }
  }
  static inline float fmin(at::BFloat16 a, at::BFloat16 b) {
    return std::fmin((float)a, (float)b);
  }
  static inline float fmax(at::BFloat16 a, at::BFloat16 b) {
    return std::fmax((float)a, (float)b);
  }
  static inline at::BFloat16 frac(at::BFloat16 a) {
    return a - std::trunc(float(a));
  }
  static inline at::BFloat16 atan2(at::BFloat16 a, at::BFloat16 b) {
    return std::atan2(float(a), float(b));
  }
  static inline at::BFloat16 hypot(at::BFloat16 a, at::BFloat16 b) {
    return std::hypot(float(a), float(b));
  }
  static inline at::BFloat16 nextafter(at::BFloat16 a, at::BFloat16 b) {
    return nextafteri(a, b);
  }
  static inline at::BFloat16 cinv(at::BFloat16 a) {
    return 1.0f / a;
  }

  static inline at::BFloat16 add(at::BFloat16 a, at::BFloat16 b) {
    return a + b;
  }
  static inline at::BFloat16 div(at::BFloat16 a, at::BFloat16 b) {
    return a / b;
  }
  static inline at::BFloat16 mul(at::BFloat16 a, at::BFloat16 b) {
    return a * b;
  }
  static inline at::BFloat16 sub(at::BFloat16 a, at::BFloat16 b) {
    return a - b;
  }

  static inline at::BFloat16 pow(at::BFloat16 a, at::BFloat16 b) {
    return std::pow(float(a), float(b));
  }
  static inline at::BFloat16 abs(at::BFloat16 a) {
    return std::fabs(float(a));
  }
  static inline at::BFloat16 fabs(at::BFloat16 a) {
    return std::fabs(float(a));
  }
  static inline bool isnan(at::BFloat16 a) {
    return std::isnan((float)a);
  }
  static inline bool isinf(at::BFloat16 a) {
    return std::isinf((float)a);
  }
  static inline at::BFloat16 copysign(at::BFloat16 a, at::BFloat16 b) {
    return std::copysign((float)a, (float)b);
  }
  static inline at::BFloat16 fmod(at::BFloat16 a, at::BFloat16 b) {
    return std::fmod((float)a, (float)b);
  }
};

template <>
struct Numerics<float> {
  static inline float lower_bound() {
    return -std::numeric_limits<float>::infinity();
  }
  static inline float upper_bound() {
    return std::numeric_limits<float>::infinity();
  }
  static inline constexpr float pi() {
    return pi_i<float>();
  }

  static inline bool lt(float a, float b) {
    return std::isless(a, b);
  }
  static inline bool le(float a, float b) {
    return std::islessequal(a, b);
  }
  static inline bool gt(float a, float b) {
    return std::isgreater(a, b);
  }
  static inline bool ge(float a, float b) {
    return std::isgreaterequal(a, b);
  }
  static inline bool eq(float a, float b) {
    return a == b;
  }
  static inline bool ne(float a, float b) {
    return a != b;
  }

  static inline float exp(float a) {
    return std::exp(a);
  }
  static inline float exp2(float a) {
    return std::exp2(a);
  }
  static inline float log(float a) {
    return std::log(a);
  }
  static inline float log10(float a) {
    return std::log10(a);
  }
  static inline float log1p(float a) {
    return std::log1p(a);
  }
  static inline float log2(float a) {
    return std::log2(a);
  }
  static inline float expm1(float a) {
    return std::expm1(a);
  }

  static inline float neg(float a) {
    return -a;
  }
  static inline float sin(float a) {
    return std::sin(a);
  }
  static inline float cos(float a) {
    return std::cos(a);
  }
  static inline float sqrt(float a) {
    return std::sqrt(a);
  }
  static inline float rsqrt(float a) {
    return sycl::rsqrt(a);
  }
  static inline float ceil(float a) {
    return std::ceil(a);
  }
  static inline float floor(float a) {
    return std::floor(a);
  }
  static inline float trunc(float a) {
    return std::trunc(a);
  }

  static inline float acos(float a) {
    return std::acos(a);
  }
  static inline float cosh(float a) {
    return std::cosh(a);
  }
  static inline float acosh(float a) {
    return std::acosh(a);
  }
  static inline float asin(float a) {
    return std::asin(a);
  }
  static inline float sinh(float a) {
    return std::sinh(a);
  }
  static inline float asinh(float a) {
    return std::asinh(a);
  }
  static inline float tan(float a) {
    return std::tan(a);
  }
  static inline float atan(float a) {
    return std::atan(a);
  }
  static inline float tanh(float a) {
    return std::tanh(a);
  }
  static inline float atanh(float a) {
    return std::atanh(a);
  }
  static inline float erf(float a) {
    return std::erf(a);
  }
  static inline float erfc(float a) {
    return std::erfc(a);
  }
  static inline float lgamma(float a) {
    return std::lgamma(a);
  }
  static inline float round(float a) {
    return std::round(a);
  }

  static inline float frac(float a) {
    return a - std::trunc(a);
  }
  static inline float cinv(float a) {
    return 1.0f / a;
  }
  static inline float add(float a, float b) {
    return a + b;
  }
  static inline float div(float a, float b) {
    return a / b;
  }
  static inline float mul(float a, float b) {
    return a * b;
  }
  static inline float sub(float a, float b) {
    return a - b;
  }
  static inline float pow(float a, float b) {
    return std::pow(a, b);
  }
  static inline float atan2(float a, float b) {
    return std::atan2(a, b);
  }
  static inline float hypot(float a, float b) {
    return std::hypot(a, b);
  }
  static inline float nextafter(float a, float b) {
    return std::nextafter(a, b);
  }
  static inline float min(float a, float b) {
    if (a != a) {
      return a;
    } else if (b != b) {
      return b;
    } else {
      return std::fmin(a, b);
    }
  }
  static inline float max(float a, float b) {
    if (a != a) {
      return a;
    } else if (b != b) {
      return b;
    } else {
      return std::fmax(a, b);
    }
  }
  static inline float fmin(float a, float b) {
    return std::fmin(a, b);
  }
  static inline float fmax(float a, float b) {
    return std::fmax(a, b);
  }
  static inline float abs(float a) {
    return std::fabs(a);
  }
  static inline float fabs(float a) {
    return std::fabs(a);
  }
  static inline bool isnan(float a) {
    return std::isnan(a);
  }
  static inline bool isinf(float a) {
    return std::isinf(a);
  }
  static inline float copysign(float a, float b) {
    return std::copysign(a, b);
  }
  static inline float fmod(float a, float b) {
    return std::fmod(a, b);
  }
};

template <>
struct Numerics<double> {
  static inline double lower_bound() {
    return -std::numeric_limits<double>::infinity();
  }
  static inline double upper_bound() {
    return std::numeric_limits<double>::infinity();
  }
  static inline constexpr double pi() {
    return pi_i<double>();
  }

  static inline bool lt(double a, double b) {
    return std::isless(a, b);
  }
  static inline bool le(double a, double b) {
    return std::islessequal(a, b);
  }
  static inline bool gt(double a, double b) {
    return std::isgreater(a, b);
  }
  static inline bool ge(double a, double b) {
    return std::isgreaterequal(a, b);
  }
  static inline bool eq(double a, double b) {
    return a == b;
  }
  static inline bool ne(double a, double b) {
    return a != b;
  }

  static inline double exp(double a) {
    return std::exp(a);
  }
  static inline double exp2(double a) {
    return std::exp2(a);
  }
  static inline double log(double a) {
    return std::log(a);
  }
  static inline double log10(double a) {
    return std::log10(a);
  }
  static inline double log1p(double a) {
    return std::log1p(a);
  }
  static inline double log2(double a) {
    return std::log2(a);
  }
  static inline double expm1(double a) {
    return std::expm1(a);
  }

  static inline double neg(double a) {
    return -a;
  }
  static inline double sin(double a) {
    return std::sin(a);
  }
  static inline double cos(double a) {
    return std::cos(a);
  }
  static inline double sqrt(double a) {
    return std::sqrt(a);
  }
  static inline double rsqrt(double a) {
    return sycl::rsqrt(a);
  }
  static inline double ceil(double a) {
    return std::ceil(a);
  }
  static inline double floor(double a) {
    return std::floor(a);
  }
  static inline double trunc(double a) {
    return std::trunc(a);
  }
  static inline double acos(double a) {
    return std::acos(a);
  }
  static inline double cosh(double a) {
    return std::cosh(a);
  }
  static inline double acosh(double a) {
    return std::acosh(a);
  }
  static inline double asin(double a) {
    return std::asin(a);
  }
  static inline double sinh(double a) {
    return std::sinh(a);
  }
  static inline double asinh(double a) {
    return std::asinh(a);
  }
  static inline double tan(double a) {
    return std::tan(a);
  }
  static inline double atan(double a) {
    return std::atan(a);
  }
  static inline double tanh(double a) {
    return std::tanh(a);
  }
  static inline double atanh(double a) {
    return std::atanh(a);
  }
  static inline double erf(double a) {
    return std::erf(a);
  }
  static inline double erfc(double a) {
    return std::erfc(a);
  }
  static inline double lgamma(double a) {
    return std::lgamma(a);
  }
  static inline double round(double a) {
    return std::round(a);
  }

  static inline double frac(double a) {
    return a - std::trunc(a);
  }
  static inline double cinv(double a) {
    return 1.0f / a;
  }
  static inline double add(double a, double b) {
    return a + b;
  }
  static inline double div(double a, double b) {
    return a / b;
  }
  static inline double mul(double a, double b) {
    return a * b;
  }
  static inline double sub(double a, double b) {
    return a - b;
  }
  static inline double pow(double a, double b) {
    return std::pow(a, b);
  }
  static inline double atan2(double a, double b) {
    return std::atan2(a, b);
  }
  static inline double hypot(double a, double b) {
    return std::hypot(a, b);
  }
  static inline double nextafter(double a, double b) {
    return std::nextafter(a, b);
  }
  static inline double min(double a, double b) {
    if (a != a) {
      return a;
    } else if (b != b) {
      return b;
    } else {
      return std::fmin(a, b);
    }
  }
  static inline double max(double a, double b) {
    if (a != a) {
      return a;
    } else if (b != b) {
      return b;
    } else {
      return std::fmax(a, b);
    }
  }
  static inline double fmin(double a, double b) {
    return std::fmin(a, b);
  }
  static inline double fmax(double a, double b) {
    return std::fmax(a, b);
  }
  static inline double abs(double a) {
    return std::fabs(a);
  }
  static inline double fabs(double a) {
    return std::fabs(a);
  }
  static inline bool isnan(double a) {
    return std::isnan(a);
  }
  static inline bool isinf(double a) {
    return std::isinf(a);
  }
  static inline double copysign(double a, double b) {
    return std::copysign(a, b);
  }
  static inline double fmod(double a, double b) {
    return std::fmod(a, b);
  }
};

template <>
struct Numerics<c10::complex<float>> {
  static inline c10::complex<float> acos(c10::complex<float> a) {
    return static_cast<c10::complex<float>>(
        std::acos(static_cast<std::complex<float>>(a)));
  }

  static inline c10::complex<float> sin(c10::complex<float> a) {
    return static_cast<c10::complex<float>>(
        std::sin(static_cast<std::complex<float>>(a)));
  }

  static inline c10::complex<float> cosh(c10::complex<float> a) {
    return static_cast<c10::complex<float>>(
        std::cosh(static_cast<std::complex<float>>(a)));
  }

  static inline c10::complex<float> acosh(c10::complex<float> a) {
    return static_cast<c10::complex<float>>(
        std::acosh(static_cast<std::complex<float>>(a)));
  }

  static inline c10::complex<float> sinh(c10::complex<float> a) {
    return static_cast<c10::complex<float>>(
        std::sinh(static_cast<std::complex<float>>(a)));
  }

  static inline c10::complex<float> asinh(c10::complex<float> a) {
    return static_cast<c10::complex<float>>(
        std::asinh(static_cast<std::complex<float>>(a)));
  }

  static inline c10::complex<float> asin(c10::complex<float> a) {
    return static_cast<c10::complex<float>>(
        std::asin(static_cast<std::complex<float>>(a)));
  }

  static inline c10::complex<float> cos(c10::complex<float> a) {
    return static_cast<c10::complex<float>>(
        std::cos(static_cast<std::complex<float>>(a)));
  }

  static inline c10::complex<float> atan(c10::complex<float> a) {
    return static_cast<c10::complex<float>>(
        std::atan(static_cast<std::complex<float>>(a)));
  }

  static inline c10::complex<float> tan(c10::complex<float> a) {
    return static_cast<c10::complex<float>>(
        std::tan(static_cast<std::complex<float>>(a)));
  }

  static inline c10::complex<float> tanh(c10::complex<float> a) {
    return static_cast<c10::complex<float>>(
        std::tanh(static_cast<std::complex<float>>(a)));
  }

  static inline c10::complex<float> atanh(c10::complex<float> a) {
    return static_cast<c10::complex<float>>(
        std::atanh(static_cast<std::complex<float>>(a)));
  }

  static inline c10::complex<float> neg(c10::complex<float> a) {
    return -a;
  }

  static inline c10::complex<float> sqrt(c10::complex<float> a) {
    return static_cast<c10::complex<float>>(
        std::sqrt(static_cast<std::complex<float>>(a)));
  }

  static inline c10::complex<float> rsqrt(c10::complex<float> a) {
    return c10::complex<float>(1.0, 0) /
        static_cast<c10::complex<float>>(
               std::sqrt(static_cast<std::complex<float>>(a)));
  }

  static inline float abs(c10::complex<float> a) {
    return std::abs(a);
  }

  static inline float fabs(c10::complex<float> a) {
    // std::abs will return corresponding float type when arg is complex<float>
    return std::abs(a);
  }

  static inline c10::complex<float> exp(c10::complex<float> a) {
    return static_cast<c10::complex<float>>(
        std::exp(static_cast<std::complex<float>>(a)));
  }

  static inline c10::complex<float> add(
      c10::complex<float> a,
      c10::complex<float> b) {
    return a + b;
  }
  static inline c10::complex<float> mul(
      c10::complex<float> a,
      c10::complex<float> b) {
    return a * b;
  }
  static inline c10::complex<float> sub(
      c10::complex<float> a,
      c10::complex<float> b) {
    return a - b;
  }
  static inline c10::complex<float> div(
      c10::complex<float> a,
      c10::complex<float> b) {
    return a / b;
  }

  static inline c10::complex<float> pow(
      c10::complex<float> a,
      c10::complex<float> b) {
    return static_cast<c10::complex<float>>(std::pow(
        static_cast<std::complex<float>>(a),
        static_cast<std::complex<float>>(b)));
  }

  static inline c10::complex<float> log(c10::complex<float> a) {
    return static_cast<c10::complex<float>>(
        std::log(static_cast<std::complex<float>>(a)));
  }

  static inline c10::complex<float> log2(c10::complex<float> a) {
    return std::log2(a);
  }

  static inline c10::complex<float> log10(c10::complex<float> a) {
    return static_cast<c10::complex<float>>(
        std::log10(static_cast<std::complex<float>>(a)));
  }

  static inline bool eq(c10::complex<float> a, c10::complex<float> b) {
    return a == b;
  }

  static inline bool ne(c10::complex<float> a, c10::complex<float> b) {
    return a != b;
  }

  static inline constexpr c10::complex<float> pi() {
    return c10::complex<float>(pi_i<float>(), 0);
  }
};

template <>
struct Numerics<c10::complex<double>> {
  static inline c10::complex<double> acos(c10::complex<double> a) {
    return static_cast<c10::complex<double>>(
        std::acos(static_cast<std::complex<double>>(a)));
  }

  static inline c10::complex<double> sin(c10::complex<double> a) {
    return static_cast<c10::complex<double>>(
        std::sin(static_cast<std::complex<double>>(a)));
  }

  static inline c10::complex<double> cosh(c10::complex<double> a) {
    return static_cast<c10::complex<double>>(
        std::cosh(static_cast<std::complex<double>>(a)));
  }

  static inline c10::complex<double> acosh(c10::complex<double> a) {
    return static_cast<c10::complex<double>>(
        std::acosh(static_cast<std::complex<double>>(a)));
  }

  static inline c10::complex<double> sinh(c10::complex<double> a) {
    return static_cast<c10::complex<double>>(
        std::sinh(static_cast<std::complex<double>>(a)));
  }

  static inline c10::complex<double> asinh(c10::complex<double> a) {
    return static_cast<c10::complex<double>>(
        std::asinh(static_cast<std::complex<double>>(a)));
  }

  static inline c10::complex<double> asin(c10::complex<double> a) {
    return static_cast<c10::complex<double>>(
        std::asin(static_cast<std::complex<double>>(a)));
  }

  static inline c10::complex<double> cos(c10::complex<double> a) {
    return static_cast<c10::complex<double>>(
        std::cos(static_cast<std::complex<double>>(a)));
  }

  static inline c10::complex<double> atan(c10::complex<double> a) {
    return static_cast<c10::complex<double>>(
        std::atan(static_cast<std::complex<double>>(a)));
  }

  static inline c10::complex<double> tan(c10::complex<double> a) {
    return static_cast<c10::complex<double>>(
        std::tan(static_cast<std::complex<double>>(a)));
  }

  static inline c10::complex<double> tanh(c10::complex<double> a) {
    return static_cast<c10::complex<double>>(
        std::tanh(static_cast<std::complex<double>>(a)));
  }

  static inline c10::complex<double> atanh(c10::complex<double> a) {
    return static_cast<c10::complex<double>>(
        std::atanh(static_cast<std::complex<double>>(a)));
  }

  static inline c10::complex<double> neg(c10::complex<double> a) {
    return -a;
  }

  static inline c10::complex<double> sqrt(c10::complex<double> a) {
    return static_cast<c10::complex<double>>(
        std::sqrt(static_cast<std::complex<double>>(a)));
  }

  static inline c10::complex<double> rsqrt(c10::complex<double> a) {
    return c10::complex<double>(1.0, 0) /
        static_cast<c10::complex<double>>(
               std::sqrt(static_cast<std::complex<double>>(a)));
  }

  static inline double abs(c10::complex<double> a) {
    return std::abs(a);
  }

  static inline double fabs(c10::complex<double> a) {
    // std::abs will return corresponding double type when arg is
    // complex<double>
    return std::abs(a);
  }

  static inline c10::complex<double> exp(c10::complex<double> a) {
    return static_cast<c10::complex<double>>(
        std::exp(static_cast<std::complex<double>>(a)));
  }

  static inline c10::complex<double> add(
      c10::complex<double> a,
      c10::complex<double> b) {
    return a + b;
  }
  static inline c10::complex<double> mul(
      c10::complex<double> a,
      c10::complex<double> b) {
    return a * b;
  }
  static inline c10::complex<double> sub(
      c10::complex<double> a,
      c10::complex<double> b) {
    return a - b;
  }
  static inline c10::complex<double> div(
      c10::complex<double> a,
      c10::complex<double> b) {
    return a / b;
  }

  static inline c10::complex<double> pow(
      c10::complex<double> a,
      c10::complex<double> b) {
    return static_cast<c10::complex<double>>(std::pow(
        static_cast<std::complex<double>>(a),
        static_cast<std::complex<double>>(b)));
  }

  static inline c10::complex<double> log(c10::complex<double> a) {
    return static_cast<c10::complex<double>>(
        std::log(static_cast<std::complex<double>>(a)));
  }

  static inline c10::complex<double> log2(c10::complex<double> a) {
    return std::log2(a);
  }

  static inline c10::complex<double> log10(c10::complex<double> a) {
    return static_cast<c10::complex<double>>(
        std::log10(static_cast<std::complex<double>>(a)));
  }

  static inline bool eq(c10::complex<double> a, c10::complex<double> b) {
    return a == b;
  }

  static inline bool ne(c10::complex<double> a, c10::complex<double> b) {
    return a != b;
  }

  static inline constexpr c10::complex<double> pi() {
    return c10::complex<double>(pi_i<double>(), 0);
  }
};

template <typename In, typename Out>
struct ScalarConvert {
  static Out to(const In v) {
    return (Out)v;
  }
};

template <typename T, typename U>
T scalar_cast(U u) {
  return ScalarConvert<U, T>::to(u);
}

template <typename T>
inline T Min(T a, T b) {
  return (a < b) ? a : b;
}

template <typename T>
inline T Max(T a, T b) {
  return (a > b) ? a : b;
}

template <typename T>
inline T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

/**
 *    Computes ceil(a / b) * b; i.e., rounds up `a` to the next highest
 *       multiple of b
 *       */

template <typename T>
inline T RoundUp(T a, T b) {
  return CeilDiv(a, b) * b;
}
