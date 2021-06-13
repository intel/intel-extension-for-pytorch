#pragma once

#include <ATen/ATen.h>
#include <runtime/DPCPP.h>
#include "Pointwise.h"

namespace at {
namespace AtenIpexTypeXPU {

template <typename T>
struct TensorBitAndConstantOp {
  TensorBitAndConstantOp(T v) : val(v) {}
  void operator()(T& out, T& in) const {
    out = in & val;
  }

  void operator()(T& v) const {
    v &= val;
  }

  const T val;
};

template <typename T>
struct TensorBitOrConstantOp {
  TensorBitOrConstantOp(T v) : val(v) {}
  void operator()(T& out, T& in) const {
    out = in | val;
  }

  void operator()(T& v) const {
    v |= val;
  }

  const T val;
};

template <typename T>
struct TensorRemainderOp {
  TensorRemainderOp(T v) : val(v) {}
  void operator()(T& out, T& in) const {
    out = in % val;
    if (modulo_wrap<T>(out, val)) {
      out += val;
    }
  }

  void operator()(T& v) const {
    v = v % val;
    if (modulo_wrap<T>(v, val)) {
      v += val;
    }
  }

  const T val;
};

template <>
struct TensorRemainderOp<float> {
  TensorRemainderOp(float v) : val(v) {}
  void operator()(float& out, float& in) const {
    out = in - val * DPCPP::floor(in / val);
  }

  void operator()(float& v) const {
    v = v - val * DPCPP::floor(v / val);
  }

  const float val;
};

template <>
struct TensorRemainderOp<double> {
  TensorRemainderOp(double v) : val(v) {}
  void operator()(double& out, double& in) const {
    out = in - val * DPCPP::floor(in / val);
  }

  void operator()(double& v) const {
    v = v - val * DPCPP::floor(v / val);
  }

  const double val;
};

template <>
struct TensorRemainderOp<at::Half> {
  TensorRemainderOp(at::Half v) : val(v) {}

  void operator()(at::Half& out, at::Half& in) const {
    out = in - val * DPCPP::floor(float(in / val));
  }

  void operator()(at::Half& v) const {
    v = v - val * DPCPP::floor(float(v / val));
  }

  const at::Half val;
};

template <>
struct TensorRemainderOp<at::BFloat16> {
  TensorRemainderOp(at::BFloat16 v) : val(v) {}

  void operator()(at::BFloat16& out, at::BFloat16& in) const {
    out = in - val * DPCPP::floor(float(in / val));
  }

  void operator()(at::BFloat16& v) const {
    v = v - val * DPCPP::floor(float(v / val));
  }

  const at::BFloat16 val;
};

template <typename T>
struct TensorFmodOp {
  TensorFmodOp(T v) : val((float)v) {}
  void operator()(T& out, T& in) const {
    out = (T)DPCPP::fmod((float)in, val);
  }

  void operator()(T& v) const {
    v = (T)DPCPP::fmod((float)v, val);
  }

  const float val;
};

template <>
struct TensorFmodOp<double> {
  TensorFmodOp(double v) : val(v) {}
  void operator()(double& out, double& in) const {
    out = DPCPP::fmod(in, val);
  }

  void operator()(double v) const {
    v = DPCPP::fmod(v, val);
  }

  const double val;
};

template <typename T, int Upper>
struct TensorTriOp {
  TensorTriOp(int64_t stride0_, int64_t stride1_, int64_t k_)
      : stride0(stride0_), stride1(stride1_), k(k_) {}

  int mask(T& out, int64_t offset) const {
    if (offset < 1) {
      DPCPP_PRINT("offset is %ld, mask is not supported!\n", offset);
    }

    ptrdiff_t n = offset - 1;
    int64_t row, col;

    if (stride0 > stride1) {
      row = (int64_t)(n / stride0);
      col = (int64_t)((n % stride0) / stride1);
    } else {
      row = (int64_t)((n % stride1) / stride0);
      col = (int64_t)(n / stride1);
    }

    return Upper ? (col - row >= k) : (col - row <= k);
  }

  void operator()(T& out, T& in, int64_t offset) const {
    out = mask(out, offset) ? in : ScalarConvert<int, T>::to(0);
  }

  void operator()(T& v, int64_t offset) const {
    if (!mask(v, offset))
      v = ScalarConvert<int, T>::to(0);
  }

  const int64_t stride0, stride1, k;
};

} // namespace AtenIpexTypeXPU
} // namespace at
