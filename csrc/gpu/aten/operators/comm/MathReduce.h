#pragma once

#include <ATen/ATen.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include <cmath>

#include "Numerics.h"
#include "Pointwise.h"
#include "SimpleReduce.h"
using namespace at::native;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t>
DPCPP_DEVICE struct AddOp {
  scalar_t operator()(const scalar_t& lhs, const scalar_t& rhs) const {
    return lhs + rhs;
  }
};

template <typename scalar_t>
DPCPP_DEVICE struct MulOp {
  scalar_t operator()(const scalar_t& lhs, const scalar_t& rhs) const {
    return lhs * rhs;
  }
};

DPCPP_DEVICE struct LogicalAll {
  unsigned char operator()(unsigned char const x, unsigned char const y) const {
    return (x && y);
  }
};

DPCPP_DEVICE struct LogicalAny {
  unsigned char operator()(unsigned char const x, unsigned char const y) const {
    return (x || y);
  }
};

template <typename T>
DPCPP_DEVICE struct ReduceAdd {
  T operator()(const T a, const T b) const {
    return Numerics<T>::add(a, b);
  }
};

template <typename T>
DPCPP_DEVICE struct ReduceMin {
  T operator()(T a, T b) const {
    return (Numerics<T>::lt(a, b) || Numerics<T>::isnan(a)) ? a : b;
  }
};

template <typename T>
DPCPP_DEVICE struct ReduceMax {
  T operator()(T a, T b) const {
    return (Numerics<T>::gt(a, b) || Numerics<T>::isnan(a)) ? a : b;
  }
};

template <typename T, typename Index>
struct MaxValuePair {
  std::pair<T, Index> operator()(
      const std::pair<T, Index> a,
      const std::pair<T, Index> b) const {
    return (Numerics<T>::ge(a.first, b.first) || Numerics<T>::isnan(a.first))
        ? a
        : b;
  }
};

template <typename T, typename Index>
struct MinValuePair {
  std::pair<T, Index> operator()(
      const std::pair<T, Index> a,
      const std::pair<T, Index> b) const {
    return (Numerics<T>::le(a.first, b.first) || Numerics<T>::isnan(a.first))
        ? a
        : b;
  }
};

} // namespace AtenIpexTypeXPU
} // namespace at
