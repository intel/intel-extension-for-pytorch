#pragma once

#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <utils/DPCPP.h>
#include "ATDispatch.h"

#include "Numerics.h"
#include "zmath.h"

namespace at {
namespace AtenIpexTypeXPU {

// Unary
#define IPEX_UNARY_AND_ALL_OPS(op, func, creator, types) \
  template <typename scalar_t>                           \
  struct op##_functor {                                  \
    scalar_t operator()(scalar_t a) const {              \
      return func(a);                                    \
    }                                                    \
  };                                                     \
  Tensor& op(const Tensor& self, Tensor& out) {          \
    auto iter = TensorIterator::creator(out, self);      \
    IPEX_DISPATCH_##types##_AND2(                        \
        at::ScalarType::Half,                            \
        at::ScalarType::BFloat16,                        \
        iter.dtype(),                                    \
        #op,                                             \
        [&]() {                                          \
          op##_functor<scalar_t> f;                      \
          dpcpp_kernel_for_tensor_iter(iter, f);         \
        });                                              \
    return out;                                          \
  }

#define IPEX_UNARY_BASE_OPS(op, func, creator, types) \
  template <typename scalar_t>                        \
  struct op##_functor {                               \
    scalar_t operator()(scalar_t a) const {           \
      return func(a);                                 \
    }                                                 \
  };                                                  \
  Tensor& op(const Tensor& self, Tensor& out) {       \
    auto iter = TensorIterator::creator(out, self);   \
    IPEX_DISPATCH_##types(iter.dtype(), #op, [&]() {  \
      op##_functor<scalar_t> f;                       \
      dpcpp_kernel_for_tensor_iter(iter, f);          \
    });                                               \
    return out;                                       \
  }

#define IPEX_UNARY_AND_ALL_OPS_COMMON(op, func, creator, types) \
  template <typename scalar_t>                                  \
  struct op##_functor {                                         \
    scalar_t operator()(scalar_t a) const {                     \
      return func(a);                                           \
    }                                                           \
  };                                                            \
  Tensor& op(const Tensor& self, Tensor& out) {                 \
    auto iter = TensorIterator::creator(out, self);             \
    IPEX_DISPATCH_##types##_AND2(                               \
        at::ScalarType::Half,                                   \
        at::ScalarType::BFloat16,                               \
        iter.common_dtype(),                                    \
        #op,                                                    \
        [&]() {                                                 \
          op##_functor<scalar_t> f;                             \
          dpcpp_kernel_for_tensor_iter(iter, f);                \
        });                                                     \
    return out;                                                 \
  }

#define IPEX_UNARY_BASE_OPS_COMMON(op, func, creator, types) \
  template <typename scalar_t>                               \
  struct op##_functor {                                      \
    scalar_t operator()(scalar_t a) const {                  \
      return func(a);                                        \
    }                                                        \
  };                                                         \
  Tensor& op(const Tensor& self, Tensor& out) {              \
    auto iter = TensorIterator::creator(out, self);          \
    IPEX_DISPATCH_##types(iter.common_dtype(), #op, [&]() {  \
      op##_functor<scalar_t> f;                              \
      dpcpp_kernel_for_tensor_iter(iter, f);                 \
    });                                                      \
    return out;                                              \
  }

#define IPEX_UNARY_LOOPS_FUNC_FLOAT_ALL(op, func, creator) \
  IPEX_UNARY_AND_ALL_OPS(op, func, creator, FLOATING_TYPES)

#define IPEX_UNARY_LOOPS_FUNC_ALL_ALL(op, func, creator) \
  IPEX_UNARY_AND_ALL_OPS(op, func, creator, ALL_TYPES)

#define IPEX_UNARY_LOOPS_FUNC_FLOAT_ALL_COMPLEX(op, func, creator) \
  IPEX_UNARY_AND_ALL_OPS(op, func, creator, FLOATING_AND_COMPLEX_TYPES)

#define IPEX_UNARY_LOOPS_FUNC_ALL_ALL_COMPLEX(op, func, creator) \
  IPEX_UNARY_AND_ALL_OPS(op, func, creator, ALL_TYPES_AND_COMPLEX)

#define IPEX_UNARY_LOOPS_FUNC_FLOAT_BASE(op, func, creator) \
  IPEX_UNARY_BASE_OPS(op, func, creator, FLOATING_TYPES)

#define IPEX_UNARY_LOOPS_FUNC_ALL_BASE(op, func, creator) \
  IPEX_UNARY_BASE_OPS(op, func, creator, ALL_TYPES)

#define IPEX_UNARY_LOOPS_FUNC_FLOAT_BASE_COMPLEX(op, func, creator) \
  IPEX_UNARY_BASE_OPS(op, func, creator, FLOATING_AND_COMPLEX_TYPES)

#define IPEX_UNARY_LOOPS_FUNC_ALL_BASE_COMPLEX(op, func, creator) \
  IPEX_UNARY_BASE_OPS(op, func, creator, ALL_TYPES_AND_COMPLEX)

#define IPEX_UNARY_LOOPS_FUNC_COMPLEX(op, func, creator) \
  IPEX_UNARY_BASE_OPS(op, func, creator, COMPLEX_TYPES)

// common_dtype
#define IPEX_UNARY_LOOPS_FUNC_FLOAT_ALL_COMMON(op, func, creator) \
  IPEX_UNARY_AND_ALL_OPS_COMMON(op, func, creator, FLOATING_TYPES)

#define IPEX_UNARY_LOOPS_FUNC_ALL_ALL_COMMON(op, func, creator) \
  IPEX_UNARY_AND_ALL_OPS_COMMON(op, func, creator, ALL_TYPES)

#define IPEX_UNARY_LOOPS_FUNC_FLOAT_ALL_COMPLEX_COMMON(op, func, creator) \
  IPEX_UNARY_AND_ALL_OPS_COMMON(op, func, creator, FLOATING_AND_COMPLEX_TYPES)

#define IPEX_UNARY_LOOPS_FUNC_ALL_ALL_COMPLEX_COMMON(op, func, creator) \
  IPEX_UNARY_AND_ALL_OPS_COMMON(op, func, creator, ALL_TYPES_AND_COMPLEX)

#define IPEX_UNARY_LOOPS_FUNC_FLOAT_BASE_COMMON(op, func, creator) \
  IPEX_UNARY_BASE_OPS_COMMON(op, func, creator, FLOATING_TYPES)

#define IPEX_UNARY_LOOPS_FUNC_ALL_BASE_COMMON(op, func, creator) \
  IPEX_UNARY_BASE_OPS_COMMON(op, func, creator, ALL_TYPES)

#define IPEX_UNARY_LOOPS_FUNC_FLOAT_BASE_COMPLEX_COMMON(op, func, creator) \
  IPEX_UNARY_BASE_OPS_COMMON(op, func, creator, FLOATING_AND_COMPLEX_TYPES)

#define IPEX_UNARY_LOOPS_FUNC_ALL_BASE_COMPLEX_COMMON(op, func, creator) \
  IPEX_UNARY_BASE_OPS_COMMON(op, func, creator, ALL_TYPES_AND_COMPLEX)

#define IPEX_UNARY_LOOPS_FUNC_COMPLEX_COMMON(op, func, creator) \
  IPEX_UNARY_BASE_OPS_COMMON(op, func, creator, COMPLEX_TYPES)

} // namespace AtenIpexTypeXPU
} // namespace at
