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
#define IPEX_UNARY_AND_ALL_OPS(op, func, creator, types)             \
  Tensor& op(const Tensor& self, Tensor& out) {                      \
    auto iter = TensorIterator::creator(out, self);                  \
    IPEX_DISPATCH_##types##_AND2(                                    \
        at::ScalarType::Half,                                        \
        at::ScalarType::BFloat16,                                    \
        iter.dtype(),                                                \
        #op,                                                         \
        [&]() {                                                      \
          dpcpp_kernel_for_tensor_iter(                              \
              iter, [](scalar_t a) -> scalar_t { return func(a); }); \
        });                                                          \
    return out;                                                      \
  }

#define IPEX_UNARY_BASE_OPS(op, func, creator, types)            \
  Tensor& op(const Tensor& self, Tensor& out) {                  \
    auto iter = TensorIterator::creator(out, self);              \
    IPEX_DISPATCH_##types(iter.dtype(), #op, [&]() {             \
      dpcpp_kernel_for_tensor_iter(                              \
          iter, [](scalar_t a) -> scalar_t { return func(a); }); \
    });                                                          \
    return out;                                                  \
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

// Binary
#define IPEX_BINARY_AND_ALL_OPS(op, func, creator, types)                     \
  Tensor& op(const Tensor& self, const Tensor& other, Tensor& out) {          \
    /* to handle op(tensor, scalar) redispatch path. */                       \
    /* by default, scalar will be wrapped as a CPU tensor in default */       \
    /* catchall implememtation. here convert to XPU lazily. */                \
    Tensor other_maybe_scalar = other;                                        \
    if (other.device().type() == at::kCPU && other.numel() == 1) {            \
      other_maybe_scalar = other.to("xpu");                                   \
    }                                                                         \
                                                                              \
    auto iter = TensorIterator::creator(out, self, other_maybe_scalar);       \
    IPEX_DISPATCH_##types##_AND2(                                             \
        at::ScalarType::Half,                                                 \
        at::ScalarType::BFloat16,                                             \
        iter.dtype(),                                                         \
        #op,                                                                  \
        [&]() {                                                               \
          dpcpp_kernel_for_tensor_iter(                                       \
              iter,                                                           \
              [](scalar_t a, scalar_t b) -> scalar_t { return func(a, b); }); \
        });                                                                   \
    return out;                                                               \
  }

#define IPEX_BINARY_BASE_OPS(op, func, creator, types)                    \
  Tensor& op(const Tensor& self, const Tensor& other, Tensor& out) {      \
    /* to handle op(tensor, scalar) redispatch path. */                   \
    /* by default, scalar will be wrapped as a CPU tensor in default */   \
    /* catchall implememtation. here convert to XPU lazily. */            \
    Tensor other_maybe_scalar = other;                                    \
    if (other.device().type() == at::kCPU && other.numel() == 1) {        \
      other_maybe_scalar = other.to("xpu");                               \
    }                                                                     \
                                                                          \
    auto iter = TensorIterator::creator(out, self, other_maybe_scalar);   \
    IPEX_DISPATCH_##types(iter.dtype(), #op, [&]() {                      \
      dpcpp_kernel_for_tensor_iter(                                       \
          iter,                                                           \
          [](scalar_t a, scalar_t b) -> scalar_t { return func(a, b); }); \
    });                                                                   \
    return out;                                                           \
  }

#define IPEX_BINARY_LOOPS_FUNC_FLOAT_ALL(op, func, creator) \
  IPEX_BINARY_AND_ALL_OPS(op, func, creator, FLOATING_TYPES)

#define IPEX_BINARY_LOOPS_FUNC_ALL_ALL(op, func, creator) \
  IPEX_BINARY_AND_ALL_OPS(op, func, creator, ALL_TYPES)

#define IPEX_BINARY_LOOPS_FUNC_FLOAT_ALL_COMPLEX(op, func, creator) \
  IPEX_BINARY_AND_ALL_OPS(op, func, creator, FLOATING_AND_COMPLEX_TYPES)

#define IPEX_BINARY_LOOPS_FUNC_ALL_ALL_COMPLEX(op, func, creator) \
  IPEX_BINARY_AND_ALL_OPS(op, func, creator, ALL_TYPES_AND_COMPLEX)

#define IPEX_BINARY_LOOPS_FUNC_FLOAT_BASE(op, func, creator) \
  IPEX_BINARY_BASE_OPS(op, func, creator, FLOATING_TYPES)

#define IPEX_BINARY_LOOPS_FUNC_ALL_BASE(op, func, creator) \
  IPEX_BINARY_BASE_OPS(op, func, creator, ALL_TYPES)

#define IPEX_BINARY_LOOPS_FUNC_FLOAT_BASE_COMPLEX(op, func, creator) \
  IPEX_BINARY_BASE_OPS(op, func, creator, FLOATING_AND_COMPLEX_TYPES)

#define IPEX_BINARY_LOOPS_FUNC_ALL_BASE_COMPLEX(op, func, creator) \
  IPEX_BINARY_BASE_OPS(op, func, creator, ALL_TYPES_AND_COMPLEX)

#define IPEX_BINARY_LOOPS_FUNC_COMPLEX(op, func, creator) \
  IPEX_BINARY_BASE_OPS(op, func, creator, COMPLEX_TYPES)

} // namespace AtenIpexTypeXPU
} // namespace at
