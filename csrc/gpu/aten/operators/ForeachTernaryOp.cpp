/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Fill.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/TensorIterator.h>
#include <aten/core/detail/IndexUtils.h>
#include <aten/operators/comm/Numerics.h>
#include <runtime/Utils.h>
#include "ATen/OpMathType.h"
#include "comm/ATDispatch.h"
#include "comm/ApplyUtils.h"
#include "comm/RegistrationDeclarations.h"
#include "utils/CustomOperatorRegistration.h"

#include <iostream>
#include "ForeachFunctors.h"
#include "Loops.h"
#include "MultiTensorApply.h"
#include "comm/Numerics.h"

namespace at {
namespace AtenIpexTypeXPU {

template <typename scalar_t>
inline bool is_lerp_weight_small(scalar_t weight) {
  return std::abs(weight) < scalar_t(0.5f);
}
template <typename scalar_t>
inline bool is_lerp_weight_small(c10::complex<scalar_t> weight) {
  // Avoid the sqrt in abs(weight)
  return (weight.real() * weight.real() + weight.imag() * weight.imag()) <
      scalar_t(0.25f);
}

template <typename scalar_t, typename weight_t>
inline scalar_t lerp(scalar_t self_, scalar_t end_, weight_t weight_) {
  using opmath_t = at::opmath_type<scalar_t>;
  using opmath_weight_t = at::opmath_type<weight_t>;

  opmath_t self = self_;
  opmath_t end = end_;
  opmath_weight_t weight = weight_;

  // Conditional for better numeric. This has been discussed in
  // https://github.com/pytorch/pytorch/pull/18871
  return is_lerp_weight_small(weight)
      ? self + weight * (end - self)
      : end - (end - self) * (opmath_t(1) - weight);
}

template <typename scalar_t>
struct LerpFunctor {
  inline scalar_t operator()(
      const scalar_t self,
      const scalar_t end,
      const scalar_t weight) {
    return lerp(self, end, weight);
  }
};

std::vector<at::Tensor> _foreach_lerp(
    TensorList tensors1,
    TensorList tensors2,
    TensorList tensors3) {
  at::native::check_foreach_api_restrictions(tensors1, tensors2, tensors3);
  if (!at::native::can_use_fast_route({tensors1, tensors2, tensors3})) {
    return at::native::foreach_tensor_ternary_lerp_slow(
        tensors1, tensors2, tensors3);
  }

  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors1.size());
  for (const auto& t : tensors1) {
    vec_res.emplace_back(at::native::empty_like(t));
  }
  std::vector<std::vector<at::Tensor>> tensor_lists{
      tensors1.vec(), tensors2.vec(), tensors3.vec(), vec_res};

  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "foreach_tensor_lerp_ternary_",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<4>(
            tensor_lists,
            TernaryOpListFunctor<
                scalar_t,
                /* depth */ 4,
                /* r_args_depth */ 3,
                /* res_arg_index */ 3>(),
            LerpFunctor<opmath_t>());
      });

  return tensor_lists[3];
}

void _foreach_lerp_(
    TensorList tensors1,
    TensorList tensors2,
    TensorList tensors3) {
  at::native::check_foreach_api_restrictions(tensors1, tensors2, tensors3);
  if (!at::native::can_use_fast_route({tensors1, tensors2, tensors3})) {
    return at::native::foreach_tensor_ternary_lerp_slow_(
        tensors1, tensors2, tensors3);
  }

  std::vector<std::vector<at::Tensor>> tensor_lists{
      tensors1.vec(), tensors2.vec(), tensors3.vec()};
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "foreach_tensor_lerp_ternary_",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<3>(
            tensor_lists,
            TernaryOpListFunctor<
                scalar_t,
                /* depth */ 3,
                /* r_args_depth */ 3,
                /* res_arg_index */ 0>(),
            LerpFunctor<opmath_t>());
      });
  increment_version(tensors1);
}

std::vector<at::Tensor> _foreach_lerp(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& weight) {
  at::native::check_foreach_api_restrictions(tensors1, tensors2);
  if (!at::native::can_use_fast_route({tensors1, tensors2})) {
    return at::native::foreach_tensor_lerp_list_kernel_slow(
        tensors1, tensors2, weight);
  }

  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors1.size());
  for (const auto& t : tensors1) {
    vec_res.emplace_back(at::native::empty_like(t));
  }
  std::vector<std::vector<at::Tensor>> tensor_lists{
      tensors1.vec(), tensors2.vec(), vec_res};

  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "foreach_tensor_lerp_scalar_",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<3>(
            tensor_lists,
            TernaryOpScalarFunctor<
                scalar_t,
                /* depth */ 3,
                /* r_args_depth */ 2,
                /* res_arg_index */ 2>(),
            LerpFunctor<opmath_t>(),
            weight.to<opmath_t>());
      });

  return tensor_lists[2];
}

void _foreach_lerp_(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& weight) {
  at::native::check_foreach_api_restrictions(tensors1, tensors2);
  if (!at::native::can_use_fast_route({tensors1, tensors2})) {
    return at::native::foreach_tensor_lerp_list_kernel_slow_(
        tensors1, tensors2, weight);
  }

  std::vector<std::vector<at::Tensor>> tensor_lists{
      tensors1.vec(), tensors2.vec()};
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "foreach_tensor_lerp_scalar__",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<2>(
            tensor_lists,
            TernaryOpScalarFunctor<
                scalar_t,
                /* depth */ 2,
                /* r_args_depth */ 2,
                /* res_arg_index */ 0>(),
            LerpFunctor<opmath_t>(),
            weight.to<opmath_t>());
      });
}

} // namespace AtenIpexTypeXPU
} // namespace at
