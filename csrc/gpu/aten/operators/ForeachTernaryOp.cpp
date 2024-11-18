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

} // namespace AtenIpexTypeXPU
} // namespace at
