/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Complex number math operations that act as no-ops for other dtypes.
#include <ATen/NumericUtils.h>
#include <c10/util/MathConstants.h>
#include <c10/util/complex.h>

namespace at {
namespace AtenIpexTypeXPU {
namespace {

template <typename TYPE>
inline TYPE round_impl(TYPE z) {
  return std::nearbyint(z);
}

template <>
inline c10::complex<float> round_impl(c10::complex<float> z) {
  return c10::complex<float>(
      std::nearbyint(z.real()), std::nearbyint(z.imag()));
}

template <>
inline c10::complex<double> round_impl(c10::complex<double> z) {
  return c10::complex<double>(
      std::nearbyint(z.real()), std::nearbyint(z.imag()));
}

} // end namespace
} // namespace AtenIpexTypeXPU
} // namespace at
