#pragma once

#include <sycl/sycl.hpp>

#include <c10/util/Float8_e5m2.h>

#include "utils.h"

using namespace at;

namespace at {
namespace AtenIpexTypeXPU {

template <bool is_scale_inverted, typename fp8_type>
inline fp8_type scaled_fp8_conversion(float const val,
                                      float const scale) {
  float x = 0.0f;
  if constexpr (is_scale_inverted) {
    x = val * scale;
  } else {
    x = val / scale;
  }

  float r =
      fmax(-quant_type_max_v<fp8_type>, fmin(x, quant_type_max_v<fp8_type>));
  return static_cast<fp8_type>(r);
}

template <typename scalar_t, bool is_scale_inverted, typename fp8_type>
inline void scaled_fp8_conversion_vec(fp8_type* __restrict__ out,
                                      scalar_t* const* __restrict__ input,
                                      float const scale,
                                      int64_t const num_elems,
                                      int const tid, int const step) {
  using float8x4_t = q8x4_t<fp8_type>;
  // Vectorized input/output to better utilize memory bandwidth.
  auto const* vectorized_in = reinterpret_cast<vec4_t<scalar_t> const*>(input);
  auto* vectorized_out = reinterpret_cast<float8x4_t*>(out);

  int64_t const num_vec_elems = num_elems >> 2;

#pragma unroll 4
  for (int64_t i = tid; i < num_vec_elems; i += step) {
    vec4_t<scalar_t> in_vec = vectorized_in[i];
    float8x4_t out_vec;

    out_vec.x = scaled_fp8_conversion<is_scale_inverted, fp8_type>(
        static_cast<float>(in_vec.x), scale);
    out_vec.y = scaled_fp8_conversion<is_scale_inverted, fp8_type>(
        static_cast<float>(in_vec.y), scale);
    out_vec.z = scaled_fp8_conversion<is_scale_inverted, fp8_type>(
        static_cast<float>(in_vec.z), scale);
    out_vec.w = scaled_fp8_conversion<is_scale_inverted, fp8_type>(
        static_cast<float>(in_vec.w), scale);
    vectorized_out[i] = out_vec;
  }

  // Handle the remaining elements if num_elems is not divisible by 4
  for (int64_t i = num_vec_elems * 4 + tid; i < num_elems; i += step) {
    out[i] = scaled_fp8_conversion<is_scale_inverted, fp8_type>(
        static_cast<float>(input[i]), scale);
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at
