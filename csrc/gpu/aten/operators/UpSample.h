#pragma once

#include <math.h>

#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>

#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include "comm/Atomics.h"
#include "comm/ParamUtils.h"
namespace at {
namespace AtenIpexTypeXPU {
template <typename scalar_t>
inline scalar_t min(scalar_t a, scalar_t b) {
  return a < b ? a : b;
}

template <typename scalar_t>
inline scalar_t max(scalar_t a, scalar_t b) {
  return a > b ? a : b;
}
template <typename scalar_t>
static inline scalar_t area_pixel_compute_scale(
    int input_size,
    int output_size,
    bool align_corners) {
  /* We view each pixel as an area, idx + 0.5 as its center index.
   * Here is an example formula in 1D case.
   * if align_corners: center of two corner pixel areas are preserved,
   *     (0.5, 0.5) -> (0.5, 0.5),
   *     (input_size - 0.5, 0.5) -> (output_size - 0.5)
   *     scale = (input_size - 0.5 - 0.5) / (output_size - 0.5 - 0.5)
   *     src_index + 0.5 - 0.5 = scale * (dst_index + 0.5 - 0.5)
   * if not align_corners: the whole range is scaled accordingly
   *     scale = input_size / output_size
   *     src_idx + 0.5 = scale * (dst_index + 0.5)
   */
  if (output_size > 1) {
    return align_corners
        ? static_cast<scalar_t>(input_size - 1) / (output_size - 1)
        : static_cast<scalar_t>(input_size) / output_size;
  } else {
    return scalar_t(0);
  }
}
template <typename accscalar_t>
static inline accscalar_t nearest_compute_scales_value(
    const c10::optional<double> scale,
    int64_t src_size,
    int64_t dst_size) {
  return (scale.has_value() && scale.value() > 0.)
      ? (accscalar_t)(1.0 / scale.value())
      : (accscalar_t)src_size / dst_size;
}

template <typename accscalar_t>
static inline accscalar_t area_pixel_compute_scale(
    int input_size,
    int output_size,
    bool align_corners,
    const c10::optional<double> scale) {
  if (align_corners) {
    if (output_size > 1) {
      return (accscalar_t)(input_size - 1) / (output_size - 1);
    } else {
      return static_cast<accscalar_t>(0);
    }
  } else {
    return nearest_compute_scales_value<accscalar_t>(
        scale, input_size, output_size);
  }
}

template <typename accscalar_t>
static inline accscalar_t area_pixel_compute_source_index(
    accscalar_t scale,
    int dst_index,
    bool align_corners,
    bool cubic) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    accscalar_t src_idx = scale * (dst_index + static_cast<accscalar_t>(0.5)) -
        static_cast<accscalar_t>(0.5);
    return (!cubic && src_idx < static_cast<accscalar_t>(0))
        ? static_cast<accscalar_t>(0)
        : src_idx;
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at
