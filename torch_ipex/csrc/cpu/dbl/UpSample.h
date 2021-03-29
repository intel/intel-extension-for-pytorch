#pragma once

#include <ATen/ATen.h>
#include "Common.h"
#include "cpu/dil/dil.hpp"

namespace torch_ipex {
namespace cpu {
namespace dbl {
namespace upsample {

template <typename T>
bool hasValue(T first_scale) {
  return first_scale.has_value();
}

template <typename T, typename... Args>
bool hasValue(T first_scale, Args&&... other_scales) {
  return hasValue(first_scale) && hasValue(other_scales...);
}

template <typename T>
T unpackOptional(c10::optional<T> v) {
  return v.value();
}

template <typename... Args>
std::vector<float> getScales(Args&&... args) {
  if (hasValue(std::forward<Args>(args)...)) {
    return {unpackOptional(args)...};
  } else {
    return {};
  }
}

auto calculate_resample_size(
    const at::Tensor& self,
    at::IntArrayRef output_size) {
  const auto nd = self.dim();
  const auto spatial_nd = output_size.size();
  auto out_size = self.sizes().vec();
  for (size_t i = 0; i < spatial_nd; i++)
    out_size[nd - spatial_nd + i] = output_size[i];
  return out_size;
}

template <typename... Args>
at::Tensor dil_upsample(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    dil::algorithm algorithm,
    Args&&... args) {
  dbl::comm::reorder_to_bf16_for_mix_prec(self);
  dil::tensor x = dbl::comm::try_gen_dil_tensor(self);
  dil::tensor y;

  auto out_size = calculate_resample_size(self, output_size);
  auto scales = getScales(args...);
  dil::resampling_forward::compute(x, y, out_size, scales, algorithm);

  return dbl::comm::gen_aten_tensor_by(std::move(y));
}

template <typename... Args>
at::Tensor dil_upsample_backward(
    const at::Tensor & grad_output,
    at::IntArrayRef input_size,
    dil::algorithm algorithm,
    Args&&... args) {
  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output);
  dil::tensor dy = dbl::comm::try_gen_dil_tensor(grad_output);
  dil::tensor dx;
  auto scales = getScales(args...);
  dil::resampling_backward::compute(dy, dx, input_size.vec(), scales, algorithm);

  return dbl::comm::gen_aten_tensor_by(std::move(dx));
}

} // namespace upsample
} // namespace dbl
} // namespace cpu
} // namespace torch_ipex
