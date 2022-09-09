#pragma once

#include <ATen/ATen.h>

namespace torch_ipex {
namespace int8 {
namespace utils {

inline std::tuple<double, int64_t> get_mkldnn_input_scale_zp(
    const at::Tensor& input) {
  TORCH_CHECK(
      input.qscheme() == c10::QScheme::PER_TENSOR_AFFINE,
      "should use per_tensor_affine quantization for input of LSTM");

  double scale = input.q_scale();

  // PyTorch scale: (max - min) / (qmax - qmin)
  // oneDNN scale: (qmax - qmin) / (max - min)
  double mkldnn_scale = 1. / scale;

  int64_t zp = input.q_zero_point();
  return std::make_tuple(mkldnn_scale, zp);
}

inline at::Tensor get_weight_scale_tensor(const at::Tensor& weight) {
  TORCH_CHECK(
      weight.qscheme() == c10::QScheme::PER_CHANNEL_AFFINE,
      "should use per_channel_affine quantization for weight of LSTM");
  at::Tensor weight_scales_tensor = weight.q_per_channel_scales();
  TORCH_CHECK(
      weight_scales_tensor.dim() == 1,
      "expect weight_scales tensor to be 1d, got dim = ",
      weight_scales_tensor.dim());
  return weight_scales_tensor;
}

} // namespace utils
} // namespace int8
} // namespace torch_ipex