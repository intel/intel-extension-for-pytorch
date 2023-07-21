#pragma once
#include <ATen/ATen.h>
#include "CustomOperatorRegistration.h"
#include "Settings.h"

namespace at {
namespace AtenIpexTypeXPU {

using ComputeEngine = xpu::COMPUTE_ENG;

// T would be Tensor, TensorList, ITensorListRef
template <typename T, typename... Args>
ComputeEngine choose_compute_eng(
    ComputeEngine recommend,
    const T& first_input,
    const Args&... inputs) {
  bool has_onednn_layout_tensor =
      xpu::oneDNN::has_onednn_layout(first_input, inputs...);
  if (has_onednn_layout_tensor) {
    return xpu::COMPUTE_ENG::ONEDNN;
  }
  ComputeEngine user_eng = Settings::I().get_compute_eng();
  if (user_eng != xpu::COMPUTE_ENG::RECOMMEND)
    return user_eng;
  else
    return recommend;
}
} // namespace AtenIpexTypeXPU
} // namespace at
