#pragma once

#include <ATen/Tensor.h>

namespace torch_ipex {

class AtenIpexPrepack {
 public:
  static void prepack_conv_weight(at::Tensor &weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups);
};

class AtenIpexJITPrepack {
 public:
  static at::Tensor prepack_conv_weight(const at::Tensor &weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups);
};

}  // namespace torch_ipex

