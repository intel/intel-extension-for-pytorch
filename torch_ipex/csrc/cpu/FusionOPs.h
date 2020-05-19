#pragma once

#include <ATen/Tensor.h>

#include <torch/csrc/jit/runtime/custom_operator.h>

#include "dil/dil.hpp"

namespace torch { namespace jit {

// XXX: PyTorch does not support nesting namespace
// And the alias analysis is not working for namespace other than aten ...
// So we fake some op namespaces to workaround that.
namespace dnnl {
  static auto conv2d_relu = Symbol::fromQualString("dnnl::conv2d_relu");
  static auto conv2d_sum = Symbol::fromQualString("dnnl::conv2d_sum");
  static auto conv2d_relu_sum = Symbol::fromQualString("dnnl::conv2d_relu_sum");
  static auto conv2d_sum_relu = Symbol::fromQualString("dnnl::conv2d_sum_relu");

}

}} // namespace torch::jit

namespace torch_ipex {
namespace cpu {

class AtenIpexJITDev {
 public:
  // for JIT ops
  static at::Tensor dil_convolution_relu(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups);

};

}  // namespace cpu
}  // namespace torch_ipex
