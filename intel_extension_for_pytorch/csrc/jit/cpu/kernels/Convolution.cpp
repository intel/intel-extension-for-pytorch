#include "Convolution.h"
#include "csrc/aten/cpu/Conv.h"
#include "csrc/aten/cpu/ConvTranspose.h"
#include "csrc/utils/utils.h"

#include <ATen/Context.h>
#include <ATen/InferSize.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/csrc/autograd/function.h>

#include <limits>

#include "csrc/cpu/ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {

at::Tensor dil_convolution_base(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("dil_convolution_base", std::vector<c10::IValue>({}));
#endif
  return convolution_impl(
      input, weight, bias, stride, padding, dilation, groups, ideep::attr_t());
}

at::Tensor dil_convolution_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("dil_convolution_relu", std::vector<c10::IValue>({}));
#endif
  return convolution_impl(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      groups,
      ideep::attr_t::fuse_relu());
}

at::Tensor& dil_convolution_sum(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    at::Tensor& accumu,
    at::Scalar alpha) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("dil_convolution_sum", std::vector<c10::IValue>({}));
#endif
  auto scale = alpha.to<float>();
  convolution_inplace_impl(
      input,
      weight,
      bias,
      accumu,
      stride,
      padding,
      dilation,
      groups,
      ideep::attr_t::fuse_sum(scale));
  return accumu;
}

at::Tensor& dil_convolution_sum_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    at::Tensor& accumu,
    at::Scalar alpha) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("dil_convolution_sum_relu", std::vector<c10::IValue>({}));
#endif
  auto scale = alpha.to<float>();
  convolution_inplace_impl(
      input,
      weight,
      bias,
      accumu,
      stride,
      padding,
      dilation,
      groups,
      ideep::attr_t::residual(scale));
  return accumu;
}

} // namespace cpu
} // namespace torch_ipex
