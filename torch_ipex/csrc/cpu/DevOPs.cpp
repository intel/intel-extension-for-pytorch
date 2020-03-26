#include "torch_ipex/csrc/cpu/DevOPs.h"

#include <ATen/Context.h>
#include <ATen/CPUGenerator.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>

#include <limits>

#include "torch_ipex/csrc/aten_ipex_bridge.h"
#include "torch_ipex/csrc/ipex_tensor_impl.h"
#include "torch_ipex/csrc/utils.h"
#include "dbl/Common.h"
#include "dbl/Conv.h"
#include "ShadeDataContext.h"

#include "dil/dil.hpp"

namespace torch_ipex {
namespace cpu {

//#define DBG
#if defined(DBG)
#define DEBUG(fmt) printf(fmt);
#else
#define DEBUG(fmt)
#endif

at::Tensor dil_convolution(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
  DEBUG("AtenIpexCPUDev::dil_convolution\n");
  dil::tensor dil_input;
  dil::tensor dil_weight;
  c10::optional<dil::tensor> dil_bias{c10::nullopt};

  TORCH_INTERNAL_ASSERT(input.defined());
  TORCH_INTERNAL_ASSERT(input.device().type() == at::DeviceType::DPCPP);
  TORCH_INTERNAL_ASSERT(input.is_contiguous());

  TORCH_INTERNAL_ASSERT(weight.defined());
  TORCH_INTERNAL_ASSERT(weight.device().type() == at::DeviceType::DPCPP);
  TORCH_INTERNAL_ASSERT(weight.is_contiguous());

  dil_input = dbl::comm::try_gen_dil_tensor(input);
  dil_weight = dbl::comm::try_gen_dil_tensor(weight);
  if (bias.defined()) {
    TORCH_INTERNAL_ASSERT(bias.is_contiguous());
    TORCH_INTERNAL_ASSERT(bias.device().type() == at::DeviceType::DPCPP);
    dil_bias = dbl::comm::try_gen_dil_tensor(bias);
  }

  dil::tensor dil_output = dbl::conv::conv2d_impl(
    dil_input,
    dil_weight,
    dil_bias,
    padding,
    stride,
    dilation,
    groups);

  return dbl::comm::gen_aten_tensor_by(dil_output);
}

at::Tensor AtenIpexCPUDev::convolution_overrideable(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups) {
  DEBUG("AtenIpexCPUDev::convolution_overrideable\n");
  // NOTE: DO NOT always call contiguous. It may break lazy-reorder. Because contiguous will call reorder instantly.
  if (check_force_dnnl_env()) {
    return dil_convolution(
      input.is_contiguous() ? input : input.contiguous(),
      weight.is_contiguous() ? weight : weight.contiguous(),
      bias.defined() ? (bias.is_contiguous() ? bias :bias.contiguous()) : bias,
      stride,
      padding,
      dilation,
      groups);
  } else {
    return mkldnn_convolution(input, weight, bias, padding, stride, dilation, groups);
  }
}

at::Tensor AtenIpexCPUDev::mkldnn_convolution(const at::Tensor & self, const at::Tensor & weight, const at::Tensor & bias, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups) {
  DEBUG("AtenIpexCPUDev::mkldnn_convolution\n");
  TORCH_INTERNAL_ASSERT(self.defined());
  TORCH_INTERNAL_ASSERT(weight.defined());
  TORCH_INTERNAL_ASSERT(self.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT(weight.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT(!(bias.defined()) || (bias.defined() && bias.layout() == c10::kStrided));
  auto&& _ipex_self = bridge::shallowFallbackToCPUTensor(self);
  auto&& _ipex_weight = bridge::shallowFallbackToCPUTensor(weight);
  auto&& _ipex_bias = bridge::shallowFallbackToCPUTensor(bias);
  auto&& _ipex_result = at::mkldnn_convolution(_ipex_self.contiguous(), _ipex_weight.contiguous(), _ipex_bias.contiguous(), padding, stride, dilation, groups);
  static_cast<void>(_ipex_result); // Avoid warnings in case not used
  TORCH_INTERNAL_ASSERT(_ipex_result.is_contiguous());
  TORCH_INTERNAL_ASSERT(_ipex_result.layout() == c10::kStrided);
  return bridge::shallowUpgradeToDPCPPTensor(_ipex_result);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> AtenIpexCPUDev::convolution_backward_overrideable(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, std::array<bool,3> output_mask) {
  DEBUG("AtenIpexCPUDev::convolution_backward_overrideable\n");
  return mkldnn_convolution_backward(input, grad_output, weight, padding, stride, dilation, groups, output_mask);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> AtenIpexCPUDev::mkldnn_convolution_backward(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask) {
  DEBUG("AtenIpexCPUDev::mkldnn_convolution_backward\n");
  TORCH_INTERNAL_ASSERT(self.defined());
  TORCH_INTERNAL_ASSERT(grad_output.defined());
  TORCH_INTERNAL_ASSERT(weight.defined());
  TORCH_INTERNAL_ASSERT(self.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT(grad_output.defined());
  TORCH_INTERNAL_ASSERT(weight.layout() == c10::kStrided);
  auto&& _ipex_self = bridge::shallowFallbackToCPUTensor(self);
  auto&& _ipex_grad_output = bridge::shallowFallbackToCPUTensor(grad_output);
  auto&& _ipex_weight = bridge::shallowFallbackToCPUTensor(weight);
  auto&& _ipex_result = at::mkldnn_convolution_backward(_ipex_self.contiguous(), _ipex_grad_output.contiguous(), _ipex_weight.contiguous(), padding, stride, dilation, groups, output_mask);
  static_cast<void>(_ipex_result); // Avoid warnings in case not used
  return std::tuple<at::Tensor,at::Tensor,at::Tensor>(bridge::shallowUpgradeToDPCPPTensor(std::get<0>(_ipex_result)), bridge::shallowUpgradeToDPCPPTensor(std::get<1>(_ipex_result)), bridge::shallowUpgradeToDPCPPTensor(std::get<2>(_ipex_result)));
}

}  // namespace cpu
}  // namespace torch_ipex
