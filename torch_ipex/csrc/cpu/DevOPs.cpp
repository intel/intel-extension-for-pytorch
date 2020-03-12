#include "torch_ipex/csrc/cpu/DevOPs.h"

#include <ATen/Context.h>
#include <ATen/CPUGenerator.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>

#include <limits>

#include "torch_ipex/csrc/aten_ipex_bridge.h"
#include "torch_ipex/csrc/ipex_tensor_impl.h"
#include "torch_ipex/csrc/utils.h"
#include "dil/dil.hpp"
#include "ShadeDataContext.h"

namespace torch_ipex {
namespace cpu {

//#define DBG
#if defined(DBG)
#define DEBUG(fmt) printf(fmt);
#else
#define DEBUG(fmt)
#endif

std::vector<int64_t> conv_output_size(
    at::IntArrayRef input_size,
    at::IntArrayRef kernel_size,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation) {
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[0];
  output_size[1] = kernel_size[0];
  for (size_t d = 2; d < dim; ++d) {
    auto kernel = dilation[d - 2] * (kernel_size[d] - 1) + 1;
    output_size[d] = (input_size[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}

dil::tensor _dil_conv2d(
    const dil::tensor& x,
    const dil::tensor& w,
    const c10::optional<dil::tensor>& b,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups) {
  std::vector<int64_t> kernel_size(x.ndims());
  // mkldnn conv2d weights could have been re-ordered to 5d by
  // mkldnn_reorder_conv2d_weight
  if (w.ndims() == x.ndims() + 1) {
    AT_ASSERTM(
      groups > 1,
      "Only group _mkldnn_conv2d weights could have been reordered to 5d");
    kernel_size[0] = w.get_dim(0) * w.get_dim(1);
    std::copy_n(w.get_dims().cbegin() + 2, x.ndims() - 1, kernel_size.begin() + 1);
  } else {
    std::copy_n(w.get_dims().cbegin(), x.ndims(), kernel_size.begin());
  }

  const dil::dims x_dims = x.get_dims();
  std::vector<int64_t> input_size{x_dims.cbegin(), x_dims.cend()};
  std::vector<int64_t> output_sizes = conv_output_size(input_size, kernel_size, padding, stride, dilation);

  dil::tensor y;
  if (b.has_value()) {
    dil::convolution_forward::compute(
      x,
      w,
      b.value(),
      {output_sizes.cbegin(), output_sizes.cend()},
      y,
      {stride.begin(), stride.end()},
      {dilation.begin(), dilation.end()},
      {padding.begin(), padding.end()},
      {padding.begin(), padding.end()},
      groups);
  } else {
    dil::convolution_forward::compute(
      x,
      w,
      {output_sizes.cbegin(), output_sizes.cend()},
      y,
      {stride.begin(), stride.end()},
      {dilation.begin(), dilation.end()},
      {padding.begin(), padding.end()},
      {padding.begin(), padding.end()},
      groups);
  }
  return y;
}

dil::tensor dil_tensor_view_from_dense(const at::Tensor& tensor) {
  AT_ASSERTM(
    tensor.device().type() == at::DeviceType::DPCPP,
    "dil_tensor_view_from_dense expects CPU tensor input");
  AT_ASSERTM(
    tensor.layout() == at::Layout::Strided,
    "dil_tensor_view_from_dense expects dense tensor input");
  AT_ASSERTM(
    !tensor.is_variable(),
    "dil_tensor_view_from_dense: should not be a variable");
  at::ScalarType cur_type = tensor.scalar_type();
  return {{{tensor.sizes().cbegin(), tensor.sizes().cend()}, get_dil_data_type(cur_type)},
          tensor.data_ptr()};
}

at::Tensor dil_convolution(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups) {
  DEBUG("AtenIpexCPUOptimized::dil_convolution\n");
  dil::tensor dil_input;
  dil::tensor dil_weight;
  c10::optional<dil::tensor> dil_bias{c10::nullopt};

  if (ShadeDataContext::isDilTensor(input)) {
    TORCH_INTERNAL_ASSERT(ShadeDataContext::isDilTensor(weight));
    dil_input = ShadeDataContext::getDilTensor(input);
    dil_weight = ShadeDataContext::getDilTensor(weight);
    if (bias.defined()) {
      TORCH_INTERNAL_ASSERT(ShadeDataContext::isDilTensor(bias));
      dil_bias = ShadeDataContext::getDilTensor(bias);
    }
  } else {
    TORCH_INTERNAL_ASSERT(input.is_contiguous());
    TORCH_INTERNAL_ASSERT(weight.is_contiguous());
    dil_input = dil_tensor_view_from_dense(input);
    dil_weight = dil_tensor_view_from_dense(weight);
    if (bias.defined()) {
      TORCH_INTERNAL_ASSERT(bias.is_contiguous());
      dil_bias = dil_tensor_view_from_dense(bias);
    }
  }

  dil::tensor dil_output = _dil_conv2d(
    dil_input,
    dil_weight,
    dil_bias,
    padding,
    stride,
    dilation,
    groups);
  // Generate new CPU Tensor and store dil tensor at its storage
  cpu::ShadeDataContext *shade_data_context = cpu::ShadeDataContext::allocShadeDataContext();
  shade_data_context->dil_tensor = dil_output;
  shade_data_context->data_type = cpu::SHADE_DATA_TYPE::DIL;
  c10::DataPtr shade_data_ptr(
    nullptr,
    shade_data_context,
    cpu::ShadeDataContext::freeShadeDataContext,
    at::DeviceType::DPCPP);
  auto dims = dil_output.get_dims();
  auto at_data_type = get_at_data_type(dil_output.get_data_type());
  auto storage_impl = c10::make_intrusive<at::StorageImpl>(
    at::scalarTypeToTypeMeta(at_data_type),
    dil_output.get_nelems(),
    std::move(shade_data_ptr),
    nullptr,
    /*resizeable=*/false);
  return at::detail::make_tensor<IPEXTensorImpl>(storage_impl, at::TensorTypeId::DPCPPTensorId);
}

at::Tensor AtenIpexCPUDev::convolution_overrideable(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups) {
  DEBUG("AtenIpexCPUOptimized::convolution_overrideable\n");
  return mkldnn_convolution(input, weight, bias, padding, stride, dilation, groups);
}

at::Tensor AtenIpexCPUDev::mkldnn_convolution(const at::Tensor & self, const at::Tensor & weight, const at::Tensor & bias, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups) {
  DEBUG("AtenIpexCPUOptimized::mkldnn_convolution\n");
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
  DEBUG("AtenIpexCPUOptimized::convolution_backward_overrideable\n");
  return mkldnn_convolution_backward(input, grad_output, weight, padding, stride, dilation, groups, output_mask);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> AtenIpexCPUDev::mkldnn_convolution_backward(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask) {
  DEBUG("AtenIpexCPUOptimized::mkldnn_convolution_backward\n");
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
