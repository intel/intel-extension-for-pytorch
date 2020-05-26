#include "init_python_bindings.h"
#include "version.h"

#include <c10/core/Device.h>
#include <c10/util/Optional.h>
#include <torch/csrc/utils/pybind.h>

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "aten_ipex_type.h"
#include "auto_opt_config.h"
#include "cpu/dil/dil.hpp"
#include "cpu/ShadeDataContext.h"
#include "cpu/ExtendOPs.h"
#include "cpu/MlpOPs.h"

namespace torch_ipex {
namespace {

py::object GetRevisions() {
  auto py_dict = py::dict();
  py_dict["ipex"] = std::string(IPEX_GITREV);
  py_dict["torch"] = std::string(TORCH_GITREV);
  return py_dict;
}

void setAutoDNNL(bool val) {
  AutoOptConfig::singleton().set_auto_dnnl(val);
}

/// **** Only for unit test ****
bool isDilTensor(const at::Tensor &tensor) {
  return cpu::ShadeDataContext::isDilTensor(tensor);
}

dil::dims getDilTensorSizes(const at::Tensor &tensor) {
  if (isDilTensor(tensor)) {
    auto dil_tensor = cpu::ShadeDataContext::getDilTensor(tensor);
    return dil_tensor.get_dims();
  }
  return dil::dims();
}

dil::dims getDilTensorStrides(const at::Tensor &tensor) {
  if (isDilTensor(tensor)) {
    auto dil_tensor = cpu::ShadeDataContext::getDilTensor(tensor);
    return dil_tensor.get_strides();
  }
  return dil::dims();
}
/// ****************************

void InitIpexModuleBindings(py::module m) {
  m.def("_get_git_revs", []() { return GetRevisions(); });
  m.def("enable_auto_dnnl", []() { AutoOptConfig::singleton().set_auto_dnnl(true); });
  m.def("disable_auto_dnnl", []() { AutoOptConfig::singleton().set_auto_dnnl(false); });
  m.def("get_auto_dnnl", []() { return AutoOptConfig::singleton().get_auto_dnnl(); });
  m.def("enable_mix_bf16_fp32", []() { AutoOptConfig::singleton().set_mix_bf16_fp32(true); });
  m.def("disable_mix_bf16_fp32", []() { AutoOptConfig::singleton().set_mix_bf16_fp32(false); });
  m.def("get_mix_bf16_fp32", []() { return AutoOptConfig::singleton().get_mix_bf16_fp32(); });
  m.def("enable_pure_bf16", []() { AutoOptConfig::singleton().set_pure_bf16(true); });
  m.def("disable_pure_bf16", []() { AutoOptConfig::singleton().set_pure_bf16(false); });
  m.def("get_pure_bf16", []() { return AutoOptConfig::singleton().get_pure_bf16(); });
  m.def("packed_add_",
        [](at::Tensor &top_half, at::Tensor &bot_half,
           const at::Tensor &grad, float alpha) {
          AtenIpexTypeExt::packed_add_(top_half, bot_half, grad, alpha);
        });
  m.def("interaction_forward",
        [](const std::vector<at::Tensor> &input) {
          return AtenIpexTypeExt::interaction_forward(input);
        });
  m.def("interaction_backward",
        [](const at::Tensor &grad_out, const std::vector<at::Tensor> &input) {
          return AtenIpexTypeExt::interaction_backward(grad_out, input);
        });
  m.def("embedding_bag_forward",
        [](const at::Tensor &weights, const at::Tensor &inputs, const at::Tensor &offsets) {
          return AtenIpexTypeExt::embedding_bag_forward(weights, inputs, offsets);
        });
  m.def("embedding_bag_backward",
        [](const at::Tensor &grad_out, const at::Tensor &weights,
           const at::Tensor &inputs, const at::Tensor &offsets) {
          return AtenIpexTypeExt::embedding_bag_backward(grad_out, weights, inputs, offsets);
        });
  m.def("linear",
        [](const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias) {
          return AtenIpexTypeExt::linear(input, weight, bias);
        });
  m.def("linear_backward",
        [](const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight, std::array<bool,3> output_mask) {
          return AtenIpexTypeExt::linear_backward(input, grad_output, weight, output_mask);
        });
  m.def("adaptive_avg_pool2d",
        [](at::Tensor const& input, at::IntArrayRef output_size) {
          return AtenIpexTypeExt::adaptive_avg_pool2d(input, output_size);
        });
  m.def("adaptive_avg_pool2d_backward",
        [](const at::Tensor& grad_output, const at::Tensor& input) {
          return AtenIpexTypeExt::adaptive_avg_pool2d_backward(grad_output, input);
        });
  m.def("max_pooling",
        [](const at::Tensor& input, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
          return AtenIpexTypeExt::max_pooling(input, kernel_size, stride, padding, dilation, ceil_mode);
        });
  m.def("max_pooling_backward",
        [](const at::Tensor& grad_output, const at::Tensor& output, const at::Tensor& input, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
          return AtenIpexTypeExt::max_pooling_backward(grad_output, output, input, kernel_size, stride, padding, dilation, ceil_mode);
        });
  m.def("reshape",
        [](const at::Tensor& input, at::IntArrayRef size) {
          return AtenIpexTypeExt::reshape(input, size);
        });
  m.def("mlp_forward", &AtenIpexTypeMLPExt::forward);
  m.def("mlp_backward", &AtenIpexTypeMLPExt::backward);
  m.def("mlp_create_handle", &AtenIpexTypeMLPExt::create_handle);
  m.def("mlp_set_relu_mask", &AtenIpexTypeMLPExt::set_relu_mask);
  m.def("mlp_release_handle", &AtenIpexTypeMLPExt::release_handle);

  m.def("is_dil_tensor", &isDilTensor);
  m.def("get_dil_tensor_sizes", &getDilTensorSizes);
  m.def("get_dil_tensor_strides", &getDilTensorStrides);
}

}  // namespace

void InitIpexBindings(py::module m) { InitIpexModuleBindings(m); }

}  // namespace torch_ipex

PYBIND11_MODULE(_torch_ipex, m) { torch_ipex::InitIpexBindings(m); }
