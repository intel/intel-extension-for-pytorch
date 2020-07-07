#include "init_python_bindings.h"
#include "version.h"

#include <c10/core/Device.h>
#include <c10/util/Optional.h>
#include <torch/csrc/utils/pybind.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator_options.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include "jit/fusion_pass.h"

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

bool isBF16DilTensor(const at::Tensor &tensor) {
  if (isDilTensor(tensor)) {
    auto dil_tensor = cpu::ShadeDataContext::getDilStorage(tensor);
    return dil_tensor.get_data_type() == dil::data_type::bf16;
  }

  return false;
}

bool isFP32DilTensor(const at::Tensor &tensor) {
  if (isDilTensor(tensor)) {
    auto dil_tensor = cpu::ShadeDataContext::getDilStorage(tensor);
    return dil_tensor.get_data_type() == dil::data_type::f32;
  }

  return false;
}

dil::dims getDilStorageSizes(const at::Tensor &tensor) {
  if (isDilTensor(tensor)) {
    auto dil_tensor = cpu::ShadeDataContext::getDilStorage(tensor);
    return dil_tensor.get_dims();
  }
  return dil::dims();
}

dil::dims getDilStorageStrides(const at::Tensor &tensor) {
  if (isDilTensor(tensor)) {
    auto dil_tensor = cpu::ShadeDataContext::getDilStorage(tensor);
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
        [](const at::Tensor& weight, const at::Tensor& indices, const at::Tensor& offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor>& per_sample_weights, bool include_last_offset) {
          return AtenIpexTypeExt::embedding_bag_forward(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset);
        });

  m.def("embedding_bag_backward",
        [](const at::Tensor& grad, const at::Tensor& indices, const at::Tensor& offsets, const at::Tensor offset2bag, const at::Tensor& bag_size, const at::Tensor& maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor>& per_sample_weights) {
          return AtenIpexTypeExt::embedding_bag_backward(grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights);
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
  m.def("is_bf16_dil_tensor", &isBF16DilTensor);
  m.def("is_fp32_dil_tensor", &isFP32DilTensor);
  m.def("get_dil_tensor_sizes", &getDilStorageSizes);
  m.def("get_dil_tensor_strides", &getDilStorageStrides);
  m.def("enable_jit_opt", []() { AutoOptConfig::singleton().set_jit_fuse(true); });
  m.def("disable_jit_opt", []() { AutoOptConfig::singleton().set_jit_fuse(false); });
  m.def("get_jit_opt", []() { return AutoOptConfig::singleton().get_jit_fuse(); });
}

}  // namespace
using namespace torch::jit;

void InitIpexBindings(py::module m) {
  InitIpexModuleBindings(m);
  // jit fusion pass
  RegisterPass pass([](std::shared_ptr<Graph>& g) {
    if (AutoOptConfig::singleton().get_jit_fuse()) {
      torch::jit::FusionPass(g);
    }
  });
}

}  // namespace torch_ipex

PYBIND11_MODULE(_torch_ipex, m) { torch_ipex::InitIpexBindings(m); }
