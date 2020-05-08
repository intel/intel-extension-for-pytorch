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

void InitIpexModuleBindings(py::module m) {
  m.def("_initialize_aten_bindings",
        []() { AtenIpexType::InitializeAtenBindings(); });
  m.def("_get_git_revs", []() { return GetRevisions(); });
  m.def("enable_auto_dnnl", []() { setAutoDNNL(true); });
  m.def("disable_auto_dnnl", []() { setAutoDNNL(false); });
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
  m.def("mlp_forward", &AtenIpexTypeMLPExt::forward);
  m.def("mlp_backward", &AtenIpexTypeMLPExt::backward);
  m.def("mlp_create_handle", &AtenIpexTypeMLPExt::create_handle);
  m.def("mlp_set_relu_mask", &AtenIpexTypeMLPExt::set_relu_mask);
  m.def("mlp_release_handle", &AtenIpexTypeMLPExt::release_handle);
}

}  // namespace

void InitIpexBindings(py::module m) { InitIpexModuleBindings(m); }

}  // namespace torch_ipex

PYBIND11_MODULE(_torch_ipex, m) { torch_ipex::InitIpexBindings(m); }
