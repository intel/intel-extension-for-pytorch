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
#include "utils.h"
#include "auto_opt_config.h"

#include "cpu/dil/dil.hpp"
#include "cpu/ShadeDataContext.h"
#include "cpu/ExtendOPs.h"
#include "cpu/MlpOPs.h"
#include "quantization/Observer.h"

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
  m.def("set_execution_mode", [](bool train) { AutoOptConfig::singleton().set_train(train); }, py::arg("train"));
  m.def("get_train", []() { return AutoOptConfig::singleton().get_train(); });
  
  // int8 path

  m.def("enable_mix_int8_fp32", []() { AutoOptConfig::singleton().set_mix_int8_fp32(true); });
  m.def("disable_mix_int8_fp32", []() { AutoOptConfig::singleton().set_mix_int8_fp32(false); });
  m.def("get_mix_int8_fp32", []() { return AutoOptConfig::singleton().get_mix_int8_fp32(); });
  m.def("enable_int8_calibration", []() { AutoOptConfig::singleton().set_int8_calibration(true); });
  m.def("disable_int8_calibration", []() { AutoOptConfig::singleton().set_int8_calibration(false); });
  m.def("get_int8_calibration", []() { return AutoOptConfig::singleton().get_int8_calibration(); });
  m.def("calibration_reset", []() { AutoOptConfig::singleton().calibration_reset(); });
  m.def("add_indicators", []() { AutoOptConfig::singleton().add_indicators(); });
  m.def("print_observer", []() { AutoOptConfig::singleton().print_observer(); });
  m.def("print_indicator", []() { AutoOptConfig::singleton().print_indicator(); });
  m.def("get_int8_configures", []() {
      py::list output_list;
      auto indicators = AutoOptConfig::singleton().get_indicators();
      for (auto indicator: indicators) {
        py::dict d;
        d["id"] = indicator.get_indicator_id();
        d["name"] = indicator.get_indicator_name();
        std::vector<float> scales = indicator.get_indicator_scales();
        d["input_scale"] = scales[0];
        d["output_scale"] = scales[1];
        std::vector<bool> uint8_used= indicator.get_indicator_uint8_status();
        d["input_uint8_used"] = (bool)uint8_used[0];
        d["output_uint8_used"] = (bool)uint8_used[1];
        d["quantized"] = indicator.get_indicator_quantized_status();
        output_list.append(d);
      }
      return output_list; } );
  m.def("load_indicators_file", [](const py::list& l) {
      IPEX_CHECK(py::len(l) > 0, "can't load a empty configures, please first do calibration setp"); 
      std::vector<Indicator> indicators;
      for (py::handle i : l) {
        int64_t id = py::cast<std::int64_t>(i["id"]);
        std::string op_name = py::cast<std::string>(i["name"]);
        float input_scale = py::cast<float>(i["input_scale"]);
        float output_scale = py::cast<float>(i["output_scale"]);
        bool input_uint8_used = py::cast<bool>(i["input_uint8_used"]);
        bool output_uint8_used = py::cast<bool>(i["output_uint8_used"]);
        bool quantized  = py::cast<bool>(i["quantized"]);
        Indicator temp(id, op_name, {input_scale, output_scale},
          {input_uint8_used, output_uint8_used}, quantized);
        indicators.push_back(temp);
      }
      AutoOptConfig::singleton().set_indicators(indicators); } );
}

}  // namespace
using namespace torch::jit;

void InitIpexBindings(py::module m) {
  InitIpexModuleBindings(m);
  // jit fusion pass
  RegisterPreFusionPass pre_pass([](std::shared_ptr<Graph>& g) {
    if (AutoOptConfig::singleton().get_jit_fuse()) {
      torch::jit::FusionPass(g);
    }
  });
}

}  // namespace torch_ipex

PYBIND11_MODULE(_torch_ipex, m) { torch_ipex::InitIpexBindings(m); }
