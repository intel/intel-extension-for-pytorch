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
#include "cpu/dbl/Common.h"
#include "cpu/ShadeDataContext.h"
#include "cpu/ExtendOPs.h"
#include "cpu/MlpOPs.h"
#include "cpu/ExternalOPs.h"
#include "cpu/FusionOPs.h"
#include "cpu/int8/Config.h"
#include "cpu/int8/quantization/Observer.h"
#include "ProcessGroupCCL.hpp"
#include <pybind11/chrono.h>

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

void setParameterTensor(const at::Tensor &tensor) {
  cpu::ShadeDataContext::setParameterTensor(tensor);
}

bool isParameterTensor(const at::Tensor &tensor) {
  return cpu::ShadeDataContext::isParameterTensor(tensor);
}

/// **** Only for unit test ****
bool isDilTensor(const at::Tensor &tensor) {
  return cpu::ShadeDataContext::isDilTensor(tensor);
}

bool isINT8DilTensor(const at::Tensor &tensor) {
  if (isDilTensor(tensor)) {
    auto dil_tensor = cpu::ShadeDataContext::getDilStorage(tensor);
    return dil_tensor.get_data_type() == dil::data_type::s8
      || dil_tensor.get_data_type() == dil::data_type::u8;
  }

  return false;
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

void reorder_to_float32(at::Tensor &tensor){
  cpu::dbl::comm::reorder_to_dtype(tensor, at::kFloat);
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
  m.def("mlp_forward", &AtenIpexTypeMLPExt::forward);
  m.def("mlp_backward", &AtenIpexTypeMLPExt::backward);
  m.def("mlp_create_handle", &AtenIpexTypeMLPExt::create_handle);
  m.def("mlp_set_relu_mask", &AtenIpexTypeMLPExt::set_relu_mask);
  m.def("mlp_release_handle", &AtenIpexTypeMLPExt::release_handle);
  m.def("is_dil_tensor", &isDilTensor);
  m.def("is_int8_dil_tensor", &isINT8DilTensor);
  m.def("is_bf16_dil_tensor", &isBF16DilTensor);
  m.def("is_fp32_dil_tensor", &isFP32DilTensor);
  m.def("get_dil_tensor_sizes", &getDilStorageSizes);
  m.def("get_dil_tensor_strides", &getDilStorageStrides);
  m.def("set_parameter_tensor", &setParameterTensor);
  m.def("is_parameter_tensor", &isParameterTensor);
  m.def("reorder_to_float32", &reorder_to_float32);
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
  m.def("get_int8_calibration",
        []() { AutoOptConfig::singleton().get_int8_calibration(); });
  m.def("calibration_reset", []() { Int8OptConfig::calibration_reset(); });
  m.def("add_indicators",
        []() { Int8OptConfig::get_config().add_indicators(); });
  m.def("clear_indicators",
        []() { Int8OptConfig::get_config().clear_indicators(); });
  // clear indicators for case having many scopes which have different structure
  m.def("get_int8_configures", []() {
      py::list output_list;
      auto indicators = Int8OptConfig::get_config().get_indicators();
      IPEX_CHECK(indicators.size() > 0, "can't load a empty indicators, please first do calibration step");
      for (auto indicator: indicators) {
        py::dict d;
        d["id"] = indicator.get_indicator_id();
        d["name"] = indicator.get_indicator_name();
        d["algorithm"] = indicator.get_indicator_algorithm();
        d["weight_granularity"] = indicator.get_indicator_weight_granularity();
        std::vector<float> i_scale, o_scale;
        std::tie(i_scale, o_scale) = indicator.get_indicator_scales();
        d["inputs_scale"] = i_scale;
        d["outputs_scale"] = o_scale;
        std::vector<bool> i_uint8_used, o_uint8_used;
        std::tie(i_uint8_used, o_uint8_used)= indicator.get_indicator_uint8_status();
        d["inputs_uint8_used"] = i_uint8_used;
        d["outputs_uint8_used"] = o_uint8_used;
        d["quantized"] = indicator.get_indicator_quantized_status();
        output_list.append(d);
      }
      return output_list; } );
  m.def("load_indicators_file", [](const py::list &l) {
    IPEX_CHECK(
        py::len(l) > 0,
        "can't load a empty configures, please first do calibration step");
    std::vector<Indicator> indicators;
    for (py::handle i : l) {
      int64_t id = py::cast<std::int64_t>(i["id"]);
      std::string op_name = py::cast<std::string>(i["name"]);
      std::string algorithm = py::cast<std::string>(i["algorithm"]);
      std::string weight_granularity =
          py::cast<std::string>(i["weight_granularity"]);
      std::vector<float> i_scale =
          py::cast<std::vector<float>>(i["inputs_scale"]);
      std::vector<float> o_scale =
          py::cast<std::vector<float>>(i["outputs_scale"]);
      std::vector<bool> i_uint8_used =
          py::cast<std::vector<bool>>(i["inputs_uint8_used"]);
      std::vector<bool> o_uint8_used =
          py::cast<std::vector<bool>>(i["outputs_uint8_used"]);
      bool quantized = py::cast<bool>(i["quantized"]);
      Indicator temp(id, op_name, algorithm, weight_granularity, i_scale,
                     o_scale, i_uint8_used, o_uint8_used, quantized);
      indicators.push_back(temp);
    }
    Int8OptConfig::get_config().set_indicators(indicators);
  });
  
  m.def("enable_torch_ccl", [=]() {
       py::object module = py::module::import("torch.distributed");
       py::object register_backend = module.attr("Backend").attr("register_backend"); 
       register_backend("ccl", py::cpp_function(&c10d::ProcessGroupCCL::createProcessGroupCCL,
                                            py::arg("store"),
                                            py::arg("rank"),
                                            py::arg("size"),
                                            py::arg("timeout") = std::chrono::milliseconds(
                                              ::c10d::ProcessGroupCCL::OP_TIMEOUT_MILLIS)));
       
  });
  m.def("set_xpu_mode", [=](std::string mode){
       AutoOptConfig::singleton().set_xpu_mode(torch_ipex::stringToXPUMode(mode));});

  // external OPs
  m.def("roi_align_forward", &IpexExternal::ROIAlign_forward);
  m.def("roi_align_backward", &IpexExternal::ROIAlign_backward);
  m.def("nms", &IpexExternal::nms);
  m.def("batch_score_nms", &IpexExternal::batch_score_nms);
  m.def("linear_relu", &AtenIpexTypeExt::linear_relu);
}

}  // namespace
using namespace torch::jit;

void InitIpexBindings(py::module m) {
  InitIpexModuleBindings(m);
  // jit fusion pass
  torch::jit::registerPrePass([](std::shared_ptr<Graph>& g) {
    if (AutoOptConfig::singleton().get_jit_fuse()) {
      torch::jit::FusionPass(g);
    }
  });
}

}  // namespace torch_ipex

PYBIND11_MODULE(_torch_ipex, m) { torch_ipex::InitIpexBindings(m); }
