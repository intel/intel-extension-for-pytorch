#include "init_python_bindings.h"
#include "version.h"
#include "jit/codegen/onednn/interface.h"

#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/util/Optional.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/native/quantized/cpu/quant_utils.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator_options.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include "jit/fusion_pass.h"

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "auto_opt_config.hpp"
#include "quantization/AutoCast.hpp"
#include "quantization/Config.hpp"
#include "quantization/Observer.hpp"
#include "utils.h"
#include "verbose.hpp"

//#include "ProcessGroupCCL.hpp"
#include <pybind11/chrono.h>
#include "autocast_mode.h"
#include <torch/csrc/api/include/torch/python.h>
#include <c10/core/DeviceType.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/jit/passes/pass_manager.h>

#include "cpu/ExtendOPs.h"

namespace torch_ipex {
namespace {

py::object GetRevisions() {
  auto py_dict = py::dict();
  py_dict["ipex"] = std::string(IPEX_GITREV);
  py_dict["torch"] = std::string(TORCH_GITREV);
  return py_dict;
}

void InitIpexModuleBindings(py::module m) {
  m.def("_get_git_revs", []() { return GetRevisions(); });
  m.def("mkldnn_set_verbose", &torch_ipex::verbose::_mkldnn_set_verbose);
  // ipex amp autocast
  m.def("get_autocast_dtype", []() {
    at::ScalarType current_dtype = torch_ipex::autocast::get_autocast_dtype();
    return py::reinterpret_steal<py::object>(
        THPDtype_New(current_dtype, scalarTypeName(current_dtype)));
  });
  m.def("set_autocast_dtype", [](py::object dtype) {
    at::ScalarType target_dtype =
        torch::python::detail::py_object_to_dtype(dtype);
    torch_ipex::autocast::set_autocast_dtype(target_dtype);
  });
  m.def("is_quantization_enabled",
        &torch_ipex::autocast::is_quantization_enabled);
  m.def("set_quantization_enabled",
        &torch_ipex::autocast::set_quantization_enabled);

  m.def("autocast_increment_nesting",
        &torch_ipex::autocast::autocast_increment_nesting);
  m.def("autocast_decrement_nesting",
        &torch_ipex::autocast::autocast_decrement_nesting);
  m.def("clear_autocast_cache", &torch_ipex::autocast::clear_autocast_cache);

  // llga path
  m.def("_jit_set_llga_enabled", &torch::jit::RegisterLlgaFuseGraph::setEnabled);
  m.def("_jit_llga_enabled", &torch::jit::RegisterLlgaFuseGraph::isEnabled);
  m.def("_jit_llga_fuser", [](std::shared_ptr<torch::jit::Graph> g) {
        return torch::jit::fuser::onednn::fuseGraph(g);
  });

  m.def("enable_jit_opt", []() { AutoOptConfig::singleton().set_jit_fuse(true); });
  m.def("disable_jit_opt", []() { AutoOptConfig::singleton().set_jit_fuse(false); });
  m.def("get_jit_opt", []() { return AutoOptConfig::singleton().get_jit_fuse(); });

  // int8 path
  m.def("clear_autocast_cache_int8", &torch_ipex::autocast::int8::clear_autocast_cache_int8);
  m.def("enable_int8_calibration", []() { AutoOptConfig::singleton().set_int8_calibration(true); });
  m.def("disable_int8_calibration", []() { AutoOptConfig::singleton().set_int8_calibration(false); });
  m.def("get_int8_calibration",
        []() { return AutoOptConfig::singleton().get_int8_calibration(); });
  m.def("calibration_reset", []() { Int8OptConfig::calibration_reset(); });
  m.def("set_int8_qscheme", [](const int &scheme) {
    AutoOptConfig::singleton().set_int8_qscheme(scheme);
  });
  m.def("get_int8_qscheme", []() {
    return static_cast<int>(AutoOptConfig::singleton().get_int8_qscheme());
  });

  m.def("add_indicators",
        []() { Int8OptConfig::get_config().add_indicators(); });
  m.def("clear_indicators",
        []() { Int8OptConfig::get_config().clear_indicators(); });
  // clear indicators for case having many scopes which have different structure

  m.def("get_int8_configures", []() {
      py::list output_list;
      auto indicators = Int8OptConfig::get_config().get_indicators();
      for (auto indicator: indicators) {
        py::dict d;
        d["id"] = indicator.get_indicator_id();
        d["name"] = indicator.get_indicator_name();
        d["algorithm"] = indicator.get_indicator_algorithm();
        d["weight_granularity"] = indicator.get_indicator_weight_granularity();
        std::vector<float> x_scales, y_scales;
        std::vector<int64_t> x_zero_points, y_zero_points;
        std::vector<quant_utils::TensorQuantizationParams> x_params, y_params;
        std::tie(x_params, y_params) = indicator.get_indicator_scales();
        for (auto& p: x_params) {
          x_scales.push_back(p.scale);
          x_zero_points.push_back(p.zero_point);
        }
        for (auto& p: y_params) {
          y_scales.push_back(p.scale);
          y_zero_points.push_back(p.zero_point);
        }
        std::vector<std::vector<float>> w_scales = indicator.get_indicator_weight_scales();
        d["input_scales"] = x_scales;
        d["input_zero_points"] = x_zero_points;
        d["output_scales"] = y_scales;
        d["output_zero_points"] = y_zero_points;
        d["weight_scales"] = w_scales;
        std::vector<std::string> i_quantized_dtypes, o_quantized_dtypes;
        std::tie(i_quantized_dtypes, o_quantized_dtypes)= indicator.get_indicator_quantized_dtypes();
        d["input_quantized_dtypes"] = i_quantized_dtypes;
        d["output_quantized_dtypes"] = o_quantized_dtypes;
        std::vector<bool> inputs_quantized, outputs_quantized;
        std::tie(inputs_quantized, outputs_quantized) =
            indicator.get_indicator_insert_quantized_status();
        d["inputs_quantized"] = inputs_quantized;
        d["outputs_quantized"] = outputs_quantized;
        std::vector<std::string> inputs_flow, outputs_flow;
        std::tie(inputs_flow, outputs_flow) = indicator.get_indicator_quantized_flow();
        d["inputs_flow"] = inputs_flow;
        d["outputs_flow"] = outputs_flow;
        output_list.append(d);
      }
      return output_list; } );
  m.def("load_indicators_file", [](const py::list &l) {
    std::vector<Indicator> indicators;
    for (py::handle i : l) {
      int64_t id = py::cast<std::int64_t>(i["id"]);
      std::string op_name = py::cast<std::string>(i["name"]);
      std::string algorithm = py::cast<std::string>(i["algorithm"]);
      std::string weight_granularity = py::cast<std::string>(i["weight_granularity"]);
      std::vector<double> x_scales = py::cast<std::vector<double>>(i["input_scales"]);
      std::vector<int32_t> x_zero_points = py::cast<std::vector<int32_t>>(i["input_zero_points"]);
      std::vector<double> y_scales = py::cast<std::vector<double>>(i["output_scales"]);
      std::vector<int32_t> y_zero_points = py::cast<std::vector<int32_t>>(i["output_zero_points"]);
      std::vector<quant_utils::TensorQuantizationParams> x_params, y_params;
      for (auto i = 0; i < x_scales.size(); i++) {
        quant_utils::TensorQuantizationParams param;
        param.scale = x_scales[i];
        param.zero_point = x_zero_points[i];
        x_params.push_back(param);
      }
      for (auto i = 0; i < y_scales.size(); i++) {
        quant_utils::TensorQuantizationParams param;
        param.scale = y_scales[i];
        param.zero_point = y_zero_points[i];
        y_params.push_back(param);
      }
      std::vector<std::vector<float>> w_scales = py::cast<std::vector<std::vector<float>>>(i["weight_scales"]);
      std::vector<std::string> i_quantized_dtypes =
          py::cast<std::vector<std::string>>(i["input_quantized_dtypes"]);
      std::vector<std::string> o_quantized_dtypes =
          py::cast<std::vector<std::string>>(i["output_quantized_dtypes"]);
      std::vector<bool> inputs_quantized = py::cast<std::vector<bool>>(i["inputs_quantized"]);
      std::vector<bool> outputs_quantized = py::cast<std::vector<bool>>(i["outputs_quantized"]);
      std::vector<std::string> inputs_flow = py::cast<std::vector<std::string>>(i["inputs_flow"]);
      std::vector<std::string> outputs_flow = py::cast<std::vector<std::string>>(i["outputs_flow"]);
      Indicator temp(id, op_name, algorithm, weight_granularity, x_params,
                     w_scales, y_params, i_quantized_dtypes, o_quantized_dtypes,
                     inputs_quantized, outputs_quantized, inputs_flow, outputs_flow);
      indicators.push_back(temp);
    }
    Int8OptConfig::get_config().set_indicators(indicators);
  });

  // extend OPs
  m.def("embedding_bag_fast_path_sum", &AtenIpexTypeExt::embedding_bag_fast_path_sum);
}
}  // namespace
using namespace torch::jit;

void InitIpexBindings(py::module m) {
  InitIpexModuleBindings(m);

  // // llga jit fusion pass
  // torch::jit::registerPrePass([](std::shared_ptr<Graph>& g) {
  //   if (torch::jit::RegisterLlgaFuseGraph::isEnabled()) {
  //     torch::jit::fuser::onednn::fuseGraph(g);
  //   }
  // });
  // jit fusion pass
  torch::jit::registerPrePass([](std::shared_ptr<Graph>& g) {
    if (AutoOptConfig::singleton().get_jit_fuse()) {
      torch::jit::FusionPass(g);
    }
  });
}

}  // namespace torch_ipex

PYBIND11_MODULE(_torch_ipex, m) { torch_ipex::InitIpexBindings(m); }
