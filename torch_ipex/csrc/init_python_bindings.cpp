#include "init_python_bindings.h"
#include "version.h"

#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/util/Optional.h>
#include <torch/csrc/utils/pybind.h>

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "utils.h"
#include "auto_opt_config.hpp"
#include "quantization/Observer.hpp"
#include "quantization/Config.hpp"
#include "quantization/AutoCast.hpp"

//#include "ProcessGroupCCL.hpp"
#include <pybind11/chrono.h>
#include "autocast_mode.h"
#include <torch/csrc/api/include/torch/python.h>
#include <c10/core/DeviceType.h>
#include <torch/csrc/Exceptions.h>

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
  // ipex amp autocast
  m.def("is_autocast_enabled", &torch_ipex::autocast::is_autocast_enabled);
  m.def("set_autocast_enabled", &torch_ipex::autocast::set_autocast_enabled);
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
  m.def("autocast_increment_nesting",
        &torch_ipex::autocast::autocast_increment_nesting);
  m.def("autocast_decrement_nesting",
        &torch_ipex::autocast::autocast_decrement_nesting);
  m.def("clear_autocast_cache", &torch_ipex::autocast::clear_autocast_cache);


  // int8 path
  m.def("clear_autocast_cache_int8", &torch_ipex::autocast::int8::clear_autocast_cache_int8);
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
      for (auto indicator: indicators) {
        py::dict d;
        d["id"] = indicator.get_indicator_id();
        d["name"] = indicator.get_indicator_name();
        d["algorithm"] = indicator.get_indicator_algorithm();
        d["weight_granularity"] = indicator.get_indicator_weight_granularity();
        std::vector<float> i_scale, o_scale;
        std::vector<float> w_scale;
        std::tie(i_scale, o_scale) = indicator.get_indicator_scales();
        w_scale = indicator.get_indicator_weight_scales();
        d["inputs_scale"] = i_scale;
        d["outputs_scale"] = o_scale;
        d["weight_scale"] = w_scale; 
        std::vector<bool> i_uint8_used, o_uint8_used;
        std::tie(i_uint8_used, o_uint8_used)= indicator.get_indicator_uint8_status();
        d["inputs_uint8_used"] = i_uint8_used;
        d["outputs_uint8_used"] = o_uint8_used;
        d["quantized"] = indicator.get_indicator_quantized_status();
        bool pre_quant = true, post_quant = true;
        std::tie(pre_quant, post_quant) = indicator.get_indicator_insert_quantized_status(); 
        d["pre_quantized"] = pre_quant;
        d["post_quantized"] = post_quant;
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
      std::string weight_granularity =
          py::cast<std::string>(i["weight_granularity"]);
      std::vector<float> i_scale =
          py::cast<std::vector<float>>(i["inputs_scale"]);
      std::vector<float> w_scale = {};
      if (i.contains("weight_scale")) {
        w_scale = py::cast<std::vector<float>>(i["weight_scale"]);

      }
      std::vector<float> o_scale =
          py::cast<std::vector<float>>(i["outputs_scale"]);
      std::vector<bool> i_uint8_used =
          py::cast<std::vector<bool>>(i["inputs_uint8_used"]);
      std::vector<bool> o_uint8_used =
          py::cast<std::vector<bool>>(i["outputs_uint8_used"]);
      bool quantized = py::cast<bool>(i["quantized"]);
      bool pre_quantized = true, post_quantized = true;
      if (i.contains("pre_quantized")) {
        pre_quantized = py::cast<bool>(i["pre_quantized"]);
      }
      if (i.contains("post_quantized")) {
        post_quantized = py::cast<bool>(i["post_quantized"]);
      }
      std::vector<std::string> inputs_flow, outputs_flow;
      if (i.contains("inputs_flow")) {
        inputs_flow = py::cast<std::vector<std::string>>(i["inputs_flow"]);
      }
      if (i.contains("outputs_flow")) {
        outputs_flow = py::cast<std::vector<std::string>>(i["outputs_flow"]);
      }
      Indicator temp(id, op_name, algorithm, weight_granularity, i_scale,
                     w_scale, o_scale, i_uint8_used, o_uint8_used, quantized,
                     pre_quantized, post_quantized, inputs_flow, outputs_flow);
      indicators.push_back(temp);
    }
    Int8OptConfig::get_config().set_indicators(indicators);
  });

  // extend OPs
  m.def("roi_align_forward", &AtenIpexTypeExt::ROIAlign_forward);
  m.def("roi_align_backward", &AtenIpexTypeExt::ROIAlign_backward);

  m.def("nms", &AtenIpexTypeExt::nms);
  m.def("batch_score_nms", &AtenIpexTypeExt::batch_score_nms);
}
}  // namespace
using namespace torch::jit;

void InitIpexBindings(py::module m) {
  InitIpexModuleBindings(m);
}

}  // namespace torch_ipex

PYBIND11_MODULE(_torch_ipex, m) { torch_ipex::InitIpexBindings(m); }
