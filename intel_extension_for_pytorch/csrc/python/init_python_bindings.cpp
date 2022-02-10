#include "init_python_bindings.h"

#include "intel_extension_for_pytorch/csrc/aten/cpu/utils/isa_help.h"
#include "intel_extension_for_pytorch/csrc/jit/codegen/onednn/interface.h"
#include "intel_extension_for_pytorch/csrc/version.h"

#include <ATen/native/quantized/cpu/quant_utils.h>
#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/util/Optional.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_stub.h>

#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator_options.h>
#include "intel_extension_for_pytorch/csrc/jit/fusion_pass.h"

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "intel_extension_for_pytorch/csrc/quantization/AutoCast.hpp"
#include "intel_extension_for_pytorch/csrc/quantization/Config.hpp"
#include "intel_extension_for_pytorch/csrc/quantization/Observer.hpp"
#include "intel_extension_for_pytorch/csrc/quantization/auto_opt_config.hpp"
#include "intel_extension_for_pytorch/csrc/utils/env_settings.h"
#include "intel_extension_for_pytorch/csrc/utils/rw_lock.h"
#include "intel_extension_for_pytorch/csrc/utils/verbose.hpp"

#include <c10/core/DeviceType.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/api/include/torch/python.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include "intel_extension_for_pytorch/csrc/autocast/autocast_mode.h"

#include "TaskModule.h"
#include "intel_extension_for_pytorch/csrc/aten/cpu/EmbeddingBag.h"
#include "intel_extension_for_pytorch/csrc/cpu/runtime/CPUPool.h"
#include "intel_extension_for_pytorch/csrc/cpu/runtime/TaskExecutor.h"
#include "intel_extension_for_pytorch/csrc/cpu/utils/CPUISA.h"

namespace torch_ipex {
namespace {

py::object GetBinaryInfo() {
  auto py_dict = py::dict();
  py_dict["__version__"] = std::string(__version__);
  py_dict["__gitrev__"] = std::string(__gitrev__);
  py_dict["__avx_version__"] = std::string(__avx_version__);
  py_dict["__torch_gitrev__"] = std::string(__torch_gitrev__);
  py_dict["__mode__"] = std::string(__mode__);
  return std::move(py_dict);
}

void InitIpexModuleBindings(py::module m) {
  m.def("_get_binary_info", []() { return GetBinaryInfo(); });

  // Check CPU ISA
  m.def("_does_support_avx2", []() {
    using namespace torch_ipex::cpu::utils;
    return CPUISA::info().does_support_avx2();
  });
  m.def("_does_support_avx512", []() {
    using namespace torch_ipex::cpu::utils;
    return CPUISA::info().does_support_avx512();
  });

  m.def("_get_current_isa_level", []() {
    using namespace torch_ipex::cpu;
    return get_current_isa_level();
  });

  m.def("mkldnn_set_verbose", &torch_ipex::verbose::_mkldnn_set_verbose);
  // ipex amp autocast
  m.def("get_autocast_dtype", []() {
    at::ScalarType current_dtype = torch_ipex::autocast::get_autocast_dtype();
    auto dtype = (PyObject*)torch::getTHPDtype(current_dtype);
    Py_INCREF(dtype);
    return py::reinterpret_steal<py::object>(dtype);
  });
  m.def("set_autocast_dtype", [](py::object dtype) {
    at::ScalarType target_dtype =
        torch::python::detail::py_object_to_dtype(dtype);
    torch_ipex::autocast::set_autocast_dtype(target_dtype);
  });
  m.def(
      "is_quantization_enabled",
      &torch_ipex::autocast::is_quantization_enabled);
  m.def(
      "set_quantization_enabled",
      &torch_ipex::autocast::set_quantization_enabled);
  m.def(
      "is_llga_fp32_bf16_enabled",
      &torch_ipex::autocast::is_llga_fp32_bf16_enabled);
  m.def(
      "set_llga_fp32_bf16_enabled",
      &torch_ipex::autocast::set_llga_fp32_bf16_enabled);

  m.def(
      "autocast_increment_nesting",
      &torch_ipex::autocast::autocast_increment_nesting);
  m.def(
      "autocast_decrement_nesting",
      &torch_ipex::autocast::autocast_decrement_nesting);
  m.def("clear_autocast_cache", &torch_ipex::autocast::clear_autocast_cache);

  // llga path
  m.def(
      "_jit_set_llga_weight_cache_enabled",
      &torch::jit::fuser::onednn::setLlgaWeightCacheEnabled);
  m.def(
      "_jit_llga_weight_cache_enabled",
      &torch::jit::fuser::onednn::getLlgaWeightCacheEnabled);

  m.def("enable_jit_opt", []() {
    AutoOptConfig::singleton().set_jit_fuse(true);
  });
  m.def("disable_jit_opt", []() {
    AutoOptConfig::singleton().set_jit_fuse(false);
  });
  m.def("get_jit_opt", []() {
    return AutoOptConfig::singleton().get_jit_fuse();
  });

  // int8 path
  m.def(
      "clear_autocast_cache_int8",
      &torch_ipex::autocast::int8::clear_autocast_cache_int8);
  m.def("enable_int8_calibration", []() {
    AutoOptConfig::singleton().set_int8_calibration(true);
  });
  m.def("disable_int8_calibration", []() {
    AutoOptConfig::singleton().set_int8_calibration(false);
  });
  m.def("get_int8_calibration", []() {
    return AutoOptConfig::singleton().get_int8_calibration();
  });
  m.def("calibration_reset", []() { Int8OptConfig::calibration_reset(); });
  m.def("set_int8_qscheme", [](const int& scheme) {
    AutoOptConfig::singleton().set_int8_qscheme(scheme);
  });
  m.def("get_int8_qscheme", []() {
    return static_cast<int>(AutoOptConfig::singleton().get_int8_qscheme());
  });

  m.def(
      "add_indicators", []() { Int8OptConfig::get_config().add_indicators(); });
  m.def("clear_indicators", []() {
    Int8OptConfig::get_config().clear_indicators();
  });
  // clear indicators for case having many scopes which have different structure

  m.def("get_int8_configures", []() {
    py::list output_list;
    auto indicators = Int8OptConfig::get_config().get_indicators();
    for (auto indicator : indicators) {
      py::dict d;
      d["id"] = indicator.get_indicator_id();
      d["name"] = indicator.get_indicator_name();
      d["algorithm"] = indicator.get_indicator_algorithm();
      d["weight_granularity"] = indicator.get_indicator_weight_granularity();
      std::vector<float> x_scales, y_scales;
      std::vector<int64_t> x_zero_points, y_zero_points;
      std::vector<quant_utils::TensorQuantizationParams> x_params, y_params;
      std::tie(x_params, y_params) = indicator.get_indicator_scales();
      for (auto& p : x_params) {
        x_scales.push_back(p.scale);
        x_zero_points.push_back(p.zero_point);
      }
      for (auto& p : y_params) {
        y_scales.push_back(p.scale);
        y_zero_points.push_back(p.zero_point);
      }
      std::vector<std::vector<float>> w_scales =
          indicator.get_indicator_weight_scales();
      d["input_scales"] = x_scales;
      d["input_zero_points"] = x_zero_points;
      d["output_scales"] = y_scales;
      d["output_zero_points"] = y_zero_points;
      d["weight_scales"] = w_scales;
      std::vector<std::string> i_quantized_dtypes, o_quantized_dtypes;
      std::tie(i_quantized_dtypes, o_quantized_dtypes) =
          indicator.get_indicator_quantized_dtypes();
      d["input_quantized_dtypes"] = i_quantized_dtypes;
      d["output_quantized_dtypes"] = o_quantized_dtypes;
      std::vector<bool> inputs_quantized, outputs_quantized;
      std::tie(inputs_quantized, outputs_quantized) =
          indicator.get_indicator_insert_quantized_status();
      d["inputs_quantized"] = inputs_quantized;
      d["outputs_quantized"] = outputs_quantized;
      std::vector<std::string> inputs_flow, outputs_flow;
      std::tie(inputs_flow, outputs_flow) =
          indicator.get_indicator_quantized_flow();
      d["inputs_flow"] = inputs_flow;
      d["outputs_flow"] = outputs_flow;
      output_list.append(d);
    }
    return output_list;
  });
  m.def("load_indicators_file", [](const py::list& l) {
    std::vector<Indicator> indicators;
    for (py::handle i : l) {
      int64_t id = py::cast<std::int64_t>(i["id"]);
      std::string op_name = py::cast<std::string>(i["name"]);
      std::string algorithm = py::cast<std::string>(i["algorithm"]);
      std::string weight_granularity =
          py::cast<std::string>(i["weight_granularity"]);
      std::vector<double> x_scales =
          py::cast<std::vector<double>>(i["input_scales"]);
      std::vector<int32_t> x_zero_points =
          py::cast<std::vector<int32_t>>(i["input_zero_points"]);
      std::vector<double> y_scales =
          py::cast<std::vector<double>>(i["output_scales"]);
      std::vector<int32_t> y_zero_points =
          py::cast<std::vector<int32_t>>(i["output_zero_points"]);
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
      std::vector<std::vector<float>> w_scales =
          py::cast<std::vector<std::vector<float>>>(i["weight_scales"]);
      std::vector<std::string> i_quantized_dtypes =
          py::cast<std::vector<std::string>>(i["input_quantized_dtypes"]);
      std::vector<std::string> o_quantized_dtypes =
          py::cast<std::vector<std::string>>(i["output_quantized_dtypes"]);
      std::vector<bool> inputs_quantized =
          py::cast<std::vector<bool>>(i["inputs_quantized"]);
      std::vector<bool> outputs_quantized =
          py::cast<std::vector<bool>>(i["outputs_quantized"]);
      std::vector<std::string> inputs_flow =
          py::cast<std::vector<std::string>>(i["inputs_flow"]);
      std::vector<std::string> outputs_flow =
          py::cast<std::vector<std::string>>(i["outputs_flow"]);
      Indicator temp(
          id,
          op_name,
          algorithm,
          weight_granularity,
          x_params,
          w_scales,
          y_params,
          i_quantized_dtypes,
          o_quantized_dtypes,
          inputs_quantized,
          outputs_quantized,
          inputs_flow,
          outputs_flow);
      indicators.push_back(temp);
    }
    Int8OptConfig::get_config().set_indicators(indicators);
  });

  // extend OPs
  m.def(
      "embedding_bag_fast_path_sum", &torch_ipex::embedding_bag_fast_path_sum);

  // runtime
  py::class_<torch_ipex::runtime::FutureTensor>(m, "FutureTensor")
      .def("get", &torch_ipex::runtime::FutureTensor::get);

  // The holder type is std::shared_ptr<torch_ipex::runtime::CPUPool>.
  // Please use std::shared_ptr<torch_ipex::runtime::CPUPool> as funtion
  // parameter. If you pass it as parameter from python into C++.
  py::class_<
      torch_ipex::runtime::CPUPool,
      std::shared_ptr<torch_ipex::runtime::CPUPool>>(m, "CPUPool")
      .def(py::init([](const py::list& core_list) {
        return std::make_shared<torch_ipex::runtime::CPUPool>(
            py::cast<std::vector<int32_t>>(core_list));
      }))
      .def("get_core_list", [](torch_ipex::runtime::CPUPool& self) {
        return self.get_cpu_core_list();
      });

  py::class_<
      torch_ipex::runtime::TaskModule,
      std::shared_ptr<torch_ipex::runtime::TaskModule>>(m, "TaskModule")
      .def(py::init([](const py::object& module, const py::list& core_list) {
        return std::make_shared<torch_ipex::runtime::TaskModule>(
            module, py::cast<std::vector<int32_t>>(core_list));
      }))
      .def(py::init([](const torch::jit::Module& module,
                       const py::list& core_list,
                       bool traced_module) {
        return std::make_shared<torch_ipex::runtime::TaskModule>(
            module, py::cast<std::vector<int32_t>>(core_list), traced_module);
      }))
      .def(
          "run_sync",
          [](torch_ipex::runtime::TaskModule& self,
             py::args& args,
             py::kwargs& kwargs) {
            // Depending on this being ScriptModule of nn.Module we will
            // release the GIL or not further down in the stack
            return self.run_sync(std::move(args), std::move(kwargs));
          })
      .def(
          "run_async",
          [](torch_ipex::runtime::TaskModule& self,
             py::args& args,
             py::kwargs& kwargs) {
            // Depending on this being ScriptModule of nn.Module we will release
            // the GIL or not further down in the stack
            return self.run_async(std::move(args), std::move(kwargs));
          });

  m.def("is_runtime_ext_enabled", &torch_ipex::runtime::is_runtime_ext_enabled);
  m.def("init_runtime_ext", &torch_ipex::runtime::init_runtime_ext);
  m.def("pin_cpu_cores", [](const py::list& core_list) {
    torch_ipex::runtime::_pin_cpu_cores(
        py::cast<std::vector<int32_t>>(core_list));
    return;
  });
  m.def("is_same_core_affinity_setting", [](const py::list& core_list) {
    return torch_ipex::runtime::is_same_core_affinity_setting(
        // Here converting py::list to std::vector<int32_t> will have the data
        // copy.
        py::cast<std::vector<int32_t>>(core_list));
  });
  m.def("get_current_cpu_pool", []() {
    return std::make_shared<torch_ipex::runtime::CPUPool>(
        torch_ipex::runtime::get_cpu_pool_from_mask_affinity());
  });
  m.def(
      "set_cpu_pool",
      [](std::shared_ptr<torch_ipex::runtime::CPUPool> cpu_pool) {
        torch_ipex::runtime::set_mask_affinity_from_cpu_pool((*cpu_pool));
        return;
      });
}
} // namespace
using namespace torch::jit;

void InitIpexBindings(py::module m) {
  EnvSettings::get_instance().initialize_all_settings();
  InitIpexModuleBindings(m);
}

} // namespace torch_ipex

PYBIND11_MODULE(_C, m) {
  torch_ipex::InitIpexBindings(m);
}
