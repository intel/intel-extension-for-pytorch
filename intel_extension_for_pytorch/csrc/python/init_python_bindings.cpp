#include "init_python_bindings.h"

#include "intel_extension_for_pytorch/csrc/aten/cpu/utils/isa_help.h"
#include "intel_extension_for_pytorch/csrc/jit/codegen/onednn/interface.h"
#include "intel_extension_for_pytorch/csrc/version.h"

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

#include "intel_extension_for_pytorch/csrc/jit/auto_opt_config.h"
#include "intel_extension_for_pytorch/csrc/utils/env_settings.h"
#include "intel_extension_for_pytorch/csrc/utils/fpmath_mode.h"
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

namespace torch_ipex {
namespace {

py::object GetBinaryInfo() {
  auto py_dict = py::dict();
  py_dict["__version__"] = __version__();
  py_dict["__gitrev__"] = __gitrev__();
  py_dict["__torch_gitrev__"] = __torch_gitrev__();
  py_dict["__mode__"] = __mode__();
  return std::move(py_dict);
}

void InitIpexModuleBindings(py::module m) {
  m.def("_get_binary_info", []() { return GetBinaryInfo(); });

  m.def("_get_current_isa_level", []() {
    using namespace torch_ipex::cpu;
    return get_current_isa_level();
  });

  m.def("_get_highest_cpu_support_isa_level", []() {
    using namespace torch_ipex::cpu;
    return get_highest_cpu_support_isa_level();
  });

  m.def("_get_highest_binary_support_isa_level", []() {
    using namespace torch_ipex::cpu;
    return get_highest_binary_support_isa_level();
  });

  m.def("set_profile_op_enabled", [](bool b_enable) {
    using namespace torch_ipex;
    EnvSettings::get_instance().set_settings_profile_op(b_enable);
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
  m.def("clear_autocast_cache", &torch_ipex::autocast::clear_autocast_cache);
  m.def("set_fp32_low_precision_mode", [](IPEXLowPrecisionMode mode) {
    torch_ipex::setFP32LowPrecisionModeCpu(mode);
  });

  m.def("get_fp32_low_precision_mode", &torch_ipex::getFP32LowPrecisionModeCpu);

  // llga path
  m.def(
      "is_llga_fp32_bf16_enabled",
      &torch::jit::fuser::onednn::is_llga_fp32_bf16_enabled);
  m.def(
      "set_llga_fp32_bf16_enabled",
      &torch::jit::fuser::onednn::set_llga_fp32_bf16_enabled);
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

  py::enum_<IPEXLowPrecisionMode>(m, "IPEXLowPrecisionMode")
      .value("BF32", IPEXLowPrecisionMode::BF32)
      .value("FP32", IPEXLowPrecisionMode::FP32)
      .export_values();

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
