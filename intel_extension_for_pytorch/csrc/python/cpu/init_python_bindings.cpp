#include "init_python_bindings.h"

#include "csrc/cpu/aten/utils/isa_help.h"
#include "csrc/jit/codegen/onednn/interface.h"
#include "csrc/utils/version.h"

#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/util/Optional.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_stub.h>

#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator_options.h>
#include "csrc/jit/fusion_pass.h"

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "csrc/cpu/utils/fpmath_mode.h"
#include "csrc/cpu/utils/onednn_utils.h"
#include "csrc/cpu/utils/rw_lock.h"
#include "csrc/jit/auto_opt_config.h"

#include <c10/core/DeviceType.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/api/include/torch/python.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include "csrc/cpu/autocast/autocast_kernels.h"
#include "csrc/cpu/autocast/autocast_mode.h"

#include "TaskModule.h"
#include "csrc/cpu/aten/EmbeddingBag.h"
#include "csrc/cpu/runtime/CPUPool.h"
#include "csrc/cpu/runtime/TaskExecutor.h"

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

  m.def("mkldnn_set_verbose", &torch_ipex::utils::onednn_set_verbose);
  m.def("onednn_has_bf16_support", []() {
    return torch_ipex::utils::onednn_has_bf16_type_support();
  });
  m.def("onednn_has_fp16_support", []() {
    return torch_ipex::utils::onednn_has_fp16_type_support();
  });

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

  m.def("set_fp32_math_mode", [](FP32MathMode mode) {
    torch_ipex::setFP32MathModeCpu(mode);
  });

  m.def("get_fp32_math_mode", &torch_ipex::getFP32MathModeCpu);

  m.def("_amp_update_scale_", &torch_ipex::autocast::_amp_update_scale_cpu_);
  m.def(
      "_amp_foreach_non_finite_check_and_unscale_",
      &torch_ipex::autocast::_amp_foreach_non_finite_check_and_unscale_cpu_);

  // llga path
  m.def(
      "is_llga_fp32_bf16_enabled",
      &torch_ipex::jit::fuser::onednn::is_llga_fp32_bf16_enabled);
  m.def(
      "set_llga_fp32_bf16_enabled",
      &torch_ipex::jit::fuser::onednn::set_llga_fp32_bf16_enabled);
  m.def(
      "_jit_set_llga_weight_cache_enabled",
      &torch_ipex::jit::fuser::onednn::setLlgaWeightCacheEnabled);
  m.def(
      "_jit_llga_weight_cache_enabled",
      &torch_ipex::jit::fuser::onednn::getLlgaWeightCacheEnabled);

  m.def("enable_jit_opt", []() {
    AutoOptConfig::singleton().set_jit_fuse(true);
  });
  m.def("disable_jit_opt", []() {
    AutoOptConfig::singleton().set_jit_fuse(false);
  });
  m.def("get_jit_opt", []() {
    return AutoOptConfig::singleton().get_jit_fuse();
  });

  // BF32
  py::enum_<FP32MathMode>(m, "FP32MathMode")
      .value("FP32", FP32MathMode::FP32)
      .value("TF32", FP32MathMode::TF32)
      .value("BF32", FP32MathMode::BF32)
      .export_values();

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
      .def(py::init([](const py::object& module,
                       std::shared_ptr<torch_ipex::runtime::CPUPool> cpu_pool) {
        return std::make_shared<torch_ipex::runtime::TaskModule>(
            module, (*cpu_pool));
      }))
      .def(py::init([](const torch::jit::Module& module,
                       std::shared_ptr<torch_ipex::runtime::CPUPool> cpu_pool,
                       bool traced_module) {
        return std::make_shared<torch_ipex::runtime::TaskModule>(
            module, (*cpu_pool), traced_module);
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

  m.def(
      "get_process_available_cores",
      &torch_ipex::runtime::get_process_available_cores);
  m.def("is_runtime_ext_enabled", &torch_ipex::runtime::is_runtime_ext_enabled);
  m.def("init_runtime_ext", &torch_ipex::runtime::init_runtime_ext);
  m.def(
      "pin_cpu_cores",
      [](std::shared_ptr<torch_ipex::runtime::CPUPool> cpu_pool) {
        torch_ipex::runtime::_pin_cpu_cores((*cpu_pool));
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
  InitIpexModuleBindings(m);
}

} // namespace torch_ipex