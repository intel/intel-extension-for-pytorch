#include "init_python_bindings.h"

// #include "csrc/cpu/aten/utils/isa_help.h"
// #include "csrc/jit/codegen/onednn/interface.h"
// #include "csrc/utils/version.h"

#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/util/Optional.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_stub.h>

// #include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator_options.h>
// #include "csrc/jit/fusion_pass.h"

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

// #include "csrc/cpu/utils/fpmath_mode.h"
// #include "csrc/cpu/utils/onednn_utils.h"
// #include "csrc/jit/auto_opt_config.h"
// #include "csrc/jit/cpu/tensorexpr/nnc_fuser_register.h"

#include <c10/core/DeviceType.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/api/include/torch/python.h>
#include <torch/csrc/jit/passes/pass_manager.h>
// #include "csrc/cpu/autocast/autocast_kernels.h"
// #include "csrc/cpu/autocast/autocast_mode.h"

#include "TaskModule.h"
// #include "csrc/cpu/aten/EmbeddingBag.h"
// #include "csrc/cpu/runtime/CPUPool.h"
// #include "csrc/cpu/runtime/TaskExecutor.h"
// #include "csrc/cpu/toolkit/sklearn.h"

namespace torch_ipex {
namespace {

/**********************Dummy Functions**************************/
const std::string __version__() {
  return "version";
};
const std::string __gitrev__() {
  return "abcd1234";
};
const std::string __torch_gitrev__() {
  return "ABCD5678";
};
const std::string __mode__() {
  return "dummy_mode";
};
/***************************************************************/

py::object GetBinaryInfo() {
  DUMMY_FE_TRACE
  auto py_dict = py::dict();
  py_dict["__version__"] = __version__();
  py_dict["__gitrev__"] = __gitrev__();
  py_dict["__torch_gitrev__"] = __torch_gitrev__();
  py_dict["__mode__"] = __mode__();
  return std::move(py_dict);
}

/**********************Dummy Functions**************************/
std::string get_current_isa_level() {
  DUMMY_FE_TRACE
  return "Dummy_AMX";
};
std::string get_highest_cpu_support_isa_level() {
  DUMMY_FE_TRACE
  return "Dummy_AMX";
};
std::string get_highest_binary_support_isa_level() {
  DUMMY_FE_TRACE
  return "Dummy_AMX";
};

int onednn_set_verbose(int level) {
  DUMMY_FE_TRACE
  return 0;
};
bool onednn_has_bf16_type_support() {
  DUMMY_FE_TRACE
  return false;
};
bool onednn_has_fp16_type_support() {
  DUMMY_FE_TRACE
  return false;
};
/***************************************************************/
/**********************Dummy Functions**************************/
at::ScalarType get_autocast_dtype() {
  DUMMY_FE_TRACE
  return at::ScalarType::Float;
};

void set_autocast_dtype(at::ScalarType dtype){DUMMY_FE_TRACE};
void clear_autocast_cache(){DUMMY_FE_TRACE};
/***************************************************************/
/**********************Dummy Functions**************************/
enum FP32MathMode : int { FP32 = 0, TF32 = 1, BF32 = 2 };

void setFP32MathModeCpu(FP32MathMode mode = FP32MathMode::FP32){DUMMY_FE_TRACE};

FP32MathMode getFP32MathModeCpu() {
  DUMMY_FE_TRACE
  return FP32MathMode::FP32;
};
/***************************************************************/
/**********************Dummy Functions**************************/
bool is_llga_fp32_bf16_enabled() {
  DUMMY_FE_TRACE
  return true;
}
void set_llga_fp32_bf16_enabled(bool new_enabled) {
  DUMMY_FE_TRACE
}
void setLlgaWeightCacheEnabled(bool enabled) {
  DUMMY_FE_TRACE
}

bool getLlgaWeightCacheEnabled() {
  DUMMY_FE_TRACE
  return true;
}
/***************************************************************/
/**********************Dummy Functions**************************/
inline void set_jit_fuse(bool jit_fuse) {
  DUMMY_FE_TRACE
}

inline bool get_jit_fuse() {
  DUMMY_FE_TRACE
  return true;
}
/***************************************************************/
/**********************Dummy Functions**************************/
std::vector<int32_t> init_process_available_cores() {
  DUMMY_FE_TRACE
  std::vector<int32_t> dummy;
  return dummy;
};
std::vector<int32_t> get_process_available_cores() {
  DUMMY_FE_TRACE
  std::vector<int32_t> dummy;
  return dummy;
};
std::vector<int32_t> filter_cores_by_thread_affinity(
    const std::vector<int32_t>& cpu_core_list) {
  DUMMY_FE_TRACE
  std::vector<int32_t> dummy;
  return dummy;
};
bool do_load_iomp_symbol() {
  DUMMY_FE_TRACE
  return true;
};
bool is_runtime_ext_enabled() {
  DUMMY_FE_TRACE
  return true;
};
void init_runtime_ext(){DUMMY_FE_TRACE};
void _pin_cpu_cores(const torch_ipex::runtime::CPUPool& cpu_pool){
    DUMMY_FE_TRACE};
bool is_same_core_affinity_setting(const std::vector<int32_t>& cpu_core_list) {
  DUMMY_FE_TRACE
  return true;
};
torch_ipex::runtime::CPUPool get_cpu_pool_from_mask_affinity() {
  DUMMY_FE_TRACE
  std::vector<int32_t> dummy_data;
  torch_ipex::runtime::CPUPool dummy(dummy_data);
  return dummy;
};
void set_mask_affinity_from_cpu_pool(
    const torch_ipex::runtime::CPUPool& cpu_pool){DUMMY_FE_TRACE};
/***************************************************************/

void InitIpexModuleBindings(py::module m) {
  /**********************Dummy Functions**************************/
  // m.def("enable_custom_op_2_nnc_fuser", []() {
  //   torch_ipex::jit::cpu::tensorexpr::registerCustomOp2NncFuser();
  // });
  /***************************************************************/

  m.def("_get_binary_info", []() { return GetBinaryInfo(); });

  m.def("_get_current_isa_level", []() {
    /**********************Dummy Functions**************************/
    // using namespace torch_ipex::cpu;
    /***************************************************************/
    return get_current_isa_level();
  });

  m.def("_get_highest_cpu_support_isa_level", []() {
    /**********************Dummy Functions**************************/
    // using namespace torch_ipex::cpu;
    /***************************************************************/
    return get_highest_cpu_support_isa_level();
  });

  m.def("_get_highest_binary_support_isa_level", []() {
    /**********************Dummy Functions**************************/
    // using namespace torch_ipex::cpu;
    /***************************************************************/
    return get_highest_binary_support_isa_level();
  });

  /**********************Dummy Functions**************************/
  m.def(
      "mkldnn_set_verbose",
      /*&torch_ipex::utils::onednn_set_verbose*/ &onednn_set_verbose);
  m.def("onednn_has_bf16_support", []() {
    // return torch_ipex::utils::onednn_has_bf16_type_support();
    return onednn_has_bf16_type_support();
  });
  m.def("onednn_has_fp16_support", []() {
    // return torch_ipex::utils::onednn_has_fp16_type_support();
    return onednn_has_fp16_type_support();
  });

  // ipex amp autocast
  m.def("get_autocast_dtype", []() {
    /**********************Dummy Functions**************************/
    // using namespace torch_ipex::cpu;
    // at::ScalarType current_dtype =
    // torch_ipex::autocast::get_autocast_dtype(); auto dtype =
    // (PyObject*)torch::getTHPDtype(current_dtype);
    at::ScalarType current_dtype = get_autocast_dtype();
    auto dtype = (PyObject*)torch::getTHPDtype(current_dtype);
    Py_INCREF(dtype);
    return py::reinterpret_steal<py::object>(dtype);
  });

  /**********************Dummy Functions**************************/
  m.def("set_autocast_dtype", [](py::object dtype) {
    at::ScalarType target_dtype =
        torch::python::detail::py_object_to_dtype(dtype);
    // torch_ipex::autocast::set_autocast_dtype(target_dtype);
    set_autocast_dtype(target_dtype);
  });
  m.def(
      "clear_autocast_cache", //&torch_ipex::autocast::clear_autocast_cache);
      &clear_autocast_cache);

  m.def("set_fp32_math_mode", [](FP32MathMode mode) {
    // torch_ipex::setFP32MathModeCpu(mode);
    setFP32MathModeCpu(mode);
  });

  m.def(
      "get_fp32_math_mode", //&torch_ipex::getFP32MathModeCpu);
      &getFP32MathModeCpu);

  /**********************Dummy Functions**************************/
  // m.def("_amp_update_scale_", &torch_ipex::autocast::_amp_update_scale_cpu_);
  // m.def(
  //     "_amp_foreach_non_finite_check_and_unscale_",
  //     &torch_ipex::autocast::_amp_foreach_non_finite_check_and_unscale_cpu_);

  // llga path
  m.def(
      "is_llga_fp32_bf16_enabled",
      //&torch_ipex::jit::fuser::onednn::is_llga_fp32_bf16_enabled;
      &is_llga_fp32_bf16_enabled);
  m.def(
      "set_llga_fp32_bf16_enabled",
      // &torch_ipex::jit::fuser::onednn::set_llga_fp32_bf16_enabled;
      &set_llga_fp32_bf16_enabled);
  m.def(
      "_jit_set_llga_weight_cache_enabled",
      // &torch_ipex::jit::fuser::onednn::setLlgaWeightCacheEnabled;
      &setLlgaWeightCacheEnabled);
  m.def(
      "_jit_llga_weight_cache_enabled",
      // &torch_ipex::jit::fuser::onednn::getLlgaWeightCacheEnabled;
      &getLlgaWeightCacheEnabled);

  m.def("enable_jit_opt", []() {
    // AutoOptConfig::singleton().set_jit_fuse(true);
    set_jit_fuse(true);
  });
  m.def("disable_jit_opt", []() {
    // AutoOptConfig::singleton().set_jit_fuse(false);
    set_jit_fuse(false);
  });
  m.def("get_jit_opt", []() {
    // return AutoOptConfig::singleton().get_jit_fuse();
    return get_jit_fuse();
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
      // &torch_ipex::runtime::get_process_available_cores
      &get_process_available_cores);
  m.def(
      "is_runtime_ext_enabled",
      // &torch_ipex::runtime::is_runtime_ext_enabled
      &is_runtime_ext_enabled);
  m.def(
      "init_runtime_ext",
      // &torch_ipex::runtime::init_runtime_ext
      &init_runtime_ext);
  m.def(
      "pin_cpu_cores",
      [](std::shared_ptr<torch_ipex::runtime::CPUPool> cpu_pool) {
        // torch_ipex::runtime::_pin_cpu_cores((*cpu_pool));
        _pin_cpu_cores((*cpu_pool));
        return;
      });
  m.def("is_same_core_affinity_setting", [](const py::list& core_list) {
#if 0    
    return torch_ipex::runtime::is_same_core_affinity_setting(
        // Here converting py::list to std::vector<int32_t> will have the data
        // copy.
        py::cast<std::vector<int32_t>>(core_list));
#else
    return is_same_core_affinity_setting(
        // Here converting py::list to std::vector<int32_t> will have the data
        // copy.
        py::cast<std::vector<int32_t>>(core_list));
#endif
  });
  m.def("get_current_cpu_pool", []() {
#if 0
    return std::make_shared<torch_ipex::runtime::CPUPool>(
        torch_ipex::runtime::get_cpu_pool_from_mask_affinity());
#else
    return std::make_shared<torch_ipex::runtime::CPUPool>(
        get_cpu_pool_from_mask_affinity());
#endif
  });
  m.def(
      "set_cpu_pool",
      [](std::shared_ptr<torch_ipex::runtime::CPUPool> cpu_pool) {
        // torch_ipex::runtime::set_mask_affinity_from_cpu_pool((*cpu_pool));
        set_mask_affinity_from_cpu_pool((*cpu_pool));
        return;
      });
  // m.def("roc_auc_score", &toolkit::roc_auc_score);
  // m.def("roc_auc_score_all", &toolkit::roc_auc_score_all);
}
} // namespace

using namespace torch::jit;

void InitIpexCpuBindings(py::module m) {
  torch_ipex::InitIpexModuleBindings(m);
}
} // namespace torch_ipex
