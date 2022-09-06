#include <torch/csrc/Exceptions.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/tensor/python_tensor.h>

#include <ATen/autocast_mode.h>

#include <core/Allocator.h>
#include <core/Device.h>
#include <core/Generator.h>
#include <intrinsic/intrinsic.h>
#include <pybind11/stl.h>
#include <runtime/Memory.h>
#include <runtime/Utils.h>
#include <utils/Settings.h>
#include "Event.h"
#include "Module.h"
#include "Storage.h"
#include "Stream.h"

#include <thread>

#define ASSERT_TRUE(cmd) \
  if (!(cmd))            \
  return

PyObject* module;

static bool in_bad_fork = false; // True for children forked after xpu init

#ifndef _WIN32
// Called in the forked child if xpu has already been initialized
static void forked_child() {
  in_bad_fork = true;
  set_run_yet_variable_to_false();
}
#endif

// Should be called before the first xpu call. It will be invoked in lazy_init.
static void poison_fork() {
#ifndef _WIN32
  static std::once_flag flag;
  std::call_once(flag, [] { pthread_atfork(nullptr, nullptr, forked_child); });
#endif
}

PyObject* THPModule_setDevice_wrap(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to setDevice");
  int64_t device = THPUtils_unpackLong(arg);

  xpu::dpcpp::set_device(static_cast<c10::DeviceIndex>(device));

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_getDevice_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto device = static_cast<int>(xpu::dpcpp::current_device());
  return PyLong_FromLong(device);
  END_HANDLE_TH_ERRORS
}

// Because dpcpp::device_count could call poison_fork in lazy_init,
// it is not necessary to add poison_fork here repeatedly.
PyObject* THPModule_getDeviceCount_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyLong_FromLong(xpu::dpcpp::device_count());
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_prefetchDeviceCount_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyLong_FromLong(xpu::dpcpp::prefetch_device_count());
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_getDeviceIdListForCard_wrap(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to get card id");
  int card_id = THPUtils_unpackInt(arg);
  auto deviceid_card = xpu::dpcpp::deviceIdListForCard(card_id);
  py::list deviceid_card_pylist = py::cast(deviceid_card);
  return deviceid_card_pylist.release().ptr();
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_isInBadFork(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(in_bad_fork);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_postInitExtension(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  std::vector<c10::Backend> backends = {c10::Backend::XPU};
  std::vector<c10::ScalarType> scalar_types = {
      c10::ScalarType::Byte,
      c10::ScalarType::Char,
      c10::ScalarType::Double,
      c10::ScalarType::Float,
      c10::ScalarType::Int,
      c10::ScalarType::Long,
      c10::ScalarType::Short,
      c10::ScalarType::Half,
      c10::ScalarType::Bool,
      c10::ScalarType::BFloat16};
  for (auto& backend : backends) {
    for (auto& scalar_type : scalar_types) {
      torch::tensors::register_python_tensor_type(backend, scalar_type);
    }
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_initExtension(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  TORCH_INTERNAL_ASSERT(!in_bad_fork); // Handled at python level
  poison_fork();

  auto module =
      THPObjectPtr(PyImport_ImportModule("intel_extension_for_pytorch.xpu"));
  if (!module)
    throw python_error();

  THDPStorage_postInitExtension(module);

  auto set_module_attr = [&](const char* name, PyObject* v) {
    // PyObject_SetAttrString doesn't steal reference. So no need to incref.
    if (PyObject_SetAttrString(module, name, v) < 0) {
      throw python_error();
    }
  };

  // Here is thread safety. Set run_yet TRUE before device_count() to avoid
  // circular calls.
  // Put set_run_yet_variable_to_true() here instead of in C++ API's lazy_init()
  // to avoid circular calls when directly call Python API's _lazy_init().
  set_run_yet_variable_to_true();
  auto num_gpus = xpu::dpcpp::device_count();
  auto default_dpcpp_generators =
      PyTuple_New(static_cast<Py_ssize_t>(num_gpus));
  for (int i = 0; i < num_gpus; i++) {
    auto gen = xpu::dpcpp::detail::getDefaultDPCPPGenerator(i);
    // auto cast_gen = (THPGenerator*)DPCPPGenerator_initDefaultGenerator(gen);
    auto cast_gen = (THPGenerator*)THPGenerator_initDefaultGenerator(gen);
    // This reference is meant to be given away, so no need to incref here.
    PyTuple_SetItem(default_dpcpp_generators, i, (PyObject*)cast_gen);
  }
  set_module_attr("default_generators", default_dpcpp_generators);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_getCurrentStream_wrap(
    PyObject* /* unused */,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(device_index), "invalid argument to getCurrentStream");
  int64_t device = THPUtils_unpackLong(device_index);
  return PyLong_FromUnsignedLongLong(
      xpu::dpcpp::getCurrentDPCPPStream(device).pack());
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_setCurrentStream_wrap(PyObject* self, PyObject* obj) {
  HANDLE_TH_ERRORS
  THPUtils_assert(PyLong_Check(obj), "invalid stream");
  uint64_t bits = PyLong_AsUnsignedLongLong(obj);
  if (bits == static_cast<uint64_t>(-1) && PyErr_Occurred()) {
    throw python_error();
  }
  auto stream = xpu::dpcpp::DPCPPStream::unpack(bits);
  auto device = static_cast<int>(xpu::dpcpp::current_device());
  if (device != stream.device_index()) {
    xpu::dpcpp::set_device(static_cast<c10::DeviceIndex>(device));
  }
  xpu::dpcpp::setCurrentDPCPPStream(stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_resetPeakMemoryStats(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(arg), "invalid argument to reset_peak_memory_stats");
  const int device = (int)THPUtils_unpackLong(arg);
  xpu::dpcpp::resetPeakStatsInDevAlloc(device);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* THPModule_resetAccumulatedMemoryStats(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(arg),
      "invalid argument to reset_accumulated_memory_stats");
  const int device = (int)THPUtils_unpackLong(arg);
  xpu::dpcpp::resetAccumulatedStatsInDevAlloc(device);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* THPModule_emptyCache(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  xpu::dpcpp::emptyCacheInDevAlloc();
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* THPModule_memoryStats(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(arg), "invalid argument to memory_allocated");
  const int device = (int)THPUtils_unpackLong(arg);

  using xpu::dpcpp::DeviceStats;
  using xpu::dpcpp::Stat;
  using xpu::dpcpp::StatArray;
  using xpu::dpcpp::StatType;

  const auto statToDict = [](const Stat& stat) {
    py::dict dict;

    dict["current"] = stat.current;
    dict["peak"] = stat.peak;
    dict["allocated"] = stat.allocated;
    dict["freed"] = stat.freed;
    return dict;
  };

  const auto statArrayToDict = [=](const StatArray& statArray) {
    const std::array<const char*, static_cast<size_t>(StatType::NUM_TYPES)>
        statTypeNames = {"all", "small_pool", "large_pool"};
    py::dict dict;
    for (size_t i = 0; i < statTypeNames.size(); ++i) {
      dict[statTypeNames[i]] = statToDict(statArray[i]);
    }
    return dict;
  };

  const auto stats = xpu::dpcpp::getDeviceStatsFromDevAlloc(device);

  py::dict result;
  result["num_alloc_retries"] = stats.num_alloc_retries;
  result["num_ooms"] = stats.num_ooms;
  result["allocation"] = statArrayToDict(stats.allocation);
  result["segment"] = statArrayToDict(stats.segment);
  result["active"] = statArrayToDict(stats.active);
  result["inactive_split"] = statArrayToDict(stats.inactive_split);
  result["allocated_bytes"] = statArrayToDict(stats.allocated_bytes);
  result["reserved_bytes"] = statArrayToDict(stats.reserved_bytes);
  result["active_bytes"] = statArrayToDict(stats.active_bytes);
  result["inactive_split_bytes"] = statArrayToDict(stats.inactive_split_bytes);

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_memorySnapshot(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS

  using xpu::dpcpp::BlockInfo;
  using xpu::dpcpp::SegmentInfo;

  const auto segmentInfoToDict = [](const SegmentInfo& segmentInfo) {
    py::dict segmentDict;
    segmentDict["device"] = segmentInfo.device;
    segmentDict["address"] = segmentInfo.address;
    segmentDict["total_size"] = segmentInfo.total_size;
    segmentDict["allocated_size"] = segmentInfo.allocated_size;
    segmentDict["active_size"] = segmentInfo.active_size;
    segmentDict["segment_type"] = (segmentInfo.is_large ? "large" : "small");

    py::list blocks;
    for (const auto& blockInfo : segmentInfo.blocks) {
      py::dict blockDict;
      blockDict["size"] = blockInfo.size;
      blockDict["state"] =
          (blockInfo.allocated
               ? "active_allocated"
               : (blockInfo.active ? "active_pending_free" : "inactive"));
      blocks.append(blockDict);
    }
    segmentDict["blocks"] = blocks;

    return segmentDict;
  };

  const std::vector<SegmentInfo>& snapshot = xpu::dpcpp::snapshotOfDevAlloc();
  py::list result;

  for (const auto& segmentInfo : snapshot) {
    result.append(segmentInfoToDict(segmentInfo));
  }

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

static const char* scalarTypeName(const at::ScalarType type) {
  switch (type) {
#define DEFINE_CASE(ctype, name) \
  case at::ScalarType::name:     \
    return #ctype;
    AT_FORAUTOCAST_SCALAR_TYPES(DEFINE_CASE)
#undef DEFINE_CASE
    default:
      throw std::runtime_error("unknown scalar type for autocast");
  }
}

static PyObject* set_autocast_xpu_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (!PyBool_Check(arg)) {
    throw TypeError("enabled must be a bool (got %s)", Py_TYPE(arg)->tp_name);
  }
  at::autocast::set_xpu_enabled(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* is_autocast_xpu_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (at::autocast::is_xpu_enabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* set_autocast_xpu_dtype(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (!THPDtype_Check(arg)) {
    throw TypeError(
        "dtype must be a torch.dtype (got %s)", Py_TYPE(arg)->tp_name);
  }
  at::ScalarType targetType = reinterpret_cast<THPDtype*>(arg)->scalar_type;
  at::autocast::set_autocast_xpu_dtype(targetType);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* get_autocast_xpu_dtype(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  at::ScalarType current_dtype = at::autocast::get_autocast_xpu_dtype();
  return THPDtype_New(current_dtype, scalarTypeName(current_dtype));
  END_HANDLE_TH_ERRORS
}
static struct PyMethodDef _THPModule_methods[] = {
    {"_initExtension",
     (PyCFunction)THPModule_initExtension,
     METH_NOARGS,
     nullptr},
    {"_postInitExtension",
     (PyCFunction)THPModule_postInitExtension,
     METH_NOARGS,
     nullptr},
    {"_setDevice", (PyCFunction)THPModule_setDevice_wrap, METH_O, nullptr},
    {"_getDevice", (PyCFunction)THPModule_getDevice_wrap, METH_NOARGS, nullptr},
    {"_getDeviceCount",
     (PyCFunction)THPModule_getDeviceCount_wrap,
     METH_NOARGS,
     nullptr},
    {"_prefetchDeviceCount",
     (PyCFunction)THPModule_prefetchDeviceCount_wrap,
     METH_NOARGS,
     nullptr},
    {"_getDeviceIdListForCard",
     (PyCFunction)THPModule_getDeviceIdListForCard_wrap,
     METH_O,
     nullptr},
    {"_xpu_isInBadFork",
     (PyCFunction)THPModule_isInBadFork,
     METH_NOARGS,
     nullptr},
    {"_getCurrentStream",
     (PyCFunction)THPModule_getCurrentStream_wrap,
     METH_O,
     nullptr},
    {"_setCurrentStream",
     (PyCFunction)THPModule_setCurrentStream_wrap,
     METH_O,
     nullptr},
    {"_emptyCache", (PyCFunction)THPModule_emptyCache, METH_NOARGS, nullptr},
    {"_memoryStats", (PyCFunction)THPModule_memoryStats, METH_O, nullptr},
    {"_resetAccumulatedMemoryStats",
     (PyCFunction)THPModule_resetAccumulatedMemoryStats,
     METH_O,
     nullptr},
    {"_resetPeakMemoryStats",
     (PyCFunction)THPModule_resetPeakMemoryStats,
     METH_O,
     nullptr},
    {"_memorySnapshot",
     (PyCFunction)THPModule_memorySnapshot,
     METH_NOARGS,
     nullptr},
    {"set_autocast_xpu_enabled", set_autocast_xpu_enabled, METH_O, nullptr},
    {"is_autocast_xpu_enabled", is_autocast_xpu_enabled, METH_NOARGS, nullptr},
    {"set_autocast_xpu_dtype", set_autocast_xpu_dtype, METH_O, nullptr},
    {"get_autocast_xpu_dtype", get_autocast_xpu_dtype, METH_NOARGS, nullptr},
    {nullptr}};

std::string get_dev_type(const DeviceProp& prop) {
  std::ostringstream stream;
  switch (prop.dev_type) {
    case sycl::info::device_type::cpu:
      stream << "cpu";
      break;
    case sycl::info::device_type::gpu:
      stream << "gpu";
      break;
    case sycl::info::device_type::accelerator:
      stream << "accelerator";
      break;
    default:
      stream
          << "unknown device type:"
          << static_cast<
                 typename std::underlying_type<sycl::info::device_type>::type>(
                 prop.dev_type);
      break;
  }
  return stream.str();
}

static void register_xpu_device_properties(PyObject* module) {
  // Add _DeviceProperties class to intel_extension_for_pytorch._C
  auto m = py::handle(module).cast<py::module>();
  py::class_<DeviceProp>(m, "_DeviceProperties")
      .def_readonly("name", &DeviceProp::dev_name)
      .def_readonly("platform_name", &DeviceProp::platform_name)
      .def_readonly("total_memory", &DeviceProp::global_mem_size)
      .def_readonly("max_compute_units", &DeviceProp::max_compute_units)
      .def_readonly("max_sub_devices", &DeviceProp::max_sub_devices)
      .def_property_readonly(
          "dev_type", [](const DeviceProp& prop) { return get_dev_type(prop); })
      .def("__repr__", [](const DeviceProp& prop) {
        std::ostringstream stream;
        stream << "_DeviceProperties(name='" << prop.dev_name
               << "', platform_name='" << prop.platform_name << "', dev_type='"
               << get_dev_type(prop)
               << "', max_sub_devices=" << prop.max_sub_devices
               << ", total_memory=" << prop.global_mem_size / (1024 * 1024)
               << "MB, max_compute_units=" << prop.max_compute_units << ")";
        return stream.str();
      });
}

static void bindGetDeviceProperties(PyObject* module) {
  // Add method to intel_extension_for_pytorch._C
  auto m = py::handle(module).cast<py::module>();
  m.def(
      "_get_device_properties",
      [](int device) -> DeviceProp* {
        return xpu::dpcpp::getDeviceProperties(device);
      },
      py::return_value_policy::reference);

  m.def("_synchronize", [](const int& device_index) {
    auto& dpcpp_queue = getCurrentDPCPPStream(device_index).dpcpp_queue();
    dpcpp_queue.wait_and_throw();
  });
}

at::Scalar scalar_slow(PyObject* object) {
  // Zero-dim tensors are converted to Scalars as-is. Note this doesn't
  // currently handle most NumPy scalar types except np.float64.
  if (THPVariable_Check(object)) {
    return ((THPVariable*)object)->cdata->item();
  }

  if (THPUtils_checkLong(object)) {
    return at::Scalar(static_cast<int64_t>(THPUtils_unpackLong(object)));
  }

  if (PyBool_Check(object)) {
    return at::Scalar(THPUtils_unpackBool(object));
  }

  if (PyComplex_Check(object)) {
    return at::Scalar(THPUtils_unpackComplexDouble(object));
  }
  return at::Scalar(THPUtils_unpackDouble(object));
}

void init_module(pybind11::module& m) {
  m.def(
      "linear_gelu",
      [](const at::Tensor& input,
         const at::Tensor& weight,
         const at::Tensor& bias) {
        return at::AtenIpexTypeXPU::linear_gelu(input, weight, bias);
      },
      "fused linear with gelu opt. on Intel device");
  m.def(
      "linear_relu",
      [](const at::Tensor& input,
         const at::Tensor& weight,
         const at::Tensor& bias) {
        return at::AtenIpexTypeXPU::trans_addmm_relu(input, weight, bias);
      },
      "fused linear with relu opt. on Intel device");

  m.def(
      "linear_sigmoid",
      [](const at::Tensor& input,
         const at::Tensor& weight,
         const at::Tensor& bias) {
        return at::AtenIpexTypeXPU::trans_addmm_sigmoid(input, weight, bias);
      },
      "fused linear with sigmoid opt. on Intel device");

  m.def("_synchronize", [](const int& device_index) {
    auto& dpcpp_queue = getCurrentDPCPPStream(device_index).dpcpp_queue();
    dpcpp_queue.wait();
  });

  m.def(
      "mul_add",
      [](const at::Tensor& self,
         const at::Tensor& other,
         const at::Tensor& accumu,
         float alpha) {
        return at::AtenIpexTypeXPU::mul_add(self, other, accumu, alpha);
      },
      "fused mul with add opt. on Intel device");

  m.def(
      "packed_add",
      [](at::Tensor& top_half,
         at::Tensor& bot_half,
         const at::Tensor& grad,
         float alpha) {
        return at::AtenIpexTypeXPU::packed_add(top_half, bot_half, grad, alpha);
      },
      "enable split SGD for BF16 weight update. on Intel device");

  m.def(
      "fusion_amdd",
      [](at::Tensor& p,
         at::Tensor& d_p,
         at::Tensor& buf,
         float weight_decay,
         float momentum,
         float dampening,
         float lr) {
        return at::AtenIpexTypeXPU::fusion_amdd(
            p, d_p, buf, weight_decay, momentum, dampening, lr);
      },
      "enable Fusion SGD for weight update. on Intel device");

  m.def(
      "interaction",
      [](at::Tensor& input_mlp, at::Tensor& input_emb) {
        return at::AtenIpexTypeXPU::interaction(input_mlp, input_emb);
      },
      "interaction kernel implemtation on Intel device");

  m.def(
      "fused_adamWMasterWeight",
      [](Tensor& master_weight,
         Tensor& weight,
         Tensor& grad,
         const bool amsgrad,
         Tensor& avg,
         Tensor& avg_sq,
         Tensor& max_avg_sq,
         int64_t& step,
         double lr,
         double eps,
         double beta1,
         double beta2,
         double weight_decay) {
        return at::AtenIpexTypeXPU::fused_adamWMasterWeight(
            master_weight,
            weight,
            grad,
            amsgrad,
            avg,
            avg_sq,
            max_avg_sq,
            step,
            lr,
            eps,
            beta1,
            beta2,
            weight_decay);
      },
      "optimize official torch AdamW optimizer kernel implemtation");

  m.def(
      "transformer_adamWMasterWeight",
      [](Tensor& master_weight,
         Tensor& weight,
         Tensor& grad,
         Tensor& avg,
         Tensor& avg_sq,
         Tensor& max_avg_sq,
         int64_t& step,
         double lr,
         double eps,
         double beta1,
         double beta2,
         double weight_decay,
         const bool correct_bias) {
        return at::AtenIpexTypeXPU::transformer_adamWMasterWeight(
            master_weight,
            weight,
            grad,
            avg,
            avg_sq,
            max_avg_sq,
            step,
            lr,
            eps,
            beta1,
            beta2,
            weight_decay,
            correct_bias);
      },
      "optimize transformer AdamW optimizer kernel implemtation");

  m.def(
      "fused_SGDMasterWeight",
      [](at::Tensor& master_weight,
         at::Tensor& weight,
         at::Tensor& grad,
         double weight_decay,
         bool momentum_buffer_existed,
         at::Tensor& momentum_buffer,
         double momentum,
         double dampening,
         bool nesterov,
         double lr) {
        return at::AtenIpexTypeXPU::fused_SGDMasterWeight(
            master_weight,
            weight,
            grad,
            weight_decay,
            momentum_buffer_existed,
            momentum_buffer,
            momentum,
            dampening,
            nesterov,
            lr);
      },
      "optimized SGDMasterWeight optimizer kernel implemtation on Intel device");

  m.def("to_plain", [](const at::Tensor& input) {
    return at::AtenIpexTypeXPU::to_plain_if_needed(input);
  });

  m.def("dump_memory_stat", [](const int& device_index) {
    return xpu::dpcpp::dumpMemoryStatusFromDevAlloc(device_index);
  });

  m.def(
      "_is_onemkl_enabled", []() { return Settings::I().is_onemkl_enabled(); });

  m.def("_is_channels_last_1d_enabled", []() {
    return Settings::I().is_channels_last_1d_enabled();
  });

  m.def(
      "_get_verbose_level", []() { return Settings::I().get_verbose_level(); });

  m.def("_set_verbose_level", [](int level) {
    return Settings::I().set_verbose_level(level);
  });

  py::enum_<xpu::dpcpp::XPU_BACKEND>(m, "XPUBackend")
      .value("GPU", xpu::dpcpp::XPU_BACKEND::GPU)
      .value("CPU", xpu::dpcpp::XPU_BACKEND::CPU)
      .value("AUTO", xpu::dpcpp::XPU_BACKEND::AUTO)
      .export_values();

  m.def("_get_backend", []() {
    return static_cast<int>(Settings::I().get_backend());
  });
  m.def("_set_backend", [](const int backend) {
    return Settings::I().set_backend(static_cast<XPU_BACKEND>(backend));
  });

  m.def("_is_sync_mode", []() { return Settings::I().is_sync_mode_enabled(); });

  m.def("_enable_sync_mode", []() { Settings::I().enable_sync_mode(); });

  m.def("_disable_sync_mode", []() { Settings::I().disable_sync_mode(); });

  m.def("_is_tile_as_device_enabled", []() {
    return Settings::I().is_tile_as_device_enabled();
  });

  m.def("_is_onednn_layout_enabled", []() {
    return Settings::I().is_onednn_layout_enabled();
  });

  m.def(
      "_enable_onednn_layout", []() { Settings::I().enable_onednn_layout(); });

  m.def("_disable_onednn_layout", []() {
    Settings::I().disable_onednn_layout();
  });

  m.def("_set_onednn_verbose", [](const int level) {
    return Settings::I().set_onednn_verbose(level);
  });

  m.def("_set_onemkl_verbose", [](const int level) {
    return Settings::I().set_onemkl_verbose(level);
  });

  py::enum_<xpu::dpcpp::FP32_MATH_MODE>(m, "FP32MathMode")
      .value("FP32", xpu::dpcpp::FP32_MATH_MODE::FP32)
      .value("TF32", xpu::dpcpp::FP32_MATH_MODE::TF32)
      .value("BF32", xpu::dpcpp::FP32_MATH_MODE::BF32)
      .export_values();

  m.def("_get_fp32_math_mode", []() {
    return static_cast<int>(Settings::I().get_fp32_math_mode());
  });
  m.def("_set_fp32_math_mode", [](const int mode) {
    return Settings::I().set_fp32_math_mode(static_cast<FP32_MATH_MODE>(mode));
  });

  m.def("_enable_simple_trace", []() {
#ifdef BUILD_SIMPLE_TRACE
    Settings::I().enable_simple_trace();
#endif
  });

  m.def("_disable_simple_trace", []() {
#ifdef BUILD_SIMPLE_TRACE
    Settings::I().disable_simple_trace();
#endif
  });

  m.def("_is_simple_trace_enabled", []() {
#ifdef BUILD_SIMPLE_TRACE
    return Settings::I().is_simple_trace_enabled();
#else
    return false;
#endif
  });

  auto module = m.ptr();
  THDPStream_init(module);
  THDPEvent_init(module);
  THDPStorage_init(module);
  PyModule_AddFunctions(module, _THPModule_methods);
  register_xpu_device_properties(module);
  bindGetDeviceProperties(module);
}
