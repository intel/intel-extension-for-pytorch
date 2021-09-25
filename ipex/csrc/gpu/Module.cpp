#include <torch/csrc/Exceptions.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/tensor/python_tensor.h>

#include <autograd/InferenceMode.h>
#include <core/Allocator.h>
#include <core/Device.h>
#include <core/Generator.h>
#include <intrinsic/ipex_intrinsic.h>
#include <jit/fusion_pass.h>
#include <runtime/Memory.h>
#include <utils/Settings.h>
#include "Event.h"
#include "Module.h"
#include "Storage.h"
#include "Stream.h"

#define ASSERT_TRUE(cmd) \
  if (!(cmd))            \
  return

PyObject* module;

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

PyObject* THPModule_getDeviceCount_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyLong_FromLong(xpu::dpcpp::device_count());
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
  auto module = THPObjectPtr(PyImport_ImportModule("ipex.xpu"));
  if (!module)
    throw python_error();

  THPStorage_postInitExtension(module);

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
    {nullptr}};

std::string get_dev_type(const DeviceProp& prop) {
  std::ostringstream stream;
  switch (prop.dev_type) {
    case DPCPP::info::device_type::cpu:
      stream << "cpu";
      break;
    case DPCPP::info::device_type::gpu:
      stream << "gpu";
      break;
    case DPCPP::info::device_type::accelerator:
      stream << "accelerator";
      break;
    default:
      stream
          << "unknown device type:"
          << static_cast<
                 typename std::underlying_type<DPCPP::info::device_type>::type>(
                 prop.dev_type);
      break;
  }
  return stream.str();
}

static void register_xpu_device_properties(PyObject* module) {
  // Add _DeviceProperties class to ipex._C
  auto m = py::handle(module).cast<py::module>();
  py::class_<DeviceProp>(m, "_DeviceProperties")
      .def_readonly("name", &DeviceProp::dev_name)
      .def_readonly("platform_name", &DeviceProp::platform_name)
      .def_readonly("global_memory_size", &DeviceProp::global_mem_size)
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
               << ", global_memory_size="
               << prop.global_mem_size / (1024 * 1024)
               << "MB, max_compute_units=" << prop.max_compute_units << ")";
        return stream.str();
      });
}

static void register_inference_mode(PyObject* module) {
  // Add _DeviceProperties class to torch_ipex._C
  auto m = py::handle(module).cast<py::module>();
  py::class_<InferenceMode>(m, "_InferenceMode").def(py::init<bool>());
}

static void bindGetDeviceProperties(PyObject* module) {
  // Add method to ipex._C
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

void init_module(pybind11::module& m) {
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
      "fused_adamW",
      [](at::Tensor& master_grad_input,
         at::Tensor& grad_input,
         at::Tensor& grad,
         at::Tensor& avg,
         at::Tensor& avg_sq,
         int64_t step,
         double lr,
         double eps,
         double beta1,
         double beta2,
         double weight_decay,
         const bool correct_bias) {
        return at::AtenIpexTypeXPU::fused_adamW(
            master_grad_input,
            grad_input,
            grad,
            avg,
            avg_sq,
            step,
            lr,
            eps,
            beta1,
            beta2,
            weight_decay,
            correct_bias);
      },
      "optimized adamW optimizer kernel implemtation on Intel device");

  m.def("to_plain", [](const at::Tensor& input) {
    return at::AtenIpexTypeXPU::to_plain_if_needed(input);
  });

  m.def("dump_memory_stat", [](const int& device_index) {
    return xpu::dpcpp::dumpMemoryStatusFromDevAlloc(device_index);
  });

  m.def(
      "_is_onedpl_enabled", []() { return Settings::I().is_onedpl_enabled(); });

  m.def(
      "_is_onemkl_enabled", []() { return Settings::I().is_onemkl_enabled(); });

  m.def("_is_double_disabled", []() {
    return Settings::I().is_double_disabled();
  });

  m.def(
      "_get_warning_level", []() { return Settings::I().get_warning_level(); });

  m.def("_get_xpu_backend", []() {
    return int(Settings::I().get_xpu_backend());
  });
  m.def("_set_xpu_backend", [](const int& backend) {
    return Settings::I().set_xpu_backend(static_cast<XPU_BACKEND>(backend));
  });

  m.def("_is_force_sync_exec", []() {
    return Settings::I().is_force_sync_exec();
  });

  m.def("_is_event_profiling_enabled", []() {
    return Settings::I().is_event_profiling_enabled();
  });

  m.def("_is_tile_partition_enabled", []() {
    return Settings::I().is_tile_partition_enabled();
  });

  m.def("_is_onednn_layout_enabled", []() {
    return Settings::I().is_onednn_layout_enabled();
  });

  m.def("_is_tf32_mode_enabled", []() {
    return Settings::I().is_tf32_mode_enabled();
  });

  auto set_module_attr = [&](const char* name, PyObject* v) {
    // PyObject_SetAttrString doesn't steal reference. So no need to incref.
    if (PyObject_SetAttrString(m.ptr(), name, v) < 0) {
      throw python_error();
    }
  };

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

  auto module = m.ptr();
  THDPStream_init(module);
  THDPEvent_init(module);
  THPStorage_init(module);
  PyModule_AddFunctions(module, _THPModule_methods);
  register_xpu_device_properties(module);
  register_inference_mode(module);
  bindGetDeviceProperties(module);
}
