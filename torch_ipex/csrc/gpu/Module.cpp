#include <torch/csrc/Exceptions.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/tensor/python_tensor.h>

#include <ATen/ipex_type_dpcpp_customized.h>
#include <gpu/jit/fusion_pass.h>
#include <Module.h>
#include <Stream.h>
#include <Storage.h>
#include <core/Functions.h>
#include <core/Generator.h>

#define ASSERT_TRUE(cmd) if (!(cmd)) return

PyObject* module;

PyObject * THPModule_setDevice_wrap(PyObject *self, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to setDevice");
  int64_t device = THPUtils_unpackLong(arg);

  at::dpcpp::set_device(static_cast<c10::DeviceIndex>(device));

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THPModule_getDevice_wrap(PyObject *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  auto device = static_cast<int>(at::dpcpp::current_device());
  return PyLong_FromLong(device);
  END_HANDLE_TH_ERRORS
}

PyObject * THPModule_getDeviceCount_wrap(PyObject *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  return PyLong_FromLong(at::dpcpp::device_count());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPModule_postInitExtension(PyObject *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  std::vector<c10::Backend> backends = { c10::Backend::XPU };
  std::vector<c10::ScalarType> scalar_types = { c10::ScalarType::Byte, c10::ScalarType::Char, c10::ScalarType::Double, c10::ScalarType::Float,
                                                c10::ScalarType::Int,  c10::ScalarType::Long, c10::ScalarType::Short,  c10::ScalarType::Half,
                                                c10::ScalarType::Bool, c10::ScalarType::BFloat16};
  for (auto& backend : backends) {
    for (auto& scalar_type : scalar_types) {
      torch::tensors::register_python_tensor_type(backend, scalar_type);
    }
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPModule_initExtension(PyObject *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  auto module = THPObjectPtr(PyImport_ImportModule("torch_ipex"));
  if (!module) throw python_error();

  // Register Storage Python objects with DynamicTypes.cpp
  THXPStorage_postInit<at::kHalf>(module);
  THXPStorage_postInit<at::kInt>(module);
  THXPStorage_postInit<at::kBool>(module);
  THXPStorage_postInit<at::kLong>(module);
  THXPStorage_postInit<at::kFloat>(module);
  THXPStorage_postInit<at::kDouble>(module);
  THXPStorage_postInit<at::kBFloat16>(module);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THPModule_getCurrentStream_wrap(
  PyObject * /* unused */, PyObject *device_index) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
    THPUtils_checkLong(device_index), "invalid argument to getCurrentStream");
  int64_t device = THPUtils_unpackLong(device_index);
  return PyLong_FromUnsignedLongLong(
    at::dpcpp::getCurrentDPCPPStream(device).pack());
  END_HANDLE_TH_ERRORS
}

static struct PyMethodDef _THCPModule_methods[] = {
  {"_initExtension",  (PyCFunction)THPModule_initExtension,   METH_NOARGS,       nullptr},
  {"_postInitExtension",  (PyCFunction)THPModule_postInitExtension,   METH_NOARGS,       nullptr},
  {"_setDevice",   (PyCFunction)THPModule_setDevice_wrap,   METH_O,       nullptr},
  {"_getDevice",   (PyCFunction)THPModule_getDevice_wrap,   METH_NOARGS,  nullptr},
  {"_getDeviceCount", (PyCFunction)THPModule_getDeviceCount_wrap, METH_NOARGS, nullptr},
  {"_getCurrentStream", (PyCFunction)THPModule_getCurrentStream_wrap, METH_O, nullptr},
  {nullptr}
};

void init_module(pybind11::module& m) {

  torch_ipex::jit::InitFusionPass();

  m.def("linear_relu",
        [](const at::Tensor & input,
           const at::Tensor & weight,
           const at::Tensor & bias) {
            return at::AtenIpexTypeXPU::linear_relu(input, weight, bias);
        },
        "fused linear with relu opt. on Intel device");

  m.def("linear_sigmoid",
        [](const at::Tensor & input,
           const at::Tensor & weight,
           const at::Tensor & bias) {
            return at::AtenIpexTypeXPU::linear_sigmoid(input, weight, bias);
        },
        "fused linear with sigmoid opt. on Intel device");

  m.def("mul_add",
        [](const at::Tensor & self,
           const at::Tensor & other,
           const at::Tensor & accumu,
           float alpha) {
            return at::AtenIpexTypeXPU::mul_add(self, other, accumu, alpha);
        },
        "fused mul with add opt. on Intel device");

  m.def("packed_add",
      [](at::Tensor & top_half,
        at::Tensor & bot_half,
        const at::Tensor & grad,
        float alpha) {
        return at::AtenIpexTypeXPU::packed_add(top_half, bot_half, grad, alpha);
      },
      "enable split SGD for BF16 weight update. on Intel device");

#if defined(USE_ONEDPL)
  m.def("_onedpl_is_enabled",
        []() {return true;});
#else
  m.def("_onedpl_is_enabled",
        []() {return false;});
#endif

#if defined(USE_ONEMKL)
  m.def("_onemkl_is_enabled",
        []() {return true;});
#else
  m.def("_onemkl_is_enabled",
        []() {return false;});
#endif

#if defined(BUILD_DOUBLE_KERNEL)
  m.def("_double_kernel_disabled",
        []() {return false;});
#else
  m.def("_double_kernel_disabled",
        []() {return true;});
#endif

  auto set_module_attr = [&](const char* name, PyObject* v) {
      // PyObject_SetAttrString doesn't steal reference. So no need to incref.
      if (PyObject_SetAttrString(m.ptr(), name, v) < 0) {
        throw python_error();
      }
  };

  auto num_gpus = at::dpcpp::device_count();
  auto default_dpcpp_generators = PyTuple_New(static_cast<Py_ssize_t>(num_gpus));
  for(int i = 0; i < num_gpus; i++) {
    auto gen = at::dpcpp::detail::getDefaultDPCPPGenerator(i);
    //auto cast_gen = (THPGenerator*)DPCPPGenerator_initDefaultGenerator(gen);
    auto cast_gen = (THPGenerator*)THPGenerator_initDefaultGenerator(gen);
    // This reference is meant to be given away, so no need to incref here.
    PyTuple_SetItem(default_dpcpp_generators, i, (PyObject*)cast_gen);
  }
  set_module_attr("default_generators", default_dpcpp_generators);

  auto module = m.ptr();
  THDPStream_init(module);
  PyModule_AddFunctions(module, _THCPModule_methods);
  ASSERT_TRUE(THXPStorage_init<at::kInt>(module));
  ASSERT_TRUE(THXPStorage_init<at::kLong>(module));
  ASSERT_TRUE(THXPStorage_init<at::kHalf>(module));
  ASSERT_TRUE(THXPStorage_init<at::kBool>(module));
  ASSERT_TRUE(THXPStorage_init<at::kFloat>(module));
  ASSERT_TRUE(THXPStorage_init<at::kDouble>(module));
  ASSERT_TRUE(THXPStorage_init<at::kBFloat16>(module));
}
