#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/Generator.h>

#include <ATen/aten_ipex_type_default.h>
#include <ATen/ipex_type_dpcpp_customized.h>
#include <../jit/fusion_pass.h>
#include <../jit/weight_freeze.h>
#include <core/Generator.h>

#include <pybind11/pybind11.h>


namespace py = pybind11;

PYBIND11_MODULE(torch_ipex, m) {
  m.doc() = "PyTorch Extension for Intel dGPU";

  at::RegisterAtenTypeFunctions();
  torch::jit::InitFusionPass();

  m.def("linear_relu",
      [](const at::Tensor & input,
         const at::Tensor & weight,
         const at::Tensor & bias) {
        return at::AtenIpexTypeDPCPP::linear_relu(input, weight, bias);
      },
      "fused linear with relu opt. on Intel device");

  m.def("mul_add",
      [](const at::Tensor & self,
        const at::Tensor & other,
        const at::Tensor & accumu,
        float alpha) {
        return at::AtenIpexTypeDPCPP::mul_add(self, other, accumu, alpha);
      },
      "fused mul with add opt. on Intel device");

#if defined(USE_USM)
  m.def("_usm_is_enabled",
        []() {return true;});
#else
  m.def("_usm_is_enabled",
        []() {return false;});
#endif

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

}
