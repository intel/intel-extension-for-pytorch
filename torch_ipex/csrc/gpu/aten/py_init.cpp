#include <torch/csrc/jit/python/pybind_utils.h>

#include <ATen/aten_ipex_type_default.h>
#include <ATen/ipex_type_dpcpp_customized.h>
#include <../jit/fusion_pass.h>
#include <../jit/weight_freeze.h>

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

#if defined(USE_PSTL) && defined(USE_USM)
  m.def("_usm_pstl_is_enabled",
        []() {return true;});
#else
  m.def("_usm_pstl_is_enabled",
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
}
