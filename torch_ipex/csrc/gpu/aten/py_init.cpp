#include <torch/csrc/jit/python/pybind_utils.h>

#include <ATen/aten_ipex_type_default.h>
#include <ATen/ipex_type_dpcpp_customized.h>
#include <../jit/fusion_pass.h>
#include <../jit/weight_freeze.h>

#include <pybind11/pybind11.h>


namespace py = pybind11;

PYBIND11_MODULE(torch_ipex, m) {
  // TODO:
  printf("loading torch_ipex.so ++\n");

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

  printf("loading _torch_ipex.so --\n");
}
