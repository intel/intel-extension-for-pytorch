#include <torch/csrc/jit/pybind_utils.h>

#include <ATen/aten_ipex_type_default.h>
#include <../jit/fusion_pass.h>
#include <../jit/weight_freeze.h>

#include <pybind11/pybind11.h>


namespace py = pybind11;

PYBIND11_MODULE(_torch_ipex, m) {
  // TODO:
  printf("loading _torch_ipex.so ++\n");

  m.doc() = "PyTorch Extension for Intel dGPU";

  at::RegisterAtenTypeFunctions();
  torch::jit::InitFusionPass();

  printf("loading _torch_ipex.so --\n");
}
