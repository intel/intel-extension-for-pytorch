#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "../../csrc/cpu/aten/fp8_utils.h"
#include "cpu/isa_help/isa_help.h"

namespace py = pybind11;

void InitIsaHelpModuleBindings(py::module m) {
  m.def("_check_isa_avx2", []() { return isa_help::check_isa_avx2(); });

  m.def("_check_isa_avx512", []() { return isa_help::check_isa_avx512(); });

  py::class_<torch_ipex::cpu::FP8TensorMeta>(m, "FP8TensorMeta")
      .def(py::init<>())
      .def_readwrite("scale", &torch_ipex::cpu::FP8TensorMeta::scale)
      .def_readwrite("scale_inv", &torch_ipex::cpu::FP8TensorMeta::scale_inv)
      .def_readwrite(
          "amax_history", &torch_ipex::cpu::FP8TensorMeta::amax_history);

  py::enum_<torch_ipex::cpu::FP8FwdTensors>(m, "FP8FwdTensors")
      .value("GEMM1_INPUT", torch_ipex::cpu::FP8FwdTensors::GEMM1_INPUT)
      .value("GEMM1_WEIGHT", torch_ipex::cpu::FP8FwdTensors::GEMM1_WEIGHT)
      .value("GEMM1_OUTPUT", torch_ipex::cpu::FP8FwdTensors::GEMM1_OUTPUT)
      .value("GEMM2_INPUT", torch_ipex::cpu::FP8FwdTensors::GEMM2_INPUT)
      .value("GEMM2_WEIGHT", torch_ipex::cpu::FP8FwdTensors::GEMM2_WEIGHT)
      .value("GEMM2_OUTPUT", torch_ipex::cpu::FP8FwdTensors::GEMM2_OUTPUT);

  py::enum_<torch_ipex::cpu::FP8BwdTensors>(m, "FP8BwdTensors")
      .value("GRAD_OUTPUT1", torch_ipex::cpu::FP8BwdTensors::GRAD_OUTPUT1)
      .value("GRAD_INPUT1", torch_ipex::cpu::FP8BwdTensors::GRAD_INPUT1)
      .value("GRAD_OUTPUT2", torch_ipex::cpu::FP8BwdTensors::GRAD_OUTPUT2)
      .value("GRAD_INPUT2", torch_ipex::cpu::FP8BwdTensors::GRAD_INPUT2);

  py::enum_<torch_ipex::cpu::Float8Format>(
      m, "Float8Format", py::module_local())
      .value("kFloat8_E4M3", torch_ipex::cpu::Float8Format::kFloat8_E4M3)
      .value("kFloat8_E5M2", torch_ipex::cpu::Float8Format::kFloat8_E5M2);
}

PYBIND11_MODULE(_isa_help, m) {
  InitIsaHelpModuleBindings(m);
}