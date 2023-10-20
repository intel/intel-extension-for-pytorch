#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "../../csrc/gpu/aten/operators/fp8/fp8_utils.h"
#include "cpu/isa_help/isa_help.h"

namespace py = pybind11;

void InitIsaHelpModuleBindings(py::module m) {
  m.def("_check_isa_avx2", []() { return isa_help::check_isa_avx2(); });

  m.def("_check_isa_avx512", []() { return isa_help::check_isa_avx512(); });

  py::class_<at::AtenIpexTypeXPU::FP8TensorMeta>(m, "FP8TensorMeta")
      .def(py::init<>())
      .def_readwrite("scale", &at::AtenIpexTypeXPU::FP8TensorMeta::scale)
      .def_readwrite(
          "scale_inv", &at::AtenIpexTypeXPU::FP8TensorMeta::scale_inv)
      .def_readwrite(
          "amax_history", &at::AtenIpexTypeXPU::FP8TensorMeta::amax_history);

  py::enum_<at::AtenIpexTypeXPU::FP8FwdTensors>(m, "FP8FwdTensors")
      .value("GEMM1_INPUT", at::AtenIpexTypeXPU::FP8FwdTensors::GEMM1_INPUT)
      .value("GEMM1_WEIGHT", at::AtenIpexTypeXPU::FP8FwdTensors::GEMM1_WEIGHT)
      .value("GEMM1_OUTPUT", at::AtenIpexTypeXPU::FP8FwdTensors::GEMM1_OUTPUT)
      .value("GEMM2_INPUT", at::AtenIpexTypeXPU::FP8FwdTensors::GEMM2_INPUT)
      .value("GEMM2_WEIGHT", at::AtenIpexTypeXPU::FP8FwdTensors::GEMM2_WEIGHT)
      .value("GEMM2_OUTPUT", at::AtenIpexTypeXPU::FP8FwdTensors::GEMM2_OUTPUT);

  py::enum_<at::AtenIpexTypeXPU::FP8BwdTensors>(m, "FP8BwdTensors")
      .value("GRAD_OUTPUT1", at::AtenIpexTypeXPU::FP8BwdTensors::GRAD_OUTPUT1)
      .value("GRAD_INPUT1", at::AtenIpexTypeXPU::FP8BwdTensors::GRAD_INPUT1)
      .value("GRAD_OUTPUT2", at::AtenIpexTypeXPU::FP8BwdTensors::GRAD_OUTPUT2)
      .value("GRAD_INPUT2", at::AtenIpexTypeXPU::FP8BwdTensors::GRAD_INPUT2);

  py::enum_<at::AtenIpexTypeXPU::Float8Format>(
      m, "Float8Format", py::module_local())
      .value("kFloat8_E4M3", at::AtenIpexTypeXPU::Float8Format::kFloat8_E4M3)
      .value("kFloat8_E5M2", at::AtenIpexTypeXPU::Float8Format::kFloat8_E5M2);
}

PYBIND11_MODULE(_isa_help, m) {
  InitIsaHelpModuleBindings(m);
}
