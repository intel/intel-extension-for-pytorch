#include <core/Generator.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/python_arg_parser.h>

namespace xpu {
// This is a temp solution. We will submit a PR to stock-PyTorch
//  and make XPU backend supported in torch.Generator() API.
// TO DO: remove this file and submit a PR to stock-PyTorch. We should move
// struct Generator from aten to c10. Then unify front-end torch.Geneator with
// VirtualGuardImpl
PyObject* THPGenerator_New(PyObject* _self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  auto type = (PyTypeObject*)THPGeneratorClass;
  static torch::PythonArgParser parser({"Generator(Device device=None)"});
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto device = r.deviceWithDefault(0, at::Device(at::kXPU));

  THPGeneratorPtr self((THPGenerator*)type->tp_alloc(type, 0));
  if (device.type() == at::kXPU) {
    self->cdata =
        make_generator<xpu::dpcpp::DPCPPGeneratorImpl>(device.index());
  } else {
    AT_ERROR(
        "Device type ",
        c10::DeviceTypeName(device.type()),
        " is not supported for torch.xpu.Generator() api.");
  }
  return (PyObject*)self.release();
  END_HANDLE_TH_ERRORS
}

} // namespace xpu
