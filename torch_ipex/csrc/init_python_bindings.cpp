#include "init_python_bindings.h"
#include "version.h"

#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/util/Optional.h>
#include <torch/csrc/utils/pybind.h>

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "utils.h"

//#include "ProcessGroupCCL.hpp"
#include <pybind11/chrono.h>
#include "autocast_mode.h"
#include <torch/csrc/api/include/torch/python.h>
#include <c10/core/DeviceType.h>
#include <torch/csrc/Exceptions.h>

namespace torch_ipex {
namespace {

py::object GetRevisions() {
  auto py_dict = py::dict();
  py_dict["ipex"] = std::string(IPEX_GITREV);
  py_dict["torch"] = std::string(TORCH_GITREV);
  return py_dict;
}

void InitIpexModuleBindings(py::module m) {
  m.def("_get_git_revs", []() { return GetRevisions(); });
  /*// external OPs
  m.def("roi_align_forward", &IpexExternal::ROIAlign_forward);
  m.def("roi_align_backward", &IpexExternal::ROIAlign_backward);
  m.def("nms", &IpexExternal::nms);
  m.def("batch_score_nms", &IpexExternal::batch_score_nms);
  m.def("linear_relu", &AtenIpexTypeExt::linear_relu);*/

  // ipex amp autocast
  m.def("is_autocast_enabled", &torch_ipex::autocast::is_autocast_enabled);
  m.def("set_autocast_enabled", &torch_ipex::autocast::set_autocast_enabled);
  m.def("get_autocast_dtype", []() {
    at::ScalarType current_dtype = torch_ipex::autocast::get_autocast_dtype();
    return py::reinterpret_steal<py::object>(
        THPDtype_New(current_dtype, scalarTypeName(current_dtype)));
  });
  m.def("set_autocast_dtype", [](py::object dtype) {
    at::ScalarType target_dtype =
        torch::python::detail::py_object_to_dtype(dtype);
    torch_ipex::autocast::set_autocast_dtype(target_dtype);
  });
  m.def("autocast_increment_nesting",
        &torch_ipex::autocast::autocast_increment_nesting);
  m.def("autocast_decrement_nesting",
        &torch_ipex::autocast::autocast_decrement_nesting);
  m.def("clear_autocast_cache", &torch_ipex::autocast::clear_autocast_cache);
}
}  // namespace
using namespace torch::jit;

void InitIpexBindings(py::module m) {
  InitIpexModuleBindings(m);
  // jit fusion pass
  /*torch::jit::registerPrePass([](std::shared_ptr<Graph>& g) {
    if (AutoOptConfig::singleton().get_jit_fuse()) {
      torch::jit::FusionPass(g);
    }
  });*/
}

}  // namespace torch_ipex

PYBIND11_MODULE(_torch_ipex, m) { torch_ipex::InitIpexBindings(m); }
