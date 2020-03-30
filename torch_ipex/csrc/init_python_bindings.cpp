#include "init_python_bindings.h"
#include "version.h"

#include <c10/core/Device.h>
#include <c10/util/Optional.h>

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "aten_ipex_type.h"
#include "auto_opt_config.h"

namespace torch_ipex {
namespace {

py::object GetRevisions() {
  auto py_dict = py::dict();
  py_dict["ipex"] = std::string(IPEX_GITREV);
  py_dict["torch"] = std::string(TORCH_GITREV);
  return py_dict;
}

void setAutoDNNL(bool val) {
  AutoOptConfig::singleton().set_auto_dnnl(val);
}

void InitIpeModuleBindings(py::module m) {
  m.def("_initialize_aten_bindings",
        []() { AtenIpexType::InitializeAtenBindings(); });
  m.def("_get_git_revs", []() { return GetRevisions(); });
  m.def("enable_auto_dnnl", []() { setAutoDNNL(true); });
  m.def("disable_auto_dnnl", []() { setAutoDNNL(false); });
}

}  // namespace

void InitIpeBindings(py::module m) { InitIpeModuleBindings(m); }

}  // namespace torch_ipex

PYBIND11_MODULE(_torch_ipex, m) { torch_ipex::InitIpeBindings(m); }
