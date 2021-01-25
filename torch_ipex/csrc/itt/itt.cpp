#include <torch/csrc/utils/pybind.h>
#include <itt/itt_wrapper.h>

void IttBindings_init(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto itt = m.def_submodule("_itt", "VTune ITT bindings");
  itt.def("rangePush", itt_range_push);
  itt.def("rangePop", itt_range_pop);
  itt.def("mark", itt_mark);
}
