#include <gpu/Module.h>
#include <py_init.h>

namespace py = pybind11;


static std::vector<PyMethodDef> methods;


void THDPStream_init(PyObject *module);
void torch_ipex_init(pybind11::module &m) {
  m.doc() = "PyTorch Extension for Intel XPU";
  init_module(m);
}