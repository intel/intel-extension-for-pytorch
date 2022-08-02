#include <runtime/Utils.h>

#include <core/Device.h>
#include <core/Generator.h>
#include <torch/csrc/python_headers.h>
#include <mutex>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>
namespace xpu {
namespace dpcpp {

static bool run_yet = false;

void lazy_init() {
  // Here is thread safety. There are two reasons for this:
  // 1. avoid circular calls.
  // 2. avoid GIL's overhead.
  if (run_yet)
    return;

  pybind11::gil_scoped_acquire g;
  // Protected by the GIL.  We don't use call_once because under ASAN it
  // has a buggy implementation that deadlocks if an instance throws an
  // exception.  In any case, call_once isn't necessary, because we
  // have taken a lock.
  if (!run_yet) {
    run_yet = true; // set run_yet TRUE before python API's execution to avoid
                    // circular calls.
    auto module =
        THPObjectPtr(PyImport_ImportModule("intel_extension_for_pytorch.xpu"));
    if (!module)
      throw python_error();
    auto res =
        THPObjectPtr(PyObject_CallMethod(module.get(), "_lazy_init", ""));
    if (!res) {
      throw python_error();
      run_yet =
          false; // if python API's execution fails, restore tun_yet to FALSE.
    }
  }
}

void set_run_yet_variable_to_false() {
  run_yet = false;
}

} // namespace dpcpp
} // namespace xpu
