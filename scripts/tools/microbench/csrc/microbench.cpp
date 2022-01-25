#include <torch/extension.h>
#include "argprint.h"

#define MICROBENCH_DISPATCH_KEY at::DispatchKey::PrivateUse3

#ifdef TORCH_V_1_11
at::Scalar scalar_slow(PyObject* arg) {
  // Zero-dim tensors are converted to Scalars as-is. Note this doesn't
  // currently handle most NumPy scalar types except np.float64.
  if (THPVariable_Check(arg)) {
    return THPVariable_Unpack(arg).item();
  }

  if (THPUtils_checkLong(arg)) {
    return at::Scalar(static_cast<int64_t>(THPUtils_unpackLong(arg)));
  }

  if (PyBool_Check(arg)) {
    return at::Scalar(THPUtils_unpackBool(arg));
  }

  if (PyComplex_Check(arg)) {
    return at::Scalar(THPUtils_unpackComplexDouble(arg));
  }
  return at::Scalar(THPUtils_unpackDouble(arg));
}
#endif

#ifdef TORCH_V_1_7
at::Scalar scalar_slow(PyObject* object) {
  // Zero-dim tensors are converted to Scalars as-is. Note this doesn't
  // currently handle most NumPy scalar types except np.float64.
  if (THPVariable_Check(object)) {
    return ((THPVariable*)object)->cdata.item();
  }

  if (THPUtils_checkLong(object)) {
    return at::Scalar(static_cast<int64_t>(THPUtils_unpackLong(object)));
  }

  if (PyBool_Check(object)) {
    return at::Scalar(THPUtils_unpackBool(object));
  }

  if (PyComplex_Check(object)) {
    return at::Scalar(THPUtils_unpackComplexDouble(object));
  }
  return at::Scalar(THPUtils_unpackDouble(object));
}
#endif

#include "generated.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "enable_verbose",
      []() {
        c10::impl::tls_set_dispatch_key_included(MICROBENCH_DISPATCH_KEY, true);
      },
      "enable microbench verbose");

  m.def(
      "disable_verbose",
      []() {
        c10::impl::tls_set_dispatch_key_included(
            MICROBENCH_DISPATCH_KEY, false);
      },
      "disable microbench verbose");

  MICRO_BENCH_REGISTER
}
