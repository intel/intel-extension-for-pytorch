#include <ATen/native/CPUFallback.h>

namespace {

static bool DEBUG_XPU_FALLBACK = false;

static void xpu_fallback_impl(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  if (!DEBUG_XPU_FALLBACK) {
    TORCH_WARN_ONCE(
        "Aten Op fallback from XPU to CPU happends.",
        " This may have performance implications.",
        " If need debug the fallback ops please set environment variable `PYTORCH_DEBUG_XPU_FALLBACK=1` ");
  } else {
    TORCH_WARN(
        "The operator '",
        op.schema().operator_name(),
        " on the XPU backend is falling back to run on the CPU.");
  }

  at::native::cpu_fallback(op, stack, true);
}

TORCH_LIBRARY_IMPL(aten, XPU, m) {
  static const char* debug_xpu_fallback = getenv("PYTORCH_DEBUG_XPU_FALLBACK");
  if (!debug_xpu_fallback || std::stoi(debug_xpu_fallback) == 0) {
    DEBUG_XPU_FALLBACK = false;
  } else {
    DEBUG_XPU_FALLBACK = true;
  }

  std::vector<std::string> fallback_list = {
      "_linalg_eigvals",
      "cholesky",
      "cholesky.out",
      "cholesky_inverse",
      "cholesky_inverse.out",
      "dot",
      "geqrf.a",
      "linalg_eig.out",
      "linalg_eigh.eigvals",
      "linalg_eigvals.out",
      "linalg_householder_product.out",
      "linalg_solve_triangular.out",
      "ormqr.out",
  };

  for (auto& op_name : fallback_list) {
    m.impl(
        op_name.c_str(),
        torch::CppFunction::makeFromBoxedFunction<&xpu_fallback_impl>());
  }
}

} // namespace
