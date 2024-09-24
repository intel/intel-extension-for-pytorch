#include <ATen/native/CPUFallback.h>

namespace {

static bool DEBUG_XPU_FALLBACK = false;

void check_input_devices(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  auto& schema_args = op.schema().arguments();
  const auto num_arguments = schema_args.size();
  auto arguments = torch::jit::last(stack, num_arguments);

  if (!arguments[0].isTensor() || !arguments[1].isTensor())
    return;

  auto& self = arguments[0].toTensor();
  auto& other = arguments[1].toTensor();

  if (!self.defined() || !other.defined())
    return;

  TORCH_CHECK(self.device() == other.device(),
      "Expected all tensors to be on the same device, but found at least two devices, ",
      self.device(), " and ", other.device(), "!");
}

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

  // dot and vdot require input device check on all devices except CPU.
  // The original check was generated in the wrapper.
  auto op_aten_name = toString(op.schema().operator_name());
  if (op_aten_name == "aten::dot" || op_aten_name == "aten::vdot") {
    check_input_devices(op, stack);
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
      "vdot",
  };

  for (auto& op_name : fallback_list) {
    m.impl(
        op_name.c_str(),
        torch::CppFunction::makeFromBoxedFunction<&xpu_fallback_impl>());
  }
}

} // namespace
