#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <torch/library.h>
namespace at {
namespace AtenIpexTypeXPU {

Tensor asynchronous_complete_cumsum_xpu(const Tensor& t_in) {
  TORCH_CHECK(t_in.is_contiguous());
  TORCH_CHECK(t_in.dtype() == at::kInt || t_in.dtype() == at::kLong);
  TORCH_CHECK(t_in.dim() == 1 || t_in.dim() == 2);
  Tensor t_out;
  if (t_in.dim() == 1) {
    t_out = at::zeros({t_in.numel() + 1}, t_in.options());
    auto r_out = t_out.slice(0, 1);
    at::cumsum_out(r_out, t_in, 0);
  } else {
    t_out = at::zeros({t_in.size(0), t_in.size(1) + 1}, t_in.options());
    auto r_out = t_out.slice(1, 1);
    at::cumsum_out(r_out, t_in, 1);
  }
  return t_out;
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
  m.impl(
      "asynchronous_complete_cumsum",
      torch::dispatch(
          c10::DispatchKey::XPU,
          TORCH_FN(at::AtenIpexTypeXPU::asynchronous_complete_cumsum_xpu)));
}
} // namespace
