#include "GradScaler.h"
namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(_amp_foreach_non_finite_check_and_unscale_cpu_stub);
IPEX_DEFINE_DISPATCH(_amp_update_scale_cpu_stub);

void _amp_foreach_non_finite_check_and_unscale_cpu_(
    std::vector<at::Tensor> scaled_grads,
    at::Tensor& found_inf,
    const at::Tensor& inv_scale) {
  _amp_foreach_non_finite_check_and_unscale_cpu_stub(
      found_inf.device().type(), scaled_grads, found_inf, inv_scale);
}

at::Tensor& _amp_update_scale_cpu_(
    at::Tensor& current_scale,
    at::Tensor& growth_tracker,
    const at::Tensor& found_inf,
    double growth_factor,
    double backoff_factor,
    int64_t growth_interval) {
  return _amp_update_scale_cpu_stub(
      growth_tracker.device().type(),
      current_scale,
      growth_tracker,
      found_inf,
      growth_factor,
      backoff_factor,
      growth_interval);
}

} // namespace cpu
} // namespace torch_ipex
