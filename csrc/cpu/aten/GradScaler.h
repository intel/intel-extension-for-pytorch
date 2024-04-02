#pragma once
#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>

namespace torch_ipex {

namespace cpu {

TORCH_API void _amp_foreach_non_finite_check_and_unscale_cpu_(
    std::vector<at::Tensor> scaled_grads,
    at::Tensor& found_inf,
    const at::Tensor& inv_scale);

TORCH_API at::Tensor& _amp_update_scale_cpu_(
    at::Tensor& current_scale,
    at::Tensor& growth_tracker,
    const at::Tensor& found_inf,
    double growth_factor,
    double backoff_factor,
    int64_t growth_interval);

using _amp_foreach_non_finite_check_and_unscale_cpu__fn =
    void (*)(std::vector<at::Tensor>, at::Tensor&, const at::Tensor&);

using _amp_update_scale_cpu__fn = at::
    Tensor& (*)(at::Tensor&, at::Tensor&, const at::Tensor&, double, double, int64_t);

IPEX_DECLARE_DISPATCH(
    _amp_foreach_non_finite_check_and_unscale_cpu__fn,
    _amp_foreach_non_finite_check_and_unscale_cpu_stub);
IPEX_DECLARE_DISPATCH(_amp_update_scale_cpu__fn, _amp_update_scale_cpu_stub);

} // namespace cpu
} // namespace torch_ipex
