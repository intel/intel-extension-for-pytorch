#pragma once
#include <math.h>

#include <ATen/DeviceGuard.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

namespace torch_ipex {
namespace autocast {

void _amp_foreach_non_finite_check_and_unscale_cpu_(
    std::vector<at::Tensor> scaled_grads,
    at::Tensor& found_inf,
    const at::Tensor& inv_scale);

at::Tensor& _amp_update_scale_cpu_(
    at::Tensor& current_scale,
    at::Tensor& growth_tracker,
    const at::Tensor& found_inf,
    double growth_factor,
    double backoff_factor,
    int64_t growth_interval);
} // namespace autocast
} // namespace torch_ipex
