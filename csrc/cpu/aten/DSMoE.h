#pragma once

#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>
#include "Linear.h"

namespace torch_ipex {
namespace cpu {

at::Tensor fused_experts(
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    bool,
    bool,
    bool,
    bool,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor);

using fused_experts_fn = at::Tensor (*)(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    bool inplace,
    bool is_vnni,
    bool is_distributed,
    bool is_woq,
    at::Tensor w1_scale,
    at::Tensor w1_zp,
    at::Tensor w2_scale,
    at::Tensor w2_zp);

at::Tensor fused_mlp(
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    bool,
    bool,
    bool,
    bool,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor);

using fused_mlp_fn = at::Tensor (*)(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    bool inplace,
    bool is_vnni,
    bool is_distributed,
    bool is_woq,
    at::Tensor w1_scale,
    at::Tensor w1_zp,
    at::Tensor w2_scale,
    at::Tensor w2_zp);

IPEX_DECLARE_DISPATCH(fused_experts_fn, fused_experts_impl_stub);
IPEX_DECLARE_DISPATCH(fused_mlp_fn, fused_mlp_impl_stub);
} // namespace cpu
} // namespace torch_ipex
