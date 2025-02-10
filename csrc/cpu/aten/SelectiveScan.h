#pragma once

#include <ATen/Tensor.h>
#include <dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

std::tuple<at::Tensor, at::Tensor> selective_scan(
    const at::Tensor& u,
    const at::Tensor& delta,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const c10::optional<at::Tensor>& D,
    const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& delta_bias,
    bool delta_softplus,
    bool return_last_state);
at::Tensor selective_state_update(
    const at::Tensor& state,
    const at::Tensor& x,
    const at::Tensor& dt,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const c10::optional<at::Tensor>& D,
    const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& dt_bias,
    bool dt_softplus);

using selective_scan_kernel_fn = std::tuple<at::Tensor, at::Tensor> (*)(
    const at::Tensor& u,
    const at::Tensor& delta,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const c10::optional<at::Tensor>& D,
    const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& delta_bias,
    bool delta_softplus,
    bool return_last_state);
using selective_state_update_fn = at::Tensor (*)(
    const at::Tensor& state,
    const at::Tensor& x,
    const at::Tensor& dt,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const c10::optional<at::Tensor>& D,
    const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& dt_bias,
    bool dt_softplus);
IPEX_DECLARE_DISPATCH(selective_scan_kernel_fn, selective_scan_kernel_stub);
IPEX_DECLARE_DISPATCH(
    selective_state_update_fn,
    selective_state_update_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
