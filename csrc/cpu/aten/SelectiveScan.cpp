#include <ATen/ATen.h>

#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/record_function.h>
#include <c10/util/irange.h>

#include "SelectiveScan.h"
#include "utils/library.h"

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(selective_scan_kernel_stub);
IPEX_DEFINE_DISPATCH(selective_state_update_kernel_stub);

/**
 * Does selective scan algorithm in Mamba Paper.
 * Paper: https://arxiv.org/abs/2312.00752
 * Official Python Implementation:
 * selective_scan_ref:
 * https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L113
 * @param u: (batch, dim, len) or (batch, len, dim)
 * @param delta: same shape as u
 * @param A: (dim, dstate) or (dstate, dim)
 * @param B: (batch, dstate, len) or (batch, dstate, 2len) or (battch, ngroups,
 * dstate, len)
 * @param C: (batch, dstate, len) or (batch, dstate, 2len) or (battch, ngroups,
 * dstate, len)
 * @param D: (dim,) or None
 * @param z: (batch, dim, len) or None
 * @param delta_bias: (dim,) or None
 * @param delta_softplus: bool
 * @param return_last_state: bool
 * @return: out: (batch, dim, len), last_state: (batch, dim, dstate)
 */
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
    bool return_last_state) {
  RECORD_FUNCTION("selective_scan_fn", c10::ArrayRef<c10::IValue>({}));
  return selective_scan_kernel_stub(
      kCPU,
      u,
      delta,
      A,
      B,
      C,
      D,
      z,
      delta_bias,
      delta_softplus,
      return_last_state);
}

/**
 * Official Python Implementation:
 * selective_state_update_ref:
 * https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/selective_state_update.py#L219
 * @param state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
 * @param x: (batch, dim) or (batch, nheads, dim)
 * @param dt: (batch, dim) or (batch, nheads, dim)
 * @param A: (dim, dstate) or (nheads, dim, dstate) or (dstate, dim) or (nheads,
 * dstate, dim)
 * @param B: (batch, dstate) or (batch, ngroups, dstate)
 * @param C: (batch, dstate) or (batch, ngroups, dstate)
 * @param D: (dim,) or (nheads, dim) or None
 * @param z: (batch, dim) or (batch, nheads, dim) or None
 * @param dt_bias: (dim,) or (nheads, dim) or None
 * @param dt_softplus: bool
 * @return: out: (batch, dim) or (batch, nheads, dim)
 */
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
    bool dt_softplus) {
  RECORD_FUNCTION("selective_state_update", c10::ArrayRef<c10::IValue>({}));
  return selective_state_update_kernel_stub(
      kCPU, state, x, dt, A, B, C, D, z, dt_bias, dt_softplus);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

IPEX_TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "selective_scan_fn(Tensor u, Tensor delta, Tensor A, Tensor B, Tensor C, Tensor? D, Tensor? z, Tensor? delta_bias, bool delta_softplus, bool return_last_state) -> (Tensor, Tensor)");
  m.impl(
      "selective_scan_fn",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::selective_scan);
  m.def(
      "selective_state_update(Tensor state, Tensor x, Tensor dt, Tensor A, Tensor B, Tensor C, Tensor? D, Tensor? z, Tensor? dt_bias, bool dt_softplus) -> (Tensor)");
  m.impl(
      "selective_state_update",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::selective_state_update);
}

} // namespace
