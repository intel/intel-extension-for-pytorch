#pragma once

#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>

enum BatchStatus : bool {
  Finished = true,
  UnFinished = false,
};

namespace torch_ipex {
namespace cpu {

namespace {

bool rnnt_update_batch_kernel_impl(
    const at::Tensor& k,
    const at::Tensor& out_lens,
    at::Tensor label_col,
    at::Tensor symbols_added,
    at::Tensor time_idxs,
    at::Tensor blankness_out,
    at::Tensor blankvec_out,
    at::Tensor not_blank_out,
    at::Tensor label_to_put_out,
    at::Tensor label_tensor_out,
    at::Tensor label_for_next_loop_out,
    at::Tensor hidden_0,
    at::Tensor hidden_1,
    const at::Tensor& hidden_prime_0,
    const at::Tensor& hidden_prime_1,
    at::Tensor x,
    at::Tensor f,
    int64_t max_symbols,
    int64_t blank_id,
    int64_t batch_size,
    int64_t _SOS,
    int64_t max_len);
}

using rnnt_update_batch_kernel_fn = bool (*)(
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor,
    at::Tensor,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t);
IPEX_DECLARE_DISPATCH(
    rnnt_update_batch_kernel_fn,
    rnnt_update_batch_kernel_stub);

} // namespace cpu
} // namespace torch_ipex