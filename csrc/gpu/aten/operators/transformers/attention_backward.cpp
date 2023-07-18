#include <ATen/ATen.h>
#include <utils/DPCPP.h>

#include "sdp_utils.h"
#include "sdp_utils_cpp.h"

namespace at {
namespace AtenIpexTypeXPU {
std::tuple<at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_flash_attention_backward(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const Tensor& cumulative_sequence_length_q,
    const Tensor& cumulative_sequence_length_k,
    const int64_t max_seqlen_batch_q,
    const int64_t max_seqlen_batch_k,
    double dropout_p,
    bool is_causal,
    const int64_t philox_seed,
    const int64_t philox_offset) {
  TORCH_CHECK(
      false,
      "'_scaled_dot_product_flash_attention_backward' hasn't been implemented, we should have falled back to the math path.");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_efficient_attention_backward(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    bool causal,
    bool chunk_grad_outputs) {
  TORCH_CHECK(
      false,
      "'_scaled_dot_product_efficient_attention_backward' hasn't been implemented, we should have falled back to the math path.");
}

} // namespace AtenIpexTypeXPU
} // namespace at
