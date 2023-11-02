#include "MergedEmbeddingBag.h"

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(merged_embeddingbag_backward_cpu_kernel_stub);
DEFINE_DISPATCH(merged_embeddingbag_backward_sgd_cpu_kernel_stub);

std::vector<Tensor> merged_embeddingbag_backward_cpu(
    const TensorList& grad_outs_,
    const TensorList& weights,
    const TensorList& indices,
    const TensorList& offsets,
    const int64_t pooling_mode,
    const bool include_last_offsets) {
  /*
   * pointer to merged_embeddingbag_backward_cpu_kernel_impl(
        grad_outs_, weights, offsets, indices, pooling_mode,
   include_last_offsets)
   */
  return merged_embeddingbag_backward_cpu_kernel_stub(
      kCPU,
      grad_outs_,
      weights,
      indices,
      offsets,
      pooling_mode,
      include_last_offsets);
}

void merged_embeddingbag_backward_sgd_cpu(
    const TensorList& grad_outs_,
    TensorList weights,
    const TensorList& indices,
    const TensorList& offsets,
    const int64_t pooling_mode,
    const bool include_last_offsets,
    const TensorList& bf16_trail,
    const double weight_decay,
    const double lr) {
  /*
  pointer to merged_embeddingbag_backward_sgd_cpu_kernel_impl(
      grad_outs_,
      weights,
      indices,
      offsets,
      pooling_mode,
      include_last_offsets,
      bf16_trail,
      weight_decay,
      lr);
  */
  return merged_embeddingbag_backward_sgd_cpu_kernel_stub(
      kCPU,
      grad_outs_,
      weights,
      indices,
      offsets,
      pooling_mode,
      include_last_offsets,
      bf16_trail,
      weight_decay,
      lr);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "merged_embeddingbag_backward_cpu(Tensor[] grad, Tensor[] weight, Tensor[] index, Tensor[] offsets, int pooling_mode, bool include_last_offsets) -> Tensor[]");
  m.impl(
      "merged_embeddingbag_backward_cpu",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::merged_embeddingbag_backward_cpu);
  m.def(
      "merged_embeddingbag_backward_sgd(Tensor[] grad, Tensor[] weight, Tensor[] index, Tensor[] offsets, int pooling_mode, bool include_last, Tensor[] bf16_trail, float weight_decay, float lr) -> ()");
  m.impl(
      "merged_embeddingbag_backward_sgd",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::merged_embeddingbag_backward_sgd_cpu);
}

} // namespace
