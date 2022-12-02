#include <c10/core/CPUAllocator.h>
#include <omp.h>
#include "MergedEmbeddingBag.h"

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(merged_embeddingbag_backward_sgd_cpu_kernel_stub);

void merged_embeddingbag_backward_sgd_cpu(
    const std::vector<Tensor>& grads_y_,
    const Tensor& indices,
    const Tensor& offsets,
    const std::vector<Tensor>& weights,
    const Tensor& indices_with_row_offset,
    const Tensor& row_offsets,
    std::vector<int64_t> pooling_modes,
    const std::vector<Tensor>& bf16_trail,
    double weight_decay,
    double lr) {
  /*
  pointer to merged_embeddingbag_backward_sgd_cpu_kernel_impl(
      grads_y_,
      indices,
      offsets,
      weights,
      indices_with_row_offset,
      row_offsets,
      pooling_modes,
      bf16_trail,
      weight_decay,
      lr);
  */
  return merged_embeddingbag_backward_sgd_cpu_kernel_stub(
      kCPU,
      grads_y_,
      indices,
      offsets,
      weights,
      indices_with_row_offset,
      row_offsets,
      pooling_modes,
      bf16_trail,
      weight_decay,
      lr);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "merged_embeddingbag_backward_sgd(Tensor[] grad, Tensor indices, Tensor offsets, Tensor[] weight, Tensor indices_with_row_offset,  Tensor row_offsets, int[] pooling_modes, Tensor[] bf16_trail, float weight_decay, float lr) -> ()");
  m.impl(
      "merged_embeddingbag_backward_sgd",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::merged_embeddingbag_backward_sgd_cpu);
}

} // namespace
