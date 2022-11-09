#include <c10/core/CPUAllocator.h>
#include "MergedEmbeddingBag.h"

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(merged_embeddingbag_backward_cpu_kernel_stub);

std::vector<Tensor> merged_embeddingbag_backward_cpu(
    const std::vector<Tensor>& grad_outs_,
    const Tensor& offsets,
    const std::vector<Tensor>& weights,
    const Tensor& indices_with_row_offset,
    const Tensor& row_offsets,
    const std::vector<int64_t> pooling_modes) {
  /*
   * pointer to merged_embeddingbag_backward_cpu_kernel_impl(
        grad_outs_, offsets, weights, indices_with_row_offset, row_offsets,
   pooling_modes, same_dtype);
   */
  return merged_embeddingbag_backward_cpu_kernel_stub(
      kCPU,
      grad_outs_,
      offsets,
      weights,
      indices_with_row_offset,
      row_offsets,
      pooling_modes);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "merged_embeddingbag_backward_cpu(Tensor[] grad, Tensor offsets, Tensor[] weight, Tensor indices_with_row_offset,  Tensor row_offsets, int[] pooling_modes) -> Tensor[]");
  m.impl(
      "merged_embeddingbag_backward_cpu",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::merged_embeddingbag_backward_cpu);
}

} // namespace
