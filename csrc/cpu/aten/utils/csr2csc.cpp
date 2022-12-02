#include "csr2csc.h"
#include "radix_sort.h"

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(sort_based_batched_csr2csc_opt_kernel_stub);

void sort_based_batched_csr2csc_opt(
    BatchedHyperCompressedSparseColumn& batched_csc,
    int B,
    const Tensor& offsets,
    const Tensor& indices,
    std::vector<int64_t> pooling_modes,
    int64_t max_embeddings) {
  /*
  pointer to sort_based_batched_csr2csc_opt_kernel_impl(
      batched_csc, B, offsets, indices, pooling_modes, max_embeddings);
  */
  sort_based_batched_csr2csc_opt_kernel_stub(
      kCPU, batched_csc, B, offsets, indices, pooling_modes, max_embeddings);
}

} // namespace cpu
} // namespace torch_ipex
