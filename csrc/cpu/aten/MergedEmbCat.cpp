#include "MergedEmbCat.h"
#include <ATen/Tensor.h>
#include <torch/all.h>

namespace torch_ipex {
namespace cpu {

using namespace at;

IPEX_DEFINE_DISPATCH(merged_embeddingbag_cat_fw_stub);
IPEX_DEFINE_DISPATCH(qmerged_embeddingbag_cat_fw_stub);

Tensor merged_embeddingbag_cat_forward(
    const TensorList& weights,
    const TensorList& indices,
    const TensorList& offsets,
    const Tensor& dense) {
  return merged_embeddingbag_cat_fw_stub(
      kCPU, weights, indices, offsets, dense);
}

Tensor dil_qmerged_embeddingbag_cat(
    const TensorList& qweights,
    const TensorList& indices,
    const TensorList& offsets,
    const Tensor& qdense,
    double o_scale,
    int64_t o_zp,
    at::ScalarType odtype) {
  return qmerged_embeddingbag_cat_fw_stub(
      kCPU, qweights, indices, offsets, qdense, o_scale);
}
} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "merged_embeddingbag_cat_forward(Tensor[] weights, Tensor[] indices, Tensor[] offsets, Tensor dense) -> Tensor");
  m.impl(
      "merged_embeddingbag_cat_forward",
      c10::DispatchKey::CPU,
      torch_ipex::cpu::merged_embeddingbag_cat_forward);
}

} // namespace