#include <ATen/AccumulateType.h>
#include <ATen/Tensor.h>
#include <dyndisp/DispatchStub.h>
#include <torch/all.h>

namespace torch_ipex {
namespace cpu {

using namespace at;

Tensor dil_qmerged_embeddingbag_cat(
    const TensorList& weights,
    const TensorList& indices,
    const TensorList& offsets,
    const Tensor& qdense,
    double o_scale,
    int64_t o_zp,
    at::ScalarType odtype);

namespace {

Tensor merged_embedding_cat_fw_impl(
    const TensorList& weights,
    const TensorList& indices,
    const TensorList& offsets,
    const Tensor& dense);

Tensor qmerged_embedding_cat_fw_impl(
    const TensorList& qweights,
    const TensorList& indices,
    const TensorList& offsets,
    const Tensor& qdense,
    double o_scale);

} // namespace

using merged_embeddingbag_cat_fw_fn = Tensor (*)(
    const TensorList&,
    const TensorList&,
    const TensorList&,
    const Tensor&);

using qmerged_embeddingbag_cat_fw_fn = Tensor (*)(
    const TensorList&,
    const TensorList&,
    const TensorList&,
    const Tensor&,
    double o_scale);

IPEX_DECLARE_DISPATCH(
    merged_embeddingbag_cat_fw_fn,
    merged_embeddingbag_cat_fw_stub);

IPEX_DECLARE_DISPATCH(
    qmerged_embeddingbag_cat_fw_fn,
    qmerged_embeddingbag_cat_fw_stub);

} // namespace cpu
} // namespace torch_ipex