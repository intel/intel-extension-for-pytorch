#include "Context.h"
#include <QuantizedXPUNativeFunctions.h>
#include <oneDNN/oneDNN.h>
#include <operators/comm/Scalar.h>

using namespace xpu::oneDNN;

namespace at {
namespace AtenIpexTypeXPU {

at::Tensor DPCPPTensorConvertor::to_plain(const at::Tensor& from_original) {
  if (!is_opaque_tensor(from_original))
    return from_original;

  auto from = from_original;

  // [watch out] here the dtype in tensor from_original's meta may not be equal
  // to the dtype stored in its context. When doing save, the storage is
  // regarded as pure u8 type then to be pickled down(here is the reorder happen
  // when pickling:
  // https://github.com/pytorch/pytorch/blob/0f4652f4989a2d196f36fe75e5c73cb88dc0800d/torch/serialization.py#L667)

  // Before saving, the block tensor should be converted to plain to make sure
  // the correctness, so the tensor MUST be reconstructed on the given storage
  // with the CORRECT meta dtype.
  // If saving block tensor, here tensor [from_original] is a pure u8 one-dim
  // tensor. Thus, the [from] tensor should be reconstructed with the correct
  // meta dtype associated with the dtype stored in the [from_original]'s
  // context. After reordering, the plain u8 tensor should be recovered.

  // Here is_equal(false) means the dtype in tensor meta is not equal to the
  // dtype in stored context, so the reconstruction is needed for reorder's
  // correctness.
  auto is_equal = check_equality_for_meta_dtype_and_ctx_dtype(from_original);
  auto is_opaque_u8_qtensor = is_opaque_u8(from_original);
  // Case 1:
  //   Tensor ctx has real dtype(like f32,s8) but meta has byte dtype(u8), this
  //   is for pickling tensor. Run following if statement
  // Case 2:
  //  Opaque u8 qtensor has QUInt8 meta but s8 ctx. No need for pickiling it,
  //  bypass following if statement.
  if (!is_equal && !is_opaque_u8_qtensor) {
    // Here use opaqueTypeToScalarType to deduce the meta dtype
    // [from] the context dtype, then reconstruct the tensor [from]
    from = at::empty_like(
        from_original,
        from_original.options().dtype(opaqueTypeToScalarType(from_original)));

    unsafe_get_and_set_data_ptr(from_original, from);
  }

  // use native API to break recursive call resulted by opaque guard in aten itf
  auto to = from.is_quantized()
      ? at::AtenIpexTypeQuantizedXPU::empty_like(
            from,
            c10::nullopt,
            c10::nullopt,
            c10::nullopt,
            c10::nullopt,
            c10::nullopt) // TODO: generate declaration with default arguments
      : at::empty_like(from);
  auto ctx = *(static_cast<DPCPPTensorContext*>(
      from.unsafeGetTensorImpl()->storage().data_ptr().get_context()));
  auto to_meta = ctx.aten_meta();
  auto to_ = share_storage_and_set_strided_as(
      to, to_meta.sizes_, to_meta.strides_, c10::nullopt);
  xpu::oneDNN::reorder(from, to_);

  if (!is_equal && !is_opaque_u8_qtensor) {
    // reconstruct the [to] tensor with the original tensor meta
    to = at::empty_like(from_original);

    // release [to_] context and set it to [to], now the tensor [to] is plain
    // and it has same meta with the tensor [from_original]
    unsafe_release_and_set_data_ptr(to_, to);

    // manually free [from] context
    from.unsafeGetTensorImpl()
        ->storage()
        .unsafeGetStorageImpl()
        ->data_ptr()
        .release_context();
  }

  return to;
}

at::Tensor DPCPPTensorConvertor::to_plain_(at::Tensor& from) {
  if (!is_opaque_tensor(from))
    return from;

  auto to = to_plain(from);

  auto ctx = DPCPPTensorContext::get_tensor_ctx(to);
  DPCPPTensorContext::set_tensor_ctx(from, std::move(ctx));
  return from;
}

} // namespace AtenIpexTypeXPU
} // namespace at
