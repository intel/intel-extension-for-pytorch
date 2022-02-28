#include "Context.h"
#include <oneDNN/oneDNN.h>
#include <operators/comm/Scalar.h>

using namespace xpu::oneDNN;

namespace at {
namespace AtenIpexTypeXPU {

at::Tensor DPCPPTensorConvertor::to_plain(const at::Tensor& from) {
  if (!is_opaque_tensor(from))
    return from;

  auto ctx = *(static_cast<DPCPPTensorContext*>(
      from.unsafeGetTensorImpl()->storage().data_ptr().get_context()));

  // case-1. int32 opaque tensor in plain fmt
  if (from.scalar_type() == at::ScalarType::Long) {
    mem_desc_t opaque_md = {ctx.meta().data};
    // FIXME: to decide AB or BA plain format
    mem_desc_t plain_md = {ctx.dims(), ctx.dtype(), ctx.get_plain_tag()};
    if (opaque_md == plain_md) {
      Tensor to = at::empty(ctx.dims(), from.options(), c10::nullopt);
      dtype_convert_by_scalar(
          to.data_ptr<int64_t>(), (int32_t*)from.data_ptr(), from.numel());
      return to;
    }
  }

  auto options = from.options();
  if (from.scalar_type() == at::ScalarType::Long)
    options = options.dtype(kInt);

  // reorder to plain based on current shape
  auto to = !from.is_quantized()
      ? at::AtenIpexTypeXPU::empty(ctx.dims(), options, c10::nullopt)
      : at::AtenIpexTypeXPU::new_qtensor(ctx.dims(), options, from.quantizer());
  xpu::oneDNN::reorder(from, to);

  // permute shape to original shape
  if (!ctx.permution().empty())
    to = at::native::permute(to, IntArrayRef(ctx.permution())).contiguous();

  // group convolution case 5D(oneDNN) -> 4D(PyTorch)
  if (from.ndimension() != ctx.dims().size()) {
    to = to.reshape(from.sizes());
  }

  // case-2. int32 opaque tensor in block fmt
  // 1. convert to plain 2. copy to int64
  if (from.scalar_type() == at::ScalarType::Long) {
    Tensor to_ = at::empty(ctx.dims(), from.options(), c10::nullopt);
    dtype_convert_by_scalar(
        to_.data_ptr<int64_t>(), to.data_ptr<int32_t>(), to.numel());
    to = to_;
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
