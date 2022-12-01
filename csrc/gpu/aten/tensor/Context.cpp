#include "Context.h"
#include <QuantizedXPUNativeFunctions.h>
#include <oneDNN/oneDNN.h>
#include <operators/comm/Scalar.h>

using namespace xpu::oneDNN;

namespace at {
namespace AtenIpexTypeXPU {

at::Tensor DPCPPTensorConvertor::to_plain(const at::Tensor& from) {
  if (!is_opaque_tensor(from))
    return from;

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
