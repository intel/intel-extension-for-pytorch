#include "ReQuantization.h"

using namespace dnnl;
using namespace at::native;
using namespace xpu::oneDNN;

namespace at {
namespace AtenIpexTypeQuantizedXPU {
Tensor requantize(
    const Tensor& src,
    const double& scale_out,
    const int64_t& zero_point_out) {
  auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
  memory::desc src_md = src_ctx.is_plain() ? memory::desc(
                                                 get_onednn_dims(src),
                                                 get_onednn_dtype(src),
                                                 get_onednn_strides(src))
                                           : src_ctx.meta();
  memory::desc dst_md = src_md;
  auto quantizer = xpu::dpcpp::make_per_tensor_affine_quantizer(
      scale_out, zero_point_out, typeMetaToScalarType(src.options().dtype()));
  auto dst_ = empty_opaque_qtensor(dst_md, c10::nullopt, quantizer);

  auto reorder_attr = xpu::oneDNN::ReorderAttr();
  int mask = 0;
  auto scale_in = src.is_quantized() ? static_cast<float>(src.q_scale()) : 1.f;
  auto requant_scale = static_cast<float>(1.f / (scale_out / scale_in));
  reorder_attr.set_dst_sc_and_zp(mask, {requant_scale}, 0, {zero_point_out});
  xpu::oneDNN::reorder(src, dst_, reorder_attr);

  return dst_;
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
