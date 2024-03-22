#include "ReQuantization.h"
#include <tensor/Tensor.h>

using namespace dnnl;
using namespace at::native;
using namespace torch_ipex::xpu::oneDNN;

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
  auto quantizer = dpcpp_make_per_tensor_affine_quantizer(
      scale_out, zero_point_out, typeMetaToScalarType(src.options().dtype()));
  auto dst_ = empty_opaque_qtensor(dst_md, c10::nullopt, quantizer);

  auto reorder_attr = torch_ipex::xpu::oneDNN::ReorderAttr();
  int mask = 0;
  auto sc_in = src.q_scale();
  int32_t zp_in = src.q_zero_point();
  auto quant_base_src = fetch_cached_quantizer_base(sc_in, zp_in);
  auto src_zp_ptr = (zp_in == 0) ? nullptr : quant_base_src.zero_point_ptr();

  auto quant_base_dst = fetch_cached_quantizer_base(scale_out, zero_point_out);
  auto dst_zp_ptr =
      (zero_point_out == 0) ? nullptr : quant_base_dst.zero_point_ptr();

  if (zp_in != 0)
    reorder_attr.set_src_zp_mask(mask);
  if (zero_point_out != 0)
    reorder_attr.set_dst_zp_mask(mask);
  reorder_attr.set_dst_sc_mask(mask);
  reorder_attr.set_src_sc_mask(mask);
  torch_ipex::xpu::oneDNN::quantized_reorder(
      src,
      dst_,
      quant_base_src.scale_ptr(),
      src_zp_ptr,
      quant_base_dst.scale_ptr(),
      dst_zp_ptr,
      {1},
      {1},
      reorder_attr);

  return dst_;
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
