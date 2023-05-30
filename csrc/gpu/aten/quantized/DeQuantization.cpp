#include <ATen/ATen.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/util/Exception.h>

#include <oneDNN/oneDNN.h>
#include <quantized/DeQuantization.h>

using namespace dnnl;
using namespace at::native;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;
using namespace at::AtenIpexTypeQuantizedXPU;

namespace at {
namespace AtenIpexTypeXPU {

Tensor dequantize_tensor_per_tensor_affine(
    Tensor& rtensor,
    const Tensor& qtensor,
    double scale,
    int64_t zero_point) {
  ReorderAttr rattr = ReorderAttr();
  int mask = 0;
  rattr.set_src_sc_mask(mask);
  bool asymmetric = false;
  if (asymmetric && zero_point != 0)
    rattr.set_src_zp_mask(mask);

  Tensor rtensor_ = at::empty(qtensor.sizes(), rtensor.options());
  // TODO: Revmoe WA after asymmetric
  if ((!is_opaque_u8(qtensor)) && qtensor.scalar_type() == kQUInt8) {
    scale = scale / 2.f;
  }
  auto quant_base = fetch_cached_quantizer_base(scale, 0);
  xpu::oneDNN::quantized_reorder(
      qtensor,
      rtensor_,
      (float*)quant_base.scale_ptr(),
      (int32_t*)quant_base.zero_point_ptr(),
      /*dst_scale=*/nullptr,
      /*dst_zero_point=*/nullptr,
      {1},
      {1},
      rattr);
  return rtensor_;
}

Tensor dequantize_tensor_per_channel_affine(
    Tensor& rtensor,
    const Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  auto q_eng =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  memory::dims q_dims = qtensor.dim() == 4 ? memory::dims(
                                                 {qtensor.size(0),
                                                  qtensor.size(1),
                                                  qtensor.size(2),
                                                  qtensor.size(3)})
      : qtensor.dim() == 2 ? memory::dims({qtensor.size(0), qtensor.size(1)})
                           : memory::dims({qtensor.size(0)});
  memory::data_type q_dt = get_onednn_dtype(qtensor);
  memory::format_tag q_fmt = qtensor.dim() == 4 ? memory::format_tag::nchw
      : qtensor.dim() == 2                      ? memory::format_tag::nc
                                                : memory::format_tag::x;
  memory::desc q_md = memory::desc(q_dims, q_dt, q_fmt);
  memory q_m = dpcpp_onednn_memory(q_md, q_eng, qtensor.data_ptr());

  memory::dims r_dims = q_dims;
  memory::data_type r_dt = get_onednn_dtype(rtensor);
  memory::format_tag r_fmt = q_fmt;
  engine r_eng = q_eng;
  memory::desc r_md = memory::desc(r_dims, r_dt, r_fmt);

  int num_channels = qtensor.size(axis);
  ReorderAttr rattr;
  int mask = (1 << axis);
  // See [Note: Scale setting for reorder]
  rattr.set_src_sc_mask(mask);

  Tensor rtensor_ = empty_opaque_tensor(r_md, rtensor.options(), c10::nullopt);
  xpu::oneDNN::quantized_reorder(
      qtensor,
      rtensor_,
      (float*)scales.data_ptr(),
      /*src_zero_point*/ nullptr,
      /*dst_scale=*/nullptr,
      /*dst_zero_point=*/nullptr,
      /*scale_zp_sz*/ scales.sizes().vec(),
      /*scale_zp_st*/ scales.strides().vec(),
      rattr);

  return rtensor_;
}

Tensor dequantize(const Tensor& self) {
  if (!self.is_quantized()) {
    return self;
  }
  auto qtensor = static_cast<QTensorImpl*>(self.unsafeGetTensorImpl());
  return qtensor->quantizer()->dequantize(self);
}

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {
Tensor dequantize(const Tensor& self) {
  if (!self.is_quantized()) {
    return self;
  }
  auto qtensor = static_cast<QTensorImpl*>(self.unsafeGetTensorImpl());
  return qtensor->quantizer()->dequantize(self);
}
} // namespace AtenIpexTypeQuantizedXPU

} // namespace at
