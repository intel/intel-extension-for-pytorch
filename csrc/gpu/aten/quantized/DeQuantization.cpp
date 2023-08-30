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
  if (zero_point != 0)
    rattr.set_src_zp_mask(mask);

  Tensor rtensor_ = at::empty(qtensor.sizes(), rtensor.options());
  auto quant_base = fetch_cached_quantizer_base(scale, zero_point);
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
  int num_channels = qtensor.size(axis);
  ReorderAttr rattr;
  int mask = (1 << axis);
  // See [Note: Scale setting for reorder]
  rattr.set_src_sc_mask(mask);
  // TODO [Asymmetric]:
  // Check zero point here
  rattr.set_src_zp_mask(mask);
  Tensor rtensor_ = at::empty(qtensor.sizes(), rtensor.options());
  xpu::oneDNN::quantized_reorder(
      qtensor,
      rtensor_,
      (float*)scales.data_ptr(),
      (int32_t*)zero_points.data_ptr(),
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
